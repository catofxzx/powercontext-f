# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Share Embedding work across concurrent callers and reuse cached vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.protocols import EmbeddingModel

__all__ = ["CoalescingEmbeddingModel"]


@dataclass(slots=True)
class _PendingRequest:
    """One caller waiting for the vectors of its own ordered text batch."""

    texts: tuple[str, ...]
    future: asyncio.Future[EmbeddingResult]


class CoalescingEmbeddingModel:
    """Coalesce concurrent Embedding batches and cache successful vectors.

    Callers that need the same text inside one coalescing window share a single
    provider request. Successful vectors are kept in a bounded LRU cache so later
    calls reuse them without issuing another provider request. Cancelling one
    caller never cancels the shared provider work other callers wait for.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number of milliseconds")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self.profile: EmbeddingProfile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._pending: list[_PendingRequest] = []
        self._flush_task: asyncio.Task[None] | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return exactly one ordered vector for each input text."""

        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        request = _PendingRequest(texts=ordered, future=asyncio.get_running_loop().create_future())
        self._pending.append(request)
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())
        return await request.future

    async def _flush(self) -> None:
        """Collect concurrent callers and resolve every request that waits on them."""

        try:
            while True:
                await self._wait_window()
                requests = [request for request in self._pending if not request.future.done()]
                self._pending.clear()
                if not requests:
                    break
                await self._process(requests)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.create_task(self._flush())

    async def _wait_window(self) -> None:
        if self._coalesce_seconds > 0:
            await asyncio.sleep(self._coalesce_seconds)
        else:
            await asyncio.sleep(0)

    async def _process(self, requests: list[_PendingRequest]) -> None:
        cached = dict(self._cache)
        needed: list[str] = []
        seen: set[str] = set()
        for request in requests:
            for text in request.texts:
                if text in seen or text in cached:
                    continue
                seen.add(text)
                needed.append(text)
        batches = [
            tuple(needed[index : index + self._max_batch_size])
            for index in range(0, len(needed), self._max_batch_size)
        ]
        outcomes: list[tuple[EmbeddingResult | None, Exception | None]] = []
        vectors: dict[str, tuple[float, ...]] = {}
        for batch in batches:
            try:
                result = await self._model.embed(batch)
            except Exception as error:  # noqa: BLE001 - the provider error belongs to this batch only
                outcomes.append((None, error))
                continue
            if len(result.vectors) != len(batch):
                outcomes.append(
                    (
                        None,
                        ValueError(
                            f"embedding provider returned {len(result.vectors)} vectors for {len(batch)} texts"  # noqa: TRY003
                        ),
                    )
                )
                continue
            outcomes.append((result, None))
            for text, vector in zip(batch, result.vectors, strict=True):
                vectors[text] = vector
                self._store(text, vector)
        batch_index: dict[str, int] = {}
        for index, batch in enumerate(batches):
            for text in batch:
                batch_index[text] = index
        for request in requests:
            self._deliver(request, cached, vectors, batch_index, outcomes)

    def _deliver(
        self,
        request: _PendingRequest,
        cached: dict[str, tuple[float, ...]],
        vectors: dict[str, tuple[float, ...]],
        batch_index: dict[str, int],
        outcomes: list[tuple[EmbeddingResult | None, Exception | None]],
    ) -> None:
        if request.future.done():
            return
        indexes = {batch_index[text] for text in request.texts if text in batch_index}
        for index in sorted(indexes):
            result, error = outcomes[index]
            if error is not None:
                request.future.set_exception(error)
                return
            if result is None:
                request.future.set_exception(ValueError("embedding provider returned no result"))  # noqa: TRY003
                return
        resolved: list[tuple[float, ...]] = []
        for text in request.texts:
            if text in vectors:
                resolved.append(vectors[text])
                continue
            vector = cached.get(text)
            if vector is None:
                request.future.set_exception(ValueError("embedding vector missing for one input text"))  # noqa: TRY003
                return
            self._touch(text)
            resolved.append(vector)
        usages = [outcomes[index][0].usage for index in sorted(indexes) if outcomes[index][0] is not None]
        request.future.set_result(
            EmbeddingResult(vectors=tuple(resolved), usage=_merge_usage(usages))
        )

    def _store(self, text: str, vector: tuple[float, ...]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _touch(self, text: str) -> None:
        if text in self._cache:
            self._cache.move_to_end(text)


def _merge_usage(usages: list[InferenceUsage]) -> InferenceUsage:
    """Sum the usage of the distinct provider batches one caller waited for."""

    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_merge_optional([usage.input_tokens for usage in usages]),
        output_tokens=_merge_optional([usage.output_tokens for usage in usages]),
    )


def _merge_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)
