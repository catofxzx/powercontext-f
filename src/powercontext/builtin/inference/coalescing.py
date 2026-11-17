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
from dataclasses import dataclass, field
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
    cached: dict[str, tuple[float, ...]] = field(default_factory=dict)
    waiters: list[tuple[str, asyncio.Future[tuple[float, ...]]]] = field(default_factory=list)
    future: asyncio.Future[EmbeddingResult] | None = None


class CoalescingEmbeddingModel:
    """Coalesce concurrent Embedding batches and cache successful vectors.

    Callers that need the same text share one provider request: a text that is
    already running is never sent to the provider twice, so a late caller joins
    the work that is already in flight. Successful vectors are kept in a bounded
    LRU cache so later calls reuse them without issuing another provider
    request. Cancelling one caller never cancels the shared provider work other
    callers wait for.
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
        self._inflight: dict[str, asyncio.Future[tuple[float, ...]]] = {}
        self._unassigned: list[str] = []
        self._batch_of_text: dict[str, int] = {}
        self._batch_usage: dict[int, InferenceUsage] = {}
        self._batch_counter = 0
        self._pending: list[_PendingRequest] = []
        self._flush_task: asyncio.Task[None] | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return exactly one ordered vector for each input text."""

        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        loop = asyncio.get_running_loop()
        request = _PendingRequest(texts=ordered, future=loop.create_future())
        for text in dict.fromkeys(ordered):
            if text in self._cache:
                request.cached[text] = self._cache[text]
                self._touch(text)
                continue
            future = self._inflight.get(text)
            if future is None:
                future = loop.create_future()
                self._inflight[text] = future
                self._unassigned.append(text)
            request.waiters.append((text, future))
        self._pending.append(request)
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush())
        return await request.future

    async def _flush(self) -> None:
        """Compute the texts that still need a vector and resolve their callers."""

        try:
            while True:
                await self._wait_window()
                if not self._unassigned and not self._pending:
                    break
                await self._process()
                if not self._unassigned and not self._pending:
                    break
        finally:
            self._flush_task = None
            if self._pending or self._unassigned:
                self._flush_task = asyncio.create_task(self._flush())

    async def _wait_window(self) -> None:
        if self._coalesce_seconds > 0:
            await asyncio.sleep(self._coalesce_seconds)
        else:
            await asyncio.sleep(0)

    async def _process(self) -> None:
        texts = list(self._unassigned)
        self._unassigned.clear()
        batches = [
            tuple(texts[index : index + self._max_batch_size])
            for index in range(0, len(texts), self._max_batch_size)
        ]
        computed: dict[str, tuple[float, ...]] = {}
        failures: dict[str, Exception] = {}
        for batch in batches:
            self._batch_counter += 1
            batch_id = self._batch_counter
            try:
                result = await self._model.embed(batch)
            except Exception as error:  # noqa: BLE001 - the provider error belongs to this batch only
                self._fail(batch, error, failures)
                continue
            if len(result.vectors) != len(batch):
                self._fail(
                    batch,
                    ValueError(f"embedding provider returned {len(result.vectors)} vectors for {len(batch)} texts"),  # noqa: TRY003
                    failures,
                )
                continue
            self._batch_usage[batch_id] = result.usage
            for text, vector in zip(batch, result.vectors, strict=True):
                computed[text] = vector
                self._batch_of_text[text] = batch_id
                self._store(text, vector)
                self._release(text, vector)
        requests = list(self._pending)
        self._pending.clear()
        for request in requests:
            if any(not future.done() for _text, future in request.waiters):
                self._pending.append(request)
                continue
            self._resolve(request, computed, failures)

    def _fail(self, batch: tuple[str, ...], error: Exception, failures: dict[str, Exception]) -> None:
        for text in batch:
            failures[text] = error
            self._release(text, error)

    def _release(self, text: str, result: tuple[float, ...] | Exception) -> None:
        future = self._inflight.pop(text, None)
        if future is None or future.done():
            return
        if isinstance(result, Exception):
            future.set_exception(result)
        else:
            future.set_result(result)

    def _resolve(
        self,
        request: _PendingRequest,
        computed: dict[str, tuple[float, ...]],
        failures: dict[str, Exception],
    ) -> None:
        if request.future is None or request.future.done():
            return
        vectors: list[tuple[float, ...]] = []
        for text in request.texts:
            cached = request.cached.get(text)
            if cached is not None:
                vectors.append(cached)
                continue
            vector = computed.get(text)
            if vector is None:
                error = failures.get(text)
                if error is None:
                    for waited, future in request.waiters:
                        if waited != text:
                            continue
                        if future.done() and future.exception() is not None:
                            error = future.exception()
                            break
                        if future.done():
                            vector = future.result()
                            break
                if vector is None:
                    request.future.set_exception(
                        error if error is not None else ValueError("embedding vector missing for one input text")  # noqa: TRY003
                    )
                    return
            vectors.append(vector)
        batch_ids = sorted({self._batch_of_text[text] for text, _ in request.waiters if text in self._batch_of_text})
        request.future.set_result(
            EmbeddingResult(
                vectors=tuple(vectors),
                usage=_merge_usage([self._batch_usage[batch_id] for batch_id in batch_ids]),
            )
        )

    def _store(self, text: str, vector: tuple[float, ...]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            evicted, _ = self._cache.popitem(last=False)
            self._batch_of_text.pop(evicted, None)

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
