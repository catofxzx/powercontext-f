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

"""Score coarse Memory candidates with one offline HTTP ranking service."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

HTTP_MEMORY_RERANKER_POLICY_ID = "http-cross-encoder-v1"
_SCORE_FIELDS = frozenset({"rank", "score"})
_RESPONSE_FIELDS = frozenset({"scores"})

__all__ = ["HTTPMemoryReranker", "HTTP_MEMORY_RERANKER_POLICY_ID"]


class HTTPMemoryReranker:
    """Rank coarse Memory hits with an HTTP service that scores bounded batches.

    The service receives the original one-based candidate ranks, so a batched
    request keeps the identity of the coarse pool it came from. Scores from every
    batch take part in one global selection: descending score, then ascending
    original rank for equal scores.
    """

    policy_id = HTTP_MEMORY_RERANKER_POLICY_ID

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = HTTP_MEMORY_RERANKER_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")  # noqa: TRY003
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the original ranks of the best candidates without changing identity."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in enumerate(batch, start=start + 1)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            requests += 1
            scores.update(_parse_scores(response, expected=set(range(start + 1, start + 1 + len(batch)))))
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, /, *, expected: set[int]) -> dict[int, float]:
    """Validate one batch response and map every original rank to its score."""

    try:
        body = response.json()
    except ValueError as error:
        raise ValueError("rerank response must be valid JSON") from error
    if not isinstance(body, dict) or set(body) != _RESPONSE_FIELDS:
        raise ValueError("rerank response must contain exactly one scores field")  # noqa: TRY003
    records = body["scores"]
    if not isinstance(records, list):
        raise ValueError("rerank response scores must be a list")  # noqa: TRY003
    scores: dict[int, float] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != _SCORE_FIELDS:
            raise ValueError("rerank score must contain exactly one rank and one score")  # noqa: TRY003
        rank = record["rank"]
        score = record["score"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("rerank rank must be an integer")  # noqa: TRY003
        if rank not in expected:
            raise ValueError("rerank response must cover exactly the requested ranks")  # noqa: TRY003
        if rank in scores:
            raise ValueError("rerank response must not repeat one rank")  # noqa: TRY003
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("rerank score must be a finite number")  # noqa: TRY003
        scores[rank] = float(score)
    if set(scores) != expected:
        raise ValueError("rerank response must cover exactly the requested ranks")  # noqa: TRY003
    return scores


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003
