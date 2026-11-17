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

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemorySearchRequest
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryHybridIndex,
    OceanBaseMemoryVectorIndex,
)
from powercontext.builtin.tags import TagFilter

_FTS_ROW = {
    "memory_artifact_id": "memory-1",
    "entry_id": "entry-1",
    "entry_version_id": "version-1",
    "text": "Alpha semantic record.",
    "distance": None,
    "channel": "fts",
}
_VECTOR_ROW = {
    "memory_artifact_id": "memory-1",
    "entry_id": "entry-1",
    "entry_version_id": "version-1",
    "text": "Alpha semantic record.",
    "distance": 0.0,
    "channel": "vector",
}


def _profile() -> EmbeddingProfile:
    return EmbeddingProfile(
        profile_id="hybrid-test-v1",
        model="test",
        dimension=3,
        distance="l2",
        normalization="unit",
    )


def _request(tag_filter: TagFilter | None = None) -> MemorySearchRequest:
    return MemorySearchRequest(
        query="alpha",
        analyzed_query="alpha",
        memories=(ArtifactRef(family="memory", artifact_id="memory-1", revision=2),),
        candidate_limit=32,
        mode="hybrid",
        query_vector=(1.0, 0.0, 0.0),
        embedding_profile=_profile(),
        tag_filter=tag_filter,
    )


def _connection(rows: list[dict[str, object]]) -> AsyncConnection:
    connection = AsyncMock(spec=AsyncConnection)
    result = MagicMock()
    result.mappings = MagicMock(return_value=rows)
    connection.execute.return_value = result
    return cast(AsyncConnection, connection)


def _statements(connection: AsyncConnection) -> list[str]:
    calls = connection.execute.await_args_list
    return [str(call.args[0].compile(dialect=mysql.dialect())) for call in calls]


def test_hybrid_index_reads_both_channels_with_one_statement(monkeypatch) -> None:
    async def scenario() -> None:
        vector_index = OceanBaseMemoryVectorIndex(_profile())
        monkeypatch.setattr(vector_index, "vector_complete", AsyncMock(return_value=True))
        index = OceanBaseMemoryHybridIndex(vector_index)
        connection = _connection([_FTS_ROW, _VECTOR_ROW])

        channels = await index.search(connection, "scope-a", _request())

        statements = _statements(connection)
        assert len(statements) == 1
        assert "pc_memory_entry_heads" in statements[0]
        assert "pc_memory_vector_entries" in statements[0]
        assert "AGAINST" in statements[0].upper()
        assert "l2_distance" in statements[0].lower()
        assert [hit.entry_id for hit in channels.fts] == ["entry-1"]
        assert [hit.entry_id for hit in channels.vector] == ["entry-1"]
        assert channels.fts[0].distance is None
        assert channels.vector[0].distance == 0.0

    asyncio.run(scenario())


def test_hybrid_index_applies_the_tag_filter_inside_the_database(monkeypatch) -> None:
    async def scenario() -> None:
        vector_index = OceanBaseMemoryVectorIndex(_profile())
        monkeypatch.setattr(vector_index, "vector_complete", AsyncMock(return_value=True))
        index = OceanBaseMemoryHybridIndex(vector_index)
        connection = _connection([])

        await index.search(connection, "scope-a", _request(tag_filter=TagFilter(tags=("selected",))))

        statements = _statements(connection)
        assert len(statements) == 1
        assert "pc_artifact_tags" in statements[0]
        assert "APPROXIMATE" not in statements[0].upper()

    asyncio.run(scenario())


def test_composite_memory_index_serves_hybrid_without_a_second_candidate_query(monkeypatch) -> None:
    async def scenario() -> None:
        vector_index = OceanBaseMemoryVectorIndex(_profile())
        monkeypatch.setattr(vector_index, "vector_complete", AsyncMock(return_value=True))
        composite = CompositeMemoryIndex(
            OceanBaseMemoryFTSIndex(),
            vector_index,
            hybrid=OceanBaseMemoryHybridIndex(vector_index),
        )
        connection = _connection([])

        await composite.search(connection, "scope-a", _request())

        statements = _statements(connection)
        assert len(statements) == 1
        assert "pc_memory_entry_heads" in statements[0]
        assert "pc_memory_vector_entries" in statements[0]

    asyncio.run(scenario())
