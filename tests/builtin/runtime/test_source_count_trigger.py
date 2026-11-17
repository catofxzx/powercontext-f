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

import pytest
from pydantic import ValidationError

from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.sqlite import SQLiteConfig, SQLiteProfile
from powercontext.builtin.persistence.supervision import ArtifactProcessingLeaseRepository
from powercontext.builtin.persistence.tables import SHARED_TABLES
from powercontext.builtin.runtime.artifact_processing import ArtifactProcessingBinding, ArtifactProcessingSupervisor
from powercontext.builtin.runtime.config import RuntimeConfig
from powercontext.builtin.runtime.processing_contracts import (
    ArtifactProcessingWorkAssignment,
    ArtifactProcessingWorkerCompletion,
)


async def _wait(predicate, timeout_seconds=5):
    async with asyncio.timeout(timeout_seconds):
        while not predicate():  # noqa: ASYNC110 - bounded observation of worker events
            await asyncio.sleep(0.005)


class _CountProvider:
    """Report the pending Source count of every Scope under test."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    async def pending_counts(self, connection, scope_ids, /) -> dict[str, int]:
        return {scope_id: self.counts.get(scope_id, 0) for scope_id in scope_ids}


class _Launcher:
    """Accept one invocation and acknowledge its durable request."""

    def __init__(self, database) -> None:
        self.database = database
        self.assignments: list[ArtifactProcessingWorkAssignment] = []

    async def start(self, assignment: ArtifactProcessingWorkAssignment):
        self.assignments.append(assignment)
        return _Handle(self, assignment)


class _Handle:
    def __init__(self, launcher: _Launcher, assignment: ArtifactProcessingWorkAssignment) -> None:
        self.launcher = launcher
        self.assignment = assignment

    async def wait(self) -> ArtifactProcessingWorkerCompletion:
        work = self.assignment
        async with self.launcher.database.transaction() as connection:
            await ArtifactProcessingLeaseRepository().require_fence(connection, work.fence)
            intents = ArtifactProcessingIntentRepository()
            row = await intents.load(connection, work.scope_id, work.binding_name, for_update=True)
            assert row is not None
            await intents.acknowledge(
                connection,
                work.scope_id,
                work.binding_name,
                work.claimed_request_generation,
                clean_generation=row.dirty_generation,
            )
        return ArtifactProcessingWorkerCompletion()

    async def terminate(self) -> None:
        return None


def test_count_thresholds_are_strict_positive_integers():
    for field in ("memory_source_count_threshold", "topic_memory_source_count_threshold"):
        assert getattr(RuntimeConfig(), field) is None
        for value in (1, 3, 100):
            assert getattr(RuntimeConfig(**{field: value}), field) == value
        for value in (True, False, 0, -1, 1.0, "3"):
            with pytest.raises(ValidationError):
                RuntimeConfig(**{field: value})


def test_count_threshold_admits_only_the_scope_that_reached_it(tmp_path):
    async def scenario():
        async with SQLiteProfile.open(
            SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'count.db'}"), tables=SHARED_TABLES
        ) as profile:
            launcher = _Launcher(profile.database)
            provider = _CountProvider({"below": 1, "reached": 2})
            intents = ArtifactProcessingIntentRepository()
            async with profile.database.transaction() as connection:
                for scope in ("below", "reached"):
                    await intents.mark_dirty(connection, scope, "memory-binding")
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding(
                        "memory-binding",
                        "memory",
                        launcher,
                        max_workers=1,
                        source_count_threshold=2,
                        source_count_provider=provider,
                    ),
                ),
                lease_mode="single-process",
            ):
                await _wait(lambda: launcher.assignments)
                # A later evaluation must not admit the scope below the threshold.
                await asyncio.sleep(1.2)
                assert [item.scope_id for item in launcher.assignments] == ["reached"]

    asyncio.run(scenario())
