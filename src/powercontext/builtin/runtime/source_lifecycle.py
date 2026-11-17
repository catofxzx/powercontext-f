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

"""Propagate Source availability through one Scope's evidence dependency graph."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.source_lifecycle import (
    SourceLifecycleEventRecord,
    SourceLifecycleNode,
    SourceLifecycleRepository,
)
from powercontext.builtin.sources import validate_scope_id
from powercontext.limits import MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from powercontext.builtin.persistence.database import AsyncDatabase
    from powercontext.builtin.persistence.topic_memory import TopicMemoryRepository
    from powercontext.builtin.runtime.application import BuiltinRuntime

SourceLifecycleReference = SourceRef | MemoryCitation | ArtifactRef
SourceLifecyclePurpose = Literal["generation", "audit"]

_PURPOSES = frozenset({"generation", "audit"})
_POLICIES = frozenset({"all", "any"})


class SourceLifecycleError(ValueError):
    """Report a rejected lifecycle registration, event, or resolution."""

    def __init__(self, code: str, detail: object | None = None) -> None:
        self.code = code
        self.detail = detail
        suffix = "" if detail is None else f" ({detail})"
        super().__init__(f"source lifecycle rejected: {code}{suffix}")


class SourceLifecycleEvent(BaseModel):
    """One availability transition applied to a registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    reason: str
    revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def require_non_blank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")  # noqa: TRY003
        return value


class SourceLifecycleService:
    """Own the persistence and native-state adapters used by the scoped service."""

    def __init__(self, database: AsyncDatabase, topics: TopicMemoryRepository) -> None:
        self.database = database
        self.topics = topics
        self.repository = SourceLifecycleRepository()


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope's evidence graph."""

    def __init__(self, runtime: BuiltinRuntime, service: SourceLifecycleService | None = None) -> None:
        self._runtime = runtime
        self._service = service

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id, self._service)


class ScopedSourceLifecycleApplication:
    """Register evidence and propagate Source availability inside one Scope."""

    def __init__(
        self,
        runtime: BuiltinRuntime,
        scope_id: str,
        service: SourceLifecycleService | None = None,
    ) -> None:
        self._runtime = runtime
        self._service = service
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(self, node_id: str, source_ref: SourceRef, revision: int = 1) -> None:
        """Register one Source as the root evidence of this Scope's graph."""

        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("a Source node requires a SourceRef")
        if revision < 1:
            raise SourceLifecycleError("a Source revision must be positive")
        await self._require_readable_source(source_ref)
        await self._register(
            node_id,
            family="source",
            reference=source_ref,
            parents=(),
            policy="all",
            revision=revision,
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        parents: Sequence[str] = (),
        policy: str = "all",
    ) -> None:
        """Register one Memory entry and the evidence that supports it."""

        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("a Memory node requires a MemoryCitation")
        await self._require_readable_memory(citation)
        await self._register(
            node_id,
            family="memory",
            reference=citation,
            parents=parents,
            policy=policy,
            revision=None,
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        parents: Sequence[str] = (),
        policy: str = "all",
    ) -> None:
        """Register one Topic Memory artifact and the evidence that supports it."""

        if not isinstance(artifact_ref, ArtifactRef):
            raise SourceLifecycleError("a Topic node requires an ArtifactRef")
        await self._require_readable_topic(artifact_ref)
        await self._register(
            node_id,
            family="topic",
            reference=artifact_ref,
            parents=parents,
            policy=policy,
            revision=None,
        )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source event and return every currently available node id."""

        async with self._scope_write():
            async with self._database().transaction() as connection:
                nodes = await self._repository().list_nodes(connection, self.scope_id)
                nodes_by_id = {node.node_id: node for node in nodes}
                self._validate_event(nodes_by_id, event)
                await self._record_event(connection, event)
            targets = self._resolve_targets(nodes, event)
            await self._apply_native_states(nodes, targets)
            async with self._database().transaction() as connection:
                for node in nodes:
                    await self._repository().set_availability(
                        connection,
                        self.scope_id,
                        node.node_id,
                        targets[node.node_id],
                    )
        return sorted(node.node_id for node in nodes if targets[node.node_id])

    async def resolve(self, node_id: str, purpose: str = "generation") -> SourceLifecycleReference:
        """Return the registered reference of one node for the requested purpose."""

        if purpose not in _PURPOSES:
            raise SourceLifecycleError(f"unknown purpose: {purpose}")
        async with self._database().transaction() as connection:
            node = await self._repository().get_node(connection, self.scope_id, node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown node: {node_id}")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError(f"node is not available: {node_id}")
        return node.reference

    async def _register(
        self,
        node_id: str,
        *,
        family: str,
        reference: SourceLifecycleReference,
        parents: Sequence[str],
        policy: str,
        revision: int | None,
    ) -> None:
        selected_parents = () if family == "source" else _validated_parents(parents)
        selected_policy = _validated_policy(policy)
        selected_node_id = _validated_node_id(node_id)
        async with self._scope_write():
            async with self._database().transaction() as connection:
                nodes = await self._repository().list_nodes(connection, self.scope_id)
                nodes_by_id = {node.node_id: node for node in nodes}
                existing = nodes_by_id.get(selected_node_id)
                if existing is not None:
                    self._require_same_binding(
                        existing,
                        family=family,
                        reference=reference,
                        parents=selected_parents,
                        policy=selected_policy,
                        revision=revision,
                    )
                    available = existing.available
                else:
                    self._require_supported_parents(nodes_by_id, selected_node_id, selected_parents)
                    available = (
                        True
                        if family == "source"
                        else self._derived_availability(nodes_by_id, selected_parents, selected_policy)
                    )
                    await self._repository().insert_node(
                        connection,
                        self.scope_id,
                        SourceLifecycleNode(
                            node_id=selected_node_id,
                            ordinal=await self._repository().next_ordinal(connection, self.scope_id),
                            family=family,  # type: ignore[arg-type]
                            reference=reference,
                            parents=selected_parents,
                            policy=selected_policy,  # type: ignore[arg-type]
                            available=available,
                            revision=revision,
                        ),
                    )
            await self._apply_native_state(family, reference, available)

    async def _require_readable_source(self, reference: SourceRef) -> None:
        """Reject a Source registration whose Source does not exist in this Scope."""

        try:
            await self._runtime._records().get_source(
                self.scope_id,
                reference.source_type,
                reference.source_id,
            )
        except Exception as error:
            raise SourceLifecycleError(f"unreadable Source: {reference.source_id}") from error

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        """Reject a Memory registration whose citation does not resolve in this Scope."""

        try:
            async with self._runtime._context(self.scope_id) as context:
                service = context.artifacts.memory
                memory = await service.revision(citation.memory_ref)
                if memory.artifact_id != context.artifacts.memory_artifact_id:
                    raise SourceLifecycleError("citation names another Memory")
                await service.validate_citation(citation)
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError(f"unreadable Memory entry: {citation.entry_id}") from error

    async def _require_readable_topic(self, reference: ArtifactRef) -> None:
        """Reject a Topic registration whose Revision does not exist in this Scope."""

        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:
            raise SourceLifecycleError(f"unreadable Topic artifact: {reference.artifact_id}") from error

    def _require_same_binding(
        self,
        existing: SourceLifecycleNode,
        *,
        family: str,
        reference: SourceLifecycleReference,
        parents: tuple[str, ...],
        policy: str,
        revision: int | None,
    ) -> None:
        if (
            existing.family != family
            or existing.reference != reference
            or existing.parents != parents
            or existing.policy != policy
            or existing.revision != revision
        ):
            raise SourceLifecycleError(f"node is already bound: {existing.node_id}")

    def _require_supported_parents(
        self,
        nodes_by_id: dict[str, SourceLifecycleNode],
        node_id: str,
        parents: tuple[str, ...],
    ) -> None:
        for parent in parents:
            if parent == node_id:
                raise SourceLifecycleError(f"node depends on itself: {node_id}")
            if parent not in nodes_by_id:
                raise SourceLifecycleError(f"unknown parent: {parent}")
        _reject_cycle(nodes_by_id, node_id, parents)

    def _validate_event(
        self,
        nodes_by_id: dict[str, SourceLifecycleNode],
        event: SourceLifecycleEvent,
    ) -> None:
        node = nodes_by_id.get(event.node_id)
        if node is None or node.family != "source":
            raise SourceLifecycleError(f"event does not name a registered Source: {event.node_id}")
        if node.revision != event.revision:
            raise SourceLifecycleError(f"event revision does not match the Source: {event.node_id}")

    async def _record_event(self, connection: AsyncConnection, event: SourceLifecycleEvent) -> None:
        last = await self._repository().last_event(connection, self.scope_id, event.node_id)
        if last is not None:
            if event.sequence < last.sequence:
                raise SourceLifecycleError(f"event sequence is not monotonic: {event.node_id}")
            if event.sequence == last.sequence:
                if (event.revision, event.available, event.reason) != (
                    last.revision,
                    last.available,
                    last.reason,
                ):
                    raise SourceLifecycleError(f"event sequence conflicts with a recorded event: {event.node_id}")
                return
        await self._repository().record_event(
            connection,
            self.scope_id,
            SourceLifecycleEventRecord(
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
            ),
        )

    def _resolve_targets(
        self,
        nodes: tuple[SourceLifecycleNode, ...],
        event: SourceLifecycleEvent,
    ) -> dict[str, bool]:
        """Recompute every node's availability from its parents' availability."""

        targets = {node.node_id: node.available for node in nodes}
        targets[event.node_id] = event.available
        for node in nodes:
            if node.family == "source":
                continue
            support = [targets[parent] for parent in node.parents]
            targets[node.node_id] = all(support) if node.policy == "all" else any(support)
        return targets

    def _derived_availability(
        self,
        nodes_by_id: dict[str, SourceLifecycleNode],
        parents: tuple[str, ...],
        policy: str,
    ) -> bool:
        support = [nodes_by_id[parent].available for parent in parents]
        return all(support) if policy == "all" else any(support)

    async def _apply_native_states(
        self,
        nodes: tuple[SourceLifecycleNode, ...],
        targets: dict[str, bool],
    ) -> None:
        for node in nodes:
            if targets[node.node_id] == node.available:
                continue
            await self._apply_native_state(node.family, node.reference, targets[node.node_id])

    async def _apply_native_state(
        self,
        family: str,
        reference: SourceLifecycleReference,
        available: bool,
    ) -> None:
        """Drive the native Memory or Topic Memory state for one node."""

        if family == "memory" and isinstance(reference, MemoryCitation):
            await self._apply_memory_state(reference, available)
        elif family == "topic" and isinstance(reference, ArtifactRef):
            await self._apply_topic_state(reference, available)

    async def _apply_memory_state(self, citation: MemoryCitation, available: bool) -> None:
        """Deactivate or reactivate one Memory entry through the Memory service."""

        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current, entries = await service.head_entries(context.artifacts.memory_artifact_id)
            entry = next((item for item in entries if item.entry_id == citation.entry_id), None)
            if entry is None:
                raise SourceLifecycleError(f"unknown Memory entry: {citation.entry_id}")
            state = next(
                (
                    item.state
                    for item in current.content.manifest.entries
                    if item.entry_id == citation.entry_id
                ),
                None,
            )
            if available and state == "inactive":
                await service.reactivate(current, entries=(entry,), reason="source-lifecycle")
            elif not available and state == "active":
                await service.forget(current, entries=(entry,), reason="source-lifecycle")

    async def _apply_topic_state(self, reference: ArtifactRef, available: bool) -> None:
        """Withdraw or restore one Topic Memory artifact through its repository."""

        topics = self._service.topics if self._service is not None else None
        if topics is None:
            raise SourceLifecycleError("topic memory is not configured")
        async with self._database().transaction() as connection:
            if available:
                await topics.restore_topic(connection, self.scope_id, reference)
            else:
                await topics.withdraw_topic(connection, self.scope_id, reference)

    def _database(self) -> AsyncDatabase:
        if self._service is None:
            raise SourceLifecycleError("source lifecycle is not configured")
        return self._service.database

    @asynccontextmanager
    async def _scope_write(self) -> AsyncIterator[None]:
        """Serialize one lifecycle write against the Scope's other writers."""

        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            yield

    def _repository(self) -> SourceLifecycleRepository:
        if self._service is None:
            raise SourceLifecycleError("source lifecycle is not configured")
        return self._service.repository


def _validated_node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node id must not be blank")
    if len(value) > MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("node id is too long")
    return value


def _validated_parents(parents: Sequence[str]) -> tuple[str, ...]:
    if isinstance(parents, str):
        raise SourceLifecycleError("parents must be a sequence of node ids")
    values = tuple(parents)
    if not values:
        raise SourceLifecycleError("a derived node requires at least one parent")
    for parent in values:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parent node id must not be blank")
    if len(set(values)) != len(values):
        raise SourceLifecycleError("parents must not repeat")
    return values


def _validated_policy(policy: str) -> str:
    if policy not in _POLICIES:
        raise SourceLifecycleError(f"unknown support policy: {policy}")
    return policy


def _reject_cycle(
    nodes_by_id: dict[str, SourceLifecycleNode],
    node_id: str,
    parents: tuple[str, ...],
) -> None:
    """Reject a dependency edge that closes a cycle in the Scope's graph."""

    pending = list(parents)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == node_id:
            raise SourceLifecycleError(f"node depends on itself: {node_id}")
        if current in seen:
            continue
        seen.add(current)
        node = nodes_by_id.get(current)
        if node is not None:
            pending.extend(node.parents)


__all__ = [
    "ScopedSourceLifecycleApplication",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleService",
]
