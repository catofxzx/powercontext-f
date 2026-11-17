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

"""Durable registrations, applied events, and derived availability for Source evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.codec import dump_model, load_model
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.sources import SourceRef

SourceLifecycleFamily = Literal["source", "memory", "topic"]
SourceLifecyclePolicy = Literal["all", "any"]
SourceLifecycleReference = SourceRef | MemoryCitation | ArtifactRef

_REFERENCE_TYPES: dict[str, type] = {
    "source": SourceRef,
    "memory": MemoryCitation,
    "topic": ArtifactRef,
}


@dataclass(frozen=True)
class SourceLifecycleNode:
    """One registered evidence node and the dependency edges that support it."""

    node_id: str
    ordinal: int
    family: SourceLifecycleFamily
    reference: SourceLifecycleReference
    parents: tuple[str, ...]
    policy: SourceLifecyclePolicy
    available: bool
    revision: int | None


@dataclass(frozen=True)
class SourceLifecycleEventRecord:
    """One applied lifecycle event for a registered Source."""

    node_id: str
    sequence: int
    revision: int
    available: bool
    reason: str


class SourceLifecycleRepository:
    """Store one Scope's lifecycle graph without deciding its availability rules."""

    async def list_nodes(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> tuple[SourceLifecycleNode, ...]:
        """Return every registered node in registration order."""

        rows = (
            (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODES_TABLE)
                    .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
                    .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
                )
            )
            .mappings()
            .all()
        )
        return tuple(_decode_node(row) for row in rows)

    async def get_node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> SourceLifecycleNode | None:
        """Return one registered node, or ``None`` when the Scope does not name it."""

        row = (
            (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODES_TABLE).where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                        SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        return None if row is None else _decode_node(row)

    async def next_ordinal(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> int:
        """Return the registration ordinal the next node of this Scope must take."""

        current = await connection.scalar(
            select(func.max(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)).where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
            )
        )
        return 1 if current is None else int(current) + 1

    async def insert_node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node: SourceLifecycleNode,
        /,
    ) -> None:
        """Register one node with its reference, parents, and policy."""

        await connection.execute(
            insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                scope_id=scope_id,
                node_id=node.node_id,
                ordinal=node.ordinal,
                family=node.family,
                reference=_encode_reference(node.family, node.reference, node.node_id),
                parents=_encode_parents(node.parents),
                policy=node.policy,
                available=node.available,
                revision=node.revision,
            )
        )

    async def set_availability(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        available: bool,
        /,
    ) -> None:
        """Persist the availability decided for one registered node."""

        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(available=available)
        )

    async def record_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        event: SourceLifecycleEventRecord,
        /,
    ) -> None:
        """Append one applied event for a registered Source."""

        await connection.execute(
            insert(SOURCE_LIFECYCLE_EVENTS_TABLE).values(
                scope_id=scope_id,
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
            )
        )

    async def last_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> SourceLifecycleEventRecord | None:
        """Return the highest-sequence event applied to one Source."""

        row = (
            (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_EVENTS_TABLE)
                    .where(
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
                    )
                    .order_by(SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence.desc())
                    .limit(1)
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return SourceLifecycleEventRecord(
            node_id=str(row["node_id"]),
            sequence=int(row["sequence"]),
            revision=int(row["revision"]),
            available=bool(row["available"]),
            reason=str(row["reason"]),
        )


def _encode_reference(
    family: str,
    reference: SourceLifecycleReference,
    node_id: str,
) -> bytes:
    return dump_model(reference, kind="source-lifecycle", name=f"{family}:{node_id}")


def _decode_reference(family: str, payload: object, node_id: str) -> SourceLifecycleReference:
    model = _REFERENCE_TYPES[family]
    return load_model(model, payload, kind="source-lifecycle", name=f"{family}:{node_id}")


def _encode_parents(parents: tuple[str, ...]) -> bytes:
    return json.dumps(list(parents), separators=(",", ":")).encode()


def _decode_parents(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, bytes | bytearray | memoryview):
        return ()
    decoded = json.loads(bytes(payload).decode())
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


def _decode_node(row) -> SourceLifecycleNode:
    family = str(row["family"])
    return SourceLifecycleNode(
        node_id=str(row["node_id"]),
        ordinal=int(row["ordinal"]),
        family=family,
        reference=_decode_reference(family, row["reference"], str(row["node_id"])),
        parents=_decode_parents(row["parents"]),
        policy=str(row["policy"]),
        available=bool(row["available"]),
        revision=None if row["revision"] is None else int(row["revision"]),
    )


__all__ = [
    "SourceLifecycleEventRecord",
    "SourceLifecycleFamily",
    "SourceLifecycleNode",
    "SourceLifecyclePolicy",
    "SourceLifecycleReference",
    "SourceLifecycleRepository",
]
