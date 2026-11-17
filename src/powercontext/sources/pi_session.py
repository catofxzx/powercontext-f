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

"""Capture the selected branch of a Pi session as event-level evidence Sources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.sources.connectors import ConnectorRunSession
    from powercontext.sources.definitions import SourceDefinition

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
PI_SESSION_FORMAT_VERSION = 3
_PI_IDENTITY_PREFIX = "pi:"
_CONTROL_TYPES = frozenset({"compaction", "branch_summary", "custom_message"})

__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]


class PiSessionInput(BaseModel):
    """One Pi session archive and the branch that should be captured."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event of the selected branch."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """One captured Pi event kept as retrievable evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    """One archive record with the identity that links the branch graph."""

    identifier: str
    parent_id: str | None
    event_type: str
    payload: dict[str, Any]


class PiEventSourceAdapter:
    """Resolve and read one captured Pi event."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured Pi event as text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the visible events of one Pi branch exactly once."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Import the events that the selected branch gained since the last run."""

        records = _read_archive(self._path.read_bytes())
        events = _visible_events(records, self._leaf_id)
        imported = _imported_records(session.checkpoint)
        digests = {record.identifier: _record_digest(record.payload) for record in records}
        for event_id, digest in imported.items():
            current = digests.get(event_id)
            if current is not None and current != digest:
                raise ValueError(f"imported Pi event {event_id!r} was rewritten in the session archive")  # noqa: TRY003
        merged = dict(imported)
        for event in events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                raise ValueError(f"Pi event {event.event_id!r} was not accepted by the runtime")  # noqa: TRY003
            merged[event.event_id] = digests[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"session_id": records[0].payload.get("id"), "imported": merged},
        )


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable identity of one captured Pi event."""

    identity = f"{_PI_IDENTITY_PREFIX}{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("Pi event source identity must not exceed the maximum source id length")  # noqa: TRY003
    return identity


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, from the root to the leaf."""

    records = _read_archive(value.content)
    return _visible_events(records, value.leaf_id)


def _visible_events(records: tuple[_SessionRecord, ...], leaf_id: str | None) -> tuple[PiEventInput, ...]:
    by_id = {record.identifier: record for record in records}
    leaf = records[-1].identifier if leaf_id is None else leaf_id
    if leaf not in by_id:
        raise ValueError(f"Pi session leaf {leaf!r} is not part of the archive")  # noqa: TRY003
    chain: list[_SessionRecord] = []
    seen: set[str] = set()
    current: _SessionRecord | None = by_id[leaf]
    while current is not None:
        if current.identifier in seen:
            raise ValueError("Pi session branch contains a cycle")  # noqa: TRY003
        seen.add(current.identifier)
        chain.append(current)
        current = None if current.parent_id is None else by_id.get(current.parent_id)
    chain.reverse()
    session_id = records[0].payload["id"]
    events: list[PiEventInput] = []
    for record in chain:
        rendered = _render_event(record)
        if rendered is None:
            continue
        role, content = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=record.identifier,
                parent_id=record.parent_id,
                event_type=record.event_type,
                role=role,
                content=f"{role}: {content}",
                record=record.payload,
            )
        )
    return tuple(events)


def _render_event(record: _SessionRecord) -> tuple[str, str] | None:
    payload = record.payload
    if record.event_type == "message":
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(message)
        if not body:
            return None
        return role, body
    if record.event_type in {"compaction", "branch_summary"}:
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        return record.event_type, summary
    if record.event_type == "custom_message":
        body = _blocks_text(payload.get("content"))
        if not body:
            return None
        return "custom", body
    return None


def _message_body(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        if not isinstance(command, str) or not command:
            return ""
        return f"$ {command}\n{output if isinstance(output, str) else ''}"
    if role == "toolResult":
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        text = _blocks_text(message.get("content"))
        if not text:
            return ""
        return f"[tool_result {tool_name} {tool_call_id}] {text}"
    content = message.get("content")
    if isinstance(content, str):
        return content
    return _blocks_text(content)


def _blocks_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            encoded = json.dumps(
                arguments if arguments is not None else {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            parts.append(f"[tool_call {name} {call_id}] {encoded}")
    return "\n".join(parts)


def _read_archive(content: bytes) -> tuple[_SessionRecord, ...]:
    if not content:
        raise ValueError("Pi session archive must not be empty")  # noqa: TRY003
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Pi session archive must contain records")  # noqa: TRY003
    payloads: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except ValueError as error:
            raise ValueError("Pi session archive must contain valid JSON records") from error
        if not isinstance(payload, dict):
            raise ValueError("Pi session archive records must be JSON objects")  # noqa: TRY003
        payloads.append(payload)
    header = payloads[0]
    if header.get("type") != "session" or header.get("version") != PI_SESSION_FORMAT_VERSION:
        raise ValueError("Pi session archive must start with a version 3 session record")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session archive must declare a session id")  # noqa: TRY003
    records: list[_SessionRecord] = []
    seen: set[str] = set()
    for payload in payloads[1:]:
        identifier = payload.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Pi session records must declare an id")  # noqa: TRY003
        if identifier in seen:
            raise ValueError(f"Pi session record {identifier!r} appears more than once")  # noqa: TRY003
        parent = payload.get("parentId")
        if parent is not None and not isinstance(parent, str):
            raise ValueError("Pi session parent id must be a string")  # noqa: TRY003
        if isinstance(parent, str) and not parent:
            parent = None
        if parent is not None and parent not in seen:
            raise ValueError(f"Pi session record {identifier!r} references an unknown parent")  # noqa: TRY003
        event_type = payload.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("Pi session records must declare a type")  # noqa: TRY003
        seen.add(identifier)
        records.append(
            _SessionRecord(
                identifier=identifier,
                parent_id=parent,
                event_type=event_type,
                payload=payload,
            )
        )
    return tuple(records)


def _record_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _imported_records(checkpoint: object) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    imported = checkpoint.get("imported")
    if not isinstance(imported, dict):
        return {}
    return {key: value for key, value in imported.items() if isinstance(key, str) and isinstance(value, str)}
