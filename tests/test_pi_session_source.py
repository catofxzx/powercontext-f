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

"""Behaviour of the Pi session source: the branch of the archive and its identity."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from powercontext.sources import ConnectorBinding, SourceDefinitionRegistry, SourceRef
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)

HEADER = {"type": "session", "version": 3, "id": "session-of-the-archive"}


class _AcceptingSourceSink:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def submit(
        self,
        binding: ConnectorBinding,
        item_id: str,
        definition_name: str,
        value: object,
        /,
    ) -> SourceRef:
        del binding, value
        self.events.append(f"source:{item_id}")
        return SourceRef(source_type=definition_name, source_id=item_id)


class _MemoryCheckpointStore:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def load(self, binding: ConnectorBinding, /) -> Any:
        return deepcopy(self.values.get(binding.binding_id))

    async def save(self, binding: ConnectorBinding, checkpoint: Any, /, *, expected: Any) -> None:
        assert self.values.get(binding.binding_id) == expected
        self.values[binding.binding_id] = deepcopy(checkpoint)


def archive(records: list[dict], session_id: str = "session-of-the-archive") -> bytes:
    header = {"type": "session", "version": 3, "id": session_id}
    return "\n".join(json.dumps(row) for row in [header, *records]).encode()


def message(identifier: str, parent: str | None, role: str, content: str) -> dict:
    return {"type": "message", "id": identifier, "parentId": parent, "message": {"role": role, "content": content}}


def test_an_event_carries_the_session_of_the_header_and_not_another_event() -> None:
    """The session of an event is the id of the session header, not of another event."""

    parsed = parse_pi_session(
        PiSessionInput(
            content=archive([
                message("first", None, "user", "hello"),
                message("second", "first", "assistant", "hi"),
            ])
        )
    )
    assert [event.session_id for event in parsed] == ["session-of-the-archive", "session-of-the-archive"]
    assert [event.event_id for event in parsed] == ["first", "second"]


def test_the_source_identity_follows_the_session_of_the_header() -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        parsed = parse_pi_session(PiSessionInput(content=archive([message("first", None, "user", "hello")])))
        source = await registry.resolve(parsed[0])
        assert source.session_id == "session-of-the-archive"
        assert "session-of-the-archive" in source.name
        assert source.name.endswith("first")

    asyncio.run(scenario())


def test_two_sessions_that_share_an_event_id_keep_distinct_identities() -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        first = parse_pi_session(PiSessionInput(content=archive([message("a", None, "user", "one")], "session-one")))
        second = parse_pi_session(PiSessionInput(content=archive([message("a", None, "user", "two")], "session-two")))
        names = {(await registry.resolve(event)).name for event in (*first, *second)}
        assert len(names) == 2, "events of two sessions must not collide"

    asyncio.run(scenario())


def test_the_branch_of_the_leaf_runs_from_the_root_to_the_leaf() -> None:
    records = [
        message("root", None, "user", "shared"),
        {"type": "model_change", "id": "control", "parentId": "root"},
        message("leaf", "control", "assistant", "answer"),
    ]
    parsed = parse_pi_session(PiSessionInput(content=archive(records), leaf_id="leaf"))
    assert [event.event_id for event in parsed] == ["root", "leaf"]
    # a control node keeps the graph relation of the branch
    assert parsed[1].parent_id == "control"


def test_a_session_without_an_event_has_no_branch_to_capture() -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=archive([])))


def test_a_record_without_a_parent_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=archive([{"type": "message", "id": "a"}])))


def test_the_connector_keeps_the_progress_of_every_session(tmp_path: Path) -> None:
    """A binding that visits a second archive keeps the progress of the first one."""

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_bytes(archive([message("a", None, "user", "first session")], "one"))
    second.write_bytes(archive([message("a", None, "user", "second session")], "two"))

    async def scenario() -> None:
        events: list[str] = []
        checkpoints = _MemoryCheckpointStore()
        lifecycle = ConnectorLifecycle(sink=_AcceptingSourceSink(events), checkpoints=checkpoints)
        binding = ConnectorBinding(
            scope_id="scope-a", binding_id="pi-a", connector_name="pi-session-history", connector_version="1"
        )

        await lifecycle.run(PiSessionConnector(first), binding)
        await lifecycle.run(PiSessionConnector(second), binding)
        # the second archive shares the event id of the first one
        assert events.count("source:a") == 2
        await lifecycle.run(PiSessionConnector(first), binding)
        assert events.count("source:a") == 2, "the first session keeps its own progress"

    asyncio.run(scenario())
