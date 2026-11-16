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

"""Transfer one prepared context as a complete pack or as a prefix delta."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512
DEFAULT_DELTA_CHUNK_BYTES = 128
_DIGEST_PREFIX = "sha256:"

__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
    "build_incremental_target",
]


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One prepared context payload split into ordered byte chunks."""

    items: tuple[bytes, ...]

    @classmethod
    def build(cls, items: Iterable[bytes], /) -> ContextPack:
        """Build one complete pack from already split chunks."""

        return cls(items=tuple(items))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split one text into deterministic chunks that stay valid UTF-8."""

        return cls(items=_chunk_content(content, chunk_bytes))

    @property
    def digest(self) -> str:
        """Return the digest of the declared pack content."""

        return _digest(_unsigned(self.items))

    @property
    def byte_count(self) -> int:
        """Return the exact serialized size of this pack."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this pack into its wire representation."""

        return _encode_with_byte_count({**_unsigned(self.items), "digest": self.digest})

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        """Decode and verify one complete pack."""

        payload = _decode_object(data)
        _require_fields(payload, {"schema", "items", "digest", "byte_count"})
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("context pack schema is not supported")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise ContextPackError("context pack byte count must be an integer")  # noqa: TRY003
        digest = payload["digest"]
        _require_digest(digest)
        pack = cls(items=_decode_items(payload["items"]))
        if digest != pack.digest:
            raise ContextPackError("context pack digest does not match its content")  # noqa: TRY003
        if byte_count != len(pack.to_bytes()):
            raise ContextPackError("context pack byte count does not match its content")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """The prefix that stays valid plus the suffix that completes the target pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        """Describe how the target pack differs from the base pack."""

        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """Return the exact serialized size of this delta."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this delta into its wire representation."""

        return _encode(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_encode_item(item) for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        """Decode and verify one delta."""

        payload = _decode_object(data)
        _require_fields(payload, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("context pack delta schema is not supported")  # noqa: TRY003
        prefix_items = payload["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int) or prefix_items < 0:
            raise ContextPackError("context pack delta prefix must be a non-negative integer")  # noqa: TRY003
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        _require_digest(base_digest)
        _require_digest(target_digest)
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=_decode_items(payload["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from the base pack and one delta."""

    if base.digest != delta.base_digest:
        raise ContextPackError("context pack delta does not apply to this base pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("context pack delta prefix exceeds the base pack")  # noqa: TRY003
    target = ContextPack(items=(*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("context pack delta does not reproduce the target pack")  # noqa: TRY003
    return target


def build_incremental_target(
    base: ContextPack,
    content: str,
    /,
    *,
    chunk_bytes: int = DEFAULT_DELTA_CHUNK_BYTES,
) -> ContextPack:
    """Build the target pack that keeps the base chunks and re-chunks only the changed tail.

    Reusing the base chunks keeps the reusable prefix byte-identical, while a finer
    tail keeps the transferred delta close to the bytes that actually changed.
    """

    base_bytes = b"".join(base.items)
    new_bytes = content.encode("utf-8")
    common = _character_boundary(base_bytes, new_bytes)
    kept = 0
    consumed = 0
    for item in base.items:
        if consumed + len(item) > common:
            break
        consumed += len(item)
        kept += 1
    return ContextPack(items=(*base.items[:kept], *_chunk_bytes(new_bytes[consumed:], chunk_bytes)))


def _character_boundary(left: bytes, right: bytes) -> int:
    """Return the length of the longest common prefix that ends on a character boundary."""

    limit = min(len(left), len(right))
    common = 0
    while common < limit and left[common] == right[common]:
        common += 1
    while common > 0 and (left[common - 1] & 0xC0) == 0x80:
        common -= 1
    return common


def _chunk_bytes(data: bytes, chunk_bytes: int) -> tuple[bytes, ...]:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ContextPackError("chunk size must be a positive integer")  # noqa: TRY003
    items: list[bytes] = []
    start = 0
    while start < len(data):
        end = min(start + chunk_bytes, len(data))
        while end < len(data) and (data[end] & 0xC0) == 0x80:
            end -= 1
        if end == start:
            raise ContextPackError("content cannot be split into chunks within the requested size")  # noqa: TRY003
        items.append(data[start:end])
        start = end
    return tuple(items)


def _chunk_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ContextPackError("chunk size must be a positive integer")  # noqa: TRY003
    items: list[bytes] = []
    characters: list[str] = []
    size = 0
    for character in content:
        encoded = len(character.encode("utf-8"))
        if encoded > chunk_bytes:
            raise ContextPackError("content cannot be split into chunks within the requested size")  # noqa: TRY003
        if size + encoded > chunk_bytes:
            items.append("".join(characters).encode("utf-8"))
            characters = []
            size = 0
        characters.append(character)
        size += encoded
    if characters:
        items.append("".join(characters).encode("utf-8"))
    return tuple(items)


def _unsigned(items: tuple[bytes, ...]) -> dict[str, object]:
    return {"schema": CONTEXT_PACK_SCHEMA, "items": [_encode_item(item) for item in items]}


def _encode_item(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_items(values: object) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError("context pack items must be an array")  # noqa: TRY003
    items: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("context pack item must be a base64 string")  # noqa: TRY003
        try:
            items.append(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("context pack item must be valid base64") from error
    return tuple(items)


def _encode(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _encode_with_byte_count(payload: dict[str, object]) -> bytes:
    """Serialize a pack whose byte count covers the whole payload."""

    for digits in range(1, 8):
        candidate = 10 ** (digits - 1)
        length = len(_encode({**payload, "byte_count": candidate}))
        if candidate <= length < candidate * 10:
            encoded = _encode({**payload, "byte_count": length})
            if len(encoded) == length:
                return encoded
    raise ContextPackError("context pack byte count is not representable")  # noqa: TRY003


def _digest(unsigned: dict[str, object]) -> str:
    return f"{_DIGEST_PREFIX}{hashlib.sha256(_encode(unsigned)).hexdigest()}"


def _decode_object(data: bytes) -> dict[str, object]:
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise ContextPackError("context pack payload must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("context pack payload must be a JSON object")  # noqa: TRY003
    return payload


def _require_fields(payload: dict[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ContextPackError("context pack payload fields must match the wire contract")  # noqa: TRY003


def _require_digest(value: object) -> None:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX) or len(value) != len(_DIGEST_PREFIX) + 64:
        raise ContextPackError("context pack digest must use sha256:<hex>")  # noqa: TRY003
