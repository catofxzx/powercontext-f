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

"""Startup migration for the Topic Memory active projection lifecycle state."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import MYSQL_IDENTITY_COLLATION

_TABLE_NAME = "pc_topic_memory_active_topics"
_COLUMN_NAME = "lifecycle_state"
_SQLITE_ADD = (
    f"ALTER TABLE {_TABLE_NAME} ADD COLUMN {_COLUMN_NAME} VARCHAR(16) NOT NULL DEFAULT 'active'"
)
_MYSQL_ADD = (
    f"ALTER TABLE {_TABLE_NAME} ADD COLUMN {_COLUMN_NAME} VARCHAR(16) "
    f"CHARACTER SET utf8mb4 COLLATE {MYSQL_IDENTITY_COLLATION} NOT NULL DEFAULT 'active'"
)


async def ensure_topic_memory_lifecycle_schema(connection: AsyncConnection) -> None:
    """Add the withdrawn-state column to deployments created before it existed."""

    dialect = connection.dialect.name
    if dialect not in {"sqlite", "mysql"}:
        raise ValueError(f"unsupported Topic Memory lifecycle migration dialect: {dialect}")  # noqa: TRY003
    if _COLUMN_NAME in await _column_names(connection, _TABLE_NAME):
        return
    await connection.exec_driver_sql(_SQLITE_ADD if dialect == "sqlite" else _MYSQL_ADD)


async def _column_names(connection: AsyncConnection, table_name: str) -> set[str]:
    if connection.dialect.name == "sqlite":
        rows = (await connection.exec_driver_sql(f"PRAGMA table_info('{table_name}')")).mappings()
        return {str(row["name"]) for row in rows}
    rows = (
        await connection.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = :table_name"
            ),
            {"table_name": table_name},
        )
    ).mappings()
    return {str(row["column_name"]) for row in rows}


__all__ = ["ensure_topic_memory_lifecycle_schema"]
