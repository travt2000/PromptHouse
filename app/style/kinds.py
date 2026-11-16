"""The two analysis kinds: what each one asks the model for, and the schema it must answer with.

Everything else (scanning, the model server, batching, resume, storage) is shared; a kind
only decides the prompt, the schema and its version. The prompts are files:
``prompts/analysis/<key>.system.md`` and ``prompts/analysis/<key>.user.md``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from ..prompt_files import render
from .schema import (
    CHARACTER_SCHEMA,
    CHARACTER_SCHEMA_VERSION,
    INSPIRATION_SCHEMA,
    INSPIRATION_SCHEMA_VERSION,
)

@dataclass(frozen=True)
class Kind:
    key: str
    title: str
    schema_version: int
    schema: dict

    def schema_copy(self) -> dict:
        return copy.deepcopy(self.schema)

    @property
    def system_prompt(self) -> str:
        return render(f"analysis/{self.key}.system")

    def user_prompt(self, schema_json: str = "") -> str:
        """The text sent with each image; ``schema_json`` is added when the server cannot enforce the schema."""
        return render(f"analysis/{self.key}.user", schema=schema_json)


CHARACTER = Kind("character", "My character", CHARACTER_SCHEMA_VERSION, CHARACTER_SCHEMA)
INSPIRATION = Kind("inspiration", "Inspiration", INSPIRATION_SCHEMA_VERSION, INSPIRATION_SCHEMA)

KINDS: dict[str, Kind] = {k.key: k for k in (CHARACTER, INSPIRATION)}


def get_kind(key: str) -> Kind:
    try:
        return KINDS[key]
    except KeyError:
        raise ValueError(f"Unknown analysis kind: {key}") from None
