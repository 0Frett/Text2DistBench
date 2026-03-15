"""Common JSON/JSONL helpers used across Text2DistBench scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List


def ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    """Create the parent directory for ``path`` if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_json(data: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")