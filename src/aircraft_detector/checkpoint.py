"""Checkpoint helpers shared by train and inference entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)

