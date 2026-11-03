from __future__ import annotations

from pathlib import Path

_ROOT_MARKERS = ("pyproject.toml", "site")


def _looks_like_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _ROOT_MARKERS)


def repo_root() -> Path | None:
    starts = [Path(__file__).resolve(), Path.cwd().resolve() / "_"]
    for start in starts:
        for candidate in start.parents:
            if _looks_like_root(candidate):
                return candidate
    return None


def findings_path() -> Path | None:
    root = repo_root()
    return None if root is None else root / "site" / "src" / "data" / "findings.json"


def figure_path(name: str) -> Path | None:
    root = repo_root()
    return None if root is None else root / "figures" / name
