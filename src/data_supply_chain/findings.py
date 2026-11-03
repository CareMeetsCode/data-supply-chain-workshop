from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_META_KEY = "_meta"


class SuppressedFindingError(RuntimeError):
    pass


SUPPRESSED_KEYS: dict[str, str] = {
    "crossers_mort": (
        "Mortality among the 13 masked Black diabetic patients is 100%. It is a "
        "survivorship artifact — n=13, the oldest and sickest patients in the "
        "cohort, over ~20 years of follow-up — not a causal effect of the "
        "equation. The workshop deliberately suppresses it and quotes "
        "mean eGFR instead. A 100% death rate WILL be screenshotted out of "
        "context. Quote crossers_egfr_2021_mean."
    ),
    "crossers_mort_wt": (
        "Survey-weighted variant of crossers_mort. Same artifact, and weighting "
        "a 13-person group does not make it publishable."
    ),
    "rest_mort": (
        "Only meaningful as the comparison arm for crossers_mort. Publishing it "
        "alone invites reconstructing the suppressed contrast."
    ),
    "rest_mort_wt": "Survey-weighted variant of rest_mort. Same reasoning.",
}


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for dist in ("data-supply-chain", "data-supply-chain-workshop"):
        try:
            return f"{dist} {version(dist)}"
        except PackageNotFoundError:
            continue
    return "unknown (not installed as a distribution)"


def build_meta(state: Any) -> dict[str, Any]:
    from data_supply_chain.data import COHORT_SHA256
    from data_supply_chain.preprocess import distribution_variant

    return {
        "distribution_variant": distribution_variant(),
        "package_version": _package_version(),
        "cohort_sha256": COHORT_SHA256,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": state.source,
        "is_real": bool(state.is_real),
    }


class Findings:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        if key == _META_KEY or key.startswith(_META_KEY + "."):
            raise ValueError(
                f"{_META_KEY!r} is provenance, not a finding. It is written by "
                "write(state=...), which derives every field from an existing "
                "definition. Recording it by hand is how a hand-typed digest or "
                "an is_real=True on a synthetic run would get onto the site."
            )
        leaf = key.rsplit(".", 1)[-1]
        if leaf in SUPPRESSED_KEYS:
            raise SuppressedFindingError(
                f"{key!r} must never enter findings.json.\n  {SUPPRESSED_KEYS[leaf]}"
            )
        self._data[key] = value

    def write(self, path: Path | None = None, *, state: Any = None) -> None:
        offenders = sorted(_suppressed_keys_in(self._data))
        if offenders:
            raise SuppressedFindingError(
                "refusing to write findings.json containing suppressed keys: "
                + ", ".join(offenders)
            )

        if state is None:
            raise TypeError(
                "Findings.write() requires state= (the WorkshopState from "
                "cold_start()). It supplies _meta.source and _meta.is_real, and "
                "the site build refuses any findings.json without them."
            )

        if path is None:
            from data_supply_chain.paths import findings_path

            path = findings_path()
            if path is None:
                raise RuntimeError(
                    "findings.json can only be written from a repo checkout, and "
                    "none was found — this is expected in Colab, where there is "
                    "no site/ to write into.\n"
                    "Emitting findings is a maintainer action. If you meant to "
                    "write somewhere specific, pass an explicit path."
                )

        path = Path(path)
        payload = {_META_KEY: build_meta(state), **self._data}
        payload = _preserve_timestamp_if_unchanged(payload, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.write("\n")

    @classmethod
    def open(cls, path: Path | None = None) -> "Findings":
        try:
            existing = cls.load(path)
        except (OSError, json.JSONDecodeError, RuntimeError):
            existing = {}

        if not isinstance(existing, dict):
            existing = {}

        collector = cls()
        collector._data = {k: v for k, v in existing.items() if k != _META_KEY}
        return collector

    @classmethod
    def load(cls, path: Path | None = None) -> dict[str, Any]:
        if path is None:
            from data_supply_chain.paths import findings_path

            path = findings_path()
            if path is None:
                raise RuntimeError(
                    "no repo checkout found, so there is no findings.json to "
                    "load. Pass an explicit path."
                )
        with open(path) as f:
            return json.load(f)


def _preserve_timestamp_if_unchanged(payload: dict, path: Path) -> dict:
    try:
        with open(path) as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return payload

    if not isinstance(existing, dict):
        return payload

    def without_timestamp(d: dict) -> dict:
        meta = {k: v for k, v in d.get(_META_KEY, {}).items() if k != "generated_at"}
        return {**{k: v for k, v in d.items() if k != _META_KEY}, _META_KEY: meta}

    if without_timestamp(existing) != without_timestamp(payload):
        return payload

    previous = existing.get(_META_KEY, {}).get("generated_at")
    if previous is None:
        return payload
    return {**payload, _META_KEY: {**payload[_META_KEY], "generated_at": previous}}


def _suppressed_keys_in(obj: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).rsplit(".", 1)[-1] in SUPPRESSED_KEYS:
                found.add(path)
            found |= _suppressed_keys_in(value, path)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            found |= _suppressed_keys_in(item, f"{prefix}[{i}]")
    return found
