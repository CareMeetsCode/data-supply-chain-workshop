from __future__ import annotations
import io
import urllib.error
import urllib.request
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

_CACHE_NAME = "nhanes_1999_2004_analytic.csv.gz"

COHORT_SHA256 = "6eb7cb697550d0c840507dbd0526f08206adf80e31757d0917b37e914d327d76"

_PACKAGE_DATA_DIR = "_data"

_CACHE_URL = (
    "https://raw.githubusercontent.com/WaverlyRose/data-supply-chain/main/"
    f"data/{_CACHE_NAME}"
)

_URL_TIMEOUT_SECONDS = 30

_SYNTHETIC_BANNER = (
    "\n" + "=" * 60 +
    "\n  SYNTHETIC DATA MODE — national projections disabled\n" +
    "=" * 60 + "\n"
)


class CohortFetchError(RuntimeError):
    pass


class TransientFetchError(CohortFetchError):
    pass


class PermanentFetchError(CohortFetchError):
    pass


def _candidate_dirs(data_dir: str | None) -> list[Path]:
    if data_dir is not None:
        return [Path(data_dir)]

    here = Path(__file__).resolve()
    return [
        here.parents[2] / "data",
        Path.cwd() / "data",
    ]


def load_cohort(
    data_dir: str | None = None,
    synthetic: bool = False,
) -> tuple[pd.DataFrame, str]:
    if synthetic:
        warnings.warn(_SYNTHETIC_BANNER, stacklevel=2)
        return _make_synthetic(), "synthetic"

    if data_dir is None:
        bundled = _bundled_cache_path()
        if bundled is not None:
            return _prepare(pd.read_csv(bundled)), "package"

    for candidate in _candidate_dirs(data_dir):
        cache_path = candidate / _CACHE_NAME
        if cache_path.exists():
            return _prepare(pd.read_csv(cache_path)), "cache"

    try:
        return _load_from_url()
    except TransientFetchError as exc:
        warnings.warn(
            f"Could not reach the cohort URL ({exc}). "
            "Falling back to synthetic cohort.\n" + _SYNTHETIC_BANNER,
            stacklevel=2,
        )
        return _make_synthetic(), "synthetic"


STORED_EGFR_COLUMNS = ("eGFR_2009", "eGFR_2021", "eGFR_cys")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    from data_supply_chain.egfr import add_egfr_columns

    df = df.drop(columns=[c for c in STORED_EGFR_COLUMNS if c in df.columns])
    return add_egfr_columns(df)


def _bundled_cache_path() -> Path | None:
    from importlib.resources import files

    try:
        resource = files("data_supply_chain") / _PACKAGE_DATA_DIR / _CACHE_NAME
        path = Path(str(resource))
    except (ModuleNotFoundError, TypeError, NotADirectoryError):
        return None
    return path if path.is_file() else None


def _load_from_url() -> tuple[pd.DataFrame, str]:
    try:
        with urllib.request.urlopen(_CACHE_URL, timeout=_URL_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            raise PermanentFetchError(
                f"HTTP {exc.code} fetching the cohort from {_CACHE_URL}. "
                "The URL is wrong, or the repository is private and therefore "
                "unreadable without credentials. This is a distribution bug, not "
                "an offline degradation, so it is NOT falling back to synthetic "
                "data — synthetic numbers here would silently disagree with the "
                "published findings."
            ) from exc
        raise TransientFetchError(f"HTTP {exc.code} from {_CACHE_URL}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransientFetchError(f"{type(exc).__name__}: {exc}") from exc

    return _prepare(pd.read_csv(io.BytesIO(payload), compression="gzip")), "url"


def _make_synthetic(n: int = 6_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 80, n).astype(float)
    female = rng.integers(0, 2, n).astype(float)
    black = (rng.random(n) < 0.12).astype(float)
    scr = rng.lognormal(0.05, 0.25, n)
    scys = rng.lognormal(-0.1, 0.3, n)
    acr = rng.lognormal(2.5, 1.2, n)
    diabetes = (rng.random(n) < 0.12).astype(float)
    hypertension = (rng.random(n) < 0.30).astype(float)
    insured = (rng.random(n) < 0.82).astype(float)
    death = (rng.random(n) < 0.08).astype(float)
    followup_mo = rng.uniform(1, 120, n)

    return pd.DataFrame({
        "age": age,
        "female": female,
        "black": black,
        "scr": scr,
        "scys": scys,
        "acr": acr,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "insured": insured,
        "wt_mec": np.ones(n),
        "SDMVSTRA": rng.integers(1, 16, n),
        "SDMVPSU": rng.integers(1, 3, n),
        "death": death,
        "followup_mo": followup_mo,
    })


SWEEP_NAME = "pareto_sweep.csv"


def _sweep_path() -> Path:
    from importlib.resources import files

    try:
        resource = files("data_supply_chain") / _PACKAGE_DATA_DIR / SWEEP_NAME
        if resource.is_file():
            return Path(str(resource))
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    from data_supply_chain.paths import repo_root

    root = repo_root()
    if root is not None and (root / "data" / SWEEP_NAME).exists():
        return root / "data" / SWEEP_NAME

    raise FileNotFoundError(
        f"{SWEEP_NAME} is neither bundled in the package nor in a checkout's "
        "data/. In a checkout run `make sweep` (~3.5 minutes, full distribution "
        "only). In Colab it means the installed wheel predates the artifact."
    )


def load_pareto_sweep() -> pd.DataFrame:
    return pd.read_csv(_sweep_path(), comment="#")


def pareto_sweep_meta() -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in _sweep_path().read_text().splitlines():
        if not line.startswith("#"):
            break
        if ":" in line:
            key, _, value = line.lstrip("# ").partition(":")
            meta[key.strip()] = value.strip()
    return meta
