"""Loading and validating the rating matrices.

A rating matrix is a float64 array of shape (N_USERS, N_ITEMS): rows are users,
columns are movies in the order of namesngenre.npy. NaN marks an unobserved
cell; every observed cell holds one of RATING_VALUES.
"""

import hashlib
from pathlib import Path

import numpy as np

# The repository root: the course files sit next to generate.py, so that the
# platform's command `python3 generate.py --name ratings_train.npy` finds them.
DATA_DIR = Path(__file__).resolve().parent.parent

N_USERS = 610
N_ITEMS = 4980

# Half stars from 0.5 to 5. Multiples of 0.5 are exact in floating point,
# so membership can be tested with exact equality.
RATING_VALUES = np.arange(1, 11) / 2

# Checksums of the course files: a teammate with a different copy gets an
# error instead of results that cannot be compared with everyone else's.
SHA256 = {
    "ratings_train.npy": "103faa68f3aaf7d98e6e34f8ef7be0ade882b4da6c6bd7f0ee45a6a407cdc3a6",
    "ratings_test.npy": "8f48d552dce594cf3051a2965d6b3c16896f5b11d2622cc9d061938689b6c2e0",
    "namesngenre.npy": "d2d169c1e602b9430867280aac6a76e05aec689b9724e70243650b6b1b3ec7a7",
}
RATING_FILES = ("ratings_train.npy", "ratings_test.npy")


def _checked_path(name: str) -> Path:
    """Path of the course file <name>, after checking its checksum."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: copy the course files into {DATA_DIR}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        raise ValueError(f"{path} does not match the course file (sha256 {digest})")
    return path


def validate_ratings(R: np.ndarray) -> None:
    """Raise ValueError unless R follows the rating-matrix convention above.

    Takes any matrix, including the train+test matrix the platform passes to
    generate.py, which has no known checksum.
    """
    if R.shape != (N_USERS, N_ITEMS):
        raise ValueError(f"shape {R.shape}, expected {(N_USERS, N_ITEMS)} (users × items)")
    if R.dtype != np.float64:
        raise ValueError(f"dtype {R.dtype}, expected float64")
    observed = R[~np.isnan(R)]
    if observed.size == 0:
        raise ValueError("no observed rating")
    invalid = ~np.isin(observed, RATING_VALUES)
    if invalid.any():
        raise ValueError(f"values outside the rating scale: {np.unique(observed[invalid])[:10]}")


def load_ratings(name: str) -> np.ndarray:
    """Load the course file <name>, check its checksum and convention, return it read-only.

    Read-only because the protocol hands the same matrix to many models: an
    in-place write by one of them would silently change what the next one sees.
    """
    if name not in RATING_FILES:
        raise ValueError(f"unknown rating file {name!r}, expected one of {RATING_FILES}")
    R = np.load(_checked_path(name))
    validate_ratings(R)
    R.setflags(write=False)
    return R


def load_movies() -> tuple[np.ndarray, list[list[str]]]:
    """Return the movie titles and the genre list of each movie, in column order."""
    table = np.load(_checked_path("namesngenre.npy"))
    if table.shape != (N_ITEMS, 2):
        raise ValueError(f"namesngenre.npy has shape {table.shape}, expected {(N_ITEMS, 2)}")
    titles = table[:, 0]
    genres = [g.split("|") for g in table[:, 1]]
    return titles, genres
