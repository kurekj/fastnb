"""fastnb: High-performance C++ Naive Bayes for 16S rRNA taxonomy."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from fastnb_cpp import NbClassifier, NbConfig, NbResult  # type: ignore
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

__version__ = "1.0.0"

_cached_nb: "NbClassifier | None" = None
_cached_params_dir: "str | None" = None


def classify(
    sequences: dict[str, str],
    params_dir: "str | Path" = "model/nb_params",
    threads: int = 1,
    confidence: float = 0.7,
) -> pd.DataFrame:
    """Classify ASV sequences using C++ Naive Bayes.

    Args:
        sequences: Dict mapping feature IDs to DNA sequences.
        params_dir: Path to directory with .npy model files.
        threads: Number of OpenMP threads.
        confidence: Confidence threshold for hierarchical truncation.

    Returns:
        DataFrame with columns: Taxon, Confidence (QIIME2-compatible).
    """
    if not sequences:
        return pd.DataFrame(columns=["Taxon", "Confidence"])

    if not _CPP_AVAILABLE:
        raise ImportError(
            "fastnb C++ extension not available. "
            "Build with: pip install fastnb"
        )

    global _cached_nb, _cached_params_dir
    pd_str = str(params_dir)
    if _cached_nb is None or _cached_params_dir != pd_str:
        _cached_nb = NbClassifier()
        _cached_nb.load(pd_str)
        _cached_params_dir = pd_str

    conf = NbConfig()
    conf.num_threads = threads
    conf.confidence_threshold = confidence

    pairs = list(sequences.items())
    results = _cached_nb.classify(pairs, conf)

    rows = {}
    for i, (asv_id, _seq) in enumerate(pairs):
        r = results[i]
        rows[asv_id] = {"Taxon": r.taxonomy, "Confidence": r.confidence}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Feature ID"
    return df[["Taxon", "Confidence"]]
