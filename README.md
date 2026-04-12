# fastnb

High-performance C++17 Naive Bayes classifier for SILVA-based 16S rRNA taxonomy, drop-in replacement for scikit-learn with 13–21× speedup.

## Quick Start

```python
import fastnb

results = fastnb.classify(
    sequences={"ASV1": "ATCGATCG...", "ASV2": "GCTAGCTA..."},
    params_dir="model/nb_params",
    threads=8,
    confidence=0.7,
)
print(results)  # DataFrame with Taxon, Confidence columns
```

## Installation

```bash
pip install git+https://github.com/kurekj/fastnb.git
```

Or build from source:

```bash
git clone https://github.com/kurekj/fastnb.git
cd fastnb
pip install .
```

### Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16+
- Python 3.10+
- pybind11, NumPy, pandas

## Performance

Benchmarked on 6 clinical patient samples (1,993 ASVs total, SILVA 138.2, 79,664 taxa). Hardware: 24-core Intel workstation, 94 GB DDR4 RAM.

| Metric | sklearn | fastnb (1 thread) | fastnb (8 threads) |
|--------|---------|-------------------|---------------------|
| Mean time (332 ASVs) | 49.4 s | 18.5 s (2.7×) | 3.0 s (16.8×) |
| Speedup range | — | 2.0–3.4× | 12.4–20.8× |
| Model memory | ~3.6 GB (float64) | ~2.5 GB (float32) | ~2.5 GB (shared) |

Key advantage: OpenMP shared-memory threading — all threads share a single 2.5 GB model copy. sklearn's `n_jobs` forks processes, duplicating ~3.6 GB per worker.

## Accuracy

99.85% genus-level agreement with sklearn across 1,993 clinical ASVs (3 of 6 patients: 100% exact match). The 3 disagreements favor fastnb — correct genus resolved where sklearn abstained, confirmed by independent BLAST validation.

## How It Works

1. **Feature extraction** — MurmurHash3 7-gram hashing (8,192 features, L2 norm), identical to sklearn's `HashingVectorizer(analyzer='char_wb')`
2. **Batch GEMM** — all sequences classified in one matrix multiplication pass (reads 2.45 GB model once, not once per sequence). Transposed layout enables AVX2 auto-vectorization. float32 doubles SIMD throughput.
3. **Hierarchical confidence truncation** — softmax (float64) + QIIME 2 rank-level grouping, identical to `classify_sklearn`

## Compatibility

- Loads the same `.npy` parameter files exported from trained sklearn classifiers
- Returns pandas DataFrame compatible with QIIME 2 `FeatureData[Taxonomy]`
- No model retraining or downstream pipeline changes needed
- Works with any `HashingVectorizer` + `MultinomialNB` model (16S, 18S, ITS)

## Citation

If you use fastnb in your research, please cite:

> M. Nawalny, J. Kurek, "fastnb: A high-performance C++ Naive Bayes classifier for SILVA-based 16S rRNA taxonomy," *SoftwareX*, 2026 (submitted).

## License

MIT
