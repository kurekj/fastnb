# fastnb

High-performance C++ Naive Bayes classifier for SILVA-based 16S rRNA taxonomy.

**98x faster** and **72,000x less RAM** than sklearn MultinomialNB — identical algorithm, drop-in replacement.

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
pip install fastnb
```

Or build from source:

```bash
git clone https://github.com/kurekj/fastnb.git
cd fastnb
pip install .
```

## Performance

| Metric | sklearn | fastnb (8 threads) | Improvement |
|--------|---------|-------------------|-------------|
| Time (300 ASVs) | 72.97 s | 0.74 s | 98x faster |
| Peak RAM | 15,186 MB | 0.21 MB | 72,000x less |

## License

MIT
