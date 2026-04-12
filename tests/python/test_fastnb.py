"""Integration tests for fastnb Python API."""
import pytest
from unittest.mock import patch, MagicMock


def test_classify_empty_sequences():
    """classify() with empty dict returns empty DataFrame."""
    import fastnb
    result = fastnb.classify(sequences={})
    assert result.empty
    assert list(result.columns) == ["Taxon", "Confidence"]


def test_classify_returns_dataframe():
    """classify() returns DataFrame with correct columns and index."""
    import fastnb as fb

    mock_result = MagicMock()
    mock_result.taxonomy = "k__Bacteria; p__Firmicutes"
    mock_result.confidence = 0.85

    mock_nb = MagicMock()
    mock_nb.classify.return_value = [mock_result]

    mock_cpp = MagicMock()
    mock_cpp.NbClassifier.return_value = mock_nb
    mock_cpp.NbConfig.return_value = MagicMock()

    with patch.object(fb, '_CPP_AVAILABLE', True), \
         patch.object(fb, '_cached_nb', None), \
         patch.object(fb, '_cached_params_dir', None), \
         patch('fastnb.NbClassifier', mock_cpp.NbClassifier, create=True), \
         patch('fastnb.NbConfig', mock_cpp.NbConfig, create=True):

        result = fb.classify(
            sequences={"ASV1": "ATCGATCG"},
            params_dir="/fake",
            threads=1,
            confidence=0.7,
        )

    assert "Taxon" in result.columns
    assert "Confidence" in result.columns
    assert result.index.name == "Feature ID"
    assert len(result) == 1
    assert result.loc["ASV1", "Taxon"] == "k__Bacteria; p__Firmicutes"


def test_model_cached_across_calls():
    """Two calls with same params_dir load model only once."""
    import fastnb as fb

    fb._cached_nb = None
    fb._cached_params_dir = None

    mock_nb = MagicMock()
    mock_result = MagicMock()
    mock_result.taxonomy = "k__Bacteria"
    mock_result.confidence = 0.9
    mock_nb.classify.return_value = [mock_result]

    mock_cpp = MagicMock()
    mock_cpp.NbClassifier.return_value = mock_nb
    mock_cpp.NbConfig.return_value = MagicMock()

    with patch.object(fb, '_CPP_AVAILABLE', True), \
         patch('fastnb.NbClassifier', mock_cpp.NbClassifier, create=True), \
         patch('fastnb.NbConfig', mock_cpp.NbConfig, create=True):

        seqs = {"ASV1": "ATCGATCG"}
        fb.classify(seqs, params_dir="/fake", threads=1, confidence=0.7)
        fb.classify(seqs, params_dir="/fake", threads=1, confidence=0.7)

        assert mock_cpp.NbClassifier.call_count == 1
        assert mock_nb.load.call_count == 1


def test_import_error_when_cpp_unavailable():
    """classify() raises ImportError when C++ extension missing."""
    import fastnb as fb

    with patch.object(fb, '_CPP_AVAILABLE', False):
        with pytest.raises(ImportError, match=r"C\+\+ extension not available"):
            fb.classify({"ASV1": "ATCG"}, params_dir="/fake")
