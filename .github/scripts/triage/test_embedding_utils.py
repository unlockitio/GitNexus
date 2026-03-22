"""Tests for embedding_utils.py — all embedding model calls are mocked."""
from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Mock fastembed before importing the module under test (persistent)
if "fastembed" not in sys.modules:
    sys.modules["fastembed"] = MagicMock()

from embedding_utils import (
    embed_texts,
    normalize_rows,
    reduce_dimensions,
    detect_outliers,
    find_duplicate_pairs,
    suggest_labels,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    LABEL_SIMILARITY_THRESHOLD,
    MAX_LABELS_PER_ITEM,
)


class TestEmbedTexts:
    """Tests for the embed_texts function."""

    def test_empty_list_returns_empty_array(self):
        result = embed_texts([])
        assert result.shape == (0, EMBEDDING_DIM)
        assert result.dtype == np.float32

    @patch("embedding_utils.TextEmbedding")
    def test_single_text(self, mock_cls):
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        mock_model.embed.return_value = iter([vec])

        result = embed_texts(["hello world"])

        mock_cls.assert_called_once_with(model_name=EMBEDDING_MODEL)
        mock_model.embed.assert_called_once_with(
            ["hello world"], batch_size=EMBEDDING_BATCH_SIZE
        )
        assert result.shape == (1, EMBEDDING_DIM)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result[0], vec)

    @patch("embedding_utils.TextEmbedding")
    def test_multiple_texts(self, mock_cls):
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        vecs = [
            np.random.randn(EMBEDDING_DIM).astype(np.float32)
            for _ in range(5)
        ]
        mock_model.embed.return_value = iter(vecs)

        result = embed_texts(["a", "b", "c", "d", "e"])
        assert result.shape == (5, EMBEDDING_DIM)
        assert result.dtype == np.float32


class TestNormalizeRows:
    """Tests for L2 row normalization."""

    def test_empty_matrix(self):
        m = np.empty((0, 10), dtype=np.float32)
        result = normalize_rows(m)
        assert result.shape == (0, 10)

    def test_single_row(self):
        m = np.array([[3.0, 4.0]], dtype=np.float32)
        result = normalize_rows(m)
        # Norm should be ~1.0
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_multiple_rows(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((10, 50)).astype(np.float32)
        result = normalize_rows(m)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_row_stays_near_zero(self):
        m = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        result = normalize_rows(m)
        # Zero row divided by eps -> very small values
        assert np.linalg.norm(result[0]) < 1e-3
        # Non-zero row should be unit norm
        assert abs(np.linalg.norm(result[1]) - 1.0) < 1e-5

    def test_preserves_direction(self):
        m = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        result = normalize_rows(m)
        np.testing.assert_allclose(result[0], [1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [0.0, 1.0], atol=1e-5)


class TestReduceDimensions:
    """Tests for PCA dimensionality reduction."""

    def test_single_sample_returns_unchanged(self):
        m = np.random.randn(1, 50).astype(np.float32)
        result = reduce_dimensions(m, 0.95, 10)
        np.testing.assert_array_equal(result, m)

    def test_reduces_dimensions(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((100, 50)).astype(np.float32)
        result = reduce_dimensions(m, 0.95, 10)
        assert result.shape == (100, 10)
        assert result.dtype == np.float32

    def test_caps_at_n_minus_1(self):
        rng = np.random.default_rng(42)
        # 5 samples, 20 features -> max components = 4 (n-1)
        m = rng.standard_normal((5, 20)).astype(np.float32)
        result = reduce_dimensions(m, 0.95, 50)
        assert result.shape == (5, 4)

    def test_caps_at_d(self):
        rng = np.random.default_rng(42)
        # 100 samples, 3 features -> max components = 3
        m = rng.standard_normal((100, 3)).astype(np.float32)
        result = reduce_dimensions(m, 0.95, 50)
        assert result.shape == (100, 3)

    def test_max_components_respected(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((50, 30)).astype(np.float32)
        result = reduce_dimensions(m, 0.95, 5)
        assert result.shape[1] == 5


class TestDetectOutliers:
    """Tests for Mahalanobis-based outlier detection."""

    def test_single_sample_returns_empty(self):
        m = np.random.randn(1, 5).astype(np.float32)
        result = detect_outliers(m, 3.0)
        assert result == []

    def test_empty_returns_empty(self):
        # n < 2 case
        m = np.empty((0, 5), dtype=np.float32)
        result = detect_outliers(m, 3.0)
        assert result == []

    def test_finds_outliers_in_synthetic_data(self):
        rng = np.random.default_rng(42)
        # Create a tight cluster with one obvious outlier
        cluster = rng.standard_normal((50, 3)).astype(np.float32) * 0.1
        outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        m = np.vstack([cluster, outlier])
        result = detect_outliers(m, 3.0)
        # The outlier (index 50) should be detected
        assert 50 in result

    def test_returns_list_of_ints(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((20, 3)).astype(np.float32)
        result = detect_outliers(m, 3.0)
        assert isinstance(result, list)
        for idx in result:
            assert isinstance(idx, (int, np.integer))

    def test_low_threshold_flags_more(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((30, 3)).astype(np.float32)
        low = detect_outliers(m, 1.0)
        high = detect_outliers(m, 10.0)
        assert len(low) >= len(high)


class TestFindDuplicatePairs:
    """Tests for cosine similarity duplicate detection."""

    def test_single_item_returns_empty(self):
        m = np.random.randn(1, 10).astype(np.float32)
        result = find_duplicate_pairs(m, 0.9)
        assert result == []

    def test_empty_returns_empty(self):
        m = np.empty((0, 10), dtype=np.float32)
        result = find_duplicate_pairs(m, 0.9)
        assert result == []

    def test_identical_vectors_detected(self):
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        m = np.vstack([vec, vec, np.random.randn(10).astype(np.float32)])
        result = find_duplicate_pairs(m, 0.99)
        # Items 0 and 1 are identical, should be found
        assert any(i == 0 and j == 1 for i, j, _ in result)

    def test_orthogonal_vectors_not_detected(self):
        m = np.eye(5, dtype=np.float32)
        result = find_duplicate_pairs(m, 0.5)
        assert result == []

    def test_returns_correct_format(self):
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        m = np.vstack([vec, vec])
        result = find_duplicate_pairs(m, 0.5)
        assert len(result) >= 1
        for item in result:
            assert len(item) == 3
            i, j, sim = item
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert isinstance(sim, float)
            assert i < j

    def test_i_less_than_j(self):
        rng = np.random.default_rng(42)
        # Create some similar vectors
        base = rng.standard_normal(10).astype(np.float32)
        m = np.vstack([base + rng.standard_normal(10) * 0.01 for _ in range(5)])
        result = find_duplicate_pairs(m, 0.5)
        for i, j, _ in result:
            assert i < j

    def test_high_threshold_fewer_pairs(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((10, 20)).astype(np.float32)
        # Normalize for meaningful cosine similarities
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / norms
        low = find_duplicate_pairs(m, 0.3)
        high = find_duplicate_pairs(m, 0.9)
        assert len(low) >= len(high)


class TestSuggestLabels:
    """Tests for embedding-based label suggestion."""

    def test_empty_items_returns_empty_lists(self):
        items = np.empty((0, 10), dtype=np.float32)
        labels = np.random.randn(3, 10).astype(np.float32)
        result = suggest_labels(items, labels, ["a", "b", "c"])
        assert result == []

    def test_empty_labels_returns_empty_per_item(self):
        items = np.random.randn(5, 10).astype(np.float32)
        labels = np.empty((0, 10), dtype=np.float32)
        result = suggest_labels(items, labels, [])
        assert len(result) == 5
        assert all(s == [] for s in result)

    def test_identical_embedding_gets_that_label(self):
        """If an item embedding equals a label embedding, it should suggest that label."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        items = np.array([vec], dtype=np.float32)
        labels = np.array([vec, [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = suggest_labels(items, labels, ["bug", "feature", "docs"], threshold=0.5)
        assert len(result) == 1
        assert result[0][0][0] == "bug"
        assert result[0][0][1] > 0.99

    def test_threshold_filters_low_similarity(self):
        """With a high threshold, orthogonal vectors should get no suggestions."""
        items = np.eye(3, dtype=np.float32)
        labels = np.eye(3, dtype=np.float32)
        # threshold=0.99 means only near-exact matches
        result = suggest_labels(items, labels, ["a", "b", "c"], threshold=0.99)
        # Each item should match exactly one label (itself)
        for sugs in result:
            assert len(sugs) == 1

    def test_max_per_item_respected(self):
        """Even if all labels are similar, max_per_item caps the results."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(10).astype(np.float32)
        items = np.array([base])
        # All labels very similar to item
        labels = np.array([base + rng.standard_normal(10) * 0.01 for _ in range(10)])
        names = [f"label-{i}" for i in range(10)]
        result = suggest_labels(items, labels, names, threshold=0.1, max_per_item=2)
        assert len(result[0]) <= 2

    def test_returns_sorted_by_similarity_descending(self):
        """Suggestions should be ordered highest similarity first."""
        items = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
        labels = np.array([
            [1.0, 0.0, 0.0],  # decent match
            [1.0, 0.5, 0.0],  # exact match
            [0.0, 0.0, 1.0],  # poor match
        ], dtype=np.float32)
        result = suggest_labels(items, labels, ["a", "b", "c"], threshold=0.1)
        scores = [s for _, s in result[0]]
        assert scores == sorted(scores, reverse=True)

    def test_returns_correct_format(self):
        rng = np.random.default_rng(42)
        items = rng.standard_normal((3, 10)).astype(np.float32)
        labels = rng.standard_normal((5, 10)).astype(np.float32)
        names = ["bug", "feature", "docs", "ci", "test"]
        result = suggest_labels(items, labels, names, threshold=0.0)
        assert len(result) == 3
        for sugs in result:
            for name, score in sugs:
                assert isinstance(name, str)
                assert isinstance(score, float)
                assert name in names
