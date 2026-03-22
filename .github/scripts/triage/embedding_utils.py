"""Pure math utilities for triage sweep embedding analysis.

All functions are stateless and perform no I/O (except model loading by FastEmbed).
Each function operates on numpy arrays and returns numpy arrays or plain Python types.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from fastembed import TextEmbedding
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics.pairwise import cosine_similarity

# FastEmbed model — BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings.
# ~46MB quantized ONNX, runs on CPU in ~0.5s per batch of 32.
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# Embedding dimensionality (determined by model choice).
EMBEDDING_DIM: int = 384

# Batch size for FastEmbed. 32 balances memory and throughput on
# a 2-vCPU GitHub Actions runner with ~7GB RAM.
EMBEDDING_BATCH_SIZE: int = 32


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    """Embed a list of texts into dense vectors using FastEmbed.

    Returns an array of shape (len(texts), 384) with dtype float32.
    Empty input returns a (0, 384) array.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    vectors = list(model.embed(texts, batch_size=EMBEDDING_BATCH_SIZE))
    return np.vstack(vectors).astype(np.float32)


def normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize each row to unit length.

    Zero-norm rows (e.g. from empty text) remain zero vectors.
    Uses eps=1e-10 in the denominator to avoid division by zero.
    """
    if matrix.shape[0] == 0:
        return matrix

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-10)


def reduce_dimensions(
    matrix: NDArray[np.float32],
    variance_ratio: float,
    max_components: int,
) -> NDArray[np.float32]:
    """Reduce dimensionality via PCA.

    Computes n_components = min(max_components, n-1, d). If n_components < 1,
    returns the matrix unchanged. Logs explained variance for observability.
    The variance_ratio parameter documents intent but is not strictly enforced;
    the actual retained variance depends on the data and component cap.
    """
    n, d = matrix.shape
    if n <= 1:
        return matrix

    n_components = min(max_components, n - 1, d)
    if n_components < 1:
        return matrix

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {d}d -> {n_components}d, explained variance: {explained:.3f}")
    return reduced.astype(np.float32)


def detect_outliers(
    matrix: NDArray[np.float32],
    threshold: float,
) -> list[int]:
    """Flag items whose Mahalanobis distance exceeds the threshold.

    Uses EllipticEnvelope (robust covariance via MCD) to estimate the
    multivariate Gaussian, then computes sqrt(squared Mahalanobis distance)
    for each sample. Returns indices of outliers sorted ascending.
    """
    n = matrix.shape[0]
    if n < 2:
        return []

    envelope = EllipticEnvelope(contamination=0.1, random_state=42)
    envelope.fit(matrix)

    # .mahalanobis() returns squared Mahalanobis distances
    distances = np.sqrt(envelope.mahalanobis(matrix))
    outlier_mask = distances > threshold
    return list(np.where(outlier_mask)[0])


def find_duplicate_pairs(
    matrix: NDArray[np.float32],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Find pairs of items with cosine similarity above threshold.

    Returns (i, j, similarity) tuples where i < j. The input should be
    L2-normalized embeddings (full dimensionality, not PCA-reduced) so
    cosine similarity equals the dot product.
    """
    n = matrix.shape[0]
    if n <= 1:
        return []

    sim_matrix = cosine_similarity(matrix)
    # Upper triangle indices (i < j), excluding diagonal
    rows, cols = np.triu_indices(n, k=1)
    similarities = sim_matrix[rows, cols]

    mask = similarities > threshold
    pairs: list[tuple[int, int, float]] = []
    for idx in np.where(mask)[0]:
        pairs.append((int(rows[idx]), int(cols[idx]), float(similarities[idx])))

    return pairs


# ── Label suggestion via embedding similarity ────────────────────────

# Minimum similarity between an item and a label to suggest it.
# 0.4 is intentionally permissive — the report is for human review.
LABEL_SIMILARITY_THRESHOLD: float = 0.4

# Maximum number of labels to suggest per item.
MAX_LABELS_PER_ITEM: int = 3


def suggest_labels(
    item_embeddings: NDArray[np.float32],
    label_embeddings: NDArray[np.float32],
    label_names: list[str],
    threshold: float = LABEL_SIMILARITY_THRESHOLD,
    max_per_item: int = MAX_LABELS_PER_ITEM,
) -> list[list[tuple[str, float]]]:
    """Suggest labels for each item based on embedding similarity.

    Computes cosine similarity between item embeddings (n, 384) and
    label embeddings (m, 384). For each item, returns the top-k labels
    whose similarity exceeds the threshold, sorted by similarity descending.

    Returns a list of length n, where each element is a list of
    (label_name, similarity) tuples. Empty list if no label exceeds threshold.
    """
    n = item_embeddings.shape[0]
    m = label_embeddings.shape[0]
    if n == 0 or m == 0:
        return [[] for _ in range(n)]

    # (n, m) similarity matrix: each row is one item vs all labels
    sim_matrix = cosine_similarity(item_embeddings, label_embeddings)

    suggestions: list[list[tuple[str, float]]] = []
    for i in range(n):
        row = sim_matrix[i]
        # Indices sorted by similarity descending
        ranked = np.argsort(row)[::-1]
        item_labels: list[tuple[str, float]] = []
        for idx in ranked[:max_per_item]:
            score = float(row[idx])
            if score < threshold:
                break
            item_labels.append((label_names[idx], score))
        suggestions.append(item_labels)

    return suggestions
