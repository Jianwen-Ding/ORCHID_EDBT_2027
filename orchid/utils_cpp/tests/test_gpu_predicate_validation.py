"""Validate GPU bitmap buffer dimensions before launching the CUDA kernel."""

import numpy as np
import pytest

from utils_cpp import gpu


@pytest.fixture
def uploaded_inputs():
    def prepare(**changes):
        values = dict(
            posting_lists=np.array([1, 3], dtype=np.uint32),
            posting_sizes=np.array([2], dtype=np.int64),
            posting_queries=np.array([0], dtype=np.int64),
            dense_masks=np.array([1, 0, 4], dtype=np.uint8),
            dense_mask_queries=np.array([1], dtype=np.int64),
        )
        values.update(changes)
        try:
            spans = [gpu.gpu_prep(value) for value in values.values()]
            promise = gpu.upload_to_gpu(*spans)
        except RuntimeError as error:
            if "GPU backend not compiled" in str(error):
                pytest.skip("GPU backend not compiled")
            raise
        return spans, promise
    return prepare


@pytest.mark.parametrize("changes,n_queries,n_docs,message", [
    ({}, -1, 19, "n_queries must be nonnegative"),
    ({}, 2, 0, "n_docs must be positive"),
    ({}, 2, -1, "n_docs must be positive"),
    ({"posting_sizes": np.array([], dtype=np.int64)}, 2, 19, "same number of elements"),
    ({}, 1, 19, "total n_queries"),
    ({}, 3, 19, "total n_queries"),
    ({"dense_masks": np.array([1, 0], dtype=np.uint8)}, 2, 19, "bytes per dense query"),
    ({"dense_masks": np.array([1, 0, 4, 0], dtype=np.uint8)}, 2, 19, "bytes per dense query"),
    ({"posting_sizes": np.array([], dtype=np.int64),
      "posting_queries": np.array([], dtype=np.int64)}, 1, 19, "posting_lists must be empty"),
    ({"dense_mask_queries": np.array([], dtype=np.int64)}, 2, 19, "total n_queries"),
])
def test_rejects_inconsistent_span_sizes(uploaded_inputs, changes, n_queries, n_docs, message):
    spans, promise = uploaded_inputs(**changes)
    with pytest.raises(ValueError, match=message):
        gpu.predicate_results_to_bitmap(*spans, n_queries, n_docs)
