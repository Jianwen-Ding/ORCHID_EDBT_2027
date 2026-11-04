"""Check the GPU components used by a small ORCHID SIFT run."""

import sys
from pathlib import Path


def probe_gpu() -> None:
    import faiss
    import numpy as np

    required = (
        "get_num_gpus", "StandardGpuResources", "GpuClonerOptions",
        "index_cpu_to_gpu", "INDICES_64_BIT",
    )
    if any(not hasattr(faiss, name) for name in required):
        raise RuntimeError("installed FAISS is CPU-only; rebuild with CUDA support")
    if faiss.get_num_gpus() < 1:
        raise RuntimeError("no CUDA GPU is visible on this host")

    # Exercise the same cloning options as the benchmark before clustering.
    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.use_cuvs = False
    index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(1), options)
    if index is None:
        raise RuntimeError("FAISS could not create a GPU index")

    from predicate_index import PackedBitmap, PostingList, materialize_results_gpu

    results = [
        PostingList(np.array([0, 8, 18], dtype=np.uint32)),
        PackedBitmap(np.array([2, 128, 0], dtype=np.uint8)),
    ]
    actual = np.asarray(
        materialize_results_gpu(results, 19).collect_to_cpu()
    ).reshape(2, 3)
    np.testing.assert_array_equal(
        actual, np.array([[1, 1, 4], [2, 128, 0]], dtype=np.uint8)
    )


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchid"))
    try:
        probe_gpu()
    except (
        ImportError, AttributeError, RuntimeError, ValueError, TypeError,
        AssertionError, OSError,
    ) as exc:
        print(f"SIFT GPU unavailable: {exc}", flush=True)
        return 1
    print("SIFT GPU probe passed: FAISS cloning and mask materialization", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
