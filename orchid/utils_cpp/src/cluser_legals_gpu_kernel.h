#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <variant>
#include <functional>
#include <vector>

inline void checkGpuStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

inline int GpuAlignment = 16;

inline size_t roundToAlignment(size_t number) {
    number = (number + GpuAlignment - 1);
    number /= GpuAlignment;
    number *= GpuAlignment;
    return number;
}

struct PromiseCapsule {
    void* ptr;
    std::function<void(void*)> free;
};

struct PromiseData {
    void* m_data = nullptr;
    size_t m_size = 0;
    size_t m_usedSize = 0;
    cudaEvent_t m_event = nullptr;
    cudaStream_t m_stream = nullptr;

    PromiseData(size_t size);
    ~PromiseData();

    void syncGpuData();
    void* pushDataArena(size_t size);
    PromiseCapsule collectRawCpuCopy();
};

PromiseCapsule copyToCpu(const void* gpuData, size_t size);

std::unique_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const void* posting_lists,
    const void* posting_sizes,
    const void* posting_queries,
    const void* dense_masks,
    const void* dense_mask_queries,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs);


std::unique_ptr<PromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const void* mask_packed_buffer,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs,
    const void* cluster_offsets,
    const void* cluster_counts,
    int32_t n_list);
