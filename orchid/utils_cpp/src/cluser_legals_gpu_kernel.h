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

// Owns staging bytes without Python references or CUDA resources.
struct CpuUploadCapsule {
    std::vector<uint8_t> data;

    explicit CpuUploadCapsule(size_t size) : data(size) {}
};

struct PromiseData {
    void* m_data = nullptr;
    size_t m_size = 0;
    size_t m_usedSize = 0;
    cudaEvent_t m_event = nullptr;
    cudaStream_t m_stream = nullptr;

    PromiseData(size_t size);
    ~PromiseData();

    const void* getData() const;
    // Appends an aligned upload. Caller checks capacity and retains the source until sync.
    void uploadCpuDataAsync(const void* data, size_t size);
    // Takes ownership on success; deletes the capsule after preceding stream work.
    void finishCpuUploadAsync(CpuUploadCapsule* capsule);
    void syncGpuData();
    PromiseCapsule collectRawCpuCopy();
};

PromiseCapsule copyToCpu(const void* gpuData, size_t size);

std::shared_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const uint32_t* posting_lists,
    const int64_t* posting_sizes,
    const int64_t* posting_queries,
    const uint8_t* dense_masks,
    const int64_t* dense_mask_queries,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs);


std::shared_ptr<PromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const uint8_t* mask_packed_buffer,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs,
    const int32_t* cluster_offsets,
    const int32_t* cluster_counts,
    int32_t n_list);
