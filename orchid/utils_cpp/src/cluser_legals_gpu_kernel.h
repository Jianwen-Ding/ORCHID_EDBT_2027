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

inline size_t round(size_t number, int roundedNumber) {
    number = (number + roundedNumber - 1);
    number /= roundedNumber;
    number *= roundedNumber;
    return number;
}
inline size_t roundToAlignment(size_t number) {
    return round(number, GpuAlignment);
}

struct PromiseCapsule {
    void* ptr;
    std::function<void(void*)> free;
};

struct StreamCapsule {
    cudaEvent_t m_streamStartEvent = nullptr;
    cudaStream_t m_stream = nullptr;

    void sync();
    StreamCapsule();
    ~StreamCapsule();
};

// Represents a block of data in the 
struct PromiseData {
private:
    void* m_data{ nullptr };
    size_t m_size{ 0 }; // Logical byte count; allocation is aligned.

    // Event that marks the completion of the promise
    cudaEvent_t m_event{ nullptr };
    std::shared_ptr<StreamCapsule> m_stream{ nullptr };

public:
    PromiseData(size_t size, std::shared_ptr<StreamCapsule> stream = nullptr);
    ~PromiseData();

    // Sets an event at the end of the stream.
    void finalizePromise();

    // Getters and setters
    const void* getData() const;
    void* getData();
    size_t getSize() const;


    // Uploads at a byte offset. Caller checks capacity and retains the source until sync.
    void uploadValueAsync(int value, size_t size, size_t offset);
    void uploadCpuDataAsync(const void* h_data, size_t size, size_t offset);
    void syncGpuData() const;
    PromiseCapsule collectRawCpuCopy();
};

PromiseCapsule copyToCpu(const void* d_gpuData, size_t size);

std::shared_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<std::shared_ptr<StreamCapsule>> blockingPromises,
    const uint32_t* d_postingLists,
    const int32_t* d_postingSizes,
    const int32_t* d_postingQueries,
    const uint8_t* d_denseMasks,
    const int64_t* h_denseMaskQueries,
    int32_t nQueries,
    int32_t bytesPerQuery,
    int32_t postingCount,
    int32_t denseMaskCount,
    int64_t postingEntries);


std::shared_ptr<PromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<std::shared_ptr<StreamCapsule>> blockingPromises,
    const uint8_t* d_maskPackedBuffer,
    int32_t nQueries,
    int32_t bytesPerQuery,
    const int32_t* d_clusterOffsets,
    const int32_t* d_clusterCounts,
    int32_t nList);
