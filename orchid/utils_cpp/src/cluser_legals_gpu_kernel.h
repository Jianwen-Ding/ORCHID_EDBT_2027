#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <thread>
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

// Allows for the creation and deletion of 
// cuda page host allocation
struct PromiseCapsule {
    void* ptr;
    std::function<void(void*)> free;
};

// Owns a cuda stream.
struct StreamCapsule {
    cudaEvent_t m_streamStartEvent = nullptr;
    cudaStream_t m_stream = nullptr;

    void sync() const;
    StreamCapsule();
    ~StreamCapsule();
};

// Represents a block of data held up before a cuda event.
struct PromiseData {
protected:
    void* m_data{ nullptr };
    size_t m_size{ 0 }; // Logical byte count; allocation is aligned.

    // Event that marks the completion of the promise
    cudaEvent_t m_event{ nullptr };
    std::shared_ptr<const StreamCapsule> m_stream{ nullptr };

    PromiseData(std::shared_ptr<const StreamCapsule> stream);
    ~PromiseData();

public:
    PromiseData(const PromiseData&) = delete;
    PromiseData& operator=(const PromiseData&) = delete;

    // Sets an event at the end of the stream.
    void finalizePromise();

    // Getters and setters
    std::shared_ptr<const StreamCapsule> getStream() const;
    size_t getSize() const;
    const void* getData() const;
    void* getData();

    // Forces event to be synced
    void sync() const;
};

// Owns pinned Cpu data that may be in transit from the Gpu.
class ReadbackPromiseData : public PromiseData {
public:
    ReadbackPromiseData(size_t size, std::shared_ptr<const StreamCapsule> stream = nullptr);
    ~ReadbackPromiseData();

    // Caller checks capacity and retains d_data until the copy completes.
    void uploadGpuDataAsync(const void* d_data, size_t size, size_t offset);
};


// Represents a block of data to be completed within the Gpu
class GpuPromiseData : public PromiseData {
public:
    GpuPromiseData(size_t size, std::shared_ptr<const StreamCapsule> stream = nullptr);
    ~GpuPromiseData();

    void uploadValueAsync(int value, size_t size, size_t offset);
    // Up to caller to retain data within h_data until data has been fully copied in
    void uploadCpuDataAsync(const void* h_data, size_t size, size_t offset);
    // Returns a completed CPU copy.
    std::shared_ptr<const ReadbackPromiseData> collectRawCpuCopy() const;
    // Copies a portion of the gpu data within bounds to cpu
    std::shared_ptr<const ReadbackPromiseData> copyToCpu(const void* d_gpuData, size_t size) const;
};

std::shared_ptr<GpuPromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<std::shared_ptr<const StreamCapsule>> blockingPromises,
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


std::shared_ptr<GpuPromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<std::shared_ptr<const StreamCapsule>> blockingPromises,
    const uint8_t* d_maskPackedBuffer,
    int32_t nQueries,
    int32_t bytesPerQuery,
    const int32_t* d_clusterOffsets,
    const int32_t* d_clusterCounts,
    int32_t nList);
