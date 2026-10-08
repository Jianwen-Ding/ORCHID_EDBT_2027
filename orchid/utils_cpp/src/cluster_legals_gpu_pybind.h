#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <type_traits>
#include <algorithm>
#include <stdexcept> 
#include <cmath>
#include <memory>
#include <cstdint>
#include <cstring>

namespace py = pybind11;

class GpuPromise;

// A non owning buffer of Gpu data
class GpuBuffer {
public:
    void* m_data;
    size_t m_size;
    std::shared_ptr<GpuPromise> m_promise;

    GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise);
};

// A typed span of Gpu data
template<typename T>
class GpuSpan : public GpuBuffer {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a primitive type.");

public:
    GpuSpan(void* data, size_t size, std::shared_ptr<GpuPromise> promise);
    
    // This is a hard cpu/gpu sync which may not be the most efficent
    // Ideally the entire pipeline is GPU until FAISS so it ought to be fine.
    py::array_t<T, py::array::c_style | py::array::forcecast> collectToCpu() const;
};

// Hidden implementation of Gpu Data
struct PromiseData;

// A set of Gpu data that may or may not be in transit
class GpuPromise : public std::enable_shared_from_this<GpuPromise> {
public:
    // >>> C++/Cuda exclusive <<<
    std::unique_ptr<PromiseData> m_data;

    GpuPromise(std::unique_ptr<PromiseData> data);
    const GpuPromise& operator=(const GpuPromise& promise) = delete;

    // >>> Pybind <<<
    GpuPromise(py::args sizingData);

    GpuSpan<uint8_t> collectBufferAsync();
    GpuSpan<uint8_t> collectBufferSync();
};

template <typename T>
GpuSpan<T> collectSpan(std::shared_ptr<GpuPromise> promise, py::array_t<T, py::array::c_style | py::array::forcecast> cpuData);

#if defined(ENABLE_GPU)

#include "cluser_legals_gpu_kernel.h"
#include <vector>
static inline std::vector<PromiseData*> gatherPromises(std::vector<const GpuBuffer*> buffers) {
    std::vector<PromiseData*> ret;
    for (size_t i = 0 ; i < buffers.size() ; i += 1) {
        PromiseData* promise = buffers[i]->m_promise->m_data.get();
        if (std::find(ret.begin(), ret.end(), promise) == ret.end()) {
            ret.push_back(promise);
        }
    }
    return ret;
}
inline GpuBuffer::GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise)
    : m_data(data), m_size(size), m_promise(std::move(promise)) {}
template <typename T>
inline GpuSpan<T>::GpuSpan(void* data, size_t size, std::shared_ptr<GpuPromise> promise)
    : GpuBuffer(data, size, std::move(promise)) {}
inline GpuPromise::GpuPromise(std::unique_ptr<PromiseData> data) : m_data(std::move(data)) {
    if (!m_data) {
        throw std::runtime_error("GPU kernel returned no data");
    }
}
inline GpuPromise::GpuPromise(py::args sizingData) {
    size_t cumulative_size = 0;
    for (auto item : sizingData) {
        if (py::hasattr(item, "nbytes")) {
            cumulative_size += roundToAlignment(item.attr("nbytes").cast<size_t>());
        }
    }
    
    m_data = std::make_unique<PromiseData>(cumulative_size);
}
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> GpuSpan<T>::collectToCpu() const {
    if (m_data == nullptr && m_size != 0) {
        throw std::runtime_error("Span not given valid data");
    }

    const size_t rawDataSize = m_size;

    if (rawDataSize % sizeof(T) != 0) {
        throw std::runtime_error("Buffer size is not divisible by sizeof(T)");
    }

    m_promise->m_data->syncGpuData();
    auto rawDataCapsule = std::make_unique<PromiseCapsule>(copyToCpu(m_data, m_size));
    void* cpuData = rawDataCapsule->ptr;
    py::capsule rawData(rawDataCapsule.get(), [](void* raw) {
        std::unique_ptr<PromiseCapsule> capsule(static_cast<PromiseCapsule*>(raw));
        capsule->free(capsule->ptr);
    });
    rawDataCapsule.release();

    return py::array_t<T, py::array::c_style | py::array::forcecast>(
            { static_cast<py::ssize_t>(rawDataSize / sizeof(T)) },      // shape
            { static_cast<py::ssize_t>(sizeof(T)) }, // strides in bytes
            static_cast<T*>(cpuData),
            rawData);
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    return GpuSpan<uint8_t>(m_data->m_data, m_data->m_size, shared_from_this());
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    m_data->syncGpuData();
    return collectBufferAsync();
}
template <typename T>
inline GpuSpan<T> collectSpan(std::shared_ptr<GpuPromise> promise, py::array_t<T, py::array::c_style | py::array::forcecast> cpuData) {
    size_t size = cpuData.nbytes();
    void* data = promise->m_data->pushDataArena(size);
    if (size != 0) {
        checkGpuStatus(cudaMemcpyAsync(data, cpuData.data(), size, cudaMemcpyHostToDevice,
                                      promise->m_data->m_stream));
        // Keep the CPU input alive until its transfer completes.
        promise->m_data->syncGpuData();
    }
    return GpuSpan<T>(data, size, promise);
}

inline GpuSpan<uint8_t> predicate_results_to_bitmap_gpu_out_py (
    const GpuSpan<uint32_t> posting_lists,
    const GpuSpan<int64_t> posting_sizes,
    const GpuSpan<int64_t> posting_queries,
    const GpuSpan<uint8_t> dense_masks,
    const GpuSpan<int64_t> dense_mask_queries,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs) {
    // TODO: Add more checking 
    auto result = std::make_shared<GpuPromise>(predicate_results_to_bitmap_gpu_impl(
        gatherPromises({&posting_lists, &posting_sizes, &posting_queries, &dense_masks, &dense_mask_queries}),
        posting_lists.m_data, posting_sizes.m_data, posting_queries.m_data,
        dense_masks.m_data, dense_mask_queries.m_data, n_queries, bytes_per_query, n_docs));
    return result->collectBufferAsync();
}

inline GpuSpan<float> cluster_legals_preordered_packed_gpu_out_py (
    const GpuSpan<uint8_t> mask_packed_buffer,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs,
    const GpuSpan<int32_t> cluster_offsets,
    const GpuSpan<int32_t> cluster_counts,
    int32_t n_list) {
    // TODO: Add more checking 
    auto result = std::make_shared<GpuPromise>(cluster_legals_preordered_packed_gpu_impl(
        gatherPromises({&mask_packed_buffer, &cluster_offsets, &cluster_counts}),
        mask_packed_buffer.m_data, n_queries, bytes_per_query, n_docs,
        cluster_offsets.m_data, cluster_counts.m_data, n_list));
    return GpuSpan<float>(result->m_data->m_data, result->m_data->m_size, result);
}

#else

// If Gpu logic is not support it should throw on any GPU specific logic 
// that attempts to be called. 
struct PromiseData { };

inline GpuBuffer::GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise) {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline GpuSpan<T>::GpuSpan(void* data, size_t size, std::shared_ptr<GpuPromise> promise) : GpuBuffer(data, size, promise) {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuPromise::GpuPromise(std::unique_ptr<PromiseData> data) {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuPromise::GpuPromise(py::args sizingData) {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> GpuSpan<T>::collectToCpu() const {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline GpuSpan<T> collectSpan(std::shared_ptr<GpuPromise> promise, py::array_t<T, py::array::c_style | py::array::forcecast> cpuData) {
    throw std::runtime_error("GPU backend not compiled");
}

inline GpuSpan<uint8_t> predicate_results_to_bitmap_gpu_out_py (
    const GpuSpan<uint32_t> posting_lists,
    const GpuSpan<int64_t> posting_sizes,
    const GpuSpan<int64_t> posting_queries,
    const GpuSpan<uint8_t> dense_masks,
    const GpuSpan<int64_t> mask_queries,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs) {
    throw std::runtime_error("GPU backend not compiled");
}

inline GpuSpan<float> cluster_legals_preordered_packed_gpu_out_py (
    const GpuSpan<uint8_t> mask_packed_buffer,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs,
    const GpuSpan<int32_t> cluster_offsets,
    const GpuSpan<int32_t> cluster_counts,
    int32_t n_list) {
    throw std::runtime_error("GPU backend not compiled");
}

#endif
