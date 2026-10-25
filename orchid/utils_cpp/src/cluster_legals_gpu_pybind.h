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
    void* const m_data{ nullptr };
    size_t m_size{ 0 };
    std::shared_ptr<GpuPromise> m_promise{ nullptr };

    GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise);
};

// A typed span of Gpu data
template<typename T>
class GpuSpan {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a primitive type.");

    bool m_uploaded{ 0 };
    py::array_t<T, py::array::c_style> m_cpuInformation{ };
    std::unique_ptr<GpuBuffer> m_gpuBuffer{ };

    friend class detail::GpuSpanTemplateUtil;
public:
    GpuSpan(void* data, size_t size, std::shared_ptr<GpuPromise> promise);

    // Pybind API
    
    // This is a hard cpu/gpu sync which may not be the most efficent
    // Ideally the entire pipeline is GPU until FAISS so it ought to be fine.
    py::array_t<T, py::array::c_style > collectToCpu() const;
    
    // Gets underlying buffer, Spassuming it has already been uploaded.
    GpuBuffer getBuffer();
};

// Helper function for adding GPU span
template<typename T>
void pymoduleAddGpuSpan(py::module_ &m) {
    std::string typeName = py::type_id<T>();
    py::class_<GpuSpan<T>, GpuBuffer>(m, typeName + "GpuSpan")
        .def("collect_to_cpu", &GpuSpan<T>::collectToCpu);
}

namespace detail {
    // Since GpuSpan is a templated type, this util allows
    // a limited amount of templates to be sent to pybind.
    template <typename... T>
    class GpuSpanTemplateUtil {    
    public:
        // Gpu span upload information

        // Sizes a single gpu span
        static size_t getSpanSizesForUpload(py::handle item) {
            bool isSupportedSpan = false;
            size_t accumulatedSize = 0;
            ([&]{
                if (py::isinstance<GpuSpan<T>>(item))
                    GpuSpan<T>& span = item.cast<GpuSpan<T>&>();
                    isSupportedSpan = true;

                    if (span.m_uploaded) {
                        throw std::runtime_error("Spans cannot be reuploaded again");
                    }

                    span.
                })
            
            if (!isSupportedSpan) {
                throw std::runtime_error("Uploaded type is not supported");
            }
        }

        // Initializes buffer data for a gpu span.
        static void setSpanBuffer(py::handle item, GpuBuffer&& buffer) {
            bool successfulSet = false;

            // Gets a shallow copy if it already is in c style
            (..., ([&]{
                if (!successfulSet && py::isinstance<GpuSpan<T>>(item)) {
                    GpuSpan<T>& span = item.cast<GpuSpan<T>&>();
                    span.m_gpuBuffer = buffer;
                    successfulSet = true;
                }
            }()));

            if (successfulSet) {
                throw std::runtime_error("Given item is not a supported span");
            }
        }

        // Gpu span initialization logic

        // Formats object to be uploaded to gpu
        static py::object formatGpuUpload(py::object item) {
            std::optional<std::variant<py::array_t<T, py::array::c_style>...>> retArray;

            // Gets a shallow copy if it already is in c style
            (..., ([&]{
                if (py::isinstance<py::array_t<T>>(item)) {
                    py::array arr = py::reinterpret_borrow<py::array>(item);
                    if (arr.flags() & py::array::c_style) {

                        retArray = py::reinterpret_borrow<py::array_t<T, py::array::c_style>>(item);
                    }
                }
            }()));

            // Casts and causes a deep copy if non c style 
            if (!retArray.has_value()) {
                (..., ([&]{
                    if (py::isinstance<py::array_t<T>>(item))
                        retArray = item.cast<py::array_t<T, py::array::c_style>>();
                }()));
            }

            // Deals with direct scalar conversion
            if (!retArray.has_value()) {
                (..., ([&]{
                    if (py::isinstance<T>(item))
                        retArray = py::array_t<T, py::array::c_style>({}, item.cast<T>());
                }()));
            }
                
            // If none of the attempted casts work it means that an invalid object was loaded in
            if (!retArray.has_value()) {
                throw std::runtime_error("Uploaded type is not supported");
            }

            // Extract a corresponding span from the cpu data type
            py::object result = py::none();
            bool hasAddedSpan = false;
            (..., ([&]{
                if (!hasAddedSpan && py::array_t<T, py::array::c_style>* ptr = 
                    std::get_if<py::array_t<T, py::array::c_style>>(&*retArray)) {

                    GpuSpan<T> retSpan;
                    retSpan.m_uploaded = false;
                    retSpan.m_cpuInformation = *ptr;
                    result = py::cast(retSpan);
                    bool hasGottenSpan = true;
                }
            }()));

            if (result.is_none()) {
                throw runtime_error("Unreachable point in formatGpuUpload, should have already uploaded");
            }
            
            return result;
        }

        // Gpu span binding logic 

        static void addAllSupportedGpuSpan(py::module_ &m) {
            ([&]{
                pymoduleAddGpuSpan<T>(m);
            }(), ...)
        }
    };

    typedef GpuSpanTemplateUtil
        <uint8_t,uint32_t, int32_t, int64_t> SpanUtil;
}


// Hidden implementation of Gpu Data
struct PromiseData;

// A set of Gpu data that may or may not be in transit
class GpuPromise {
public:
    // >>> C++/Cuda exclusive <<<
    std::unique_ptr<PromiseData> m_data;

    GpuPromise(std::unique_ptr<PromiseData> data);
    const GpuPromise& operator=(const GpuPromise& promise) = delete;

    // >>> Pybind <<<
    GpuSpan<uint8_t> collectBufferAsync();
    GpuSpan<uint8_t> collectBufferSync();
};

// Prepares a buffer of numberic values to be uploaded to the gpu
// Doesn't upload the buffer just yet prepares data to be uploaded
py::object gpuPrep_out_py(py::args args);

// Actually collects the buffers to be sent together in a single GPU promise.
std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args);

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
template <typename T>
py::array_t<T, py::array::c_style> GpuSpan<T>::collectToCpu() const {
    if (!m_uploaded) {
        throw std::runtime_error("Span has not been uploaded to the CPU yet");
    }
    if (m_gpuBuffer.m_data == nullptr && m_gpuBuffer.m_size != 0) {
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
    return GpuSpan<uint8_t>(m_gpuBuffer.m_data->m_data, m_gpuBuffer.m_data->m_size, shared_from_this());
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    m_data->syncGpuData();
    return collectBufferAsync();
}
inline py::object gpuPrep_out_py(py::args args) {
    for (py::object item : args) {
        detail::SpanUtils::formatGpuUpload(item);
    }
}
inline std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args) {
    size_t argLength = py::len(args);
    std::vector<py::handle> dataHandles;
    dataHandles.resize(argLength);
    std::vector<size_t> dataSizes;
    dataSize.resize(arglength);

    // Ensures item sizes are properly accounted for
    for (py::handle item : sizingData) {
        dataHandles.push_back(item);
        size_t itemSize = detail::SpanUtil::getSpanSizesForUpload(item);
        dataSizes.push_back(itemSize);
        cumulative_size += roundToAlignment(itemSize);
    }

    // Creates a block of data with needed size
    std::unique_ptr<PromiseData> promiseData = std::make_unique<PromiseData>(cumulative_size);
    void* rawDataIter = m_data.getData();
    std::shared_ptr<GpuPromise> promise = std::make_shared(std::move(promiseData));

    // Begins partitioning data along block of data
    for (int i = 0 ; i < argLength ; i++) {
        GpuBuffer addBuffer {rawDataIter, dataSizes[i], promise};
        detail::SpanUtil::setSpanBuffer(dataHandles[i], addBuffer);

        rawDataIter = static_cast<void*>(static_cast<uint8_t*>(rawDataIter) + roundToAlignment(dataSize));
    }
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
inline GpuSpan<T>::setGpuBuffer()
template <typename T>
inline py::array_t<T, py::array::c_style> GpuSpan<T>::collectToCpu() const {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline const GpuBuffer GpuSpan<T>::getBuffer() {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline py::object gpuPrep_out_py() {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuPromise uploadToGpu_out_py(py::args args) {
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

inline void addGpuModule(py::module_ &m) {
    py::class_<GpuBuffer>(m, "GpuBuffer")
        .def_readonly("data", &GpuBuffer::m_data)
        .def_readonly("size", &GpuBuffer::m_size)
        .def_readonly("promise", &GpuBuffer::m_promise);

    detail::SpanUtil::addAllSupportedGpuSpan(m);
}
