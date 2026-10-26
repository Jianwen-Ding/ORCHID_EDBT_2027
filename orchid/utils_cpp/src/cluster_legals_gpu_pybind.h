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
#include <optional>
#include <variant>
#include <string>
#include <vector>
#include <limits>
#include <utility>

#if defined(ENABLE_GPU)
#include "cluser_legals_gpu_kernel.h"
#endif

namespace py = pybind11;

class GpuPromise;
namespace detail { template <typename... T> class GpuSpanTemplateUtil; }

// A non owning buffer of Gpu data
class GpuBuffer {
public:
    void* m_data{ nullptr };
    size_t m_size{ 0 };
    std::shared_ptr<GpuPromise> m_promise{ nullptr };

    GpuBuffer() = default;
    GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise);
};

// A typed span of Gpu data
template<typename T>
class GpuSpan : public GpuBuffer {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a primitive type.");

    bool m_uploaded{ 0 };
    py::array_t<T, py::array::c_style> m_cpuInformation{ };

    template <typename...> friend class detail::GpuSpanTemplateUtil;
public:
    GpuSpan() = default;
    GpuSpan(void* data, size_t size, std::shared_ptr<GpuPromise> promise);

    // Pybind API
    
    // This is a hard cpu/gpu sync which may not be the most efficent
    // Ideally the entire pipeline is GPU until FAISS so it ought to be fine.
    py::array_t<T, py::array::c_style > collectToCpu() const;
    
    // Gets underlying buffer, assuming it has already been uploaded.
    const GpuBuffer& getBuffer() const;
};


namespace detail {
    // Helper function for adding GPU span
    template<typename T>
    void pymoduleAddGpuSpan(py::module_ &m, const std::string& name) {
        py::class_<GpuSpan<T>, GpuBuffer>(m, (name + "_gpu_span").c_str())
            .def("collect_to_cpu", &GpuSpan<T>::collectToCpu);
    }

    // Structural string type for C++20 non-type template parameters.
    template <std::size_t N>
    struct FixedString {
        char value[N];

        constexpr FixedString(const char (&text)[N]) {
            std::copy_n(text, N, value);
        }
    };

    template <typename T, FixedString Name>
    struct NamedType {
        using type = T;
        static constexpr auto name = Name;
    };

    // Since GpuSpan is a templated type, this util allows
    // a limited amount of templates to be sent to pybind.
    template <typename... Entries>
    class GpuSpanTemplateUtil {    
    public:
        // Gpu span upload information

        // Sizes a single gpu span
        static size_t getSpanSizesForUpload(py::handle item) {
            bool isSupportedSpan = false;
            size_t accumulatedSize = 0;
            ([&]{
                if (py::isinstance<GpuSpan<typename Entries::type>>(item)) {
                    GpuSpan<typename Entries::type>& span = item.cast<GpuSpan<typename Entries::type>&>();
                    isSupportedSpan = true;

                    if (span.m_uploaded) {
                        throw std::runtime_error("Spans cannot be reuploaded again");
                    }

                    accumulatedSize += span.m_cpuInformation.nbytes();
                }
            }(), ...);
            
            if (!isSupportedSpan) {
                throw std::runtime_error("Uploaded type is not supported");
            }
            return accumulatedSize;
        }

        // Attaches the GPU buffer and returns its CPU upload source.
        static py::array setSpanBuffer(py::handle item, GpuBuffer&& buffer) {
            bool successfulSet = false;
            py::array source;

            // Gets a shallow copy if it already is in c style
            (..., ([&]{
                if (!successfulSet && py::isinstance<GpuSpan<typename Entries::type>>(item)) {
                    GpuSpan<typename Entries::type>& span = item.cast<GpuSpan<typename Entries::type>&>();
                    if (span.m_uploaded) {
                        throw std::runtime_error("Spans cannot be reuploaded again");
                    }
                    source = std::move(span.m_cpuInformation);
                    static_cast<GpuBuffer&>(span) = std::move(buffer);
                    span.m_uploaded = true;
                    successfulSet = true;
                }
            }()));

            if (!successfulSet) {
                throw std::runtime_error("Given item is not a supported span");
            }
            return source;
        }

        // Gpu span initialization logic

        // Formats object to be uploaded to gpu
        static py::object formatGpuUpload(py::object item) {
            std::optional<std::variant<py::array_t<typename Entries::type, py::array::c_style>...>> retArray;

            // Gets a shallow copy if it already is in c style
            (..., ([&]{
                if (py::isinstance<py::array_t<typename Entries::type>>(item)) {
                    py::array arr = py::reinterpret_borrow<py::array>(item);
                    if (arr.flags() & py::array::c_style) {

                        retArray = py::reinterpret_borrow<py::array_t<typename Entries::type, py::array::c_style>>(item);
                    }
                }
            }()));

            // Casts and causes a deep copy if non c style 
            if (!retArray.has_value()) {
                (..., ([&]{
                    if (py::isinstance<py::array_t<typename Entries::type>>(item))
                        retArray = item.cast<py::array_t<typename Entries::type, py::array::c_style>>();
                }()));
            }

            // Let NumPy infer scalar dtypes, then retry the supported array checks.
            if (!py::isinstance<py::array>(item) && PyNumber_Check(item.ptr())) {
                py::array scalar = py::array::ensure(item);
                if (scalar) {
                    return formatGpuUpload(std::move(scalar));
                }
            }

            // If none of the attempted casts work it means that an invalid object was loaded in
            if (!retArray.has_value()) {
                throw std::runtime_error("Uploaded type is not supported");
            }

            // Extract a corresponding span from the cpu data type
            py::object result = py::none();
            bool hasAddedSpan = false;
            (..., ([&]{
                if (auto* ptr = std::get_if<py::array_t<typename Entries::type, py::array::c_style>>(&*retArray);
                    !hasAddedSpan && ptr) {

                    GpuSpan<typename Entries::type> retSpan{ };
                    retSpan.m_uploaded = false;
                    retSpan.m_cpuInformation = *ptr;
                    result = py::cast(std::move(retSpan));
                    hasAddedSpan = true;
                }
            }()));

            if (result.is_none()) {
                throw std::runtime_error("Unreachable point in formatGpuUpload, should have already uploaded");
            }
            
            return result;
        }

        // Gpu span binding logic 

        static void addAllSupportedGpuSpan(py::module_ &m) {
            ([&]{
                pymoduleAddGpuSpan<typename Entries::type>(m, Entries::name.value);
            }(), ...);
        }
    };

    // Extract a type's written name from the macro's stringified argument list.
    template <FixedString Names, std::size_t Index>
    constexpr auto gpuSpanTypeName() {
        std::size_t begin = 0;
        for (std::size_t i = 0; i < Index; ++i) {
            while (Names.value[begin] != ',' && Names.value[begin] != '\0') ++begin;
            if (Names.value[begin] == ',') ++begin;
        }
        const auto isSpace = [](char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        };
        while (isSpace(Names.value[begin])) ++begin;
        std::size_t end = begin;
        while (Names.value[end] != ',' && Names.value[end] != '\0') ++end;
        while (end > begin && isSpace(Names.value[end - 1])) --end;

        auto name = Names;
        std::fill_n(name.value, sizeof(name.value), '\0');
        std::copy_n(Names.value + begin, end - begin, name.value);
        return name;
    }

    template <FixedString Names, typename... Types>
    struct GpuSpanTypes {
        template <std::size_t... Indices>
        static auto make(std::index_sequence<Indices...>)
            -> GpuSpanTemplateUtil<NamedType<Types, gpuSpanTypeName<Names, Indices>()>...>;

        using type = decltype(make(std::index_sequence_for<Types...>{}));
    };

    // Preserve the spelling of each type, including typedefs, as its Python name.
    // Example: using u8 = uint8_t; ORCHID_GPU_SPAN_UTIL(MySpans, u8, int32_t);
    #define ORCHID_GPU_SPAN_UTIL(Alias, ...) \
        typedef typename ::detail::GpuSpanTypes<#__VA_ARGS__, __VA_ARGS__>::type Alias

    using byte = uint8_t;
    using u32 = uint32_t;
    using i32 = int32_t;
    using i64 = int64_t;
    ORCHID_GPU_SPAN_UTIL(SpanUtil, byte, u32, i32, i64, float);
}


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
    GpuSpan<uint8_t> collectBufferAsync();
    GpuSpan<uint8_t> collectBufferSync();
};

// Prepares a buffer of numberic values to be uploaded to the gpu
// Doesn't upload the buffer just yet prepares data to be uploaded
py::object gpuPrep_out_py(py::args args);

// Actually collects the buffers to be sent together in a single GPU promise.
std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args);

#if defined(ENABLE_GPU)

#include <vector>
static inline std::vector<PromiseData*> gatherPromises(std::vector<const GpuBuffer*> buffers) {
    std::vector<PromiseData*> ret;
    for (size_t i = 0 ; i < buffers.size() ; i += 1) {
        if (!buffers[i]->m_promise) {
            throw std::runtime_error("Span has not been uploaded to the GPU yet");
        }
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
    : GpuBuffer(data, size, std::move(promise)), m_uploaded(true) {}
inline GpuPromise::GpuPromise(std::unique_ptr<PromiseData> data) : m_data(std::move(data)) {
    if (!m_data) {
        throw std::runtime_error("GPU kernel returned no data");
    }
}
template <typename T>
py::array_t<T, py::array::c_style> GpuSpan<T>::collectToCpu() const {
    if (!m_uploaded) {
        throw std::runtime_error("Span has not been uploaded to the GPU yet");
    }
    if (m_data == nullptr && m_size != 0) {
        throw std::runtime_error("Span not given valid data");
    }

    const size_t rawDataSize = m_size;

    if (rawDataSize % sizeof(T) != 0) {
        throw std::runtime_error("Buffer size is not divisible by sizeof(T)");
    }

    if (!m_promise) {
        throw std::runtime_error("Span has no GPU promise");
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
template <typename T>
inline const GpuBuffer& GpuSpan<T>::getBuffer() const {
    if (!m_uploaded) {
        throw std::runtime_error("Span has not been uploaded to the GPU yet");
    }
    return *this;
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    return GpuSpan<uint8_t>(m_data->m_data, m_data->m_size, shared_from_this());
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    m_data->syncGpuData();
    return collectBufferAsync();
}
inline py::object gpuPrep_out_py(py::args args) {
    py::tuple spans(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        spans[i] = detail::SpanUtil::formatGpuUpload(py::reinterpret_borrow<py::object>(args[i]));
    }
    return std::move(spans);
}
inline std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args) {
    size_t argLength = args.size();
    std::vector<py::handle> dataHandles;
    dataHandles.reserve(argLength);
    std::vector<size_t> dataSizes;
    dataSizes.reserve(argLength);
    size_t cumulative_size = 0;

    // Ensures item sizes are properly accounted for
    for (py::handle item : args) {
        dataHandles.push_back(item);
        size_t itemSize = detail::SpanUtil::getSpanSizesForUpload(item);
        dataSizes.push_back(itemSize);
        const size_t maxSize = std::numeric_limits<size_t>::max();
        size_t alignedSize = roundToAlignment(itemSize);
        cumulative_size += alignedSize;
    }

    if (cumulative_size > std::numeric_limits<size_t>::max() - (GpuAlignment - 1)) {
        throw std::overflow_error("GPU allocation size overflow");
    }

    // Stage plain bytes so the completion callback never needs Python's GIL.
    auto uploadCapsule = std::make_unique<CpuUploadCapsule>(cumulative_size);

    // Creates a block of data with needed size
    std::unique_ptr<PromiseData> promiseData = std::make_unique<PromiseData>(cumulative_size);
    void* rawDataIter = promiseData->m_data;
    std::shared_ptr<GpuPromise> promise = std::make_shared<GpuPromise>(std::move(promiseData));

    try {
        // Begins partitioning data along block of data
        for (size_t i = 0; i < argLength; ++i) {
            PromiseData& data = *promise->m_data;
            size_t alignedSize = roundToAlignment(dataSizes[i]);
            if (data.m_usedSize > data.m_size || alignedSize > data.m_size - data.m_usedSize) {
                throw std::runtime_error("GPU data arena is full");
            }
            GpuBuffer addBuffer{rawDataIter, dataSizes[i], promise};
            py::array source = detail::SpanUtil::setSpanBuffer(dataHandles[i], std::move(addBuffer));
            const void* uploadSource = nullptr;
            if (dataSizes[i] != 0) {
                uint8_t* destination = uploadCapsule->data.data() + data.m_usedSize;
                std::memcpy(destination, source.data(), dataSizes[i]);
                uploadSource = destination;
            }
            data.uploadCpuDataAsync(uploadSource, dataSizes[i]);

            if (rawDataIter != nullptr) {
                rawDataIter = static_cast<uint8_t*>(rawDataIter) + roundToAlignment(dataSizes[i]);
            }
        }
        // Orphan the capsule once its deletion has been queued after every upload.
        promise->m_data->finishCpuUploadAsync(uploadCapsule.get());
        uploadCapsule.release();
    } catch (...) {
        // Spans may retain the promise after a failure; finish copies before freeing staging.
        promise->m_data->syncGpuData();
        throw;
    }
    return promise;
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
        static_cast<const uint32_t*>(posting_lists.getBuffer().m_data), 
        static_cast<const int64_t*>(posting_sizes.getBuffer().m_data), 
        static_cast<const int64_t*>(posting_queries.getBuffer().m_data),
        static_cast<const uint8_t*>(dense_masks.getBuffer().m_data), 
        static_cast<const int64_t*>(dense_mask_queries.getBuffer().m_data), 
        n_queries, bytes_per_query, n_docs));
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
    

    auto result = std::make_shared<GpuPromise>(cluster_legals_preordered_packed_gpu_impl(
        gatherPromises({&mask_packed_buffer, &cluster_offsets, &cluster_counts}),
        static_cast<const uint8_t*>(mask_packed_buffer.getBuffer().m_data), n_queries, bytes_per_query, n_docs,
        static_cast<const int32_t*>(cluster_offsets.getBuffer().m_data), 
        static_cast<const int32_t*>(cluster_counts.getBuffer().m_data), 
        n_list));
    return GpuSpan<float>(result->m_data->m_data, result->m_data->m_size, result);
}

#else

// If Gpu logic is not support it should throw on any GPU specific logic 
// that attempts to be called. 
struct PromiseData { };

inline GpuBuffer::GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise) {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuPromise::GpuPromise(std::unique_ptr<PromiseData> data) {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline py::array_t<T, py::array::c_style> GpuSpan<T>::collectToCpu() const {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline const GpuBuffer& GpuSpan<T>::getBuffer() const {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline py::object gpuPrep_out_py(py::args args) {
    throw std::runtime_error("GPU backend not compiled");
}
inline std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args) {
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

    py::class_<GpuPromise, std::shared_ptr<GpuPromise>>(m, "GpuPromise")
        .def("collect_buffer_async", &GpuPromise::collectBufferAsync)
        .def("collect_buffer_sync", &GpuPromise::collectBufferSync);
    detail::SpanUtil::addAllSupportedGpuSpan(m);
    m.def("gpu_prep", &gpuPrep_out_py);
    m.def("upload_to_gpu", &uploadToGpu_out_py);
}
