#include "cluser_legals_gpu_kernel.h"

#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

void StreamCapsule::sync() const {
    checkGpuStatus(cudaStreamSynchronize(m_stream));
}

StreamCapsule::StreamCapsule() {
    checkGpuStatus(cudaStreamCreate(&m_stream));
    cudaEventCreate(&m_streamStartEvent);
    cudaEventRecord(m_streamStartEvent, m_stream);
}

StreamCapsule::~StreamCapsule() {
    cudaStreamDestroy(m_stream);
}


// Funnels a number of promises into a single gpu promise
static std::shared_ptr<const StreamCapsule> funnelStreams(const std::vector<std::shared_ptr<const StreamCapsule>>& streams) {
    // TODO: A bit arbitrary might need some specification or something
    // given that only one group of promises is expected for now its fine
    std::shared_ptr<const StreamCapsule> extendedStream = streams[0];
    for (int i = 1 ; i < streams.size() ; i++) {
        streams[i]->sync();
    }
    return extendedStream;
}
GpuPromiseData::GpuPromiseData(size_t size, std::shared_ptr<const StreamCapsule> stream = nullptr) : 
    m_size(roundToAlignment(size)),
    m_stream(stream) {

    if (!m_stream) {
        m_stream = std::make_shared<const StreamCapsule>();
    }

    if (m_size != 0) {
        checkGpuStatus(cudaMallocAsync(&m_data, m_size, m_stream->m_stream));
    }
}
GpuPromiseData::~GpuPromiseData() {
    cudaFree(m_data);
    cudaEventDestroy(m_event);
}
void GpuPromiseData::finalizePromise() {
    cudaEventCreate(&m_event);
    cudaEventRecord(m_event, m_stream->m_stream);
}
const void* GpuPromiseData::getData() const {
    return m_data;
}
void* GpuPromiseData::getData() {
    return m_data;
}
std::shared_ptr<const StreamCapsule> GpuPromiseData::getStream() const {
    return m_stream;
}
size_t GpuPromiseData::getSize() const {
    return m_size;
}
void GpuPromiseData::uploadValueAsync(int value, size_t size, size_t offset) {
    if (size != 0) {
        void* d_destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemsetAsync(d_destination, value, size, m_stream->m_stream));
    }
}
void GpuPromiseData::uploadCpuDataAsync(const void* h_data, size_t size, size_t offset) {
    if (size != 0) {
        void* d_destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemcpyAsync(d_destination, h_data, size,
                                       cudaMemcpyHostToDevice, m_stream->m_stream));
    }
}

void GpuPromiseData::syncGpuData() const {
    if (m_event) {
        checkGpuStatus(cudaEventSynchronize(m_event));
    } else {
        checkGpuStatus(cudaStreamSynchronize(m_stream->m_stream));
    }
}

PromiseCapsule GpuPromiseData::collectRawCpuCopy() {
    syncGpuData();
    return copyToCpu(m_data, m_size);
}

PromiseCapsule copyToCpu(const void* d_gpuData, size_t size) {
    void* h_cpuPtr = nullptr;
    checkGpuStatus(cudaMallocHost(&h_cpuPtr, size == 0 ? 1 : size));
    cudaError_t status = cudaMemcpy(h_cpuPtr, d_gpuData, size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cudaFreeHost(h_cpuPtr);
        checkGpuStatus(status);
    }
    return {h_cpuPtr, [](void* h_raw) { cudaFreeHost(h_raw); }};
}

// Assigns each thread a chunk of one posting list.
__global__ void postingListKernel (
    const uint32_t* d_postingLists,
    const int64_t* d_postingThreads,
    const int64_t* d_postingSizes,
    const int32_t* d_postingQueries,
    uint8_t* d_outMask,
    int32_t bytesPerQuery,
    int32_t threadWidth,
    int32_t postingCount) {
    extern __shared__ int64_t d_sharedMem[];
    
    // Sets up shared memory
    auto block = cg::this_thread_block();
    int64_t* d_postingSizesShared = d_sharedMem;
    int64_t* d_postingThreadsShared = d_postingSizesShared + postingCount;
    cg::memcpy_async(block, d_postingThreadsShared, d_postingThreads, postingCount * sizeof(int64_t));
    cg::memcpy_async(block, d_postingSizesShared, d_postingSizes, postingCount * sizeof(int64_t));

    // Gets the query the thread will be operating in
    const int64_t threadGlobalIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    cg::wait_prior<1>(block);
    const int32_t queryIdx = cub::UpperBound(d_postingThreadsShared, postingCount, threadGlobalIdx);

    // Finish the collective before idle threads exit.
    cg::wait(block);
    if (postingCount == queryIdx) {
        return;
    }

    // Gets the posting list bounds of the thread
    const int64_t prevPostingSizes = queryIdx == 0 ? 0 : d_postingSizesShared[queryIdx - 1];
    const int64_t prevPostingThreads = queryIdx == 0 ? 0 : d_postingThreadsShared[queryIdx - 1];
    const int64_t queryStartIdx = prevPostingSizes + (threadGlobalIdx - prevPostingThreads) * threadWidth;
    const int64_t queryEndIdx = queryStartIdx + threadWidth < d_postingSizesShared[queryIdx]
        ? queryStartIdx + threadWidth : d_postingSizesShared[queryIdx];

    const int64_t maskBegin = static_cast<int64_t>(bytesPerQuery) * d_postingQueries[queryIdx];

    // Sets the bits for this thread's posting range.
    for (int64_t i = queryStartIdx ; i < queryEndIdx; i++) {
        const uint32_t id = d_postingLists[i];
        const int64_t byteIdx = maskBegin + (id / 8);
        const unsigned int bitIdx = static_cast<unsigned int>((byteIdx & 3) * 8 + (id & 7));
        atomicOr(reinterpret_cast<unsigned int*>(d_outMask) + (byteIdx / sizeof(unsigned int)), 1u << bitIdx);
    }
}

struct ThreadWidthDivision {
    int32_t threadWidth{1};

    ThreadWidthDivision(int32_t width) : threadWidth(width) {}

    __host__ __device__
    int64_t operator()(const int32_t& x) const {
        return x / threadWidth + (x % threadWidth != 0);
    }
};

std::shared_ptr<GpuPromiseData> predicate_results_to_bitmap_gpu_impl(
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
    int64_t postingEntries) {
    const size_t totalSize = static_cast<size_t>(nQueries) * bytesPerQuery;
    auto extendedPromise = funnelStreams(blockingPromises);
    auto ret = std::make_shared<GpuPromiseData>(totalSize, extendedPromise);
    const cudaStream_t stream = extendedPromise->m_stream;
    
    if (ret->getData()) {
        ret->uploadValueAsync(0, ret->getSize(), 0);
    }

    int64_t* d_postingMetadata = nullptr;
    try {
        if (postingCount) {
            const size_t metadataBytes = static_cast<size_t>(postingCount) * 2 * sizeof(int64_t);
            checkGpuStatus(cudaMallocAsync(
                reinterpret_cast<void**>(&d_postingMetadata), metadataBytes, stream));
            int64_t* d_threadCounts = d_postingMetadata;
            int64_t* d_postingOffsets = d_postingMetadata + postingCount;
            const int32_t threadWidth = 128;
            const auto policy = thrust::cuda::par_nosync.on(stream);
            thrust::transform(
                policy, d_postingSizes, d_postingSizes + postingCount,
                d_threadCounts, ThreadWidthDivision{threadWidth});
            thrust::inclusive_scan(policy, d_threadCounts, d_threadCounts + postingCount, d_threadCounts);
            // Preserve input sizes and accumulate offsets without 32-bit overflow.
            thrust::transform(
                policy, d_postingSizes, d_postingSizes + postingCount,
                d_postingOffsets, ThreadWidthDivision{1});
            thrust::inclusive_scan(policy, d_postingOffsets, d_postingOffsets + postingCount, d_postingOffsets);
    
            // Sum ceil(size / threadWidth) <= postingCount + floor(entries / threadWidth).
            // The uploaded span provides this bound without copying a scan result to the CPU.
            const int64_t totalThreads = postingCount + postingEntries / threadWidth;
            const int blocks = static_cast<int>((totalThreads + 255) / 256);

            postingListKernel<<<blocks, 256, metadataBytes, stream>>>(
                d_postingLists, d_threadCounts, d_postingOffsets, d_postingQueries,
                static_cast<uint8_t*>(ret->getData()), bytesPerQuery,
                threadWidth, postingCount);

            checkGpuStatus(cudaGetLastError());
            checkGpuStatus(cudaFreeAsync(d_postingMetadata, stream));
            d_postingMetadata = nullptr;
        }
        // Run after posting atomics: adjacent rows may share an atomic word.
        for (int32_t i = 0; i < denseMaskCount; ++i) {
            checkGpuStatus(cudaMemcpyAsync(
                static_cast<uint8_t*>(ret->getData()) + h_denseMaskQueries[i] * bytesPerQuery,
                d_denseMasks + static_cast<int64_t>(i) * bytesPerQuery,
                bytesPerQuery, cudaMemcpyDeviceToDevice, stream));
        }
        ret->finalizePromise();
    } catch (...) {
        if (d_postingMetadata) cudaFreeAsync(d_postingMetadata, stream);
        throw;
    }
    return ret;
}


std::shared_ptr<GpuPromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<std::shared_ptr<const StreamCapsule>> blockingPromises,
    const uint8_t* d_maskPackedBuffer,
    int32_t nQueries,
    int32_t bytesPerQuery,
    const int32_t* d_clusterOffsets,
    const int32_t* d_clusterCounts,
    int32_t nList) {
    throw std::runtime_error("GPU cluster legals kernel not implemented");
}
