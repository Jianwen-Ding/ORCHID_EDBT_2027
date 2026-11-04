#include "cluser_legals_gpu_kernel.h"

#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

#include <stdexcept>

#define TILE_WIDTH 32

void StreamCapsule::sync() {
    if (m_event) {
        checkGpuStatus(cudaEventSynchronize(m_event));
    } else {
        checkGpuStatus(cudaStreamSynchronize(m_stream));
    }
}

StreamCapsule::StreamCapsule() {}
StreamCapsule::~StreamCapsule() {
    if (m_event) {
        cudaEventDestroy(m_event);
    }
    cudaStreamDestroy(m_stream);
}


// Funnels a number of promises into a single gpu promise
static std::shared_ptr<StreamCapsule> funnelStreams(const std::vector<std::shared_ptr<StreamCapsule>>& streams) {
    // TODO: A bit arbitrary might need some specification or something
    // given that only one group of promises is expected for now its fine
    std::shared_ptr<StreamCapsule> extendedStream = streams[0];
    for (int i = 1 ; i < streams.size() ; i++) {
        streams[i]->sync();
    }
    return extendedStream;
}
PromiseData::PromiseData(size_t size, std::shared_ptr<StreamCapsule> stream = nullptr) : 
    m_size(roundToAlignment(size)),
    m_stream(stream) {

    if (!m_stream) {
        m_stream = std::make_shared<StreamCapsule>();
        checkGpuStatus(cudaStreamCreate(&m_stream->m_stream));
    }

    if (m_size != 0) {
        cudaError_t status = cudaMallocAsync(&m_data, m_size, m_stream);
        if (status != cudaSuccess) {
            cudaStreamDestroy(m_stream->m_stream);
            checkGpuStatus(status);
        }
    }
}
PromiseData::~PromiseData() {
    cudaFree(m_data);
}
void PromiseData::finalizePromise() {
    if (m_stream->m_event) {
        cudaEventDestroy(m_stream->m_event);
    }
    cudaEventCreate(&m_stream->m_event);
    cudaEventRecord(m_stream->m_event, m_stream->m_stream);
}
const void* PromiseData::getData() const {
    return m_data;
}
void PromiseData::uploadCpuDataAsync(const void* data, size_t size, size_t offset) {
    if (size != 0) {
        void* destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemcpyAsync(destination, data, size,
                                       cudaMemcpyHostToDevice, m_stream->m_stream));
    }
}

void PromiseData::syncGpuData() {
    if (m_stream->m_event) {
        checkGpuStatus(cudaEventSynchronize(m_stream->m_event));
    } else {
        checkGpuStatus(cudaStreamSynchronize(m_stream->m_stream));
    }
}

PromiseCapsule PromiseData::collectRawCpuCopy() {
    syncGpuData();
    return copyToCpu(m_data, m_size);
}

PromiseCapsule copyToCpu(const void* gpuData, size_t size) {
    void* cpuPtr = nullptr;
    checkGpuStatus(cudaMallocHost(&cpuPtr, size == 0 ? 1 : size));
    cudaError_t status = cudaMemcpy(cpuPtr, gpuData, size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cudaFreeHost(cpuPtr);
        checkGpuStatus(status);
    }
    return {cpuPtr, [](void* raw) { cudaFreeHost(raw); }};
}

// Dedicates at most a single thread per 
__global__ void postingListKernel (
    const uint32_t* postingLists,
    const int32_t* postingThreads,
    const int32_t* postingSizes,
    const int32_t* postingQueries,
    int32_t* outMask,
    int32_t nQueries,
    int32_t bytesPerQuery,
    int32_t nDocs,
    int32_t threadWidth,
    int32_t postingCount) {
    extern __shared__ int32_t sharedMem[];
    
    // Sets up shared memory
    auto block = cg::this_thread_block();
    int32_t* postingSizesShared = sharedMem;
    int32_t* postingThreadsShared = postingSizesShared + nQueries;
    cg::memcpy_async(block, postingThreads, postingThreadsShared, nQueries * sizeof(int32_t));
    cg::memcpy_async(block, postingSizesShared, postingSizes, nQueries * sizeof(int32_t));

    // Gets the query the thread will be operating in
    const int32_t theadGlobalIdx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    cg::wait_prior<1>(block);
    const int32_t* queryLoc =
        cub::LowerBound(
            postingThreads,
            postingThreads + nQueries,
            theadGlobalIdx);
    const int32_t queryIdx = static_cast<int32_t>(std::distance(postingThreads, queryLoc));

    // Exits early if thread is sent
    // TODO: Currently approach simply accepts that a certain amount of threads will be completely useless
    if (nQueries == queryIdx) {
        return;
    }

    // Gets the posting list bounds of the thread
    cg::wait(block);
    const int32_t prevPostingSizes = queryIdx == 0 ? 0 : postingSizes[queryIdx - 1];
    const int32_t queryStartIdx = prevPostingSizes + (theadGlobalIdx - *queryLoc) * threadWidth;
    const int32_t queryEndIdx = std::min(queryStartIdx + threadWidth, postingSizes[queryIdx]);

    const int32_t maskBegin = bytesPerQuery * queryIdx;

    // Takes the setup and begins to send 
    for (int32_t i = queryStartIdx ; i < queryEndIdx; i++) {
        const int32_t byteIdx = postingLists[i] / 32;
        const int32_t byteIdx = 
    }
}


struct ThreadWidthDivision {
    int32_t threadWidth{ 1 };

    ThreadWidthDivision(int32_t _threadWidth) : threadWidth(_threadWidth) {}

    __host__ __device__
    int32_t operator()(const int32_t& x) const {
        return x / threadWidth + (x % threadWidth != 0);
    }
}

std::shared_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<std::shared_ptr<StreamCapsule>> blockingPromises,
    const uint32_t* postingLists,
    const int32_t* postingSizes,
    const int32_t* postingQueries,
    const uint8_t* denseMasks,
    const int64_t* dense_mask_queries,
    int32_t nQueries,
    int32_t bytesPerQuery,
    int32_t nDocs,
    int32_t postingCount,
    int32_t denseMaskCount) {
    // Funnels relevant streams from blocking promises
    std::shared_ptr<StreamCapsule> extendedPromise = funnelStreams(blockingPromises);
    std::shared_ptr<PromiseData> ret = std::make_shared<PromiseData>(round(bytesPerQuery, sizeof(int32_t), extendedPromise));
    cudaStream_t stream = extendedPromise->m_stream;

    // Provides easier format for each thread to find corresponding query index
    int32_t* threadCounts;
    cudaMallocAsync(&threadCounts, nQueries * sizeof(int32_t), stream);

    const int32_t threadWidth = 128;
    thrust::transform(
        thrust::cuda::par_nosync.on(stream),
        postingSizes, postingSizes + nQueries,
        threadCounts,
        ThreadWidthDivision{ threadWidth });
    thrust::inclusive_scan(
        thrust::cuda::par_nosync.on(stream),
        threadCounts, 
        threadCounts + nQueries, 
        threadCounts);
    thrust::inclusive_scan(
        thrust::cuda::par_nosync.on(stream),
        postingSizes, 
        postingSizes + nQueries, 
        postingSizes);
    
    // Divides the postings into different blocks
    int32_t totalThreads = nQueries + ((postingCount + threadWidth - 1) / threadWidth);
    int32_t totalBlocks = (totalThreads + 127) / 128;
    // postingListKernel<<<totalBlocks, 128, nQueries * 2 * sizeof>>>();

    return extendedPromise;
}


std::shared_ptr<PromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<std::shared_ptr<PromiseData>> blockingPromises,
    const uint8_t* mask_packed_buffer,
    int32_t nQueries,
    int32_t bytesPerQuery,
    int32_t nDocs,
    const int32_t* cluster_offsets,
    const int32_t* cluster_counts,
    int32_t n_list) {
    throw std::runtime_error("GPU cluster legals kernel not implemented");
}
