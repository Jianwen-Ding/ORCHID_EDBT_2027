#include "cluser_legals_gpu_kernel.h"

#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

#include <stdexcept>

#define TILE_WIDTH 32

// Funnels a number of promises into a single gpu promise
static std::shared_ptr<PromiseData> funnelPromises(const std::vector<std::shared_ptr<PromiseData>>& promises) {
    // TODO: A bit arbitrary might need some specification or something
    // given that only one group of promises is expected for now its fine
    std::shared_ptr<PromiseData> extendedPromise = promises[0];
    for (int i = 1 ; i < promises.size() ; i++) {
        promises[i]->syncGpuData();
    }
    return extendedPromise;
}
PromiseData::PromiseData(size_t size) : m_size(roundToAlignment(size)) {
    checkGpuStatus(cudaStreamCreate(&m_stream));
    if (m_size != 0) {
        cudaError_t status = cudaMalloc(&m_data, m_size);
        if (status != cudaSuccess) {
            cudaStreamDestroy(m_stream);
            checkGpuStatus(status);
        }
    }
}
PromiseData::~PromiseData() {
    cudaStreamSynchronize(m_stream);
    cudaFree(m_data);
    if (m_event) {
        cudaEventDestroy(m_event);
    }
    cudaStreamDestroy(m_stream);
}
void PromiseData::finalizePromise() {
    if (m_event) {
        cudaEventDestroy(m_event);
    }
    cudaEventCreate(&m_event);
    cudaEventRecord(m_event, m_stream);
}
const void* PromiseData::getData() const {
    return m_data;
}
void PromiseData::uploadCpuDataAsync(const void* data, size_t size, size_t offset) {
    if (size != 0) {
        void* destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemcpyAsync(destination, data, size,
                                       cudaMemcpyHostToDevice, m_stream));
    }
}

void PromiseData::syncGpuData() {
    if (m_event) {
        checkGpuStatus(cudaEventSynchronize(m_event));
    } else {
        checkGpuStatus(cudaStreamSynchronize(m_stream));
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
__global__ void posting_list_kernel (
    const uint32_t* postingLists,
    const int64_t* postingThreads,
    const int64_t* postingSizes,
    const int64_t* postingQueries,
    uint8_t* out_mask,
    int32_t nQueries,
    int32_t bytesPerQuery,
    int32_t threadWidth,
    int32_t postingCount) {
    const int32_t theadGlobalIdx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const int32_t queryIdx = cub::UpperBound(
        postingThreads,
        nQueries,
        theadGlobalIdx);
    const in
    for (int32_t i = idx ; i < idx + threadWidth  && i < postingCount; i++) {

    }
}

std::shared_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<std::shared_ptr<PromiseData>> blockingPromises,
    const uint32_t* postingLists,
    const int64_t* postingSizes,
    const int64_t* postingQueries,
    const uint8_t* denseMasks,
    const int64_t* dense_mask_queries,
    int32_t nQueries,
    int32_t bytesPerQuery,
    int32_t nDocs,
    int32_t postingCount,
    int32_t dense_mask_count) {
    // Funnels relevant streams from blocking promises
    std::shared_ptr<PromiseData> extendedPromise = funnelPromises(blockingPromises);
    cudaStream_t stream = extendedPromise->m_stream;

    // Provides easier format for each thread to find corresponding query index
    const int32_t threadWidth = 32;
    thrust::transform(
        thrust::cuda::par_nosync.on(stream),
        d_input, d_input + n,
        d_output,
        DoubleValue{}
    );
    
    // Divides the postings into different blocks
    

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
