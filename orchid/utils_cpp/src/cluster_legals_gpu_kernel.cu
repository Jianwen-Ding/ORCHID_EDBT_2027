#include "cluser_legals_gpu_kernel.h"

#include <stdexcept>

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
void PromiseData::syncGpuData() {
    if (m_event) {
        checkGpuStatus(cudaEventSynchronize(m_event));
    } else {
        checkGpuStatus(cudaStreamSynchronize(m_stream));
    }
}

void* PromiseData::pushDataArena(size_t size) {
    size_t alignedSize = roundToAlignment(size);
    if (alignedSize > m_size - m_usedSize) {
        throw std::runtime_error("GPU data arena is full");
    }
    void* data = m_data ? static_cast<char*>(m_data) + m_usedSize : nullptr;
    m_usedSize += alignedSize;
    return data;
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


std::unique_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const void* posting_lists,
    const void* posting_sizes,
    const void* posting_queries,
    const void* dense_masks,
    const void* dense_mask_queries,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs) {
    throw std::runtime_error("GPU predicate bitmap kernel not implemented");
}


std::unique_ptr<PromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const void* mask_packed_buffer,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs,
    const void* cluster_offsets,
    const void* cluster_counts,
    int32_t n_list) {
    throw std::runtime_error("GPU cluster legals kernel not implemented");
}
