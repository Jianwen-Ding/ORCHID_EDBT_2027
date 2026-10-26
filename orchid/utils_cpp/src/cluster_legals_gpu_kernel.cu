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
const void* PromiseData::getData() const {
    return m_data;
}
void PromiseData::uploadCpuDataAsync(const void* data, size_t size) {
    if (size != 0) {
        void* destination = static_cast<uint8_t*>(m_data) + m_usedSize;
        checkGpuStatus(cudaMemcpyAsync(destination, data, size,
                                       cudaMemcpyHostToDevice, m_stream));
    }
    m_usedSize += roundToAlignment(size);
}

void PromiseData::finishCpuUploadAsync(CpuUploadCapsule* capsule) {
    checkGpuStatus(cudaLaunchHostFunc(m_stream, [](void* raw) {
        delete static_cast<CpuUploadCapsule*>(raw);
    }, capsule));
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


std::shared_ptr<PromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const uint32_t* posting_lists,
    const int64_t* posting_sizes,
    const int64_t* posting_queries,
    const uint8_t* dense_masks,
    const int64_t* dense_mask_queries,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs) {
    throw std::runtime_error("GPU predicate bitmap kernel not implemented");
}


std::shared_ptr<PromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<PromiseData*> blockingPromises,
    const uint8_t* mask_packed_buffer,
    int32_t n_queries,
    int32_t bytes_per_query,
    int32_t n_docs,
    const int32_t* cluster_offsets,
    const int32_t* cluster_counts,
    int32_t n_list) {
    throw std::runtime_error("GPU cluster legals kernel not implemented");
}
