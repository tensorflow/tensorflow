/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h"

#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator(
    PlatformGpuId platform_gpu_id, size_t pool_size,
    bool reserve_memory, bool compute_stats)
    : name_(absl::StrCat("gpu_async_", platform_gpu_id.value())) {
  stream_exec_ =
      DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                platform_gpu_id).ValueOrDie();

#if CUDA_VERSION < 11020
  LOG(FATAL) << "TF_GPU_ALLOCATOR=cuda_malloc_async need CUDA 11.2 or higher to compile.";
#elif !defined(GOOGLE_CUDA)
  LOG(FATAL) << "GOOGLE_CUDA not defined";
#else
  // Initialized here as it only exist if compiled with a recent
  // enough CUDA.
  pool_ = nullptr;
  cuda_stream_ = nullptr;
  // WAR an CUDA 11.2 driver bug for multiple-GPU. It currently
  // request that the context on GPU 0 is initialized. Which isn't the
  // case for TF+horovod.
  if (platform_gpu_id.value() > 0) {
    CUcontext pctx; // We loose track of it. But this is fine.
    CUresult err = cuDevicePrimaryCtxRetain(&pctx, 0);
    if (err != CUDA_SUCCESS){
      LOG(FATAL) << "Failed to create the context on device 0.";
    }
  }

  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  int cuda_malloc_async_supported;
  cudaDeviceGetAttribute(&cuda_malloc_async_supported,
                         cudaDevAttrMemoryPoolsSupported,
                         platform_gpu_id.value());
  if (!cuda_malloc_async_supported) {
    LOG(FATAL) << "TF_GPU_ALLOCATOR=cuda_malloc_async isn't currently supported."
               << " Possible causes: device not supported, driver too old, "
               << " OS not supported, CUDA version too old.";
  }

  cudaError_t cerr = cudaStreamCreate(&cuda_stream_);
  if (cerr != cudaSuccess) {
    LOG(FATAL) << "could not allocate CUDA stream for context : "
               << cudaGetErrorString(cerr);
  }

  cerr = cudaDeviceGetDefaultMemPool(&pool_, platform_gpu_id.value());
  if (cerr != cudaSuccess) {
    LOG(FATAL) << "could not get the default CUDA pool : "
               << cudaGetErrorString(cerr);
  }
  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_gpu_id.value() << " with pool size of: "
          << pool_size << " this ptr: " << this;
  cerr = cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                 reinterpret_cast<void*>(&pool_size));
  if (compute_stats) {
    mutex_lock lock(lock_);
    stats_.bytes_limit = static_cast<int64>(pool_size);
  } // If not set, it means we do not compute stats.
  if (cerr != cudaSuccess) {
    LOG(FATAL) << "could not set the default CUDA pool memory threshold : "
               << cudaGetErrorString(cerr);
    pool_ = nullptr;
  }

  // If in TF_DETERMINISTIC_OPS is set, then make the allocator behave
  // determistically.
  bool deterministic_ops = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                             /*default_val=*/false,
                                             &deterministic_ops));
  if (deterministic_ops) {
    cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseAllowOpportunistic, 0);
    cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseAllowInternalDependencies, 0);
  }
#endif

  VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator PoolSize " << pool_size;
  if (reserve_memory) {
    void* ptr = AllocateRaw(0, pool_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator reserved the pool";
    ClearStats();
  }
}

GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator() {
#ifdef GOOGLE_CUDA
  cuStreamDestroy(cuda_stream_);
#endif
}

void* GpuCudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
#if CUDA_VERSION < 11020 || !defined(GOOGLE_CUDA)
  return nullptr;
#else
  if (pool_ == nullptr) {
    LOG(FATAL) << "The instantiation of GpuCudaMallocAsyncAllocator failed."
               << " See previous errors.";
  }
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* ptr = 0;
  cudaError_t res = cudaMallocFromPoolAsync(&ptr, num_bytes, pool_, cuda_stream_);
  if (res != cudaSuccess) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    mutex_lock lock(lock_);
    LOG(ERROR) << Name() << " cudaMallocAsync failed to allocate " << num_bytes
               << "\n Error name: " << cudaGetErrorName(res)
               << "\n Error string: " << cudaGetErrorString(res)
               << "\n Free memory/Total memory: " << free << "/" << total
               << "\n Stats: \n" << stats_.DebugString();
    return nullptr;
  }

  // Update stats.
  if (stats_.bytes_limit.has_value()) {
    mutex_lock lock(lock_);
    ++stats_.num_allocs;
    stats_.bytes_in_use += num_bytes;
    stats_.peak_bytes_in_use =
        std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
    stats_.largest_alloc_size =
        std::max<std::size_t>(stats_.largest_alloc_size, num_bytes);
    size_map_[ptr] = num_bytes;
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << ptr;
  return ptr;
#endif
}
void GpuCudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
#if CUDA_VERSION < 11020 || !defined(GOOGLE_CUDA)
#else
  cudaError_t res = cudaFreeAsync(ptr, cuda_stream_);
  if (res == cudaErrorCudartUnloading) {
    // It happens with multi-GPU that TF free the GPU allocation after
    // the driver is unloaded. It is safe to ignore this error here.
    // TODO: Find how to fix the shutdown steps in TF.
    VLOG(1) << "Ignoring Error: " << cudaGetErrorName(res)
	    << " \nError string: " << cudaGetErrorString(res);
  } else if (res != cudaSuccess) {
    size_t free, total;
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
    cudaMemGetInfo(&free, &total);
    LOG(ERROR) << "cudaFreeAsync failed to free " << ptr
               << "\n Error name: " << cudaGetErrorName(res)
               << "\n Error string: " << cudaGetErrorString(res)
               << "\n Free memory/Total memory: " << free << "/" << total
               << "\n Stats: \n" << stats_.DebugString();
  }

  // Updates the stats.
  if (stats_.bytes_limit.has_value()) {
    mutex_lock lock(lock_);
    DCHECK(size_map_.contains(ptr));
    size_t size = size_map_[ptr];
    stats_.bytes_in_use -= size;
    size_map_.erase(ptr);
  }

  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // GOOGLE_CUDA
}

bool GpuCudaMallocAsyncAllocator::TracksAllocationSizes() const {
  return stats_.bytes_limit.has_value();
}

size_t GpuCudaMallocAsyncAllocator::RequestedSize(const void* ptr) const {
  CHECK(ptr);
  return size_map_.at(ptr);
}

size_t GpuCudaMallocAsyncAllocator::AllocatedSize(const void* ptr) const {
  CHECK(ptr);
  return size_map_.at(ptr);
}

absl::optional<AllocatorStats> GpuCudaMallocAsyncAllocator::GetStats() {
  mutex_lock l(lock_);
  return stats_;
}

void GpuCudaMallocAsyncAllocator::ClearStats() {
  mutex_lock l(lock_);
  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = stats_.bytes_in_use;
  stats_.largest_alloc_size = 0;
}

}  // namespace tensorflow
