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

#include "absl/types/optional.h"
#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

#if GOOGLE_CUDA
static std::string GetCudaErrorMessage(CUresult result) {
  const char* error;
  cuGetErrorString(result, &error);
  const char* name;
  cuGetErrorName(result, &name);
  return absl::StrCat("CUDA error: ", error ? error : "<unknown>", " (",
                      name ? name : "Unknown", ")");
}
#endif  // GOOGLE_CUDA

GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator(
    PlatformDeviceId platform_device_id, size_t pool_size, bool reserve_memory,
    bool compute_stats)
    : name_(absl::StrCat("gpu_async_", platform_device_id.value())) {
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .ValueOrDie();
  // Initialized here as it only exist if compiled with a recent
  // enough CUDA.
  pool_ = nullptr;
  cuda_stream_ = nullptr;
  // WAR an CUDA 11.2 driver bug for multiple-GPU. It currently
  // request that the context on GPU 0 is initialized. Which isn't the
  // case for TF+horovod.
  int driverVersion;
  cuDriverGetVersion(&driverVersion);
  VLOG(2) << "DRIVER VERSION: " << driverVersion;
  if (platform_device_id.value() > 0 && driverVersion < 11030) {
    CUcontext pctx;  // We loose track of it. But this is fine.
    if (auto result = cuDevicePrimaryCtxRetain(&pctx, 0))
      LOG(FATAL)  // Crash OK.
          << "Failed to retain context: " << GetCudaErrorMessage(result);
  }

  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  int cuda_malloc_async_supported;
  if (auto status =
          cuDeviceGetAttribute(&cuda_malloc_async_supported,
                               CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                               platform_device_id.value()))
    LOG(FATAL) <<  // Crash OK.
        "Failed to get device attribute: " << GetCudaErrorMessage(status);
  if (!cuda_malloc_async_supported)
    LOG(FATAL)  // Crash OK.
        << "TF_GPU_ALLOCATOR=cuda_malloc_async isn't currently supported."
        << " Possible causes: device not supported, driver too old, "
        << " OS not supported, CUDA version too old.";

  if (auto status =
          cuDeviceGetDefaultMemPool(&pool_, platform_device_id.value()))
    LOG(FATAL) <<  // Crash OK.
        "Failed to get default CUDA pool: " << GetCudaErrorMessage(status);

  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_device_id.value() << " with pool size of: " << pool_size
          << " this ptr: " << this;
  uint64_t pool_size_64 = pool_size;
  if (auto status = cuMemPoolSetAttribute(
          pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &pool_size_64))
    LOG(FATAL) <<  // Crash OK.
        "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);

  if (compute_stats) {
    stats_ = std::make_unique<AllocatorStats>();
    stats_->bytes_limit = static_cast<int64>(pool_size);
  }  // If not set, it means we do not compute stats.

  // If in TF_DETERMINISTIC_OPS is set, then make the allocator behave
  // determistically.
  bool deterministic_ops = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                             /*default_val=*/false,
                                             &deterministic_ops));
  if (deterministic_ops) {
    int disable = 0;
    cuMemPoolSetAttribute(pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                          &disable);
    cuMemPoolSetAttribute(
        pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &disable);
  }

  // Set read/write access to all GPUs.
  static auto* all_pools_ = new std::vector<CUmemoryPool*>();
  static auto* all_ids_ = new std::vector<PlatformDeviceId>();
  DCHECK(all_pools_->size() == all_ids_->size());
  for (int i = 0; i < all_pools_->size(); ++i) {
    // Set the current pool access to the previous GPUs.
    CUmemAccessDesc map;
    map.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map.location.id = (*all_ids_)[i].value();

    map.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    VLOG(2) << "Setting access of the current pool to "
            << " location id: " << map.location.id;
    int canAccessPeer;
    if (auto status = cuDeviceCanAccessPeer(
            &canAccessPeer, platform_device_id.value(), map.location.id)) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed: " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(pool_, &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }

    // Set the previous pools access to the current GPU.
    map.location.id = platform_device_id.value();

    VLOG(2) << "Set access to the pool id: " << i
            << " location id: " << map.location.id;
    if (auto status = cuDeviceCanAccessPeer(&canAccessPeer, i,
                                            platform_device_id.value())) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed: " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(*(*all_pools_)[i], &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }
  }
  all_pools_->push_back(&pool_);
  all_ids_->push_back(platform_device_id);

  VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator PoolSize " << pool_size;
  int64 prealloc_size = 0;
  // TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1 is a special value that
  // preallocates the total pool size.
  TF_CHECK_OK(ReadInt64FromEnvVar("TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC", 0,
                                  &prealloc_size));
  if (prealloc_size == -1) {
    prealloc_size = pool_size;
  } else if (reserve_memory) {
    prealloc_size = pool_size;
  }

  if (prealloc_size != 0) {
    void* ptr = AllocateRaw(0, prealloc_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator reserved the pool for "
            << prealloc_size << " bytes"
            << ". First ptr: " << ptr;
    ClearStats();
  }
#else   // TF_CUDA_MALLOC_ASYNC_SUPPORTED
  LOG(FATAL) << "GpuCudaMallocAsyncAllocator requires CUDA 11.2+";  // Crash OK.
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}

GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator() {
}

void* GpuCudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  CHECK(cuda_stream_ != nullptr)
      << "A stream must be added to the GpuCudaMallocAsync allocator";
  if (pool_ == nullptr) {
    LOG(FATAL)  // Crash OK.
        << "The instantiation of GpuCudaMallocAsyncAllocator failed."
        << " See previous errors.";
  }
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* ptr = nullptr;
  if (auto result =
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr),
                                  num_bytes, pool_, cuda_stream_)) {
    size_t free, total;
    cuMemGetInfo(&free, &total);
    mutex_lock lock(lock_);
    LOG(ERROR) << Name() << " cuMemAllocAsync failed to allocate " << num_bytes
               << ": " << GetCudaErrorMessage(result)
               << "\n Free memory/Total memory: " << free << "/" << total;
    if (auto stats = GetStats())
      LOG(ERROR) << "Stats: " << stats->DebugString();
    return nullptr;
  }

  // Update stats.
  if (stats_) {
    mutex_lock lock(lock_);
    ++(stats_->num_allocs);
    stats_->bytes_in_use += num_bytes;
    stats_->peak_bytes_in_use =
        std::max(stats_->peak_bytes_in_use, stats_->bytes_in_use);
    stats_->largest_alloc_size =
        std::max<std::size_t>(stats_->largest_alloc_size, num_bytes);
    size_map_[ptr] = num_bytes;
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << ptr;
  return ptr;
#else   // TF_CUDA_MALLOC_ASYNC_SUPPORTED
  return nullptr;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}
void GpuCudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  if (auto result = cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(ptr),
                                   cuda_stream_)) {
    if (result == CUDA_ERROR_DEINITIALIZED) {
      // It happens with multi-GPU that TF free the GPU allocation after
      // the driver is unloaded. It is safe to ignore this error here.
      // TODO: Find how to fix the shutdown steps in TF.
      VLOG(1) << "Ignoring CUDA error: " << GetCudaErrorMessage(result);
    } else {
      size_t free, total;
      se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
      cuMemGetInfo(&free, &total);
      LOG(ERROR) << "cudaFreeAsync failed to free " << ptr << ": "
                 << GetCudaErrorMessage(result)
                 << "\n Free memory/Total memory: " << free << "/" << total;
      if (auto stats = GetStats())
        LOG(ERROR) << "Stats: " << stats->DebugString();
    }
  }

  // Updates the stats.
  if (stats_) {
    mutex_lock lock(lock_);
    DCHECK(size_map_.contains(ptr));
    size_t size = size_map_[ptr];
    stats_->bytes_in_use -= size;
    size_map_.erase(ptr);
  }

  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}

bool GpuCudaMallocAsyncAllocator::TracksAllocationSizes() const {
  return static_cast<bool>(stats_);
}

size_t GpuCudaMallocAsyncAllocator::RequestedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  mutex_lock l(lock_);
  return size_map_.at(ptr);
}

size_t GpuCudaMallocAsyncAllocator::AllocatedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  mutex_lock l(lock_);
  return size_map_.at(ptr);
}

absl::optional<AllocatorStats> GpuCudaMallocAsyncAllocator::GetStats() {
  if (!stats_) return absl::nullopt;
  mutex_lock l(lock_);
  return *stats_;
}

bool GpuCudaMallocAsyncAllocator::ClearStats() {
  if (!stats_) return false;
  mutex_lock l(lock_);
  stats_->num_allocs = 0;
  stats_->peak_bytes_in_use = stats_->bytes_in_use;
  stats_->largest_alloc_size = 0;
  return true;
}

}  // namespace tensorflow
