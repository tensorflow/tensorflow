/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace stream_executor {

struct GpuCudaMallocAsyncAllocator::CudaState {
  // cudaMallocAsync is stream aware. But TF StreamExecutor use only 1
  // compute stream and already synchronize with the h2d, d2h and d2d
  // stream. So we do not need to ask cudaMallocAsync to add extra
  // synchronization.
  // Not owned.
  CUstream cuda_stream{};

  // Not owned. The default pool of the associated GPU.
  // If null, then the instantiation failed and the first allocation
  // will return an error.
  CUmemoryPool pool{};
};

void GpuCudaMallocAsyncAllocator::PrintAllocatorStatisticsNoLock() {
  std::map<size_t, int> size_map_histogram;
  std::vector<std::string> ptr_size_string;
  for (auto p : size_map_) {
    if (VLOG_IS_ON(8)) {
      ptr_size_string.push_back(
          absl::StrCat("(", absl::Hex(p.first), ",", p.second) + ")");
    }
    size_map_histogram[p.second]++;
  }
  LOG(ERROR) << "Histogram of current allocation: (allocation_size_in_bytes, "
             << "nb_allocation_of_that_sizes), ...;";
  for (auto p : size_map_histogram) {
    LOG(ERROR) << p.first << ", " << p.second;
  }

  VLOG(8) << "\nThe sorted list of (ptr,size):";
  VLOG(8) << absl::StrJoin(ptr_size_string, ",");

  cuuint64_t mem_reserved_current;
  if (auto result = cuMemPoolGetAttribute(cuda_state_->pool,
                                          CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                                          &mem_reserved_current)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << cuda::ToStatus(result);
  }
  cuuint64_t mem_used_current;
  if (auto result = cuMemPoolGetAttribute(cuda_state_->pool,
                                          CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                                          &mem_used_current)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << cuda::ToStatus(result);
  }
  cuuint64_t mem_reserved_high;
  if (auto result = cuMemPoolGetAttribute(cuda_state_->pool,
                                          CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                                          &mem_reserved_high)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << cuda::ToStatus(result);
  }
  cuuint64_t mem_used_high;
  if (auto result = cuMemPoolGetAttribute(
          cuda_state_->pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH, &mem_used_high)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << cuda::ToStatus(result);
  }
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: "
             << mem_reserved_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_CURRENT: " << mem_used_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: " << mem_reserved_high;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_HIGH: " << mem_used_high;
}

std::atomic<int> GpuCudaMallocAsyncAllocator::number_instantiated_(0);

GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator(
    tsl::PlatformDeviceId platform_device_id, bool create_new_pool,
    size_t new_pool_size, bool reserve_memory, size_t reserve_memory_size,
    bool sync_mode, bool compute_stats)
    : cuda_state_{std::make_unique<CudaState>()},
      name_(absl::StrCat("gpu_async_", platform_device_id.value())),
      reserve_memory_(reserve_memory),
      create_new_pool_(create_new_pool),
      sync_mode_(sync_mode) {
  ++number_instantiated_;

  stream_exec_ = GPUMachineManager()
                     ->ExecutorForDevice(platform_device_id.value())
                     .value();

  int driverVersion;
  cuDriverGetVersion(&driverVersion);
  VLOG(2) << "DRIVER VERSION: " << driverVersion;
  if (driverVersion < 11020) {
    LOG(FATAL)  // Crash OK.
        << "Disable cuda_malloc_async or update your CUDA driver to a version"
        << " compatible with CUDA 11.2 or higher."
        << " We detected a version compatible with: " << driverVersion;
  }

  // WAR an CUDA 11.2 driver bug for multiple-GPU. It currently
  // request that the context on GPU 0 is initialized. Which isn't the
  // case for TF+horovod.
  if (platform_device_id.value() > 0 && driverVersion < 11030) {
    CUcontext pctx;  // We loose track of it. But this is fine.
    if (auto result = cuDevicePrimaryCtxRetain(&pctx, 0))
      LOG(FATAL)  // Crash OK.
          << "Failed to retain context: " << cuda::ToStatus(result);
  }

  std::unique_ptr<ActivateContext> scoped_activation = stream_exec_->Activate();

  // Check the CUDA runtime is recent enough.
  if (auto status2 = cuDriverGetVersion(&driverVersion)) {
    LOG(FATAL)  // Crash OK.
        << "Error while fetching driver version: " << cuda::ToStatus(status2);
  }

  // Check that cudaMallocAsync is supported.
  int cuda_malloc_async_supported;
  if (auto status =
          cuDeviceGetAttribute(&cuda_malloc_async_supported,
                               CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                               platform_device_id.value())) {
    LOG(FATAL)  // Crash OK.
        << "On device: " << platform_device_id.value()
        << " Current driver: " << driverVersion
        << ". Failed to get device attribute : " << cuda::ToStatus(status);
  }
  if (!cuda_malloc_async_supported)
    LOG(FATAL)  // Crash OK.
        << "TF_GPU_ALLOCATOR=cuda_malloc_async isn't currently supported on "
        << "GPU id " << platform_device_id.value() << ":"
        << " Possible causes: device not supported (request SM60+), driver too "
           "old, "
        << " OS not supported, CUDA version too old(request CUDA11.2+).";

  size_t pool_size;
  if (create_new_pool_) {
    pool_size = new_pool_size;
    CUmemPoolProps pool_props;
    memset(reinterpret_cast<void*>(&pool_props), 0, sizeof(pool_props));
    pool_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    pool_props.location.id = platform_device_id.value();
    pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
#if CUDA_VERSION >= 12030
    pool_props.maxSize = new_pool_size;
#endif  // CUDA_VERSION >= 12030
    if (auto status = cuMemPoolCreate(&cuda_state_->pool, &pool_props))
      LOG(FATAL) <<  // Crash OK.
          "Failed to create CUDA pool: " << cuda::ToStatus(status);
  } else {
    pool_size = reserve_memory_size;
    if (auto status = cuDeviceGetDefaultMemPool(&cuda_state_->pool,
                                                platform_device_id.value()))
      LOG(FATAL) <<  // Crash OK.
          "Failed to get default CUDA pool: " << cuda::ToStatus(status);
    VLOG(2) << "using default memory pool " << cuda_state_->pool;
  }

  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_device_id.value() << " with pool size of: " << pool_size
          << " this ptr: " << this;
  uint64_t release_threshold_64 = reserve_memory_size;
  if (auto status = cuMemPoolSetAttribute(cuda_state_->pool,
                                          CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                          &release_threshold_64))
    LOG(FATAL) <<  // Crash OK.
        "Failed to set CUDA pool attribute: " << cuda::ToStatus(status);

  if (compute_stats) {
    stats_ = std::make_unique<tsl::AllocatorStats>();
    stats_->bytes_limit = static_cast<int64_t>(pool_size);
  }  // If not set, it means we do not compute stats.

  // If in TF_DETERMINISTIC_ALLOCATOR is set, then make the allocator behave
  // determistically.
  bool deterministic = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_DETERMINISTIC_ALLOCATOR",
                                      /*default_val=*/false, &deterministic));
  if (deterministic) {
    int disable = 0;
    if (auto status = cuMemPoolSetAttribute(
            cuda_state_->pool, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << cuda::ToStatus(status);
    }
    if (auto status = cuMemPoolSetAttribute(
            cuda_state_->pool,
            CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << cuda::ToStatus(status);
    }
  }

  // Set read/write access to all GPUs.
  static auto* const all_pools_ = new std::vector<CUmemoryPool>();
  static auto* const all_ids_ = new std::vector<tsl::PlatformDeviceId>();
  DCHECK(all_pools_->size() == all_ids_->size());

  // If the cuda_state_->pool is found in all_pools_, it means it has been
  // initialized before. This can happen in some cases, such as when multiple
  // virtual devices are created from one physical GPU, the virtual devices
  // will actually share the same CUDA memory pool. So the following pool
  // initialization steps should be skipped to avoid duplicated initialization
  // of the same pool.
  for (auto pool_item_ : *all_pools_) {
    if (pool_item_ == cuda_state_->pool) {
      VLOG(2) << Name()
              << " GpuCudaMallocAsyncAllocator pool already initialized. "
                 "PoolSize "
              << pool_size;
      return;
    }
  }
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
      cuda_state_->pool = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed to know if GPU id "
          << map.location.id << " can access GPU id "
          << platform_device_id.value() << ": " << cuda::ToStatus(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(cuda_state_->pool, &map, 1)) {
        cuda_state_->pool = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << cuda::ToStatus(status);
      }
    }

    // Set the previous pools access to the current GPU.
    map.location.id = platform_device_id.value();

    int previous_pool_id = (*all_ids_)[i].value();
    VLOG(2) << "Set access to the pool id: " << previous_pool_id
            << " location id: " << map.location.id;
    if (auto status = cuDeviceCanAccessPeer(&canAccessPeer, previous_pool_id,
                                            platform_device_id.value())) {
      cuda_state_->pool = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed: " << cuda::ToStatus(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess((*all_pools_)[i], &map, 1)) {
        cuda_state_->pool = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << previous_pool_id
            << " location id: " << map.location.id
            << " error: " << cuda::ToStatus(status);
      }
    }
  }
  all_pools_->push_back(cuda_state_->pool);
  all_ids_->push_back(platform_device_id);

  VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator PoolSize " << pool_size;
}

GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator(
    tsl::PlatformDeviceId platform_device_id, size_t release_threshold,
    bool reserve_memory, bool compute_stats)
    : GpuCudaMallocAsyncAllocator(platform_device_id, false, 0, reserve_memory,
                                  release_threshold, false, compute_stats) {}

GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator() {
  if (create_new_pool_) {
    VLOG(2) << "Delete memory pool "
            << reinterpret_cast<void*>(cuda_state_->pool);
    if (auto status = cuMemPoolDestroy(cuda_state_->pool))
      LOG(FATAL) << "Failed to destroy memory pool:" << cuda::ToStatus(status);
  }
}

void* GpuCudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
  CHECK(cuda_state_->cuda_stream != nullptr)
      << "A stream must be added to the GpuCudaMallocAsync allocator";
  if (cuda_state_->pool == nullptr) {
    LOG(FATAL)  // Crash OK.
        << "The instantiation of GpuCudaMallocAsyncAllocator failed."
        << " See previous errors.";
  }
  // The lock is only needed when stats are enabled, but it must be around
  // the cuMemAllocFromPoolAsync call as well to ensure consistency of the stats
  // update.
  std::optional<absl::MutexLock> lock;
  if (stats_) {
    lock.emplace(&mutex_);
  }
  std::unique_ptr<ActivateContext> scoped_activation = stream_exec_->Activate();
  void* ptr = nullptr;
  auto result =
      cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr), num_bytes,
                              cuda_state_->pool, cuda_state_->cuda_stream);
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    // Doing a stream synchronization give the driver more flexibility
    // for blocks coalescing and doing memory remapping. So it can
    // solve some OOM cases when memory is tight.
    cuStreamSynchronize(cuda_state_->cuda_stream);
    result =
        cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr), num_bytes,
                                cuda_state_->pool, cuda_state_->cuda_stream);
  }
  if (result) {
    size_t free, total;
    cuMemGetInfo(&free, &total);
    LOG(ERROR) << Name() << " cuMemAllocAsync failed to allocate " << num_bytes
               << " bytes: " << cuda::ToStatus(result)
               << "\n Reported by CUDA: Free memory/Total memory: " << free
               << "/" << total;
    if (stats_) {
      LOG(ERROR) << "Stats: " << stats_->DebugString();
      PrintAllocatorStatisticsNoLock();
    }

    return nullptr;
  }

  if (sync_mode_) {
    cuStreamSynchronize(cuda_state_->cuda_stream);
  }

  // Update stats.
  if (stats_) {
    ++(stats_->num_allocs);
    stats_->bytes_in_use += num_bytes;
    if (stats_->bytes_in_use > stats_->peak_bytes_in_use) {
      VLOG(9) << "New Peak memory usage of " << stats_->bytes_in_use
              << " bytes.";
    }
    stats_->peak_bytes_in_use =
        std::max(stats_->peak_bytes_in_use, stats_->bytes_in_use);
    stats_->largest_alloc_size =
        std::max<std::size_t>(stats_->largest_alloc_size, num_bytes);
    bool ptr_inserted = size_map_.emplace(ptr, num_bytes).second;
    DCHECK(ptr_inserted);
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << ptr;
  return ptr;
}

void GpuCudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
  if (ptr == nullptr) return;
  // The lock is only needed when stats are enabled, but it must be around
  // the cuMemFreeAsync call as well to ensure consistency of the stats update.
  std::optional<absl::MutexLock> lock;
  if (stats_) {
    lock.emplace(&mutex_);
  }
  if (auto result = cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(ptr),
                                   cuda_state_->cuda_stream)) {
    if (result == CUDA_ERROR_DEINITIALIZED) {
      // It happens with multi-GPU that TF free the GPU allocation after
      // the driver is unloaded. It is safe to ignore this error here.
      // TODO: Find how to fix the shutdown steps in TF.
      VLOG(1) << "Ignoring CUDA error: " << cuda::ToStatus(result);
    } else {
      size_t free, total;
      std::unique_ptr<ActivateContext> scoped_activation =
          stream_exec_->Activate();
      cuMemGetInfo(&free, &total);
      LOG(ERROR) << "cudaFreeAsync failed to free " << ptr << ": "
                 << cuda::ToStatus(result)
                 << "\n Free memory/Total memory: " << free << "/" << total;
      if (stats_) {
        LOG(ERROR) << "Stats: " << stats_->DebugString();
      }
    }
  }

  if (sync_mode_) {
    cuStreamSynchronize(cuda_state_->cuda_stream);
  }

  // Updates the stats.
  if (stats_) {
    DCHECK(size_map_.contains(ptr));
    size_t size = size_map_[ptr];
    stats_->bytes_in_use -= size;
    size_map_.erase(ptr);
  }

  VLOG(10) << Name() << " Freed ptr: " << ptr;
}

bool GpuCudaMallocAsyncAllocator::TracksAllocationSizes() const {
  return static_cast<bool>(stats_);
}

size_t GpuCudaMallocAsyncAllocator::RequestedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  absl::MutexLock l(&mutex_);
  return size_map_.at(ptr);
}

size_t GpuCudaMallocAsyncAllocator::AllocatedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  absl::MutexLock l(&mutex_);
  return size_map_.at(ptr);
}

std::optional<tsl::AllocatorStats> GpuCudaMallocAsyncAllocator::GetStats() {
  if (!stats_) return std::nullopt;
  absl::MutexLock l(&mutex_);
  return *stats_;
}

bool GpuCudaMallocAsyncAllocator::ClearStats() {
  if (!stats_) return false;
  absl::MutexLock l(&mutex_);
  stats_->num_allocs = 0;
  stats_->peak_bytes_in_use = stats_->bytes_in_use;
  stats_->largest_alloc_size = 0;
  return true;
}

void GpuCudaMallocAsyncAllocator::SetStreamAndPreallocateMemory(void* stream) {
  auto new_cuda_stream = static_cast<CUstream>(stream);
  // We don't need to re-set the CUDA stream if this is the same stream
  if (cuda_state_->cuda_stream != nullptr &&
      new_cuda_stream != cuda_state_->cuda_stream) {
    LOG(FATAL) <<  // Crash OK.
        "Trying to set the stream twice. This isn't supported. ";
  }

  uint64_t pool_size_64 = 0;
  if (auto status = cuMemPoolGetAttribute(cuda_state_->pool,
                                          CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                          &pool_size_64)) {
    LOG(FATAL) <<  // Crash OK.
        "Failed to get CUDA pool attribute: " << cuda::ToStatus(status);
  }
  cuda_state_->cuda_stream = new_cuda_stream;
  int64_t prealloc_size = 0;
  // TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1 is a special value that
  // preallocates the total pool size.
  TF_CHECK_OK(tsl::ReadInt64FromEnvVar(
      "TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC", 0, &prealloc_size));
  if (prealloc_size == -1) {
    prealloc_size = pool_size_64;
  } else if (reserve_memory_) {
    prealloc_size = pool_size_64;
  }

  if (prealloc_size != 0) {
    void* ptr = AllocateRaw(0, prealloc_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator reserved the pool for "
            << prealloc_size << " bytes" << ". First ptr: " << ptr;
    ClearStats();
  }
}

}  // namespace stream_executor
