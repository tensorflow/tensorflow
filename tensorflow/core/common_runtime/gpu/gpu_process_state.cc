/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/stream_executor/device_id_utils.h"
#include "tensorflow/compiler/xla/stream_executor/device_mem_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_init.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/tsl/framework/allocator.h"
#include "tensorflow/tsl/framework/bfc_allocator.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/framework/device_id_utils.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/util/env_var.h"

namespace tensorflow {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMallocAllocator() {
  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  return allocator_env != nullptr &&
         std::strcmp(allocator_env, "cuda_malloc") == 0;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMemoryGuardAllocator() {
  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  return allocator_env != nullptr &&
         std::strcmp(allocator_env, "memory_guard") == 0;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMallocAsyncAllocator() {
  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  auto result = allocator_env != nullptr &&
                std::strcmp(allocator_env, "cuda_malloc_async") == 0;
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  return result;
#else
  if (result)
    LOG(ERROR) << "TF_GPU_ALLOCATOR=cuda_malloc_async environment found, "
               << "but TensorFlow was not compiled with CUDA 11.2+.";
  return false;
#endif
}

/*static*/ GPUProcessState* GPUProcessState::singleton(GPUProcessState* ps) {
  static GPUProcessState* instance = ps ? ps : new GPUProcessState;
  DCHECK((!ps) || (ps == instance))
      << "Multiple calls to GPUProcessState with non-null ps";
  return instance;
}

GPUProcessState::GPUProcessState() : gpu_device_enabled_(false) {
  process_state_ = ProcessState::singleton();
}

int GPUProcessState::BusIdForGPU(tsl::TfDeviceId tf_device_id) {
  // Return the NUMA node associated with the GPU's StreamExecutor.
  se::StreamExecutor* se =
      se::DeviceIdUtil::ExecutorForTfDeviceId(
          DEVICE_GPU, se::GPUMachineManager(), tf_device_id)
          .value();
  int numa_node = se->GetDeviceDescription().numa_node();
  // bus_id must be non-negative.  If the numa_node is not known,
  // use 0.
  return numa_node >= 0 ? numa_node : 0;
}

// NOLINTNEXTLINE: clang-tidy complains this is unused because of build flags.
static std::unique_ptr<SubAllocator> CreateSubAllocator(
    const GPUOptions& options, tsl::PlatformDeviceId platform_device_id,
    const std::vector<SubAllocator::Visitor>& alloc_visitors,
    size_t total_bytes, const std::vector<tsl::TfDeviceId>& peer_gpu_ids) {
  auto executor = se::DeviceIdUtil::ExecutorForPlatformDeviceId(
                      se::GPUMachineManager(), platform_device_id)
                      .value();

  // FIXME(imintz): Observed OOM issues when using the virtual memory
  // allocators. This should be reenabled when resolved.
#if 0 && defined(GOOGLE_CUDA) && CUDA_VERSION >= 10020
  // Use the old allocator when unified memory is required.
  // TODO(imintz): Remove the cuMemAlloc capability of this allocator.
  if (options.per_process_gpu_memory_fraction() > 1.0 ||
      options.experimental().use_unified_memory()) {
    return new se::DeviceMemAllocator(executor, platform_device_id,
                                  /*use_unified_memory=*/true, alloc_visitors,
                                  {});
  } else {
    auto* gpu_context = reinterpret_cast<stream_executor::gpu::GpuContext*>(
        executor->implementation()->GpuContextHack());

    absl::flat_hash_set<tsl::PlatformDeviceId> platform_peer_gpu_ids;
    platform_peer_gpu_ids.reserve(peer_gpu_ids.size());
    for (const tsl::TfDeviceId tf_device_id : peer_gpu_ids) {
      tsl::PlatformDeviceId platform_device_id;
      TF_CHECK_OK(GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
      platform_peer_gpu_ids.insert(platform_device_id);
    }
    std::vector<tsl::PlatformDeviceId> platform_peer_gpu_ids_vec(
        platform_peer_gpu_ids.begin(), platform_peer_gpu_ids.end());

    // Adjust virtual address space to be slightly larger than the physical
    // address space in case the BFC allocator performs suboptimal garbage
    // collection.
    // TODO(imintz): Update BFC allocator to ensure it doesn't create holes in
    // the va space.
    return GpuVirtualMemAllocator::Create(
               alloc_visitors, {}, *gpu_context, platform_device_id,
               /*virtual_address_space_size=*/total_bytes * 2,
               platform_peer_gpu_ids_vec)
        .value()
        .release();
  }
#else
  return absl::WrapUnique(new se::DeviceMemAllocator(
      executor, platform_device_id,
      (options.per_process_gpu_memory_fraction() > 1.0 ||
       options.experimental().use_unified_memory()),
      alloc_visitors, {}));
#endif
}

Allocator* GPUProcessState::GetGPUAllocator(
    const GPUOptions& options, tsl::TfDeviceId tf_device_id, size_t total_bytes,
    const std::vector<tsl::TfDeviceId>& peer_gpu_ids) {
  CHECK(process_state_);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  tsl::CheckValidTfDeviceId(
      DEVICE_GPU, se::GPUMachineManager()->VisibleDeviceCount(), tf_device_id);

  if (tf_device_id.value() >= static_cast<int64_t>(gpu_allocators_.size())) {
    gpu_allocators_.resize(tf_device_id.value() + 1);
  }

  AllocatorParts& allocator_parts = gpu_allocators_[tf_device_id.value()];
  if (allocator_parts.allocator == nullptr) {
    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    tsl::PlatformDeviceId platform_device_id;
    TF_CHECK_OK(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
    int bus_id = BusIdForGPU(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= gpu_visitors_.size()) {
      gpu_visitors_.push_back({});
    }
    std::unique_ptr<SubAllocator> sub_allocator =
        CreateSubAllocator(options, platform_device_id, gpu_visitors_[bus_id],
                           total_bytes, peer_gpu_ids);
    SubAllocator* sub_allocator_ptr = sub_allocator.get();

    auto gpu_bfc_allocator = absl::make_unique<GPUBFCAllocator>(
        std::move(sub_allocator), total_bytes,
        strings::StrCat("GPU_", tf_device_id.value(), "_bfc"), [&] {
          GPUBFCAllocator::Options o;
          o.allow_growth = options.allow_growth();
          o.allow_retry_on_failure =
              !options.experimental().disallow_retry_on_allocation_failure();
          o.fragmentation_fraction =
              options.experimental().internal_fragmentation_fraction();
          return o;
        }());
    Allocator* gpu_allocator = gpu_bfc_allocator.get();

    SharedCounter* timing_counter = nullptr;
    if (options.experimental().timestamped_allocator()) {
      timing_counter = new SharedCounter;
      gpu_bfc_allocator->SetTimingCounter(timing_counter);
    }

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    if (UseCudaMemoryGuardAllocator()) {
      LOG(INFO) << "Using memory guard allocator for GPU.";
      gpu_allocator = new GPUNanResetAllocator(
          new GPUDebugAllocator(gpu_allocator, platform_device_id),
          platform_device_id);
    } else if (UseCudaMallocAllocator()) {
      LOG(INFO) << "Using CUDA malloc allocator for GPU.";
      // If true, passes all allocation requests through to cudaMalloc
      // useful for doing memory debugging with tools like cuda-memcheck
      // **WARNING** probably will not work in a multi-gpu scenario
      gpu_bfc_allocator.reset();
      gpu_allocator = new GPUcudaMallocAllocator(platform_device_id);
    } else if (UseCudaMallocAsyncAllocator() ||
               options.experimental().use_cuda_malloc_async()) {
      LOG(INFO) << "Using CUDA malloc Async allocator for GPU: "
                << platform_device_id;
      // If true, passes all allocation requests through to cudaMallocAsync
      // TODO: useful for doing memory debugging with tools like
      // compute-sanitizer.
      // TODO: **WARNING** probably will not work in a multi-gpu scenario
      gpu_bfc_allocator.reset();
      gpu_allocator =
          new se::GpuCudaMallocAsyncAllocator(platform_device_id, total_bytes);
    }

    Allocator* recording_allocator = nullptr;
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = platform_device_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      recording_allocator = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, gpu_allocator, md, &mu_);
    }
    allocator_parts = {
        std::unique_ptr<Allocator>(gpu_allocator),
        std::unique_ptr<SharedCounter>(timing_counter),
        gpu_bfc_allocator.release(),
        sub_allocator_ptr,
        std::unique_ptr<Allocator>(recording_allocator),
    };
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return allocator_parts.recording_allocator.get();
  } else {
    return allocator_parts.allocator.get();
  }
#else
  LOG(FATAL) << "GPUAllocator unavailable. Not compiled with --config=cuda or "
                "--config=rocm.";
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

SharedCounter* GPUProcessState::GPUAllocatorCounter(
    tsl::TfDeviceId tf_device_id) {
  DCHECK(process_state_);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  tsl::CheckValidTfDeviceId(
      DEVICE_GPU, se::GPUMachineManager()->VisibleDeviceCount(), tf_device_id);
  mutex_lock l(mu_);
  if (tf_device_id.value() >= static_cast<int64_t>(gpu_allocators_.size())) {
    LOG(ERROR) << "Asked for counter for GPU allocator " << tf_device_id.value()
               << " but only have " << gpu_allocators_.size();
    return nullptr;
  }

  AllocatorParts& allocator_parts = gpu_allocators_[tf_device_id.value()];
  if (allocator_parts.counter.get() == nullptr) {
    if (allocator_parts.bfc_allocator == nullptr) {
      return nullptr;
    }
    SharedCounter* timing_counter = new SharedCounter;
    allocator_parts.bfc_allocator->SetTimingCounter(timing_counter);
    allocator_parts.counter.reset(timing_counter);
  }
  return allocator_parts.counter.get();
#else
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

Allocator* GPUProcessState::GetGpuHostAllocator(const GPUOptions& options,
                                                int numa_node) {
  CHECK(process_state_);
  if (!HasGPUDevice() ||
      !process_state_->ProcessState::FLAGS_brain_mem_reg_gpu_dma) {
    return process_state_->GetCPUAllocator(numa_node);
  }
  if (numa_node == port::kNUMANoAffinity) {
    numa_node = 0;
  }
  {
    // Here we optimize the most common use case where gpu_host_allocators_
    // have already been populated and since we're only reading
    // these vectors, we can get by with a shared lock. In the slower case,
    // we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);

    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types &&
        !gpu_host_allocators_.empty() &&
        gpu_host_allocators_[0].recording_allocator != nullptr) {
      return gpu_host_allocators_[0].recording_allocator.get();
    }
    if (static_cast<int>(gpu_host_allocators_.size()) > numa_node) {
      return gpu_host_allocators_[0].allocator.get();
    }
  }

  mutex_lock lock(mu_);
  // Find the first valid StreamExecutor to request CUDA or ROCm host memory
  // through, since any will work.
  //
  // This search isn't super clean, and it would be nice to use a
  // better source of information about which executor to use.  For
  // example, process_state could maybe save the first stream executor
  // it knows is valid.
  se::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(gpu_allocators_.size()); ++i) {
    if (gpu_allocators_[i].allocator != nullptr) {
      se = se::DeviceIdUtil::ExecutorForTfDeviceId(
               DEVICE_GPU, se::GPUMachineManager(), tsl::TfDeviceId(i))
               .value();
      break;
    }
  }

  CHECK_NE(nullptr, se);

  int64_t mem_limit_bytes =
      options.experimental().gpu_host_mem_limit_in_mb() * (1LL << 20);
  if (mem_limit_bytes <= 0) {
    int64_t limit_mb = -1;
    Status status =
        tsl::ReadInt64FromEnvVar("TF_GPU_HOST_MEM_LIMIT_IN_MB",
                                 1LL << 17 /*2^17 MB == 128GB*/, &limit_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetGpuHostAllocator: " << status.message();
    }
    mem_limit_bytes = limit_mb * (1LL << 20);
  }

  while (static_cast<int>(gpu_host_allocators_.size()) <= numa_node) {
    while (gpu_host_alloc_visitors_.size() <= numa_node) {
      gpu_host_alloc_visitors_.push_back({});
    }
    while (gpu_host_free_visitors_.size() <= numa_node) {
      gpu_host_free_visitors_.push_back({});
    }
    SubAllocator* sub_allocator = new DeviceHostAllocator(
        se, numa_node, gpu_host_alloc_visitors_[numa_node],
        gpu_host_free_visitors_[numa_node]);

    tsl::BFCAllocator::Options allocator_opts;
    allocator_opts.allow_growth =
        !options.experimental().gpu_host_mem_disallow_growth();
    tsl::Allocator* allocator =
        new tsl::BFCAllocator(absl::WrapUnique(sub_allocator), mem_limit_bytes,
                              /*name=*/"gpu_host_bfc", allocator_opts);

    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    gpu_host_allocators_.push_back({std::unique_ptr<Allocator>(allocator),
                                    std::unique_ptr<SharedCounter>(nullptr),
                                    nullptr, sub_allocator,
                                    std::unique_ptr<Allocator>(nullptr)});
    AllocatorParts& allocator_parts = gpu_host_allocators_.back();
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::CPU;
      md.dev_index = 0;
      md.gpu_registered = true;
      md.nic_registered = false;
      allocator_parts.recording_allocator.reset(
          new internal::RecordingAllocator(&process_state_->mem_desc_map_,
                                           allocator_parts.allocator.get(), md,
                                           &mu_));
    }
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return gpu_host_allocators_[0].recording_allocator.get();
  } else {
    return gpu_host_allocators_[0].allocator.get();
  }
}

void GPUProcessState::AddGPUAllocVisitor(int bus_id,
                                         const SubAllocator::Visitor& visitor) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_allocators_.empty())  // Crash OK
      << "AddGPUAllocVisitor must be called before "
         "first call to GetGPUAllocator.";
  DCHECK_GE(bus_id, 0);
  while (bus_id >= static_cast<int64_t>(gpu_visitors_.size())) {
    gpu_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_visitors_[bus_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGpuHostAllocVisitor(
    int numa_node, const SubAllocator::Visitor& visitor) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_host_allocators_.empty())  // Crash OK
      << "AddGpuHostAllocVisitor must be called before "
         "first call to GetGpuHostAllocator.";
  while (numa_node >= static_cast<int64_t>(gpu_host_alloc_visitors_.size())) {
    gpu_host_alloc_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_host_alloc_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGpuHostFreeVisitor(
    int numa_node, const SubAllocator::Visitor& visitor) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_host_allocators_.empty())  // Crash OK
      << "AddGpuHostFreeVisitor must be called before "
         "first call to GetGpuHostAllocator.";
  while (numa_node >= static_cast<int64_t>(gpu_host_free_visitors_.size())) {
    gpu_host_free_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_host_free_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::TestOnlyReset() {
  if (process_state_) {
    process_state_->ProcessState::TestOnlyReset();
  }
  {
    mutex_lock lock(mu_);
    gpu_device_enabled_ = false;
    gpu_allocators_.clear();
    gpu_visitors_.clear();
    gpu_host_allocators_.clear();
    gpu_host_alloc_visitors_.clear();
    gpu_host_free_visitors_.clear();
  }
}

}  // namespace tensorflow
