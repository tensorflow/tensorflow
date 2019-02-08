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

#include <cstring>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/cuda_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

bool useCudaMallocAllocator() {
  const char* debug_allocator_str = std::getenv("TF_GPU_ALLOCATOR");
  return debug_allocator_str != nullptr &&
         std::strcmp(debug_allocator_str, "cuda_malloc") == 0;
}

bool useCudaMemoryGuardAllocator() {
  const char* debug_allocator_str = std::getenv("TF_GPU_ALLOCATOR");
  return debug_allocator_str != nullptr &&
         std::strcmp(debug_allocator_str, "memory_guard") == 0;
}

}  // namespace

/*static*/ GPUProcessState* GPUProcessState::singleton(GPUProcessState* ps) {
  static GPUProcessState* instance = ps ? ps : new GPUProcessState;
  DCHECK((!ps) || (ps == instance))
      << "Multiple calls to GPUProcessState with non-null ps";
  return instance;
}

GPUProcessState::GPUProcessState() : gpu_device_enabled_(false) {
  process_state_ = ProcessState::singleton();
}

int GPUProcessState::BusIdForGPU(TfGpuId tf_gpu_id) {
  // Return the NUMA node associated with the GPU's StreamExecutor.
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id).ValueOrDie();
  int numa_node = se->GetDeviceDescription().numa_node();
  // bus_id must be non-negative.  If the numa_node is not known,
  // use 0.
  return numa_node >= 0 ? numa_node : 0;
}

Allocator* GPUProcessState::GetGPUAllocator(const GPUOptions& options,
                                            TfGpuId tf_gpu_id,
                                            size_t total_bytes) {
  CHECK(process_state_);
#if GOOGLE_CUDA
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  GpuIdUtil::CheckValidTfGpuId(tf_gpu_id);

  if (tf_gpu_id.value() >= static_cast<int64>(gpu_allocators_.size())) {
    gpu_allocators_.resize(tf_gpu_id.value() + 1);
  }

  AllocatorParts& allocator_parts = gpu_allocators_[tf_gpu_id.value()];
  if (allocator_parts.allocator == nullptr) {
    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PlatformGpuId platform_gpu_id;
    TF_CHECK_OK(GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id));
    int bus_id = BusIdForGPU(tf_gpu_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= gpu_visitors_.size()) {
      gpu_visitors_.push_back({});
    }
    GPUMemAllocator* sub_allocator = new GPUMemAllocator(
        GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
        platform_gpu_id,
        (options.per_process_gpu_memory_fraction() > 1.0 ||
         options.experimental().use_unified_memory()),
        gpu_visitors_[bus_id], {});
    GPUBFCAllocator* gpu_bfc_allocator =
        new GPUBFCAllocator(sub_allocator, total_bytes, options,
                            strings::StrCat("GPU_", tf_gpu_id.value(), "_bfc"));
    Allocator* gpu_allocator = gpu_bfc_allocator;
    SharedCounter* timing_counter = nullptr;
    if (options.experimental().timestamped_allocator()) {
      timing_counter = new SharedCounter;
      gpu_bfc_allocator->SetTimingCounter(timing_counter);
    }

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    if (useCudaMemoryGuardAllocator()) {
      gpu_allocator = new GPUDebugAllocator(gpu_allocator, platform_gpu_id);
      gpu_allocator = new GPUNanResetAllocator(gpu_allocator, platform_gpu_id);
    } else if (useCudaMallocAllocator()) {
      // If true, passes all allocation requests through to cudaMalloc
      // useful for doing memory debugging with tools like cuda-memcheck
      // **WARNING** probably will not work in a multi-gpu scenario
      gpu_allocator =
          new GPUcudaMallocAllocator(gpu_allocator, platform_gpu_id);
    }

    Allocator* recording_allocator = nullptr;
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = platform_gpu_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      recording_allocator = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, gpu_allocator, md, &mu_);
    }
    allocator_parts = {std::unique_ptr<Allocator>(gpu_allocator),
                       std::unique_ptr<SharedCounter>(timing_counter),
                       sub_allocator,
                       std::unique_ptr<Allocator>(recording_allocator)};
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return allocator_parts.recording_allocator.get();
  } else {
    return allocator_parts.allocator.get();
  }
#else
  LOG(FATAL) << "GPUAllocator unavailable. Not compiled with --config=cuda.";
  return nullptr;
#endif  // GOOGLE_CUDA
}

std::unique_ptr<SharedCounter> GPUProcessState::ReleaseGPUAllocatorCounter(
    TfGpuId tf_gpu_id) {
  DCHECK(process_state_);
#if GOOGLE_CUDA
  GpuIdUtil::CheckValidTfGpuId(tf_gpu_id);
  mutex_lock l(mu_);
  if (tf_gpu_id.value() >= static_cast<int64>(gpu_allocators_.size())) {
    return nullptr;
  }

  AllocatorParts& allocator_parts = gpu_allocators_[tf_gpu_id.value()];
  return std::move(allocator_parts.counter);
#else
  return nullptr;
#endif
}

Allocator* GPUProcessState::GetCUDAHostAllocator(int numa_node) {
  CHECK(process_state_);
  if (!HasGPUDevice() ||
      !process_state_->ProcessState::FLAGS_brain_mem_reg_cuda_dma) {
    return process_state_->GetCPUAllocator(numa_node);
  }
  if (numa_node == port::kNUMANoAffinity) {
    numa_node = 0;
  }
  {
    // Here we optimize the most common use case where cuda_host_allocators_
    // and cuda_al_ have already been populated and since we're only reading
    // these vectors, we can get by with a shared lock. In the slower case,
    // we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);

    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types &&
        !cuda_host_allocators_.empty() &&
        cuda_host_allocators_[0].recording_allocator != nullptr) {
      return cuda_host_allocators_[0].recording_allocator.get();
    }
    if (static_cast<int>(cuda_host_allocators_.size()) > numa_node) {
      return cuda_host_allocators_[0].allocator.get();
    }
  }

  mutex_lock lock(mu_);
  // Find the first valid StreamExecutor to request CUDA host memory
  // through, since any will work.
  //
  // This search isn't super clean, and it would be nice to use a
  // better source of information about which executor to use.  For
  // example, process_state could maybe save the first stream executor
  // it knows is valid.
  se::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(gpu_allocators_.size()); ++i) {
    if (gpu_allocators_[i].allocator != nullptr) {
      se = GpuIdUtil::ExecutorForTfGpuId(TfGpuId(i)).ValueOrDie();
      break;
    }
  }

  CHECK_NE(nullptr, se);

  while (static_cast<int>(cuda_host_allocators_.size()) <= numa_node) {
    while (cuda_host_alloc_visitors_.size() <= numa_node) {
      cuda_host_alloc_visitors_.push_back({});
    }
    while (cuda_host_free_visitors_.size() <= numa_node) {
      cuda_host_free_visitors_.push_back({});
    }
    SubAllocator* sub_allocator = new CUDAHostAllocator(
        se, numa_node, cuda_host_alloc_visitors_[numa_node],
        cuda_host_free_visitors_[numa_node]);
    // TODO(zheng-xq): evaluate whether 64GB by default is the best choice.
    int64 cuda_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_CUDA_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &cuda_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetCUDAHostAllocator: " << status.error_message();
    }
    int64 cuda_host_mem_limit = cuda_host_mem_limit_in_mb * (1LL << 20);
    Allocator* allocator =
        new BFCAllocator(sub_allocator, cuda_host_mem_limit,
                         true /*allow_growth*/, "cuda_host_bfc" /*name*/);

    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    cuda_host_allocators_.push_back({std::unique_ptr<Allocator>(allocator),
                                     std::unique_ptr<SharedCounter>(nullptr),
                                     sub_allocator,
                                     std::unique_ptr<Allocator>(nullptr)});
    AllocatorParts& allocator_parts = cuda_host_allocators_.back();
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
    return cuda_host_allocators_[0].recording_allocator.get();
  } else {
    return cuda_host_allocators_[0].allocator.get();
  }
}

void GPUProcessState::AddGPUAllocVisitor(int bus_id,
                                         const SubAllocator::Visitor& visitor) {
#if GOOGLE_CUDA
  mutex_lock lock(mu_);
  CHECK(gpu_allocators_.empty())  // Crash OK
      << "AddGPUAllocVisitor must be called before "
         "first call to GetGPUAllocator.";
  DCHECK_GE(bus_id, 0);
  while (bus_id >= static_cast<int64>(gpu_visitors_.size())) {
    gpu_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_visitors_[bus_id].push_back(visitor);
#endif  // GOOGLE_CUDA
}

void GPUProcessState::AddCUDAHostAllocVisitor(
    int numa_node, const SubAllocator::Visitor& visitor) {
#if GOOGLE_CUDA
  mutex_lock lock(mu_);
  CHECK(cuda_host_allocators_.empty())  // Crash OK
      << "AddCUDAHostAllocVisitor must be called before "
         "first call to GetCUDAHostAllocator.";
  while (numa_node >= static_cast<int64>(cuda_host_alloc_visitors_.size())) {
    cuda_host_alloc_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  cuda_host_alloc_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA
}

void GPUProcessState::AddCUDAHostFreeVisitor(
    int numa_node, const SubAllocator::Visitor& visitor) {
#if GOOGLE_CUDA
  mutex_lock lock(mu_);
  CHECK(cuda_host_allocators_.empty())  // Crash OK
      << "AddCUDAHostFreeVisitor must be called before "
         "first call to GetCUDAHostAllocator.";
  while (numa_node >= static_cast<int64>(cuda_host_free_visitors_.size())) {
    cuda_host_free_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  cuda_host_free_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA
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
    cuda_host_allocators_.clear();
    cuda_host_alloc_visitors_.clear();
    cuda_host_free_visitors_.clear();
  }
}

}  // namespace tensorflow
