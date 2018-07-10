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

#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
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

GPUProcessState* GPUProcessState::instance_ = nullptr;

/*static*/ GPUProcessState* GPUProcessState::singleton() {
  if (instance_ == nullptr) {
    instance_ = new GPUProcessState;
  }
  CHECK(instance_->process_state_);

  return instance_;
}

GPUProcessState::GPUProcessState() : gpu_device_enabled_(false) {
  CHECK(instance_ == nullptr);
  instance_ = this;
  process_state_ = ProcessState::singleton();
}

// Normally the GPUProcessState singleton is never explicitly deleted.
// This function is defined for debugging problems with the allocators.
GPUProcessState::~GPUProcessState() {
  CHECK_EQ(this, instance_);
  for (auto p : gpu_allocators_) {
    delete p;
  }
  instance_ = nullptr;
}


Allocator* GPUProcessState::GetGPUAllocator(const GPUOptions& options,
                                            TfGpuId tf_gpu_id,
                                            size_t total_bytes) {
  CHECK(process_state_);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  GpuIdUtil::CheckValidTfGpuId(tf_gpu_id);

  if (tf_gpu_id.value() >= static_cast<int64>(gpu_allocators_.size())) {
    gpu_allocators_.resize(tf_gpu_id.value() + 1);
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types)
      gpu_al_.resize(tf_gpu_id.value() + 1);
  }

  if (gpu_allocators_[tf_gpu_id.value()] == nullptr) {
    VisitableAllocator* gpu_allocator;

    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PhysicalGpuId physical_gpu_id;
    TF_CHECK_OK(GpuIdManager::TfToPhysicalGpuId(tf_gpu_id, &physical_gpu_id));
    gpu_allocator =
        new GPUBFCAllocator(physical_gpu_id, total_bytes, options,
                            strings::StrCat("GPU_", tf_gpu_id.value(), "_bfc"));

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    if (useCudaMemoryGuardAllocator()) {
      gpu_allocator = new GPUDebugAllocator(gpu_allocator, physical_gpu_id);
      gpu_allocator = new GPUNanResetAllocator(gpu_allocator, physical_gpu_id);
    } else if (useCudaMallocAllocator()) {
      // If true, passes all allocation requests through to cudaMalloc
      // useful for doing memory debugging with tools like cuda-memcheck
      // **WARNING** probably will not work in a multi-gpu scenario
      gpu_allocator = new GPUcudaMallocAllocator(gpu_allocator, physical_gpu_id);
    }
    gpu_allocators_[tf_gpu_id.value()] = gpu_allocator;

    // If there are any pending AllocVisitors for this bus, add
    // them now.
    se::StreamExecutor* se =
        GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id).ValueOrDie();
    int bus_id = se->GetDeviceDescription().numa_node();
    if (bus_id >= 0 && bus_id < static_cast<int64>(gpu_visitors_.size())) {
      for (const auto& v : gpu_visitors_[bus_id]) {
        gpu_allocator->AddAllocVisitor(v);
      }
    }
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = physical_gpu_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      if (static_cast<int64>(gpu_al_.size()) <= tf_gpu_id.value()) {
        gpu_al_.resize(tf_gpu_id.value() + 1);
      }
      gpu_al_[tf_gpu_id.value()] = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, gpu_allocator, md, &mu_);
    }
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types)
    return gpu_al_[tf_gpu_id.value()];
  return gpu_allocators_[tf_gpu_id.value()];
#else
  LOG(FATAL) << "GPUAllocator unavailable. Not compiled with --config=cuda or --config=rocm.";
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

Allocator* GPUProcessState::GetGPUHostAllocator(int numa_node) {
  CHECK(process_state_);
  if (!HasGPUDevice() ||
      !process_state_->ProcessState::FLAGS_brain_mem_reg_gpu_dma) {
    return process_state_->GetCPUAllocator(numa_node);
  }

  CHECK_GE(numa_node, 0);
  {
    // Here we optimize the most common use case where gpu_host_allocators_
    // and gpu_host_al_ have already been populated and since we're only reading
    // these vectors, we can get by with a shared lock. In the slower case,
    // we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);

    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types &&
        static_cast<int>(gpu_host_al_.size()) > 0) {
      return gpu_host_al_[0];
    }
    if (static_cast<int>(gpu_host_allocators_.size()) > numa_node) {
      return gpu_host_allocators_[0];
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
    if (gpu_allocators_[i] != nullptr) {
      se = GpuIdUtil::ExecutorForTfGpuId(TfGpuId(i)).ValueOrDie();
      break;
    }
  }

  CHECK_NE(nullptr, se);

  while (static_cast<int>(gpu_host_allocators_.size()) <= numa_node) {
    // TODO(zheng-xq): evaluate whether 64GB by default is the best choice.
    int64 gpu_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_CUDA_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &gpu_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetGPUHostAllocator: " << status.error_message();
    }
    int64 gpu_host_mem_limit = gpu_host_mem_limit_in_mb * (1LL << 20);
    VisitableAllocator* allocator =
        new BFCAllocator(new GPUHostAllocator(se), gpu_host_mem_limit,
                         true /*allow_growth*/, "gpu_host_bfc" /*name*/);

    if (LogMemory::IsEnabled()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingVisitableAllocator(allocator, true);
    }
    gpu_host_allocators_.push_back(allocator);
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::CPU;
      md.dev_index = 0;
      md.gpu_registered = true;
      md.nic_registered = false;
      gpu_host_al_.push_back(new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, gpu_host_allocators_.back(), md,
          &mu_));
    }
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types)
    return gpu_host_al_[0];
  return gpu_host_allocators_[0];
}

void GPUProcessState::AddGPUAllocVisitor(int bus_id,
                                         const AllocVisitor& visitor) {
  CHECK(process_state_);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  mutex_lock lock(mu_);
  for (int i = 0; i < static_cast<int64>(gpu_allocators_.size()); ++i) {
    se::StreamExecutor* se =
        GpuIdUtil::ExecutorForTfGpuId(TfGpuId(i)).ValueOrDie();
    if (gpu_allocators_[i] &&
        (se->GetDeviceDescription().numa_node() + 1) == bus_id) {
      gpu_allocators_[i]->AddAllocVisitor(visitor);
    }
  }
  while (bus_id >= static_cast<int64>(gpu_visitors_.size())) {
    gpu_visitors_.push_back(std::vector<AllocVisitor>());
  }
  gpu_visitors_[bus_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::TestOnlyReset() {
  process_state_->ProcessState::TestOnlyReset();
  {
    mutex_lock lock(mu_);
    gpu_device_enabled_ = false;
    gpu_visitors_.clear();
    gtl::STLDeleteElements(&gpu_allocators_);
    gtl::STLDeleteElements(&gpu_host_allocators_);
    gtl::STLDeleteElements(&gpu_al_);
    gtl::STLDeleteElements(&gpu_host_al_);
  }
}

}  // namespace tensorflow
