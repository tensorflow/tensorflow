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

#include "tensorflow/core/common_runtime/gpu/process_state.h"

#include <cstring>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/pool_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

// If these flags need to be runtime configurable, consider adding
// options to ConfigProto.

// If true, register CPU RAM used to copy to/from GPU RAM with the
// CUDA driver.
const bool FLAGS_brain_mem_reg_cuda_dma = true;

// If true, record attributes of memory allocations and
// dynamically check for appropriate use of registered memory.
// Should only be true for debugging or diagnosis of
// performance issues.
const bool FLAGS_brain_gpu_record_mem_types = false;

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

ProcessState* ProcessState::instance_ = nullptr;

/*static*/ ProcessState* ProcessState::singleton() {
  if (instance_ == nullptr) {
    instance_ = new ProcessState;
  }

  return instance_;
}

ProcessState::ProcessState() : gpu_device_enabled_(false) {
  CHECK(instance_ == nullptr);
  instance_ = this;
}

ProcessState::~ProcessState() {
  for (auto p : gpu_allocators_) {
    delete p;
  }
  instance_ = nullptr;
}

string ProcessState::MemDesc::DebugString() {
  return strings::StrCat((loc == CPU ? "CPU " : "GPU "), dev_index, ", dma: ",
                         gpu_registered, ", nic: ", nic_registered);
}

ProcessState::MemDesc ProcessState::PtrType(const void* ptr) {
  if (FLAGS_brain_gpu_record_mem_types) {
    auto iter = mem_desc_map_.find(ptr);
    if (iter != mem_desc_map_.end()) {
      return iter->second;
    }
  }
  return MemDesc();
}

Allocator* ProcessState::GetGPUAllocator(const GPUOptions& options,
                                         TfGpuId tf_gpu_id,
                                         size_t total_bytes) {
#if GOOGLE_CUDA
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  GpuIdUtil::CheckValidTfGpuId(tf_gpu_id);

  if (tf_gpu_id.value() >= static_cast<int64>(gpu_allocators_.size())) {
    gpu_allocators_.resize(tf_gpu_id.value() + 1);
    if (FLAGS_brain_gpu_record_mem_types) gpu_al_.resize(tf_gpu_id.value() + 1);
  }

  if (gpu_allocators_[tf_gpu_id.value()] == nullptr) {
    VisitableAllocator* gpu_allocator;

    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    const CudaGpuId cuda_gpu_id = GpuIdUtil::TfToCudaGpuId(tf_gpu_id);
    gpu_allocator =
        new GPUBFCAllocator(cuda_gpu_id, total_bytes, options,
                            strings::StrCat("GPU_", tf_gpu_id.value(), "_bfc"));

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    if (useCudaMemoryGuardAllocator()) {
      gpu_allocator = new GPUDebugAllocator(gpu_allocator, cuda_gpu_id);
      gpu_allocator = new GPUNanResetAllocator(gpu_allocator, cuda_gpu_id);
    } else if (useCudaMallocAllocator()) {
      // If true, passes all allocation requests through to cudaMalloc
      // useful for doing memory debugging with tools like cuda-memcheck
      // **WARNING** probably will not work in a multi-gpu scenario
      gpu_allocator = new GPUcudaMallocAllocator(gpu_allocator, cuda_gpu_id);
    }
    gpu_allocators_[tf_gpu_id.value()] = gpu_allocator;

    // If there are any pending AllocVisitors for this bus, add
    // them now.
    gpu::StreamExecutor* se =
        GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id).ValueOrDie();
    int bus_id = se->GetDeviceDescription().numa_node();
    if (bus_id >= 0 && bus_id < static_cast<int64>(gpu_visitors_.size())) {
      for (const auto& v : gpu_visitors_[bus_id]) {
        gpu_allocator->AddAllocVisitor(v);
      }
    }
    if (FLAGS_brain_gpu_record_mem_types) {
      MemDesc md;
      md.loc = MemDesc::GPU;
      md.dev_index = cuda_gpu_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      if (static_cast<int64>(gpu_al_.size()) <= tf_gpu_id.value()) {
        gpu_al_.resize(tf_gpu_id.value() + 1);
      }
      gpu_al_[tf_gpu_id.value()] = new internal::RecordingAllocator(
          &mem_desc_map_, gpu_allocator, md, &mu_);
    }
  }
  if (FLAGS_brain_gpu_record_mem_types) return gpu_al_[tf_gpu_id.value()];
  return gpu_allocators_[tf_gpu_id.value()];
#else
  LOG(FATAL) << "GPUAllocator unavailable. Not compiled with --config=cuda.";
  return nullptr;
#endif  // GOOGLE_CUDA
}

Allocator* ProcessState::GetCPUAllocator(int numa_node) {
  // Although we're temporarily ignoring numa_node, check for legality.
  CHECK_GE(numa_node, 0);
  // TODO(tucker): actually maintain separate CPUAllocators for
  // different numa_nodes.  For now, just one.
  numa_node = 0;
  mutex_lock lock(mu_);
  while (cpu_allocators_.size() <= static_cast<size_t>(numa_node)) {
    bool use_bfc_allocator = false;
    // TODO(reedwm): Switch default to BGFAllocator if it's at least as fast and
    // efficient.
    Status status = ReadBoolFromEnvVar("TF_CPU_ALLOCATOR_USE_BFC", false,
                                       &use_bfc_allocator);
    if (!status.ok()) {
      LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
    }
    VisitableAllocator* allocator;
    if (use_bfc_allocator) {
      // TODO(reedwm): evaluate whether 64GB by default is the best choice.
      int64 cpu_mem_limit_in_mb = -1;
      Status status = ReadInt64FromEnvVar("TF_CPU_BFC_MEM_LIMIT_IN_MB",
                                          1LL << 16 /*64GB max by default*/,
                                          &cpu_mem_limit_in_mb);
      if (!status.ok()) {
        LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
      }
      int64 cpu_mem_limit = cpu_mem_limit_in_mb * (1LL << 20);
      allocator = new BFCAllocator(new BasicCPUAllocator(), cpu_mem_limit,
                                   true /*allow_growth*/,
                                   "bfc_cpu_allocator_for_gpu" /*name*/);
      VLOG(2) << "Using BFCAllocator with memory limit of "
              << cpu_mem_limit_in_mb << " MB for ProcessState CPU allocator";
    } else {
      allocator = new PoolAllocator(
          100 /*pool_size_limit*/, true /*auto_resize*/,
          new BasicCPUAllocator(), new NoopRounder, "cpu_pool");
      VLOG(2) << "Using PoolAllocator for ProcessState CPU allocator";
    }
    if (LogMemory::IsEnabled()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingVisitableAllocator(allocator, true);
    }
    cpu_allocators_.push_back(allocator);
  }
  return cpu_allocators_[0];
}

Allocator* ProcessState::GetCUDAHostAllocator(int numa_node) {
  if (!HasGPUDevice() || !FLAGS_brain_mem_reg_cuda_dma) {
    return cpu_allocator();
  }
  // Although we're temporarily ignoring numa_node, check for legality.
  CHECK_GE(numa_node, 0);
  // TODO(tucker): actually maintain separate CPUAllocators for
  // different numa_nodes.  For now, just one.
  numa_node = 0;

  {
    // Here we optimize the most common use case where cuda_host_allocators_
    // and cuda_al_ have already been populated and since we're only reading
    // these vectors, we can get by with a shared lock. In the slower case,
    // we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);

    if (FLAGS_brain_gpu_record_mem_types &&
        static_cast<int>(cuda_al_.size()) > 0) {
      return cuda_al_[0];
    }
    if (static_cast<int>(cuda_host_allocators_.size()) > numa_node) {
      return cuda_host_allocators_[0];
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
  gpu::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(gpu_allocators_.size()); ++i) {
    if (gpu_allocators_[i] != nullptr) {
      se = GpuIdUtil::ExecutorForTfGpuId(TfGpuId(i)).ValueOrDie();
      break;
    }
  }

  CHECK_NE(nullptr, se);

  while (static_cast<int>(cuda_host_allocators_.size()) <= numa_node) {
    // TODO(zheng-xq): evaluate whether 64GB by default is the best choice.
    int64 cuda_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_CUDA_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &cuda_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetCUDAHostAllocator: " << status.error_message();
    }
    int64 cuda_host_mem_limit = cuda_host_mem_limit_in_mb * (1LL << 20);
    VisitableAllocator* allocator =
        new BFCAllocator(new CUDAHostAllocator(se), cuda_host_mem_limit,
                         true /*allow_growth*/, "cuda_host_bfc" /*name*/);

    if (LogMemory::IsEnabled()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingVisitableAllocator(allocator, true);
    }
    cuda_host_allocators_.push_back(allocator);
    if (FLAGS_brain_gpu_record_mem_types) {
      MemDesc md;
      md.loc = MemDesc::CPU;
      md.dev_index = 0;
      md.gpu_registered = true;
      md.nic_registered = false;
      cuda_al_.push_back(new internal::RecordingAllocator(
          &mem_desc_map_, cuda_host_allocators_.back(), md, &mu_));
    }
  }
  if (FLAGS_brain_gpu_record_mem_types) return cuda_al_[0];
  return cuda_host_allocators_[0];
}

void ProcessState::AddGPUAllocVisitor(int bus_id, AllocVisitor visitor) {
#if GOOGLE_CUDA
  mutex_lock lock(mu_);
  for (int i = 0; i < static_cast<int64>(gpu_allocators_.size()); ++i) {
    gpu::StreamExecutor* se =
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
#endif  // GOOGLE_CUDA
}

}  // namespace tensorflow
