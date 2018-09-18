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

#include "tensorflow/core/common_runtime/process_state.h"

#include <cstring>
#include <vector>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

ProcessState* ProcessState::instance_ = nullptr;

/*static*/ ProcessState* ProcessState::singleton() {
  if (instance_ == nullptr) {
    instance_ = new ProcessState;
  }

  return instance_;
}

ProcessState::ProcessState() : numa_enabled_(false) {
  CHECK(instance_ == nullptr);
}

// Normally the ProcessState singleton is never explicitly deleted.
// This function is defined for debugging problems with the allocators.
ProcessState::~ProcessState() {
  CHECK_EQ(this, instance_);
  instance_ = nullptr;
  for (Allocator* a : cpu_allocators_) {
    delete a;
  }
}

string ProcessState::MemDesc::DebugString() {
  return strings::StrCat((loc == CPU ? "CPU " : "GPU "), dev_index,
                         ", dma: ", gpu_registered, ", nic: ", nic_registered);
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

VisitableAllocator* ProcessState::GetCPUAllocator(int numa_node) {
  CHECK_GE(numa_node, 0);
  if (!numa_enabled_) numa_node = 0;
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
      allocator = new BFCAllocator(
          new BasicCPUAllocator(numa_enabled_ ? numa_node : -1), cpu_mem_limit,
          true /*allow_growth*/, "bfc_cpu_allocator_for_gpu" /*name*/);
      VLOG(2) << "Using BFCAllocator with memory limit of "
              << cpu_mem_limit_in_mb << " MB for ProcessState CPU allocator";
    } else {
      allocator = new PoolAllocator(
          100 /*pool_size_limit*/, true /*auto_resize*/,
          new BasicCPUAllocator(numa_enabled_ ? numa_node : -1),
          new NoopRounder, "cpu_pool");
      VLOG(2) << "Using PoolAllocator for ProcessState CPU allocator "
              << "numa_enabled_=" << numa_enabled_
              << " numa_node=" << numa_node;
    }
    if (LogMemory::IsEnabled()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingVisitableAllocator(allocator, true);
    }
    cpu_allocators_.push_back(allocator);
  }
  return cpu_allocators_[numa_node];
}

void ProcessState::TestOnlyReset() {
  mutex_lock lock(mu_);
  mem_desc_map_.clear();
  gtl::STLDeleteElements(&cpu_allocators_);
  gtl::STLDeleteElements(&cpu_al_);
}

}  // namespace tensorflow
