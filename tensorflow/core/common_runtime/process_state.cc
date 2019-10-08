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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

/*static*/ ProcessState* ProcessState::singleton() {
  static ProcessState* instance = new ProcessState;
  static std::once_flag f;
  std::call_once(f, []() {
    AllocatorFactoryRegistry::singleton()->process_state_ = instance;
  });

  return instance;
}

ProcessState::ProcessState() : numa_enabled_(false) {}

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

Allocator* ProcessState::GetCPUAllocator(int numa_node) {
  if (!numa_enabled_ || numa_node == port::kNUMANoAffinity) numa_node = 0;
  mutex_lock lock(mu_);
  while (cpu_allocators_.size() <= static_cast<size_t>(numa_node)) {
    // If visitors have been defined we need an Allocator built from
    // a SubAllocator.  Prefer BFCAllocator, but fall back to PoolAllocator
    // depending on env var setting.
    const bool alloc_visitors_defined =
        (!cpu_alloc_visitors_.empty() || !cpu_free_visitors_.empty());
    bool use_bfc_allocator = false;
    Status status = ReadBoolFromEnvVar(
        "TF_CPU_ALLOCATOR_USE_BFC", alloc_visitors_defined, &use_bfc_allocator);
    if (!status.ok()) {
      LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
    }
    Allocator* allocator = nullptr;
    SubAllocator* sub_allocator =
        (numa_enabled_ || alloc_visitors_defined || use_bfc_allocator)
            ? new BasicCPUAllocator(
                  numa_enabled_ ? numa_node : port::kNUMANoAffinity,
                  cpu_alloc_visitors_, cpu_free_visitors_)
            : nullptr;
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
      DCHECK(sub_allocator);
      allocator =
          new BFCAllocator(sub_allocator, cpu_mem_limit, true /*allow_growth*/,
                           "bfc_cpu_allocator_for_gpu" /*name*/);
      VLOG(2) << "Using BFCAllocator with memory limit of "
              << cpu_mem_limit_in_mb << " MB for ProcessState CPU allocator";
    } else if (sub_allocator) {
      DCHECK(sub_allocator);
      allocator =
          new PoolAllocator(100 /*pool_size_limit*/, true /*auto_resize*/,
                            sub_allocator, new NoopRounder, "cpu_pool");
      VLOG(2) << "Using PoolAllocator for ProcessState CPU allocator "
              << "numa_enabled_=" << numa_enabled_
              << " numa_node=" << numa_node;
    } else {
      DCHECK(!sub_allocator);
      allocator = cpu_allocator_base();
    }
    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    cpu_allocators_.push_back(allocator);
    if (!sub_allocator) {
      DCHECK(cpu_alloc_visitors_.empty() && cpu_free_visitors_.empty());
    }
  }
  return cpu_allocators_[numa_node];
}

void ProcessState::AddCPUAllocVisitor(SubAllocator::Visitor visitor) {
  VLOG(1) << "AddCPUAllocVisitor";
  mutex_lock lock(mu_);
  CHECK_EQ(0, cpu_allocators_.size())  // Crash OK
      << "AddCPUAllocVisitor must be called prior to first call to "
         "ProcessState::GetCPUAllocator";
  cpu_alloc_visitors_.push_back(std::move(visitor));
}

void ProcessState::AddCPUFreeVisitor(SubAllocator::Visitor visitor) {
  mutex_lock lock(mu_);
  CHECK_EQ(0, cpu_allocators_.size())  // Crash OK
      << "AddCPUFreeVisitor must be called prior to first call to "
         "ProcessState::GetCPUAllocator";
  cpu_free_visitors_.push_back(std::move(visitor));
}

void ProcessState::TestOnlyReset() {
  mutex_lock lock(mu_);
  // Don't delete this value because it's static.
  Allocator* default_cpu_allocator = cpu_allocator_base();
  mem_desc_map_.clear();
  for (Allocator* a : cpu_allocators_) {
    if (a != default_cpu_allocator) delete a;
  }
  cpu_allocators_.clear();
  for (Allocator* a : cpu_al_) {
    delete a;
  }
  cpu_al_.clear();
}

}  // namespace tensorflow
