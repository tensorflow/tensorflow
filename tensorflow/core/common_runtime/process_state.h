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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_STATE_H_

#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Allocator;
class PoolAllocator;

// Singleton that manages per-process state, e.g. allocation of
// shared resources.
class ProcessState : public ProcessStateInterface {
 public:
  static ProcessState* singleton();

  // Descriptor for memory allocation attributes, used by optional
  // runtime correctness analysis logic.
  struct MemDesc {
    enum MemLoc { CPU, GPU };
    MemLoc loc;
    int dev_index;
    bool gpu_registered;
    bool nic_registered;
    MemDesc()
        : loc(CPU),
          dev_index(0),
          gpu_registered(false),
          nic_registered(false) {}
    string DebugString();
  };

  // If NUMA Allocators are desired, call this before calling any
  // Allocator accessor.
  void EnableNUMA() { numa_enabled_ = true; }

  // Returns what we know about the memory at ptr.
  // If we know nothing, it's called CPU 0 with no other attributes.
  MemDesc PtrType(const void* ptr);

  // Returns the one CPUAllocator used for the given numa_node.
  // Treats numa_node == kNUMANoAffinity as numa_node == 0.
  Allocator* GetCPUAllocator(int numa_node) override;

  // Registers alloc visitor for the CPU allocator(s).
  // REQUIRES: must be called before GetCPUAllocator.
  void AddCPUAllocVisitor(SubAllocator::Visitor v);

  // Registers free visitor for the CPU allocator(s).
  // REQUIRES: must be called before GetCPUAllocator.
  void AddCPUFreeVisitor(SubAllocator::Visitor v);

  typedef std::unordered_map<const void*, MemDesc> MDMap;

 protected:
  ProcessState();
  virtual ~ProcessState() {}
  friend class GPUProcessState;

  // If these flags need to be runtime configurable consider adding
  // them to ConfigProto.
  static const bool FLAGS_brain_mem_reg_gpu_dma = true;
  static const bool FLAGS_brain_gpu_record_mem_types = false;

  // Helper method for unit tests to reset the ProcessState singleton by
  // cleaning up everything. Never use in production.
  void TestOnlyReset();

  static ProcessState* instance_;
  bool numa_enabled_;

  mutex mu_;

  // Indexed by numa_node.  If we want numa-specific allocators AND a
  // non-specific allocator, maybe should index by numa_node+1.
  std::vector<Allocator*> cpu_allocators_ GUARDED_BY(mu_);
  std::vector<SubAllocator::Visitor> cpu_alloc_visitors_ GUARDED_BY(mu_);
  std::vector<SubAllocator::Visitor> cpu_free_visitors_ GUARDED_BY(mu_);

  // Optional RecordingAllocators that wrap the corresponding
  // Allocators for runtime attribute use analysis.
  MDMap mem_desc_map_;
  std::vector<Allocator*> cpu_al_ GUARDED_BY(mu_);
};

namespace internal {
class RecordingAllocator : public Allocator {
 public:
  RecordingAllocator(ProcessState::MDMap* mm, Allocator* a,
                     ProcessState::MemDesc md, mutex* mu)
      : mm_(mm), a_(a), md_(md), mu_(mu) {}

  string Name() override { return a_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* p = a_->AllocateRaw(alignment, num_bytes);
    mutex_lock l(*mu_);
    (*mm_)[p] = md_;
    return p;
  }
  void DeallocateRaw(void* p) override {
    mutex_lock l(*mu_);
    auto iter = mm_->find(p);
    mm_->erase(iter);
    a_->DeallocateRaw(p);
  }
  bool TracksAllocationSizes() const override {
    return a_->TracksAllocationSizes();
  }
  size_t RequestedSize(const void* p) const override {
    return a_->RequestedSize(p);
  }
  size_t AllocatedSize(const void* p) const override {
    return a_->AllocatedSize(p);
  }
  absl::optional<AllocatorStats> GetStats() override { return a_->GetStats(); }
  void ClearStats() override { a_->ClearStats(); }
  ProcessState::MDMap* mm_;  // not owned
  Allocator* a_;             // not owned
  ProcessState::MemDesc md_;
  mutex* mu_;
};
}  // namespace internal
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
