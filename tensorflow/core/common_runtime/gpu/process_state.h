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

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_PROCESS_STATE_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_PROCESS_STATE_H_

#include <functional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Allocator;
class VisitableAllocator;
class PoolAllocator;

// Singleton that manages per-process state, e.g. allocation
// of shared resources.
class ProcessState {
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

  // Query whether any GPU device has been created so far.
  // Disable thread safety analysis since a race is benign here.
  bool HasGPUDevice() const NO_THREAD_SAFETY_ANALYSIS {
    return gpu_device_enabled_;
  }

  // Set the flag to indicate a GPU device has been created.
  // Disable thread safety analysis since a race is benign here.
  void EnableGPUDevice() NO_THREAD_SAFETY_ANALYSIS {
    gpu_device_enabled_ = true;
  }

  // Returns what we know about the memory at ptr.
  // If we know nothing, it's called CPU 0 with no other attributes.
  MemDesc PtrType(const void* ptr);

  // Returns the one CPUAllocator used for the given numa_node.
  // TEMPORARY: ignores numa_node.
  Allocator* GetCPUAllocator(int numa_node);

  // Returns the one GPU allocator used for the indexed GPU.
  // Note that this is a system GPU index, not (necessarily) a brain
  // device index.
  //
  // 'total_bytes' is the total number of bytes that should be made
  // available to the allocator.  The first call to this function for
  // a given gpu_id creates the allocator, so only the total_bytes
  // used on that first call is used.
  //
  // "Allocator type" describes the type of algorithm to use for the
  // underlying allocator.  REQUIRES: Must be a valid type (see
  // config.proto for the list of supported strings.).
  //
  // REQUIRES: gpu_id must be a valid ordinal for a GPU available in the
  // current system environment.  Otherwise returns nullptr.
  virtual Allocator* GetGPUAllocator(const GPUOptions& options, int gpu_id,
                                     size_t total_bytes);

  virtual Allocator* GetCUDAHostAllocator(int numa_node);

  // Registers a function to be called once on every new Region
  // allocated by every GPURegionAllocator proximate to the specified
  // bus.  The AllocVisitor is provided with a memory pointer and the
  // size of the area it identifies.  The pointer is not guaranteed to
  // be valid after the call terminates.  The intention is for this
  // interface to be used for network device memory registration.
  // "bus_id" is platform-specific.  On many platforms it
  // should be 0.  On machines with multiple PCIe buses, it should be
  // the index of one of the PCIe buses.  If the bus_id is invalid,
  // results are undefined.
  typedef std::function<void(void*, size_t)> AllocVisitor;
  virtual void AddGPUAllocVisitor(int bus_id, AllocVisitor visitor);

  typedef std::unordered_map<const void*, MemDesc> MDMap;

 protected:
  ProcessState();

  static ProcessState* instance_;
  bool gpu_device_enabled_;

  mutex mu_;

  std::vector<Allocator*> cpu_allocators_ GUARDED_BY(mu_);
  std::vector<VisitableAllocator*> gpu_allocators_ GUARDED_BY(mu_);
  std::vector<std::vector<AllocVisitor>> gpu_visitors_ GUARDED_BY(mu_);
  std::vector<Allocator*> cuda_host_allocators_ GUARDED_BY(mu_);

  virtual ~ProcessState();

  // Optional RecordingAllocators that wrap the corresponding
  // Allocators for runtime attribute use analysis.
  MDMap mem_desc_map_;
  std::vector<Allocator*> cpu_al_ GUARDED_BY(mu_);
  std::vector<Allocator*> gpu_al_ GUARDED_BY(mu_);
  std::vector<Allocator*> cuda_al_ GUARDED_BY(mu_);
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
  bool TracksAllocationSizes() override { return a_->TracksAllocationSizes(); }
  size_t RequestedSize(void* p) override { return a_->RequestedSize(p); }
  size_t AllocatedSize(void* p) override { return a_->AllocatedSize(p); }
  void GetStats(AllocatorStats* stats) override { return a_->GetStats(stats); }
  ProcessState::MDMap* mm_;  // not owned
  Allocator* a_;             // not owned
  ProcessState::MemDesc md_;
  mutex* mu_;
};
}  // namespace internal
}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_PROCESS_STATE_H_
