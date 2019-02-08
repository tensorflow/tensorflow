/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PROCESS_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PROCESS_STATE_H_

#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Allocator;
class PoolAllocator;
class SharedCounter;

// Singleton that manages per-process state when GPUs are present.
class GPUProcessState {
 public:
  // If ps == nullptr, returns pointer to the single instance of this class to
  // be used within this process.
  //
  // If ps != nullptrs, accepts a value to be returned by all subsequent calls.
  // A non-null ps may ONLY be provided during program static storage
  // initialization.  Must not be called more than once with a non-null ps.
  //
  // If a derived class of GPUProcessState is ever used in a process, it must
  // always be used in place of this class.  In order to ensure that existing
  // calls to GPUProcessState::singleton() all resolve to the derived instance
  // instead, this function must be called once during startup, supplying the
  // derived instance value, prior to any accessor call to this function.
  static GPUProcessState* singleton(GPUProcessState* ps = nullptr);

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

  // Returns the one GPU allocator used for the indexed GPU.
  // Note that this is a system GPU index, not (necessarily) a brain
  // device index.
  //
  // 'total_bytes' is the total number of bytes that should be made
  // available to the allocator.  The first call to this function for
  // a given tf_gpu_id creates the allocator, so only the total_bytes
  // used on that first call is used.
  //
  // "Allocator type" describes the type of algorithm to use for the
  // underlying allocator.  REQUIRES: Must be a valid type (see
  // config.proto for the list of supported strings.).
  //
  // REQUIRES: tf_gpu_id must be a valid id for a BaseGPUDevice available in the
  // current system environment.  Otherwise returns nullptr.
  virtual Allocator* GetGPUAllocator(const GPUOptions& options,
                                     TfGpuId tf_gpu_id, size_t total_bytes);

  virtual Allocator* GetCUDAHostAllocator(int numa_node);

  // Registers a Visitor to be invoked on new chunks of memory allocated by the
  // SubAllocator of every GPU proximate to the specified bus.  The AllocVisitor
  // is provided with a memory pointer, a GPU id, and the size of the area it
  // identifies.  The pointer is not guaranteed to be valid after the call
  // terminates.  The intention is for this interface to be used for network
  // device memory registration.  "bus_id" is platform-specific.  On many
  // platforms it should be 0.  On machines with multiple PCIe buses, it should
  // be the index of one of the PCIe buses (maybe the NUMA node at which the
  // PCIe is rooted).  If the bus_id is invalid, results are undefined.
  virtual void AddGPUAllocVisitor(int bus_id,
                                  const SubAllocator::Visitor& visitor);

  // Registers a Visitor to be invoked on new chunks of memory allocated by
  // the SubAllocator of the CUDAHostAllocator for the given numa_node.
  virtual void AddCUDAHostAllocVisitor(int numa_node,
                                       const SubAllocator::Visitor& visitor);

  // Registers a Visitor to be invoked on each chunk handed back for freeing to
  // the SubAllocator of the CUDAHostAllocator for the given numa_node.
  virtual void AddCUDAHostFreeVisitor(int numa_node,
                                      const SubAllocator::Visitor& visitor);

  // Returns bus_id for the given GPU id.
  virtual int BusIdForGPU(TfGpuId tf_gpu_id);

  std::unique_ptr<SharedCounter> ReleaseGPUAllocatorCounter(TfGpuId tf_gpu_id);

 protected:
  // GPUProcessState is a singleton that should not normally be deleted except
  // at process shutdown.
  GPUProcessState();
  virtual ~GPUProcessState() {}
  friend class GPUDeviceTest;

  // Helper method for unit tests to reset the ProcessState singleton by
  // cleaning up everything. Never use in production.
  virtual void TestOnlyReset();

  ProcessState::MDMap* mem_desc_map() {
    if (process_state_) return &process_state_->mem_desc_map_;
    return nullptr;
  }

  static GPUProcessState* instance_;
  ProcessState* process_state_;  // Not owned.
  bool gpu_device_enabled_;

  mutex mu_;

  struct AllocatorParts {
    std::unique_ptr<Allocator> allocator;
    std::unique_ptr<SharedCounter> counter;
    SubAllocator* sub_allocator;  // owned by allocator
    std::unique_ptr<Allocator> recording_allocator;
  };
  std::vector<AllocatorParts> gpu_allocators_ GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> gpu_visitors_ GUARDED_BY(mu_);

  std::vector<AllocatorParts> cuda_host_allocators_ GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> cuda_host_alloc_visitors_
      GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> cuda_host_free_visitors_
      GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PROCESS_STATE_H_
