/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_PROCESS_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_PROCESS_STATE_H_

#include <functional>
#include <map>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Allocator;
class PluggableDeviceBFCAllocator;
class PluggableDeviceSimpleAllocator;
class PoolAllocator;

// Singleton that manages per-process state when PluggableDevices are present.
class PluggableDeviceProcessState {
 public:
  // Singleton that manages each platform's per-process state. e.g. allocation
  // of shared resource.
  static PluggableDeviceProcessState* singleton(const string& device_type,
                                                const string& platform_name);

  // Query whether any PluggableDevice has been created so far.
  // Disable thread safety analysis since a race is benign here.
  bool HasPluggableDevice() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return pluggable_device_enabled_;
  }

  // Set the flag to indicate a PluggableDevice has been created.
  // Disable thread safety analysis since a race is benign here.
  void EnablePluggableDevice() TF_NO_THREAD_SAFETY_ANALYSIS {
    pluggable_device_enabled_ = true;
  }

  // Returns the one PluggableDevice allocator used for the indexed
  // PluggableDevice. Note that this is a system PluggableDevice index.
  //
  // 'total_bytes' is the total number of bytes that should be made
  // available to the allocator.  The first call to this function for
  // a given tf_device_id creates the allocator, so only the
  // total_bytes used on that first call is used.
  //
  // 'allocator_type' describes the type of algorithm to use for the
  // underlying allocator.  REQUIRES: Must be a valid type (see
  // config.proto for the list of supported strings.).
  //
  // REQUIRES: tf_device_id must be a valid id for a PluggableDevice
  // available in the current system environment. Otherwise returns nullptr.
  virtual Allocator* GetPluggableDeviceAllocator(const GPUOptions& options,
                                                 TfDeviceId tf_device_id,
                                                 size_t total_bytes);

  int NumPluggableDeviceAllocators() {
    mutex_lock l(mu_);
    return pluggable_device_allocators_.size();
  }

  virtual Allocator* GetPluggableDeviceHostAllocator(int numa_node);

  // Returns bus_id for the given PluggableDevice id.
  virtual int BusIdForPluggableDevice(TfDeviceId tf_device_id);

 protected:
  // PluggableDeviceProcessState is a singleton that should not normally be
  // deleted except at process shutdown.
  PluggableDeviceProcessState(const string& device_type,
                              const string& platform_name);
  virtual ~PluggableDeviceProcessState() {}

  ProcessState::MDMap* mem_desc_map() {
    if (process_state_) return &process_state_->mem_desc_map_;
    return nullptr;
  }

  static PluggableDeviceProcessState* instance_;
  ProcessState* process_state_;  // Not owned.
  bool pluggable_device_enabled_;
  const string device_type_;
  const string platform_name_;
  mutex mu_;

  struct AllocatorParts {
    std::unique_ptr<Allocator> allocator;
    Allocator* device_allocator;
    SubAllocator* sub_allocator;  // owned by allocator
  };

  std::vector<AllocatorParts> pluggable_device_allocators_ TF_GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> pluggable_device_visitors_
      TF_GUARDED_BY(mu_);

  std::vector<AllocatorParts> pluggable_device_host_allocators_
      TF_GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>>
      pluggable_device_host_alloc_visitors_ TF_GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>>
      pluggable_device_host_free_visitors_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_PROCESS_STATE_H_
