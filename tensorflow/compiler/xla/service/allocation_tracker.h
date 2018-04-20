/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Tracks allocations for the XLA service; allocations can be registered
// with shape/device/tag and resolved from a handle for later use.
class AllocationTracker {
 public:
  // The allocator is used for deallocating memory when allocations are
  // deregistered. All registered allocations must have the same platform as the
  // allocator.
  AllocationTracker(Backend* backend) : backend_(backend), next_handle_(1) {}

  // Registers a shaped buffer of device memory, and returns a corresponding
  // handle that can be used for talking to XLA clients. The given shaped buffer
  // will be treated as the buffer corresponding to the only replica.
  StatusOr<GlobalDataHandle> Register(
      std::unique_ptr<ShapedBuffer> shaped_buffer, const string& tag);

  // Registers a vector of shaped buffers of device memory, one per replica, and
  // returns a corresponding handle that can be used for talking to XLA clients.
  StatusOr<GlobalDataHandle> RegisterReplicatedBuffers(
      std::vector<std::unique_ptr<ShapedBuffer>> replicated_buffers,
      const string& tag);

  // Unregister the allocation for the given data handle.
  Status Unregister(const GlobalDataHandle& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<GlobalDataHandle>> DeconstructTuple(
      const GlobalDataHandle& Data);

  // Resolve a handle from an XLA client to a vector of shaped buffers, one per
  // replica, or provide an error status to say whether any of those buffers
  // were not found (or found, but found deallocated).
  StatusOr<std::vector<const ShapedBuffer*>> Resolve(
      const GlobalDataHandle& data);

  // Resolves a handle from an XLA client and replica id to a shaped buffer, or
  // provide an error status to say whether it was not found (or found, but
  // found deallocated).
  StatusOr<const ShapedBuffer*> ResolveForReplica(const GlobalDataHandle& data,
                                                  int replica_id);

 private:
  // Data structure encapsulating single memory allocation on the device.
  struct Allocation {
    // The pointer to this allocation.
    perftools::gputools::DeviceMemoryBase device_memory;

    // The device that the memory is allocated on.
    int device_ordinal;

    // This is the number of times this memory allocation is referred to by
    // registered data handles.
    int ref_count;
  };

  // Internal helper which resolves the given GlobalDataHandle to a
  // ShapedBuffer.
  StatusOr<std::vector<const ShapedBuffer*>> ResolveInternal(
      const GlobalDataHandle& data) EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Internal helper which registers a vector of shaped buffers, one per
  // replica.
  StatusOr<GlobalDataHandle> RegisterInternal(
      std::vector<std::unique_ptr<ShapedBuffer>> replicated_buffers,
      const string& tag) EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Resets the shaped buffers corresponding to the given handle.
  Status Reset(const GlobalDataHandle& data) EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Adds the given device address to the allocation tracker, or if it already
  // exists, then increment it's reference count.
  void AddAllocationOrIncrementRefCount(
      perftools::gputools::DeviceMemoryBase device_memory, int device_ordinal)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Decrements the reference count of the given device memory. Then, if it is
  // zero, deallocate the memory.
  Status DecrementRefCount(perftools::gputools::DeviceMemoryBase device_memory,
                           int device_ordinal) EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // A map from device memory opaque value to allocation. One such map is
  // maintained per device ordinal.
  using AllocationMap = tensorflow::gtl::FlatMap<const void*, Allocation>;

  tensorflow::mutex mutex_;

  // Backend to use with this tracker. The backend supplies the memory allocator
  // to use when deallocating memory.
  Backend* backend_;

  // The next handle to assign to an allocation, guarded by the same mutex as
  // the mapping as they'll be mutated at the same time.
  int64 next_handle_ GUARDED_BY(mutex_);

  // A map from device ordinal to AllocationMap.
  tensorflow::gtl::FlatMap<int, AllocationMap> opaque_to_allocation_map_
      GUARDED_BY(mutex_);

  // A map from data handle to a vector of shaped buffers that represent the
  // buffers for different replicas.
  tensorflow::gtl::FlatMap<int64, std::vector<std::unique_ptr<ShapedBuffer>>>
      handle_to_shaped_buffers_ GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(AllocationTracker);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_
