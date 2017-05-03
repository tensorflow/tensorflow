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
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A global allocation in device space, tracked by the XLA service.
class Allocation {
 public:
  Allocation(Backend* backend, int device_ordinal,
             perftools::gputools::DeviceMemoryBase device_memory,
             const Shape& shape, const string& tag, int initial_ref_count)
      : backend_(backend),
        device_ordinal_(device_ordinal),
        device_memory_(device_memory),
        shape_(shape),
        tag_(tag),
        ref_count_(initial_ref_count) {}

  Backend* backend() const { return backend_; }
  int device_ordinal() const { return device_ordinal_; }
  perftools::gputools::DeviceMemoryBase device_memory() const {
    return device_memory_;
  }
  const Shape& shape() const { return shape_; }
  const string& tag() const { return tag_; }

  bool is_deallocated() const {
    CHECK_GE(ref_count_, 0);
    return ref_count_ == 0;
  }
  int ref_count() const {
    CHECK_GE(ref_count_, 0);
    return ref_count_;
  }
  void increment_ref_count(int inc) {
    CHECK_GT(ref_count_, 0);
    CHECK_LE(ref_count_, INT_MAX - inc);
    ref_count_ += inc;
  }
  void decrement_ref_count() {
    CHECK_GT(ref_count_, 0);
    --ref_count_;
  }
  perftools::gputools::DeviceMemoryBase* mutable_device_memory() {
    return &device_memory_;
  }

 private:
  // The backend that the memory is allocated on.
  Backend* backend_;

  // The device that the memory is allocated on.
  int device_ordinal_;

  // The pointer to this allocation.
  perftools::gputools::DeviceMemoryBase device_memory_;

  // The shape of this allocation.
  Shape shape_;

  // An informal description of this allocation shown in tools.
  string tag_;

  // This is the number of Allocation objects which refer to this memory
  // allocation.
  int ref_count_;

  // Return a string representation of this allocation for debugging or logging
  // purposes.
  string ToString() const;
};

// Tracks allocations for the XLA service; allocations can be registered
// with shape/device/tag and resolved from a handle for later use.
class AllocationTracker {
 public:
  AllocationTracker();

  // Registers device memory with a given shape, device identifier, and tag, and
  // returns a corresponding handle that can be used for talking to XLA
  // clients.
  GlobalDataHandle Register(Backend* backend, int device_ordinal,
                            perftools::gputools::DeviceMemoryBase device_memory,
                            const Shape& shape, const string& tag);

  // Unregister the allocation for the given data handle.
  tensorflow::Status Unregister(const GlobalDataHandle& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<GlobalDataHandle>> DeconstructTuple(
      const GlobalDataHandle& Data);

  // Resolve a handle from an XLA client to an allocation, or provide an
  // error status to say whether it was not found (or found, but found
  // deallocated).
  StatusOr<const Allocation*> Resolve(const GlobalDataHandle& data);

 private:
  // Internal helper which resolves the given GlobalDataHandle to an Allocation.
  StatusOr<Allocation*> ResolveInternal(const GlobalDataHandle& data)
      EXCLUSIVE_LOCKS_REQUIRED(allocation_mutex_);

  GlobalDataHandle RegisterInternal(
      Backend* backend, int device_ordinal,
      perftools::gputools::DeviceMemoryBase device_memory, const Shape& shape,
      const string& tag, int initial_ref_count)
      EXCLUSIVE_LOCKS_REQUIRED(allocation_mutex_);

  // Helper function which deallocates the memory buffer containing the given
  // shape referred to by device_memory. Tuples are traversed recursively
  // deallocating all nested buffers. The parameter deallocated_buffers contains
  // the set of buffers deallocated so far stored as opaque values (void *) from
  // DeviceMemoryBase. Keeping track of deallocated buffers prevents
  // double-freeing of buffers which may be referred to more than once in a
  // nested tuple.
  tensorflow::Status DeallocateShape(
      Backend* backend, int device_ordinal,
      perftools::gputools::DeviceMemoryBase* device_memory, const Shape& shape,
      std::set<void*>* deallocated_buffers)
      EXCLUSIVE_LOCKS_REQUIRED(allocation_mutex_);

  // Returns the opaque_to_handle_ map for the given device_ordinal, creating
  // a new map if there is not one for the device_ordinal.
  using HandleMap = std::map<void*, int64>;
  HandleMap& GetOrCreateOpaqueToHandleMap(int device_ordinal)
      EXCLUSIVE_LOCKS_REQUIRED(allocation_mutex_);

  tensorflow::mutex allocation_mutex_;  // Guards the allocation mapping.

  // The next handle to assign to an allocation, guarded by the same mutex as
  // the mapping as they'll be mutated at the same time.
  int64 next_handle_ GUARDED_BY(allocation_mutex_);

  // A map from DeviceMemoryBase to handle for each device_ordinal.
  std::vector<HandleMap> opaque_to_handle_ GUARDED_BY(allocation_mutex_);

  // Mapping from GlobalDataHandle handle to the corresponding registered
  // Allocation object.
  std::map<int64, std::unique_ptr<Allocation>> handle_to_allocation_
      GUARDED_BY(allocation_mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(AllocationTracker);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_
