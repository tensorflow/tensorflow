/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_ALLOCATION_TRACKER_H_
#define XLA_SERVICE_ALLOCATION_TRACKER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/service/backend.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

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
  StatusOr<GlobalDataHandle> Register(ScopedShapedBuffer shaped_buffer,
                                      const std::string& tag);

  // Registers a vector of shaped buffers of device memory, one per replica, and
  // returns a corresponding handle that can be used for talking to XLA clients.
  StatusOr<GlobalDataHandle> RegisterReplicatedBuffers(
      std::vector<ScopedShapedBuffer> replicated_buffers,
      const std::string& tag);

  // Unregister the allocation for the given data handle.
  Status Unregister(const GlobalDataHandle& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<GlobalDataHandle>> DeconstructTuple(
      const GlobalDataHandle& Data);

  // Resolve a handle from an XLA client to a vector of shaped buffers, one per
  // replica, or provide an error status to say whether any of those buffers
  // were not found (or found, but found deallocated).
  StatusOr<std::vector<const ShapedBuffer*>> Resolve(
      const GlobalDataHandle& data) const;

  // Resolves a handle from an XLA client and replica id to a shaped buffer, or
  // provide an error status to say whether it was not found (or found, but
  // found deallocated).
  StatusOr<const ShapedBuffer*> ResolveForReplica(const GlobalDataHandle& data,
                                                  int replica_id) const;

 private:
  // Data structure encapsulating single memory allocation on the device.
  struct Allocation {
    // The pointer to this allocation.
    se::OwningDeviceMemory device_memory;

    // This is the number of times this memory allocation is referred to by
    // registered data handles.
    int ref_count;
  };

  // Internal helper which resolves the given GlobalDataHandle to a
  // list of ScopedShapedBuffers.
  StatusOr<std::vector<const ShapedBuffer*>> ResolveInternal(
      const GlobalDataHandle& data) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Internal helper which registers a vector of shaped buffers, one per
  // replica.  ShapedBufferTy is either ScopedShapedBuffer or ShapedBuffer.  If
  // it's ShapedBuffer, all of the given buffers must already be tracked by this
  // object -- presumably this is a call from DeconstructTuple.
  template <typename ShapedBufferTy>
  StatusOr<GlobalDataHandle> RegisterInternal(
      std::vector<ShapedBufferTy> replicated_buffers, const std::string& tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Adds the given device address to the allocation tracker, or if it already
  // exists, then increment its reference count.
  void AddAllocationOrIncrementRefCount(se::DeviceMemoryBase device_memory,
                                        int device_ordinal)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Decrements the reference count of the given device memory. Then, if it is
  // zero, deallocate the memory.
  Status DecrementRefCount(se::DeviceMemoryBase device_memory,
                           int device_ordinal)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // A map from device memory opaque value to allocation. One such map is
  // maintained per device ordinal.
  using AllocationMap = absl::flat_hash_map<const void*, Allocation>;

  mutable absl::Mutex mutex_;

  // Backend to use with this tracker. The backend supplies the memory allocator
  // to use when deallocating memory.
  Backend* backend_;

  // The next handle to assign to an allocation, guarded by the same mutex as
  // the mapping as they'll be mutated at the same time.
  int64_t next_handle_ ABSL_GUARDED_BY(mutex_);

  // A map from device ordinal to AllocationMap.
  absl::flat_hash_map<int, AllocationMap> opaque_to_allocation_map_
      ABSL_GUARDED_BY(mutex_);

  // A map from data handle to a vector of shaped buffers that represent the
  // buffers for different replicas.
  //
  // The ShapedBuffers in this map's vectors need to be unique_ptrs, because our
  // public API returns pointers to them.  We expect the concrete class to be
  // ShapedBuffer and never ScopedShapedBuffer; deallocation of buffers is
  // handled by opaque_to_allocation_map_.
  //
  // The elements of the vectors need to be unique_ptrs because we return
  // pointers to them.  (In theory we could use std::list or something instead,
  // but we also want to be able to null out these elements.)
  //
  // The reason that the elements can't be unique_ptr<ScopedShapedBuffer>s is
  // the existence of DeconstructTuple().  This function allows us to create a
  // non-owning "view" into a tuple's sub-buffers.  The sub-buffers are then
  // free'd when both the view *and* the original tuple are Unregistered.  This
  // refcounting is managed in opaque_to_allocation_map_.
  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<ShapedBuffer>>>
      handle_to_shaped_buffers_ ABSL_GUARDED_BY(mutex_);

  AllocationTracker(const AllocationTracker&) = delete;
  AllocationTracker& operator=(const AllocationTracker&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_ALLOCATION_TRACKER_H_
