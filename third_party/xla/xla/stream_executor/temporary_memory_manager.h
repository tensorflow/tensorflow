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

// The temporary-memory-manager is a helper class for a Stream to keep track of
// temporary allocations. These allocations defer their deallocation to the next
// Stream::BlockHostUntilDone call for efficiency purposes (as deallocation
// itself generally forces synchronization to occur).

#ifndef XLA_STREAM_EXECUTOR_TEMPORARY_MEMORY_MANAGER_H_
#define XLA_STREAM_EXECUTOR_TEMPORARY_MEMORY_MANAGER_H_

#include <map>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/temporary_device_memory.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace internal {

// Record used inside the TemporaryMemoryManager as metadata for a given device
// memory region.
struct TemporaryMemoryRecord {
  // What "generation" this record was allocated in.
  //
  // Currently the generation counter is bumped for every allocation, but this
  // could be made coarser if necessary.
  uint64_t allocation_generation;

  // Notes whether the temporary memory has been marked as finalized, such that
  // we can release the DeviceMemory associated with this record at
  // synchronization time.
  bool finalized;
};

// Manages temporary memories associated with a stream -- keeps records of
// outstanding temporaries and their state, and can deallocate them
// appropriately at points in the Stream lifecycle (e.g. BlockHostUntilDone,
// destruction).
class TemporaryMemoryManager {
 public:
  explicit TemporaryMemoryManager(Stream* stream) : stream_(stream) {}

  // Allocates a temporary array that is then managed by this object.
  template <typename T>
  tsl::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>> AllocateArray(
      uint64_t element_count);

  // Forces deallocation of all managed temporary memory regions.
  //
  // Called, for example, when the Stream owning this temporary memory manager
  // is destroyed.
  //
  // Note: These calls to Deallocate will likely force synchronization.
  void ForceDeallocateAll();

  // Marks the given memory region as finalized.
  //
  // If must_exist is set, this will check-fail if the temporary memory record
  // is not found.
  void MarkFinalized(const DeviceMemoryBase& device_memory, uint64_t generation,
                     bool must_exist);

  // Deallocates temporary memories that have been finalized.
  //
  // Note: These calls to Deallocate will likely force synchronization, so it is
  // meant to be called before a "BlockHostUntilDone" is about to be performed.
  void DeallocateFinalizedTemporaries();

  // Returns whether the provided device_memory is finalized.
  //
  // In the vacuous case where the device memory doesn't appear in the temporary
  // memory records, it is either not a temporary at all, or has already been
  // deallocated, and thus returns true.
  bool IsFinalized(const DeviceMemoryBase& device_memory,
                   uint64_t allocation_generation) const;

  // Returns whether the manager has a live allocation record for the given
  // device memory pointer with the given generation counter.
  //
  // Note: this is a polling call -- there is no guarantee that the region is
  // still allocated once the call has completed.
  bool HasAllocated(const DeviceMemoryBase& device_memory,
                    uint64_t generation) const;

 private:
  // Allocates an array without type parameterization, so that the
  // implementation can live in the source file. Without this base allocation
  // method, we incur a circular dependency between the StreamExecutor
  // definition and this class' definition.
  tsl::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>> AllocateArrayBase(
      uint64_t element_count, uint64 element_size);

  // Mutex to guard temporary record state.
  mutable absl::Mutex mutex_;

  // Mapping from device memory to the current (live) temporary memory record.
  //
  // If a device memory is not in this mapping, it is not a temporary currently
  // allocated and owned by this temporary memory manager.
  std::map<DeviceMemoryBase, TemporaryMemoryRecord> records_
      ABSL_GUARDED_BY(mutex_);

  // Allocation generation -- we bump this counter to distinguish temporary
  // memory handles that have been deallocated and later reallocated at the same
  // device memory address.
  uint64_t generation_ ABSL_GUARDED_BY(mutex_);

  // The stream (parent object) for this temporary memory manager -- allocations
  // are performed through this stream handle.
  Stream* stream_;

  TemporaryMemoryManager(const TemporaryMemoryManager&) = delete;
  void operator=(const TemporaryMemoryManager&) = delete;
};

////////////
// Inlines

template <typename T>
tsl::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
TemporaryMemoryManager::AllocateArray(uint64_t element_count) {
  tsl::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>> temporary_memory =
      AllocateArrayBase(element_count, sizeof(T));
  if (!temporary_memory.ok()) {
    return temporary_memory.status();
  }

  return std::unique_ptr<TemporaryDeviceMemory<T>>(
      reinterpret_cast<TemporaryDeviceMemory<T>*>(temporary_memory->release()));
}

}  // namespace internal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TEMPORARY_MEMORY_MANAGER_H_
