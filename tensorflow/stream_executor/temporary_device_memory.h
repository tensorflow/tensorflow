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

// Temporary memories are used to allocate scratch space required by an
// operation about to be enqueued onto a stream.
//
//    std::unique_ptr<TemporaryDeviceMemory<float>> temporary_memory =
//        stream.AllocateTemporaryArray<float>(1024).ConsumeValueOrDie();
//    // ... enqueue stuff onto the stream using the temporary memory ...
//    // Note that the memory is accessible via
//    // temporary_memory->device_memory() and similar.
//
//    // Finalize the temporary memory. The underlying device memory may
//    // be released any time after this program point, as another thread may
//    // call Stream::BlockHostUntilDone, causing synchronization. This
//    // finalization also happens automatically for the user if the unique_ptr
//    // goes out of scope.
//    temporary_memory.Finalize();
//
// WARNING: do NOT hold onto the device memory associated with temporary_memory
// after finalization. If temporary_memory->device_memory() is used after the
// temporary memory is finalized, it will cause a DCHECK failure.
//
// Note that standard usage takes advantage of the type-safe wrapper,
// TemporaryDeviceMemory<T>, defined below.
//
// Also see tests for executable sample usage.

#ifndef TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
#define TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_

#include "tensorflow/stream_executor/device_memory.h"

namespace stream_executor {

class Stream;
namespace internal {
class TemporaryMemoryManager;
}

// Untyped base class (analogous to a void*) for temporary device memory
// allocations associated with a stream.
class TemporaryDeviceMemoryBase {
 public:
  // Marks the temporary memory as finalized if it is not already marked as
  // such.
  ~TemporaryDeviceMemoryBase();

  // Precondition: !IsFinalized()
  DeviceMemoryBase* mutable_device_memory();

  // Precondition: !IsFinalized()
  const DeviceMemoryBase& device_memory() const;

  // "Finalizes" this temporary memory, making it acceptable to release at the
  // next stream synchronization point -- the device memory can be reclaimed at
  // any time after the temporary memory is marked as finalized (e.g. if a
  // separate thread is calls Stream::BlockHostUntilDone). This may only be
  // called once -- see the precondition below.
  //
  // Precondition: !IsFinalized()
  void Finalize();

  // Returns true iff the temporary memory is finalized (that is, the user is
  // done referring to the temporary device memory, and thus it can be released
  // at the next stream synchronization point).
  bool IsFinalized() const;

  // Returns true iff the temporary memory is still allocated.
  //
  // Note: this is a polling call, no guarantee is made that the temporary
  // memory is still allocated after the call has completed.
  bool IsAllocated() const;

 private:
  friend class internal::TemporaryMemoryManager;
  friend class TemporaryDeviceMemoryTest;

  // Note: construction DCHECKs that the memory is known-allocated in the
  // stream's temporary-allocation-manager.
  TemporaryDeviceMemoryBase(Stream* parent, DeviceMemoryBase device_memory,
                            uint64 allocation_generation);

  // The device memory region that has allocated.
  DeviceMemoryBase device_memory_;

  // The generation counter value for the temporary memory record in the
  // temporary memory manager.
  uint64 allocation_generation_;

  // The stream that this temporary memory was allocated for.
  Stream* parent_;
};

// Type-safe wrapper around the base type (which is analogous to a void*).
template <typename T>
class TemporaryDeviceMemory : public TemporaryDeviceMemoryBase {
 public:
  // Type-safe wrapper around TemporaryDeviceMemoryBase::mutable_device_memory.
  DeviceMemory<T>* mutable_device_memory() {
    StaticSlicingAssertionDummy();
    return reinterpret_cast<DeviceMemory<T>*>(
        TemporaryDeviceMemoryBase::mutable_device_memory());
  }

  // Type-safe wrapper around TemporaryDeviceMemoryBase::device_memory.
  const DeviceMemory<T>& device_memory() const {
    StaticSlicingAssertionDummy();
    return reinterpret_cast<const DeviceMemory<T>&>(
        TemporaryDeviceMemoryBase::device_memory());
  }

 private:
  static void StaticSlicingAssertionDummy() {
    static_assert(
        sizeof(TemporaryDeviceMemory) == sizeof(TemporaryDeviceMemoryBase),
        "derived class is simply a wrapper, no members may be added due to "
        "slicing");
  }
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
