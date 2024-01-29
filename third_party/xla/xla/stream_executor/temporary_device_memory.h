/* Copyright 2015 The OpenXLA Authors.

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
//        stream.AllocateTemporaryArray<float>(1024).value();
//    // ... enqueue stuff onto the stream using the temporary memory ...
//    // Note that the memory is accessible via
//    // temporary_memory->device_memory() and similar.
//
// Note that standard usage takes advantage of the type-safe wrapper,
// TemporaryDeviceMemory<T>, defined below.
//
// Also see tests for executable sample usage.

#ifndef XLA_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
#define XLA_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_

#include "xla/stream_executor/device_memory.h"

namespace stream_executor {

class Stream;

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

  // Note: construction DCHECKs that the memory is known-allocated in the
  // stream's temporary-allocation-manager.
  TemporaryDeviceMemoryBase(Stream* parent, DeviceMemoryBase device_memory);

 private:
  // The device memory region that has allocated.
  DeviceMemoryBase device_memory_;

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

#endif  // XLA_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
