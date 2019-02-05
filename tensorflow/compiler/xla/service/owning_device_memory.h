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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_OWNING_DEVICE_MEMORY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_OWNING_DEVICE_MEMORY_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Break circular dependency between this file and device_memory_allocator.h.
class DeviceMemoryAllocator;

// Owning pointer for memory on a device.
//
// OwningDeviceMemory is an owning pointer like std::unique_ptr, but it can
// point to memory that resides on a "device" (e.g. a GPU).  When an
// OwningDeviceMemory goes out of scope, it frees the memory it owns.
//
// We say that an instance of OwningDeviceMemory is "active" if it currently
// owns a (possibly empty) slice of memory on the device.  Moving, Forget()'ing,
// Free()'ing, and other actions can deactive an active object.
//
// Note that we can't simply use stream_executor::ScopedDeviceMemory instead of
// OwningDeviceMemory, because ScopedDeviceMemory frees its pointer via a
// StreamExecutor.  This class needs to free via a xla::DeviceMemoryAllocator.
class OwningDeviceMemory {
 public:
  OwningDeviceMemory() : device_ordinal_(-1), allocator_(nullptr) {}

  explicit OwningDeviceMemory(se::DeviceMemoryBase mem, int device_ordinal,
                              DeviceMemoryAllocator* allocator)
      : mem_(mem), device_ordinal_(device_ordinal), allocator_(allocator) {
    CHECK(allocator != nullptr) << "allocator cannot be null.";
  }

  OwningDeviceMemory(OwningDeviceMemory&& other)
      : mem_(other.mem_),
        device_ordinal_(other.device_ordinal_),
        allocator_(other.allocator_) {
    other.mem_ = se::DeviceMemoryBase();
    other.allocator_ = nullptr;
  }

  OwningDeviceMemory& operator=(OwningDeviceMemory&& other) {
    if (allocator_ != nullptr) {
      Free();
    }
    mem_ = other.mem_;
    device_ordinal_ = other.device_ordinal_;
    allocator_ = other.allocator_;

    other.mem_ = se::DeviceMemoryBase();
    other.allocator_ = nullptr;
    return *this;
  }

  // Deactivates this instance if it's active.  Nop if it's not active.
  OwningDeviceMemory& operator=(std::nullptr_t) {
    if (allocator_ != nullptr) {
      Free();
    }
    return *this;
  }

  ~OwningDeviceMemory() {
    if (allocator_ != nullptr) {
      Free();
    }
  }

  // The returned allocator is nonnull iff this object is active.
  DeviceMemoryAllocator* allocator() const { return allocator_; }

  int device_ordinal() const { return device_ordinal_; }

  // Gets the device memory pointer.
  const void* opaque() const { return mem_.opaque(); }
  void* opaque() { return mem_.opaque(); }

  uint64 size() const { return mem_.size(); }

  // Determines whether this wraps a null pointer.
  //
  // !is_null() is sufficient but not necessary to imply `this` is active.
  bool is_null() const { return mem_.is_null(); }

  se::DeviceMemoryBase AsDeviceMemoryBase() {
    return se::DeviceMemoryBase(opaque(), size(), /*is_sub_buffer=*/false);
  }

  // Returns the wrapped DeviceMemoryBase without freeing it, and deactivates
  // this object.  Precondition: `this` is active.
  TF_MUST_USE_RESULT se::DeviceMemoryBase Forget() {
    CHECK(allocator_ != nullptr)
        << "Can't call Forget() on an inactive (i.e. moved from, Forget()'ten, "
           "or Free()'ed) instance.";
    allocator_ = nullptr;
    se::DeviceMemoryBase mem(mem_);
    mem_ = se::DeviceMemoryBase();
    return mem;
  }

  // Frees the wrapped DeviceMemoryBase and deactivates this object.
  // Precondition: `this` is active.
  void Free();

 private:
  se::DeviceMemoryBase mem_;
  int device_ordinal_;
  DeviceMemoryAllocator* allocator_;  // Null if this object is inactive.
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OWNING_DEVICE_MEMORY_H_
