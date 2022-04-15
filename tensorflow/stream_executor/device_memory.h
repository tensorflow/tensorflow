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

// Suite of types that represent device memory allocations. These are
// allocated by the StreamExecutor interface, which produces values appropriate
// for the underlying platform (whether it be CUDA or OpenCL).
//
// The untyped base class (like a device void*) is DeviceMemoryBase, which can
// be specialized for a given allocation type (like a device T*) using
// DeviceMemory<T>.

#ifndef TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_
#define TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_

#include <stddef.h>

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(b/77980417): Remove this once we've
// completed the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]

}  // namespace gputools
}  // namespace perftools

namespace stream_executor {

class DeviceMemoryAllocator;
class StreamExecutor;

// void*-analogous device memory allocation. For the typed variation, see
// DeviceMemory<T>.
//
// This is effectively a two-tuple of a pointer and size; however, note that the
// pointer may not be to the virtual address itself -- in OpenCL the pointer is
// to a cl_mem handle that describes the device allocation. Therefore,
// DeviceMemoryBase::opaque does not necessarily produce a pointer that can be
// referenced directly, so use it with caution.
//
// Thread-compatible.
class DeviceMemoryBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized device memory
  // region. An opaque pointer may be provided -- see header for details on the
  // opacity of that pointer.
  explicit DeviceMemoryBase(void *opaque = nullptr, uint64_t size = 0)
      : opaque_(opaque), size_(size) {}

  // Returns whether the backing memory is the null pointer.
  // A `== nullptr` convenience method is also provided.
  bool is_null() const { return opaque_ == nullptr; }
  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  // Provides a partial order between device memory values.
  //
  // This operator is provided so that this object can be used as a key in an
  // ordered map.
  bool operator<(const DeviceMemoryBase &other) const {
    return opaque() < other.opaque();
  }

  // Returns the size, in bytes, for the backing memory.
  uint64_t size() const { return size_; }

  // Warning: note that the pointer returned is not necessarily directly to
  // device virtual address space, but is platform-dependent.
  void *opaque() { return opaque_; }
  const void *opaque() const { return opaque_; }

  // Returns the payload of this memory region.
  uint64_t payload() const { return payload_; }

  // Sets payload to given value.
  void SetPayload(uint64_t payload) { payload_ = payload; }

  // Returns whether the two DeviceMemoryBase segments are identical (both in
  // their opaque pointer and size).
  bool IsSameAs(const DeviceMemoryBase &other) const {
    return opaque() == other.opaque() && size() == other.size();
  }

 protected:
  friend class StreamExecutor;

  // Resets the internal values of the opaque pointer and number of bytes in the
  // memory region, just as in the constructor.
  void Reset(void *opaque, uint64_t bytes) {
    opaque_ = opaque;
    size_ = bytes;
  }

 private:
  void *opaque_;  // Platform-dependent value representing allocated memory.
  uint64_t size_;         // Size in bytes of this allocation.
  uint64_t payload_ = 0;  // Payload data associated with this allocation.
};

// Typed wrapper around "void *"-like DeviceMemoryBase.
//
// For example, DeviceMemory<int> is a simple wrapper around DeviceMemoryBase
// that represents one or more integers in Device memory.
//
// Thread-compatible.
template <typename ElemT>
class DeviceMemory final : public DeviceMemoryBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized memory region.
  DeviceMemory() : DeviceMemoryBase(nullptr, 0) {}
  explicit DeviceMemory(std::nullptr_t) : DeviceMemory() {}

  // Typed device memory regions may be constructed from untyped device memory
  // regions, this effectively amounts to a cast from a void*.
  explicit DeviceMemory(const DeviceMemoryBase &other)
      : DeviceMemoryBase(const_cast<DeviceMemoryBase &>(other).opaque(),
                         other.size()) {
    SetPayload(other.payload());
  }

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64_t ElementCount() const { return size() / sizeof(ElemT); }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const { return ElementCount() == 1; }

  // Create a typed area of DeviceMemory with a given opaque pointer and the
  // quantity of bytes in the allocation. This function is broken out to
  // distinguish bytes from an element count.
  static DeviceMemory<ElemT> MakeFromByteSize(void *opaque, uint64_t bytes) {
    return DeviceMemory<ElemT>(opaque, bytes);
  }

  // Resets the DeviceMemory data, in MakeFromByteSize fashion.
  // This simply clobbers the prior values.
  void ResetFromByteSize(void *opaque, uint64_t bytes) {
    // TODO(leary) when NVCC is eliminated we can add this check (and the
    // logging include it requires).
    // CHECK_EQ(0, bytes % sizeof(ElemT));
    DeviceMemoryBase::Reset(opaque, bytes);
  }

  // ------------------------------------------------------------

 protected:
  // This constructor is solely used from derived classes; it is made protected
  // because it accepts a byte-size instead of an element count, which could
  // potentially be misused given the ElementCount() nature of this interface.
  //
  // In order to specify the desire to use byte size instead of element count
  // explicitly, use MakeFromByteSize.
  DeviceMemory(void *opaque, uint64_t size) : DeviceMemoryBase(opaque, size) {}
};

// A class to encapsulate the type and size of a dynamic shared memory
// buffer. Because the buffer exists solely on the device and is not copyable
// to the host, memory objects of this type do not maintain buffer pointers
// on the host.
template <typename ElemT>
class SharedDeviceMemory final : public DeviceMemoryBase {
 public:
  explicit SharedDeviceMemory(uint64_t elem_count)
      : DeviceMemoryBase(nullptr, elem_count * kElemSize) {}

  static constexpr size_t kElemSize = sizeof(ElemT);

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64_t ElementCount() const { return size() / kElemSize; }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const { return ElementCount() == 1; }
};

// Host-side representation of packed-and-aligned vector datatypes on the device
// side. Since these can appear in device kernel signatures, we support
// launching them with these datatypes in launch signatures.

struct Float2 {
  float x, y;
};

struct Float4 {
  Float2 xz, yw;
};

struct Double2 {
  double x, y;
};

static_assert(sizeof(Float2) == 2 * sizeof(float), "Float2 must be packed");
static_assert(sizeof(Float4) == 4 * sizeof(float), "Float4 must be packed");
static_assert(sizeof(Double2) == 2 * sizeof(double), "Double2 must be packed");

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_
