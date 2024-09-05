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

// Suite of types that represent device memory allocations. These are
// allocated by the StreamExecutor interface, which produces values appropriate
// for the underlying platform (whether it be CUDA or OpenCL).
//
// The untyped base class (like a device void*) is DeviceMemoryBase, which can
// be specialized for a given allocation type (like a device T*) using
// DeviceMemory<T>.

#ifndef XLA_STREAM_EXECUTOR_DEVICE_MEMORY_H_
#define XLA_STREAM_EXECUTOR_DEVICE_MEMORY_H_

#include <stddef.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "absl/base/attributes.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

class DeviceMemoryAllocator;

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
      : opaque_(opaque), size_(size) {
    // TODO(b/336267585): This constructor dangerously encourages
    //                 DeviceMemoryBase(mem) which would imply
    //                 DeviceMemoryBase(mem, 0)
    //                 We should delete & resolve any dependencies.
    //  explicit DeviceMemoryBase(void *opaque) = delete;
  }

  // Returns whether the backing memory is the null pointer.
  // A `== nullptr` convenience method is also provided.
  bool is_null() const { return opaque_ == nullptr; }

  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  bool operator==(const DeviceMemoryBase &other) const {
    return opaque_ == other.opaque_ && size_ == other.size_;
  }

  // Provides a partial order between device memory values.
  //
  // This operator is provided so that this object can be used as a key in an
  // ordered map.
  bool operator<(const DeviceMemoryBase &other) const {
    return std::tie(opaque_, size_) < std::tie(other.opaque_, other.size_);
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

  // Creates a memory region (slice) inside another allocated memory region.
  // Offset and size are in bytes.
  ABSL_ATTRIBUTE_ALWAYS_INLINE DeviceMemoryBase
  GetByteSlice(uint64_t offset_bytes, uint64_t size_bytes) const {
    DCHECK(offset_bytes + size_bytes <= size_)
        << "requested slice allocation (offset + size) is greater "
        << "than parent allocation size: (" << offset_bytes << " + "
        << size_bytes << ") vs. (" << size_ << ")";

    return DeviceMemoryBase(
        reinterpret_cast<std::byte *>(opaque_) + offset_bytes, size_bytes);
  }

 private:
  // Platform-dependent value representing allocated memory.
  //
  // User may also constructs the object with `kExternalAllocationMarker`
  // address and non-zero size, which indicates the case that buffer is
  // allocated externally (for Gpu backends we use it to allocate memory via
  // command buffer APIs).
  void *opaque_;
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

  // Returns pointer to the allocated data
  ElemT *base() { return reinterpret_cast<ElemT *>(opaque()); }
  const ElemT *base() const {
    return reinterpret_cast<const ElemT *>(opaque());
  }

  // Creates a typed area of DeviceMemory with a given opaque pointer and the
  // quantity of bytes in the allocation. This function is broken out to
  // distinguish bytes from an element count.
  static DeviceMemory<ElemT> MakeFromByteSize(void *opaque, uint64_t bytes) {
    return DeviceMemory<ElemT>(opaque, bytes);
  }

  // Creates a memory region (slice) inside another allocated memory region.
  // Offset and size are specified in terms of ElemT elements.
  DeviceMemory<ElemT> GetSlice(uint64_t element_offset,
                               uint64_t element_count) {
    return DeviceMemory<ElemT>(GetByteSlice(sizeof(ElemT) * element_offset,
                                            sizeof(ElemT) * element_count));
  }

 protected:
  // This is made protected because it accepts a byte-size instead of an element
  // count, which could potentially be misused given the ElementCount() nature
  // of this interface.
  //
  // In order to specify the desire to use byte size instead of element count
  // explicitly, use MakeFromByteSize.
  DeviceMemory(void *opaque, uint64_t size) : DeviceMemoryBase(opaque, size) {}
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_MEMORY_H_
