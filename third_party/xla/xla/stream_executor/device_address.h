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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_H_
#define XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "absl/base/attributes.h"
#include "absl/log/check.h"

namespace stream_executor {

// DeviceAddress is an addressable virtual memory region on device. It's backed
// by a physical memory allocation (which is not directly addressable by the
// user, as device access requires virtual memory mapping). Physical memory
// allocations are managed by the StreamExecutor and device allocator, and users
// interact with device memory through DeviceAddress.

// DeviceAddressBase is a void*-analogous pointer to a device memory address,
// which comes with an optional size parameter. For typed pointers check the
// typed `DeviceAddress<T>` version below.
//
// IMPORTANT: Ideally `size` would be a mandatory parameter that tells the
// addressable range from the base pointer, however there are many existing use
// cases that rely on the default constructor and size is not set. Users should
// check for `opaque` being null to determine if the device address is null.
class DeviceAddressBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized device address
  // region. An opaque pointer may be provided -- see header for details on the
  // opacity of that pointer.
  explicit DeviceAddressBase(void* opaque = nullptr, uint64_t size = 0)
      : opaque_(opaque), size_(size) {
    // TODO(b/336267585): This constructor dangerously encourages
    //                 DeviceAddressBase(mem) which would imply
    //                 DeviceAddressBase(mem, 0)
    //                 We should delete & resolve any dependencies.
    //  explicit DeviceAddressBase(void *opaque) = delete;
  }

  // Returns whether the backing address is the null pointer.
  // A `== nullptr` convenience method is also provided.
  bool is_null() const { return opaque_ == nullptr; }

  explicit operator bool() const { return !is_null(); }

  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  bool operator==(const DeviceAddressBase& other) const {
    return opaque_ == other.opaque_ && size_ == other.size_;
  }

  // Provides a partial order between device address values.
  //
  // This operator is provided so that this object can be used as a key in an
  // ordered map.
  bool operator<(const DeviceAddressBase& other) const {
    return std::tie(opaque_, size_) < std::tie(other.opaque_, other.size_);
  }

  // Returns the size, in bytes, for the backing address range.
  uint64_t size() const { return size_; }

  // Warning: note that the pointer returned is not necessarily directly to
  // device virtual address space, but is platform-dependent.
  void* opaque() const { return opaque_; }

  // Returns the payload of this address range.
  uint64_t payload() const { return payload_; }

  // Sets payload to given value.
  void SetPayload(uint64_t payload) { payload_ = payload; }

  // Returns whether the two DeviceAddressBase segments are identical (both in
  // their opaque pointer and size).
  bool IsSameAs(const DeviceAddressBase& other) const {
    return opaque() == other.opaque() && size() == other.size();
  }

  // Creates and address range slice at the given offset and size. Offset and
  // size are in bytes.
  ABSL_ATTRIBUTE_ALWAYS_INLINE DeviceAddressBase
  GetByteSlice(uint64_t offset_bytes, uint64_t size_bytes) const {
    DCHECK(offset_bytes + size_bytes <= size_)
        << "requested address slice (offset + size) is out of bounds "
        << "of parent address: (" << offset_bytes << " + " << size_bytes
        << ") vs. (" << size_ << ")";

    return DeviceAddressBase(
        reinterpret_cast<std::byte*>(opaque_) + offset_bytes, size_bytes);
  }

 private:
  void* opaque_;          // Platform-dependent value representing base address.
  uint64_t size_;         // Size in bytes of this address range.
  uint64_t payload_ = 0;  // Payload data associated with this address.
};

// Typed wrapper around "void *"-like DeviceAddressBase.
//
// For example, DeviceAddress<int32_t> is a simple wrapper around
// DeviceAddressBase that represents one or more integers on Device.
template <typename T>
class DeviceAddress final : public DeviceAddressBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized addess range.
  DeviceAddress() : DeviceAddressBase(nullptr, 0) {}
  explicit DeviceAddress(std::nullptr_t) : DeviceAddress() {}

  // Typed device address range may be constructed from untyped device address
  // range, this effectively amounts to a cast from a void*.
  explicit DeviceAddress(const DeviceAddressBase& other)
      : DeviceAddressBase(other.opaque(), other.size()) {
    SetPayload(other.payload());
  }

  // Returns the number of elements of type T that constitute this address.
  uint64_t ElementCount() const { return size() / sizeof(T); }

  // Returns a base pointer to the data.
  T* base() const { return reinterpret_cast<T*>(opaque()); }

  // Creates a typed area of DeviceAddress with a given opaque pointer and the
  // quantity of bytes in the address range. This function is broken out to
  // distinguish bytes from an element count.
  static DeviceAddress<T> MakeFromByteSize(void* opaque, uint64_t bytes) {
    return DeviceAddress<T>(opaque, bytes);
  }

  // Creates and address range slice at the given offset and count. Offset and
  // count are specified in terms of T elements.
  DeviceAddress<T> GetSlice(uint64_t element_offset, uint64_t element_count) {
    return DeviceAddress<T>(
        GetByteSlice(sizeof(T) * element_offset, sizeof(T) * element_count));
  }

 protected:
  // This is made protected because it accepts a byte-size instead of an element
  // count, which could potentially be misused given the ElementCount() nature
  // of this interface.
  //
  // In order to specify the desire to use byte size instead of element count
  // explicitly, use MakeFromByteSize.
  DeviceAddress(void* opaque, uint64_t size)
      : DeviceAddressBase(opaque, size) {}
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_H_
