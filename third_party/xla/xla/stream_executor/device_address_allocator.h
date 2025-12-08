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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>

#include "absl/base/macros.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor {

class Stream;

// Forward declaration for owned device address. See implementation below.
template <typename T>
class ScopedDeviceAddress;

// A type-erased owning device address.
using OwningDeviceAddress ABSL_DEPRECATE_AND_INLINE() =
    ScopedDeviceAddress<uint8_t>;

// An allocator that allocates physical memory on the device, and maps it into
// the adressable range and returns it as `DeviceAddress` object to the caller.
//
// This allocator hides the physical memory allocation details from the user,
// and gives back a device address that can be used immediately to access data
// on the device. This is what most users want to use.
class DeviceAddressAllocator {
 public:
  // Parameter platform indicates which platform the allocator allocates
  // addresses on. Must be non-null.
  explicit DeviceAddressAllocator(const Platform* platform)
      : platform_(platform) {}
  virtual ~DeviceAddressAllocator() = default;

  // Allocates addressable memory on the device.
  //
  // If size > 0 and the returned absl::StatusOr is OK, the wrapped
  // ScopedDeviceAddress must not be null.  If size == 0, must return a null
  // ScopedDeviceAddress.
  //
  // 'retry_on_failure': If false, and the first attempt to allocate the address
  // fails, the allocation should return immediately without retrying.  An
  // example use case is optional scratch spaces where a failure has only
  // performance impact.
  virtual absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) = 0;

  // Two-arg version of Allocate(), which sets retry-on-failure to true and
  // memory_space to default (0).
  //
  // (We don't simply use a default argument on the virtual Allocate function
  // because default args on virtual functions are disallowed by the Google
  // style guide.)
  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(int device_ordinal,
                                                        uint64_t size);

  // Three-arg version of Allocate(), which sets memory_space to default (0).
  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(int device_ordinal,
                                                        uint64_t size,
                                                        bool retry_on_failure);

  // Typed version of the allocation, returning typed address.
  template <typename T>
  absl::StatusOr<ScopedDeviceAddress<T>> Allocate(int device_ordinal,
                                                  uint64_t size,
                                                  bool retry_on_failure = true,
                                                  int64_t memory_space = 0);

  // Return the platform that the allocator allocates addresses on.
  const Platform* platform() const { return platform_; }

  // Can we call Deallocate() as soon as a computation has been scheduled on
  // a stream, or do we have to wait for the computation to complete first?
  virtual bool AllowsAsynchronousDeallocation() const { return false; }

  // Returns a stream pointer on which it is always safe to access address
  // allocated by this allocator. It is not necessary to use the returned stream
  // though, as clients may have additional information letting them safely use
  // a different stream.
  virtual absl::StatusOr<Stream*> GetStream(int device_ordinal) = 0;

  // TODO(ezhulenev): Make this method private.
  virtual absl::Status Deallocate(int device_ordinal,
                                  DeviceAddressBase mem) = 0;

 private:
  template <typename T>
  friend class ScopedDeviceAddress;

  const Platform* platform_;
};

// An owning container for device address allocated via DeviceAddressAllocator.
//
// ScopedDeviceAddress is an owning std::unique_ptr-like object, but it can
// point to address that resides on a "device" (e.g. a GPU). When a
// ScopedDeviceAddress goes out of scope, it frees the address it owns.
//
// We say that an instance of ScopedDeviceAddress is "active" if it currently
// owns a (possibly empty) address range on the device. Moving,
// Release()'ing, Free()'ing, and other actions can deactivate an active object.
template <typename T>
class ScopedDeviceAddress {
 public:
  // Default construction initializes the internal state to nullptr.  This
  // mirrors the std::unique_ptr<> functionality, where default construction
  // produces a nullptr unique_ptr, which can be assigned later.
  ScopedDeviceAddress() : device_ordinal_(-1), allocator_(nullptr) {}

  // Construct a ScopedDeviceAddress from a custom allocator.
  //
  // Parameters:
  //  mem: Already-allocated device address value for this scoped mechanism to
  //       deallocate. This address must have been allocated by parent.
  //  device_ordinal: Device on which the address was allocated.
  //  allocator: Allocator used to deallocate the address when this instance
  //             goes out of scope.
  ScopedDeviceAddress(DeviceAddressBase mem, int device_ordinal,
                      DeviceAddressAllocator* allocator)
      : wrapped_(mem), device_ordinal_(device_ordinal), allocator_(allocator) {
    DCHECK_GE(device_ordinal_, 0);
  }

  // Moves ownership of the device address from other to the constructed
  // object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceAddress(ScopedDeviceAddress&& other) noexcept
      : wrapped_(other.Release()),
        device_ordinal_(other.device_ordinal_),
        allocator_(other.allocator_) {}

  // Releases the device address that was provided in the constructor.
  ~ScopedDeviceAddress() { CHECK_OK(Free()); }

  // Moves ownership of the device address from other to this object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceAddress& operator=(ScopedDeviceAddress&& other) noexcept {
    CHECK_OK(Free());
    wrapped_ = other.Release();
    allocator_ = other.allocator_;
    device_ordinal_ = other.device_ordinal_;
    return *this;
  }

  // Returns the device address that backs this scoped allocation converted to
  // DeviceAddress<T> apparent type. This is useful for cases where the
  // DeviceAddress must be passed by const-ref, as the ScopedDeviceAddress
  // doesn't allow copying, for scoped-object-lifetime reasons.
  const DeviceAddress<T>& cref() const { return wrapped_; }

  // Returns a pointer to the DeviceAddress<T> apparent type for use in mutable
  // operations. The value returned should not be used outside the scope of this
  // ScopedDeviceAddress object's lifetime.
  DeviceAddress<T>* ptr() { return &wrapped_; }
  const DeviceAddress<T>* ptr() const { return &wrapped_; }

  // Smart-pointer-like operators for the wrapped DeviceAddress.
  // This reference must not be used outside the lifetime of this
  // ScopedDeviceAddress.
  const DeviceAddress<T>& operator*() const { return cref(); }
  DeviceAddress<T>* operator->() { return ptr(); }
  const DeviceAddress<T>* operator->() const { return ptr(); }

  bool is_null() const { return wrapped_.is_null(); }
  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  // Analogous to std::unique_ptr::release, releases ownership of the held
  // device address and transfers it to the caller.
  //
  // Postcondition: *this == nullptr
  DeviceAddress<T> Release() {
    DeviceAddress<T> tmp = wrapped_;
    wrapped_ = DeviceAddress<T>{};
    return tmp;
  }

  // The returned allocator is nonnull iff this object is active.
  DeviceAddressAllocator* allocator() const { return allocator_; }

  int device_ordinal() const { return device_ordinal_; }

  // Frees the existing device address, resets the wrapped address to null.
  absl::Status Free();

 private:
  DeviceAddress<T> wrapped_;           // Value we wrap with scoped-release.
  int device_ordinal_;                 // Negative one for inactive object.
  DeviceAddressAllocator* allocator_;  // Null if this object is inactive.

  ScopedDeviceAddress(const ScopedDeviceAddress&) = delete;
  void operator=(const ScopedDeviceAddress&) = delete;
};

//===-----------------------------------------------------------------------===/
// Implementation details.
//===-----------------------------------------------------------------------===/

inline absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressAllocator::Allocate(int device_ordinal, uint64_t size) {
  return Allocate(device_ordinal, size, /*retry_on_failure=*/true,
                  /*memory_space=*/0);
}

// Three-arg version of Allocate(), which sets memory_space to default (0).
inline absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressAllocator::Allocate(int device_ordinal, uint64_t size,
                                 bool retry_on_failure) {
  return Allocate(device_ordinal, size, retry_on_failure,
                  /*memory_space=*/0);
}

template <typename T>
absl::StatusOr<ScopedDeviceAddress<T>> DeviceAddressAllocator::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  return Allocate(device_ordinal, size, retry_on_failure, memory_space);
}

template <typename T>
absl::Status ScopedDeviceAddress<T>::Free() {
  if (!wrapped_.is_null()) {
    CHECK(allocator_ != nullptr) << "Owning pointer in inconsistent state";
    TF_RETURN_IF_ERROR(allocator_->Deallocate(device_ordinal_, wrapped_));
  }
  wrapped_ = DeviceAddress<T>{};
  return absl::OkStatus();
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_ALLOCATOR_H_
