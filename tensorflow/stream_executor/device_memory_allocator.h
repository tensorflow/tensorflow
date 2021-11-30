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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_

#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

class DeviceMemoryAllocator;

// Owning pointer for memory on a device.
//
// ScopedDeviceMemory is an owning pointer like std::unique_ptr, but it can
// point to memory that resides on a "device" (e.g. a GPU).  When a
// ScopedDeviceMemory goes out of scope, it frees the memory it owns.
//
// We say that an instance of ScopedDeviceMemory is "active" if it currently
// owns a (possibly empty) slice of memory on the device.  Moving,
// Release()'ing, Free()'ing, and other actions can deactive an active object.
template <typename ElemT>
class ScopedDeviceMemory {
 public:
  // Default construction initializes the internal state to nullptr.  This
  // mirrors the std::unique_ptr<> functionality, where default construction
  // produces a nullptr unique_ptr, which can be assigned later.
  ScopedDeviceMemory() : device_ordinal_(-1), allocator_(nullptr) {}

  // Construct a ScopedDeviceMemory from a custom allocator.
  //
  // Parameters:
  //  mem: Already-allocated device memory value for this scoped mechanism to
  //       deallocate. This memory must have been allocated by parent.
  //  device_ordinal: Device on which the memory was allocated.
  //  allocator: Allocator used to deallocate memory when this instance goes
  //             out of scope.
  ScopedDeviceMemory(DeviceMemoryBase mem, int device_ordinal,
                     DeviceMemoryAllocator *allocator)
      : wrapped_(mem), device_ordinal_(device_ordinal), allocator_(allocator) {
    DCHECK_GE(device_ordinal_, 0);
  }

  // A helper constructor to generate a scoped device memory given an already
  // allocated memory and a stream executor.
  //
  // Precondition: memory was allocated by the stream executor `parent`.
  ScopedDeviceMemory(StreamExecutor *parent, DeviceMemoryBase value);

  // Constructor overload that places a literal array into device memory.
  //
  // Relies on the allocation function exposed by the stream executor `parent`,
  // which will be also used for deallocating the memory
  ScopedDeviceMemory(StreamExecutor *parent,
                     std::initializer_list<ElemT> values);

  // Moves ownership of the memory from other to the constructed
  // object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceMemory(ScopedDeviceMemory &&other)
      : wrapped_(other.Release()),
        device_ordinal_(other.device_ordinal_),
        allocator_(other.allocator_) {}

  // Releases the memory that was provided in the constructor, through the
  // "parent" StreamExecutor.
  ~ScopedDeviceMemory() { TF_CHECK_OK(Free()); }

  // Moves ownership of the memory from other to this object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceMemory &operator=(ScopedDeviceMemory &&other) {
    TF_CHECK_OK(Free());
    wrapped_ = other.Release();
    allocator_ = other.allocator_;
    device_ordinal_ = other.device_ordinal_;
    return *this;
  }

  // Returns the memory that backs this scoped allocation converted to
  // DeviceMemory<T> apparent type. This is useful for cases where the
  // DeviceMemory must be passed by const-ref, as the ScopedDeviceMemory doesn't
  // allow copying, for scoped-object-lifetime reasons.
  const DeviceMemory<ElemT> &cref() const { return wrapped_; }

  // Returns a pointer to the DeviceMemory<T> apparent type for use in mutable
  // operations. The value returned should not be used outside the scope of this
  // ScopedDeviceMemory object's lifetime.
  DeviceMemory<ElemT> *ptr() { return &wrapped_; }
  const DeviceMemory<ElemT> *ptr() const { return &wrapped_; }

  // Smart-pointer-like operators for the wrapped DeviceMemory.
  // This reference must not be used outside the lifetime of this
  // ScopedDeviceMemory.
  const DeviceMemory<ElemT> &operator*() const { return cref(); }
  DeviceMemory<ElemT> *operator->() { return ptr(); }
  const DeviceMemory<ElemT> *operator->() const { return ptr(); }

  bool is_null() const { return wrapped_.is_null(); }
  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  // Analogous to std::unique_ptr::release, releases ownership of the held
  // memory and transfers it to the caller.
  //
  // Postcondition: *this == nullptr
  DeviceMemory<ElemT> Release() {
    DeviceMemory<ElemT> tmp = wrapped_;
    wrapped_ = DeviceMemory<ElemT>{};
    return tmp;
  }

  // The returned allocator is nonnull iff this object is active.
  DeviceMemoryAllocator *allocator() const { return allocator_; }

  int device_ordinal() const { return device_ordinal_; }

  // Frees the existing memory, resets the wrapped memory to null.
  port::Status Free();

 private:
  DeviceMemory<ElemT> wrapped_;       // Value we wrap with scoped-release.
  int device_ordinal_;                // Negative one for inactive object.
  DeviceMemoryAllocator *allocator_;  // Null if this object is inactive.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedDeviceMemory);
};

// Type alias for compatibility with the previous managed memory implementation.
using OwningDeviceMemory = ScopedDeviceMemory<uint8>;

// Memory allocator interface for the device.
//
// Intended usage is through Allocate() functions which return an owning smart
// pointer.
class DeviceMemoryAllocator {
 public:
  // Parameter platform indicates which platform the allocator allocates memory
  // on. Must be non-null.
  explicit DeviceMemoryAllocator(const Platform* platform)
      : platform_(platform) {}
  virtual ~DeviceMemoryAllocator() {}

  // Allocates memory on the device.
  //
  // If size > 0 and the returned StatusOr is OK, the wrapped OwningDeviceMemory
  // must not be null.  If size == 0, must return a null OwningDeviceMemory.
  //
  // 'retry_on_failure': If false, and the first attempt to allocate the memory
  // fails, the allocation should return immediately without retrying.  An
  // example use case is optional scratch spaces where a failure has only
  // performance impact.
  virtual port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal,
                                                      uint64_t size,
                                                      bool retry_on_failure,
                                                      int64_t memory_space) = 0;

  // Two-arg version of Allocate(), which sets retry-on-failure to true and
  // memory_space to default (0).
  //
  // (We don't simply use a default argument on the virtual Allocate function
  // because default args on virtual functions are disallowed by the Google
  // style guide.)
  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal,
                                              uint64_t size) {
    return Allocate(device_ordinal, size, /*retry_on_failure=*/true,
                    /*memory_space=*/0);
  }

  // Three-arg version of Allocate(), which sets memory_space to default (0).
  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure) {
    return Allocate(device_ordinal, size, retry_on_failure,
                    /*memory_space=*/0);
  }

  // Typed version of the allocation, returning typed memory.
  template <typename ElemT>
  port::StatusOr<ScopedDeviceMemory<ElemT>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure = true,
      int64_t memory_space = 0) {
    return Allocate(device_ordinal, size, retry_on_failure, memory_space);
  }

  // Must be a nop for null pointers. Should not be used.
  //
  // TODO(cheshire): Add deprecation notice.
  virtual port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) = 0;

  // Return the platform that the allocator allocates memory on.
  const Platform* platform() const { return platform_; }

  // Can we call Deallocate() as soon as a computation has been scheduled on
  // a stream, or do we have to wait for the computation to complete first?
  virtual bool AllowsAsynchronousDeallocation() const { return false; }

  // Returns a stream pointer on which it is always safe to access memory
  // allocated by this allocator. It is not necessary to use the returned stream
  // though, as clients may have additional information letting them safely use
  // a different stream.
  virtual port::StatusOr<Stream *> GetStream(int device_ordinal) = 0;

 protected:
  const Platform* platform_;
};

// Default memory allocator for a platform which uses
// StreamExecutor::Allocate/Deallocate.
class StreamExecutorMemoryAllocator : public DeviceMemoryAllocator {
 public:
  // Create an allocator supporting a single device, corresponding to the passed
  // executor.
  explicit StreamExecutorMemoryAllocator(StreamExecutor *executor);

  // Create an allocator supporting multiple stream executors.
  //
  // Precondition: all stream_executors have different device ordinals.
  StreamExecutorMemoryAllocator(
      const Platform *platform,
      absl::Span<StreamExecutor *const> stream_executors);

  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override;

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceMemoryAllocator::Allocate;

  port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override;

  // Gets-or-creates a stream for a given `device_ordinal` from an appropriate
  // stream executor.
  port::StatusOr<Stream *> GetStream(int device_ordinal) override;

  // Gets the stream executor for given device ordinal.
  port::StatusOr<StreamExecutor *> GetStreamExecutor(int device_ordinal) const;

 private:
  // Available stream executors. Each stream executor has a different device
  // ordinal.
  std::vector<StreamExecutor *> stream_executors_;

  absl::Mutex mutex_;

  // Cache of streams for GetStream.
  std::map<int, Stream> streams_ TF_GUARDED_BY(mutex_);
};

template <typename ElemT>
port::Status ScopedDeviceMemory<ElemT>::Free() {
  if (!wrapped_.is_null()) {
    CHECK(allocator_ != nullptr) << "Owning pointer in inconsistent state";
    TF_RETURN_IF_ERROR(allocator_->Deallocate(device_ordinal_, wrapped_));
  }
  wrapped_ = DeviceMemory<ElemT>{};
  return port::Status::OK();
}

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
