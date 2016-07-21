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

#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

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
  explicit DeviceMemoryBase(void *opaque = nullptr, uint64 size = 0,
                            bool is_sub_buffer = false)
      : opaque_(opaque), size_(size), is_sub_buffer_(is_sub_buffer) {}

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
  uint64 size() const { return size_; }

  // Warning: note that the pointer returned is not necessarily directly to
  // device virtual address space, but is platform-dependent.
  void *opaque() { return opaque_; }
  const void *opaque() const { return opaque_; }

  // Returns true if this is an offset into another primary allocation.
  bool is_sub_buffer() const { return is_sub_buffer_; }

  // Returns whether the two DeviceMemoryBase segments are identical (both in
  // their opaque pointer and size).
  bool IsSameAs(const DeviceMemoryBase &other) const {
    return opaque() == other.opaque() && size() == other.size();
  }

 protected:
  friend class StreamExecutor;

  // Resets the internal values of the opaque pointer and number of bytes in the
  // memory region, just as in the constructor.
  void Reset(void *opaque, uint64 bytes) {
    opaque_ = opaque;
    size_ = bytes;
  }

 private:
  void *opaque_;  // Platform-dependent value representing allocated memory.
  uint64 size_;   // Size in bytes of this allocation.
  bool is_sub_buffer_;  // Is this a primary allocation or a sub-buffer?
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
  DeviceMemory(std::nullptr_t) : DeviceMemory() {}

  // Typed device memory regions may be constructed from untyped device memory
  // regions, this effectively amounts to a cast from a void*.
  explicit DeviceMemory(const DeviceMemoryBase &other)
      : DeviceMemoryBase(const_cast<DeviceMemoryBase &>(other).opaque(),
                         other.size(), other.is_sub_buffer()) {}

  static constexpr size_t kElemSize = sizeof(ElemT);

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64 ElementCount() const { return size() / kElemSize; }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const { return ElementCount() == 1; }

  // Create a typed area of DeviceMemory with a given opaque pointer and the
  // quantity of bytes in the allocation. This function is broken out to
  // distinguish bytes from an element count.
  static DeviceMemory<ElemT> MakeFromByteSize(void *opaque, uint64 bytes) {
    return DeviceMemory<ElemT>(opaque, bytes);
  }

  // Resets the DeviceMemory data, in MakeFromByteSize fashion.
  // This simply clobbers the prior values.
  void ResetFromByteSize(void *opaque, uint64 bytes) {
    // TODO(leary) when NVCC is eliminated we can add this check (and the
    // logging include it requires).
    // CHECK_EQ(0, bytes % kElemSize);
    DeviceMemoryBase::Reset(opaque, bytes);
  }

  // ------------------------------------------------------------
  // DO NOT USE - FASTR TEAM-INTERNAL FUNCTIONS
  // Used internally by gcudacc.
#ifdef __GCUDACC__
  // Implicit conversion operators needed to support mixed mode. Since buffer
  // sizes aren't used in the CUDA launching process, and since the constructed
  // objects are all temporary, this is safe.
  // Linter warning disabled as we require an implicit conversion.
  DeviceMemory(const ElemT *opaque) :  // NOLINT
        DeviceMemoryBase(reinterpret_cast<void *>(const_cast<ElemT *>(opaque)),
                         0) {}

  operator ElemT *() { return reinterpret_cast<ElemT *>(opaque()); }
  operator const ElemT *() {
    return const_cast<const ElemT *>(reinterpret_cast<ElemT *>(opaque()));
  }
#endif
  // ------------------------------------------------------------

 protected:
  // This constructor is solely used from derived classes; it is made protected
  // because it accepts a byte-size instead of an element count, which could
  // potentially be misused given the ElementCount() nature of this interface.
  //
  // In order to specify the desire to use byte size instead of element count
  // explicitly, use MakeFromByteSize.
  DeviceMemory(void *opaque, uint64 size) : DeviceMemoryBase(opaque, size) {}
};

// A class to encapsulate the type and size of a dynamic shared memory
// buffer. Because the buffer exists solely on the device and is not copyable
// to the host, memory objects of this type do not maintain buffer pointers
// on the host.
template <typename ElemT>
class SharedDeviceMemory final : public DeviceMemoryBase {
 public:
  explicit SharedDeviceMemory(uint64 elem_count)
      : DeviceMemoryBase(nullptr, elem_count * kElemSize) {}

  static constexpr size_t kElemSize = sizeof(ElemT);

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64 ElementCount() const { return size() / kElemSize; }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const { return ElementCount() == 1; }
};

// Similar to the typed DeviceMemory, but is the unique owner of its
// memory, if any. ScopedDeviceMemory is thread-compatible. It is also
// movable and uncopyable to represent unique ownership.
template <typename ElemT>
class ScopedDeviceMemory {
 public:
  // Parameters:
  //  parent: Executor used to deallocate memory when this instance goes
  //          out of scope.
  //  value: Already-allocated device memory value for this scoped mechanism to
  //         deallocate. This memory must have been allocated by parent.
  ScopedDeviceMemory(StreamExecutor *parent, DeviceMemoryBase value);

  // Constructor overload that places a literal array into device memory
  ScopedDeviceMemory(StreamExecutor *parent,
                     std::initializer_list<ElemT> values);

  // Moves ownership of the memory from other to the constructed
  // object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceMemory(ScopedDeviceMemory &&other) noexcept:
      ScopedDeviceMemory(other.parent_, other.Release()) {}

  // Releases the memory that was provided in the constructor, through the
  // "parent" StreamExecutor.
  ~ScopedDeviceMemory();

  // Moves ownership of the memory from other to this object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceMemory& operator=(ScopedDeviceMemory &&other) {
    Reset(other.Release());
    parent_ = other.parent_;
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
  bool operator==(std::nullptr_t other) const { return wrapped_.is_null(); }
  bool operator!=(std::nullptr_t other) const { return !wrapped_.is_null(); }

  // Analogous to std::unique_ptr::reset, frees the existing memory held in
  // this scoped memory container and replaces it with updated. Ownership
  // of updated is transferred to this object.
  void Reset(DeviceMemory<ElemT> updated);
  void Reset(std::nullptr_t);

  // Analogous to std::unique_ptr::release, releases ownership of the held
  // memory and transfers it to the caller.
  //
  // Postcondition: *this == nullptr
  DeviceMemory<ElemT> Release() {
    auto tmp = wrapped_;
    wrapped_.ResetFromByteSize(nullptr, 0);
    return tmp;
  }

 private:
  DeviceMemory<ElemT> wrapped_;  // Value we wrap with scoped-release.
  StreamExecutor *parent_;       // See constructor.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedDeviceMemory);
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

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_
