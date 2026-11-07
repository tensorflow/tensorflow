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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Variant;

// Convenience functions to do typed allocation.  C++ constructors
// and destructors are invoked for complex types if necessary.
class TypedAllocator {
 public:
  // Pads `bytes` to the next multiple of sizeof(uint32_t) (4 bytes).
  // Returns 0 if padding would overflow size_t. Callers with a raw byte
  // count can call this directly.
  static size_t RoundUpTo4Bytes(size_t bytes) {
    if (bytes % sizeof(uint32_t) != 0) {
      size_t pad = sizeof(uint32_t) - (bytes % sizeof(uint32_t));
      if (std::numeric_limits<size_t>::max() - bytes < pad) {
        return 0;
      }
      bytes += pad;
    }
    return bytes;
  }

  // Pads `bytes` to the next multiple of sizeof(uint32_t) (4 bytes) for
  // sub-32-bit types (element_size < sizeof(uint32_t)). Callers with a raw
  // byte count can pass element_size = 1 to round up unconditionally, or
  // call RoundUpTo4Bytes directly.
  // Returns 0 if padding would overflow size_t.
  static size_t RoundUpSub32Bit(size_t bytes, size_t element_size) {
    if (element_size < sizeof(uint32_t)) {
      return RoundUpTo4Bytes(bytes);
    }
    return bytes;
  }

  // Returns the number of bytes to allocate for `num_elements` each of size
  // `element_size`. For sub-32-bit types (element_size < sizeof(uint32_t)),
  // the result is rounded up to the next multiple of sizeof(uint32_t)
  // (4 bytes). Returns 0 if multiplication or padding would overflow size_t.
  static size_t CalculateAllocationBytes(size_t num_elements,
                                         size_t element_size) {
    if (element_size > 0 &&
        num_elements > std::numeric_limits<size_t>::max() / element_size) {
      return 0;
    }
    return RoundUpSub32Bit(num_elements * element_size, element_size);
  }

  // Returns the number of bytes to allocate for `num_elements` of type T.
  // For sub-32-bit types (e.g. Eigen::half, Eigen::bfloat16), the result
  // is rounded up to the next multiple of sizeof(uint32_t) (4 bytes).
  // This is necessary because GPU atomic compare-and-swap (CAS) operations
  // on 16-bit types must operate on an enclosing 32-bit word
  // (see GpuAtomicCasHelper in gpu_device_functions.h). Without this
  // padding, the last element of an odd-length sub-32-bit buffer would
  // cause the 32-bit CAS to read 2 bytes past the end of the allocation.
  // The overhead is at most 3 bytes per tensor for 1-byte types
  // (at most 2 bytes for 16-bit types with an odd element count).
  // Returns 0 if multiplication or padding would overflow size_t.
  template <typename T>
  static size_t AllocationBytes(size_t num_elements) {
    return CalculateAllocationBytes(num_elements, sizeof(T));
  }

  // May return NULL if the tensor has too many elements to represent in a
  // single allocation or if the padded size overflows size_t.
  template <typename T>
  static T* Allocate(Allocator* raw_allocator, size_t num_elements,
                     const AllocationAttributes& allocation_attr) {
    // TODO(jeff): Do we need to allow clients to pass in alignment
    // requirements?

    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
      return nullptr;
    }

    size_t num_bytes = sizeof(T) * num_elements;
    if constexpr (sizeof(T) < sizeof(uint32_t)) {
      if (!raw_allocator->AllocatesOpaqueHandle()) {
        num_bytes = RoundUpSub32Bit(num_bytes, sizeof(T));
        if (num_elements > 0 && num_bytes == 0) {
          return nullptr;
        }
      }
    }

    void* p = raw_allocator->AllocateRawAlignedNew(
        Allocator::kAllocatorAlignment, num_bytes, allocation_attr);
    T* typed_p = reinterpret_cast<T*>(p);
    if (typed_p) RunCtor<T>(raw_allocator, typed_p, num_elements);
    return typed_p;
  }

  template <typename T>
  static void Deallocate(Allocator* raw_allocator, T* ptr,
                         size_t num_elements) {
    if (ptr) {
      RunDtor<T>(raw_allocator, ptr, num_elements);
      size_t num_bytes = sizeof(T) * num_elements;
      if constexpr (sizeof(T) < sizeof(uint32_t)) {
        if (!raw_allocator->AllocatesOpaqueHandle()) {
          num_bytes = RoundUpSub32Bit(num_bytes, sizeof(T));
        }
      }
      raw_allocator->DeallocateRawAlignedDelete(
          ptr, Allocator::kAllocatorAlignment, num_bytes);
    }
  }

 private:
  // No constructors or destructors are run for simple types
  template <typename T>
  static void RunCtor(Allocator* raw_allocator, T* p, size_t n) {
    static_assert(is_simple_type<T>::value, "T is not a simple type.");
  }

  template <typename T>
  static void RunDtor(Allocator* raw_allocator, T* p, size_t n) {}

  static void RunVariantCtor(Variant* p, size_t n);

  static void RunVariantDtor(Variant* p, size_t n);
};

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, tstring* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) tstring();
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, tstring* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) p->~tstring();
  }
}

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, ResourceHandle* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) ResourceHandle();
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, ResourceHandle* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) p->~ResourceHandle();
  }
}

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, Variant* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    RunVariantCtor(p, n);
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, Variant* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    RunVariantDtor(p, n);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
