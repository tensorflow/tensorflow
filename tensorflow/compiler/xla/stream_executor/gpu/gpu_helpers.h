/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Common helper functions used for dealing with CUDA API datatypes.
//
// These are typically placed here for use by multiple source components (for
// example, BLAS and executor components).

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_HELPERS_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_HELPERS_H_

#include <stddef.h>

#include <complex>

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#include "tensorflow/tsl/platform/logging.h"

namespace stream_executor {

template <typename ElemT>
class DeviceMemory;

namespace gpu {

// Converts a const DeviceMemory reference to its underlying typed pointer in
// CUDA device memory.
template <typename T>
const T* GpuMemory(const DeviceMemory<T>& mem) {
  return static_cast<const T*>(mem.opaque());
}

// Converts a (non-const) DeviceMemory pointer reference to its underlying typed
// pointer in CUDA device memory.
template <typename T>
T* GpuMemoryMutable(DeviceMemory<T>* mem) {
  return static_cast<T*>(mem->opaque());
}

static_assert(
    sizeof(std::complex<float>) == sizeof(GpuComplexType),
    "std::complex<float> and GpuComplexType should have the same size");
static_assert(offsetof(GpuComplexType, x) == 0,
              "The real part of GpuComplexType should appear first.");
static_assert(
    sizeof(std::complex<double>) == sizeof(GpuDoubleComplexType),
    "std::complex<double> and GpuDoubleComplexType should have the same "
    "size");
static_assert(offsetof(GpuDoubleComplexType, x) == 0,
              "The real part of GpuDoubleComplexType should appear first.");

// Type traits to get CUDA complex types from std::complex<>.

template <typename T>
struct GpuComplexT {
  typedef T type;
};

template <>
struct GpuComplexT<std::complex<float>> {
  typedef GpuComplexType type;
};

template <>
struct GpuComplexT<std::complex<double>> {
  typedef GpuDoubleComplexType type;
};

// Converts pointers of std::complex<> to pointers of
// GpuComplexType/GpuDoubleComplexType. No type conversion for non-complex
// types.

template <typename T>
inline const typename GpuComplexT<T>::type* GpuComplex(const T* p) {
  auto* result = reinterpret_cast<const typename GpuComplexT<T>::type*>(p);
  CHECK_EQ(reinterpret_cast<uintptr_t>(p) % alignof(decltype(*result)), 0)
      << "Source pointer is not aligned by " << alignof(decltype(*result));
  return result;
}

template <typename T>
inline typename GpuComplexT<T>::type* GpuComplex(T* p) {
  auto* result = reinterpret_cast<typename GpuComplexT<T>::type*>(p);
  CHECK_EQ(reinterpret_cast<uintptr_t>(p) % alignof(decltype(*result)), 0)
      << "Source pointer is not aligned by " << alignof(decltype(*result));
  return result;
}

// Converts values of std::complex<float/double> to values of
// GpuComplexType/GpuDoubleComplexType.
inline GpuComplexType GpuComplexValue(std::complex<float> val) {
  return {val.real(), val.imag()};
}

inline GpuDoubleComplexType GpuComplexValue(std::complex<double> val) {
  return {val.real(), val.imag()};
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_HELPERS_H_
