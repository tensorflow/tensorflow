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

// Common helper functions used for dealing with ROCM API datatypes.
//
// These are typically placed here for use by multiple source components (for
// example, BLAS and executor components).

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_HELPERS_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_HELPERS_H_

#include <stddef.h>
#include <complex>

#include "rocm/include/hip/hip_complex.h"

namespace stream_executor {

template <typename ElemT>
class DeviceMemory;

namespace rocm {

// Converts a const DeviceMemory reference to its underlying typed pointer in
// ROCM device memory.
template <typename T>
const T *ROCMMemory(const DeviceMemory<T> &mem) {
  return static_cast<const T *>(mem.opaque());
}

// Converts a (non-const) DeviceMemory pointer reference to its underlying typed
// pointer in ROCM device memory.
template <typename T>
T *ROCMMemoryMutable(DeviceMemory<T> *mem) {
  return static_cast<T *>(mem->opaque());
}

static_assert(sizeof(std::complex<float>) == sizeof(hipComplex),
              "std::complex<float> and hipComplex should have the same size");
static_assert(offsetof(hipComplex, x) == 0,
              "The real part of hipComplex should appear first.");
static_assert(sizeof(std::complex<double>) == sizeof(hipDoubleComplex),
              "std::complex<double> and hipDoubleComplex should have the same "
              "size");
static_assert(offsetof(hipDoubleComplex, x) == 0,
              "The real part of hipDoubleComplex should appear first.");

// Type traits to get ROCM complex types from std::complex<>.

template <typename T>
struct ROCMComplexT {
  typedef T type;
};

template <>
struct ROCMComplexT<std::complex<float>> {
  typedef hipComplex type;
};

template <>
struct ROCMComplexT<std::complex<double>> {
  typedef hipDoubleComplex type;
};

// Converts pointers of std::complex<> to pointers of
// hipComplex/hipDoubleComplex. No type conversion for non-complex types.

template <typename T>
inline const typename ROCMComplexT<T>::type *ROCMComplex(const T *p) {
  return reinterpret_cast<const typename ROCMComplexT<T>::type *>(p);
}

template <typename T>
inline typename ROCMComplexT<T>::type *ROCMComplex(T *p) {
  return reinterpret_cast<typename ROCMComplexT<T>::type *>(p);
}

// Converts values of std::complex<float/double> to values of
// hipComplex/hipDoubleComplex.
inline hipComplex ROCMComplexValue(std::complex<float> val) {
  return {val.real(), val.imag()};
}

inline hipDoubleComplex ROCMComplexValue(std::complex<double> val) {
  return {val.real(), val.imag()};
}
}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_HELPERS_H_
