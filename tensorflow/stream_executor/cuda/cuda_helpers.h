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

// Common helper functions used for dealing with CUDA API datatypes.
//
// These are typically placed here for use by multiple source components (for
// example, BLAS and executor components).

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_HELPERS_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_HELPERS_H_

#include <stddef.h>
#include <complex>

#include "cuda/include/cuComplex.h"
#include "cuda/include/cuda.h"

namespace perftools {
namespace gputools {

template <typename ElemT>
class DeviceMemory;

namespace cuda {

// Converts a const DeviceMemory reference to its underlying typed pointer in
// CUDA
// device memory.
template <typename T>
const T *CUDAMemory(const DeviceMemory<T> &mem) {
  return static_cast<const T *>(mem.opaque());
}

// Converts a (non-const) DeviceMemory pointer reference to its underlying typed
// pointer in CUDA device memory.
template <typename T>
T *CUDAMemoryMutable(DeviceMemory<T> *mem) {
  return static_cast<T *>(mem->opaque());
}

static_assert(sizeof(std::complex<float>) == sizeof(cuComplex),
              "std::complex<float> and cuComplex should have the same size");
static_assert(offsetof(cuComplex, x) == 0,
              "The real part of cuComplex should appear first.");
static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
              "std::complex<double> and cuDoubleComplex should have the same "
              "size");
static_assert(offsetof(cuDoubleComplex, x) == 0,
              "The real part of cuDoubleComplex should appear first.");

// Type traits to get CUDA complex types from std::complex<>.

template <typename T>
struct CUDAComplexT {
  typedef T type;
};

template <>
struct CUDAComplexT<std::complex<float>> {
  typedef cuComplex type;
};

template <>
struct CUDAComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};

// Converts pointers of std::complex<> to pointers of
// cuComplex/cuDoubleComplex. No type conversion for non-complex types.

template <typename T>
inline const typename CUDAComplexT<T>::type *CUDAComplex(const T *p) {
  return reinterpret_cast<const typename CUDAComplexT<T>::type *>(p);
}

template <typename T>
inline typename CUDAComplexT<T>::type *CUDAComplex(T *p) {
  return reinterpret_cast<typename CUDAComplexT<T>::type *>(p);
}

// Converts values of std::complex<float/double> to values of
// cuComplex/cuDoubleComplex.
inline cuComplex CUDAComplexValue(std::complex<float> val) {
  return {val.real(), val.imag()};
}

inline cuDoubleComplex CUDAComplexValue(std::complex<double> val) {
  return {val.real(), val.imag()};
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_HELPERS_H_
