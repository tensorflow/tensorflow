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

// Common helper functions used for dealing with CUDA API datatypes.
//
// These are typically placed here for use by multiple source components (for
// example, BLAS and executor components).
#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_HELPERS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_HELPERS_H_

#include <complex>
#include <cstdint>

#include "absl/log/check.h"
#include "third_party/gpus/cuda/include/cuComplex.h"

namespace stream_executor {
namespace cuda {

// Type traits to get CUDA complex types from std::complex<T>.
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
  auto *result = reinterpret_cast<const typename CUDAComplexT<T>::type *>(p);
  CHECK_EQ(reinterpret_cast<uintptr_t>(p) % alignof(decltype(*result)), 0)
      << "Source pointer is not aligned by " << alignof(decltype(*result));
  return result;
}
template <typename T>
inline typename CUDAComplexT<T>::type *CUDAComplex(T *p) {
  auto *result = reinterpret_cast<typename CUDAComplexT<T>::type *>(p);
  CHECK_EQ(reinterpret_cast<uintptr_t>(p) % alignof(decltype(*result)), 0)
      << "Source pointer is not aligned by " << alignof(decltype(*result));
  return result;
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
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_HELPERS_H_
