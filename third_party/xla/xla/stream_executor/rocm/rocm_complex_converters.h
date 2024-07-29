/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_COMPLEX_CONVERTERS_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_COMPLEX_CONVERTERS_H_

#include <complex>
#include <cstdint>

#include "absl/log/check.h"
#include "rocm/include/hip/hip_complex.h"

namespace stream_executor {
namespace rocm {

// Type traits to get ROCM complex types from std::complex<T>.
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
  auto *result = reinterpret_cast<const typename ROCMComplexT<T>::type *>(p);
  CHECK_EQ(reinterpret_cast<uintptr_t>(p) % alignof(decltype(*result)), 0)
      << "Source pointer is not aligned by " << alignof(decltype(*result));
  return result;
}
template <typename T>
inline typename ROCMComplexT<T>::type *ROCMComplex(T *p) {
  auto *result = reinterpret_cast<typename ROCMComplexT<T>::type *>(p);
  CHECK_EQ(reinterpret_cast<uintptr_t>(p) % alignof(decltype(*result)), 0)
      << "Source pointer is not aligned by " << alignof(decltype(*result));
  return result;
}

}  // namespace rocm
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_COMPLEX_CONVERTERS_H_
