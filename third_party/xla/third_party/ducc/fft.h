/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_DUCC_GOOGLE_FFT_H_
#define THIRD_PARTY_DUCC_GOOGLE_FFT_H_

// Wrapper around the DUCC FFT library to isolate usage of exceptions
// and RTTI.  Eliminates all direct usage of DUCC headers.

#include <complex>
#include <cstddef>
#include <vector>

#include "unsupported/Eigen/CXX11/ThreadPool"

namespace ducc0 {
namespace google {

using Shape = std::vector<std::size_t>;
using Stride = std::vector<std::ptrdiff_t>;

template <typename RealScalar>
void c2c(const std::complex<RealScalar>* in, const Shape& in_shape,
         const Stride& in_stride, std::complex<RealScalar>* out,
         const Shape& out_shape, const Stride& out_stride, const Shape& axes,
         bool forward, RealScalar scale,
         Eigen::ThreadPoolInterface* thread_pool);

template <typename RealScalar>
void r2c(const RealScalar* in, const Shape& in_shape, const Stride& in_stride,
         std::complex<RealScalar>* out, const Shape& out_shape,
         const Stride& out_stride, const Shape& axes, bool forward,
         RealScalar scale, Eigen::ThreadPoolInterface* thread_pool);

template <typename RealScalar>
void c2r(const std::complex<RealScalar>* in, const Shape& in_shape,
         const Stride& in_stride, RealScalar* out, const Shape& out_shape,
         const Stride& out_stride, const Shape& axes, bool forward,
         RealScalar scale, Eigen::ThreadPoolInterface* thread_pool);

#define FFT_DECLARATIONS(RealScalar)                                        \
  extern template void c2c<RealScalar>(                                     \
      const std::complex<RealScalar>* in, const Shape& in_shape,            \
      const Stride& in_stride, std::complex<RealScalar>* out,               \
      const Shape& out_shape, const Stride& out_stride, const Shape& axes,  \
      bool forward, RealScalar scale,                                       \
      Eigen::ThreadPoolInterface* thread_pool);                             \
  extern template void r2c<RealScalar>(                                     \
      const RealScalar* in, const Shape& in_shape, const Stride& in_stride, \
      std::complex<RealScalar>* out, const Shape& out_shape,                \
      const Stride& out_stride, const Shape& axes, bool forward,            \
      RealScalar scale, Eigen::ThreadPoolInterface* thread_pool);           \
  extern template void c2r(                                                 \
      const std::complex<RealScalar>* in, const Shape& in_shape,            \
      const Stride& in_stride, RealScalar* out, const Shape& out_shape,     \
      const Stride& out_stride, const Shape& axes, bool forward,            \
      RealScalar scale, Eigen::ThreadPoolInterface* thread_pool)
FFT_DECLARATIONS(float);
FFT_DECLARATIONS(double);
#undef FFT_DECLARATIONS

}  // namespace google
}  // namespace ducc0

#endif  // THIRD_PARTY_DUCC_GOOGLE_FFT_H_