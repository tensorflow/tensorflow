/* Copyright 2025 The OpenXLA Authors.

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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include "Eigen/Core"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

//===--------------------------------------------------------------------===//
// Generic conversion and operation
//===--------------------------------------------------------------------===//

template <typename VecType>
inline VecType VectorTanh(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.tanh();
  return *reinterpret_cast<const VecType*>(&result);
}

//===--------------------------------------------------------------------===//
// XLA entrypoints, renamed with asm in header file.
//===--------------------------------------------------------------------===//

using FloatArrayType = Eigen::Array<float, 16, 1>;

// Single precision
float tanh_f32(float x) { return Eigen::internal::ptanh_float(x); }
Vec4f tanh_v4f32(Vec4f x) {
  FloatArrayType buffer;
  *reinterpret_cast<Vec4f*>(&buffer) = x;
  buffer = buffer.tanh();
  return *reinterpret_cast<Vec4f*>(&buffer);
}
Vec8f tanh_v8f32(Vec8f x) {
  FloatArrayType buffer;
  *reinterpret_cast<Vec8f*>(&buffer) = x;
  buffer = buffer.tanh();
  return *reinterpret_cast<Vec8f*>(&buffer);
}
Vec16f tanh_v16f32(Vec16f x) { return VectorTanh(x); }

// Double precision
using DoubleArrayType = Eigen::Array<double, 8, 1>;
double tanh_f64(double x) { return Eigen::internal::ptanh_double(x); }
Vec4d tanh_v4f64(Vec4d x) {
  DoubleArrayType buffer;
  *reinterpret_cast<Vec4d*>(&buffer) = x;
  buffer = buffer.tanh();
  return *reinterpret_cast<Vec4d*>(&buffer);
}
Vec8d tanh_v8f64(Vec8d x) { return VectorTanh(x); }

}  // namespace xla::codegen
#endif  // defined(__has_attribute) && __has_attribute(vector_size) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
