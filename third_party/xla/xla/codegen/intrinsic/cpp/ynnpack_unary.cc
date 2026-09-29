/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/cpp/ynnpack_unary.h"

#include <cstddef>

#include "ynnpack/base/simd/gnu_vector.h"
#include "ynnpack/base/simd/vec.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

template <typename T, size_t N, typename VecType>
YNN_ALWAYS_INLINE VecType VectorLog(VecType x) {
  return ynn::simd::log(ynn::simd::vec<T, N>(x)).v_;
}

template <typename T, size_t N, typename VecType>
YNN_ALWAYS_INLINE VecType VectorLog1p(VecType x) {
  ynn::simd::vec<T, N> v(x);
  auto res = ynn::simd::log1p(v);
  return ynn::simd::copysign(res, v).v_;
}

// Single precision log
float xla_log_f32(float x) {
  return VectorLog<float, 4>(Vec4f{x, 0.0f, 0.0f, 0.0f})[0];
}
Vec2f xla_log_v2f32(Vec2f x) { return VectorLog<float, 2>(x); }
Vec4f xla_log_v4f32(Vec4f x) { return VectorLog<float, 4>(x); }
Vec8f xla_log_v8f32(Vec8f x) { return VectorLog<float, 8>(x); }
Vec16f xla_log_v16f32(Vec16f x) { return VectorLog<float, 16>(x); }

// Double precision log
double xla_log_f64(double x) { return VectorLog<double, 2>(Vec2d{x, 0.0})[0]; }
Vec2d xla_log_v2f64(Vec2d x) { return VectorLog<double, 2>(x); }
Vec4d xla_log_v4f64(Vec4d x) { return VectorLog<double, 4>(x); }
Vec8d xla_log_v8f64(Vec8d x) { return VectorLog<double, 8>(x); }

// Single precision log1p
float xla_log1p_f32(float x) {
  return VectorLog1p<float, 4>(Vec4f{x, 0.0f, 0.0f, 0.0f})[0];
}
Vec2f xla_log1p_v2f32(Vec2f x) { return VectorLog1p<float, 2>(x); }
Vec4f xla_log1p_v4f32(Vec4f x) { return VectorLog1p<float, 4>(x); }
Vec8f xla_log1p_v8f32(Vec8f x) { return VectorLog1p<float, 8>(x); }
Vec16f xla_log1p_v16f32(Vec16f x) { return VectorLog1p<float, 16>(x); }

// Double precision log1p
double xla_log1p_f64(double x) {
  return VectorLog1p<double, 2>(Vec2d{x, 0.0})[0];
}
Vec2d xla_log1p_v2f64(Vec2d x) { return VectorLog1p<double, 2>(x); }
Vec4d xla_log1p_v4f64(Vec4d x) { return VectorLog1p<double, 4>(x); }
Vec8d xla_log1p_v8f64(Vec8d x) { return VectorLog1p<double, 8>(x); }

}  // namespace xla::codegen
