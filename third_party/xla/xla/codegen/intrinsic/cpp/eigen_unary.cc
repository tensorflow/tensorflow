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

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "absl/base/attributes.h"
#include "Eigen/Core"  // NOLINT(misc-include-cleaner)
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

template <typename VecType>
inline VecType VectorAtan(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.atan();
  return *reinterpret_cast<const VecType*>(&result);
}

//===--------------------------------------------------------------------===//

// XLA entrypoints, renamed with asm in header file.
//===--------------------------------------------------------------------===//

// Single precision
float tanh_f32(float x) {
  return Eigen::internal::ptanh_float(x);  // NOLINT(misc-include-cleaner)
}
Vec4f tanh_v4f32(Vec4f x) { return VectorTanh(x); }
#if defined(__x86_64__)
Vec8f tanh_v8f32(Vec8f x) { return VectorTanh(x); }
Vec16f tanh_v16f32(Vec16f x) { return VectorTanh(x); }
#endif

// Double precision
double tanh_f64(double x) {
  return Eigen::internal::ptanh_double(x);  // NOLINT(misc-include-cleaner)
}
#if defined(__x86_64__)
Vec4d tanh_v4f64(Vec4d x) { return VectorTanh(x); }
Vec8d tanh_v8f64(Vec8d x) { return VectorTanh(x); }
#endif

// Single precision.
// This uses the same polynomial approximation as Eigen's vectorized version
// (generic_atan) for numerical consistency. Written manually to avoid
// inefficient generic scalar bitwise operations in Eigen.
float atan_f32(float x_in) {
  constexpr float kPiOverTwo = 1.5707963267948966f;

  float abs_x = std::abs(x_in);
  // For tiny inputs (|x| < 1e-3), atan(x) is indistinguishable from x up to
  // single precision epsilon. Bypassing Remez polynomial approximation avoids
  // intermediate squaring underflows and reciprocal division traps when invoked
  // via scalar loop expansion on ARM NEON.
  if (abs_x < 1e-3f) {
    return x_in;
  }
  const bool large_x = abs_x > 1.0f;
  // For |x| > 1, use atan(x) = sign(x)*pi/2 - atan(1/x). Use direct
  // approximation otherwise.
  float x = large_x ? (1.0f / abs_x) : abs_x;

  constexpr float kAlpha[] = {1.12026982009410858154296875e-01f,
                              7.296695709228515625e-01f,
                              8.109951019287109375e-01f};

  constexpr float kBeta[] = {1.00917108356952667236328125e-02f,
                             2.8318560123443603515625e-01f, 1.0f,
                             8.109951019287109375e-01f};

  float x2 = x * x;
  float p = (kAlpha[0] * x2 + kAlpha[1]) * x2 + kAlpha[2];
  float q = ((kBeta[0] * x2 + kBeta[1]) * x2 + kBeta[2]) * x2 + kBeta[3];
  float r = x * (p / q);

  float result = large_x ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4f atan_v4f32(Vec4f x) { return VectorAtan(x); }
#if defined(__x86_64__)
Vec8f atan_v8f32(Vec8f x) { return VectorAtan(x); }
Vec16f atan_v16f32(Vec16f x) { return VectorAtan(x); }
#endif

// Double precision.
// This uses the same polynomial approximation as Eigen's vectorized version
// (generic_atan) for numerical consistency. Written manually to avoid
// inefficient generic scalar bitwise operations in Eigen.
double atan_f64(double x_in) {
  constexpr double kPiOverTwo = 1.57079632679489661923;

  double abs_x = std::abs(x_in);
  // For tiny inputs (|x| < 1e-9), atan(x) is indistinguishable from x.
  // Short-circuit bypass avoids intermediate Remez approximation underflows and
  // reciprocal division anomalies.
  if (abs_x < 1e-9) {
    return x_in;
  }
  const bool large_x = abs_x > 1.0;
  // For |x| > 1, use atan(x) = sign(x)*pi/2 - atan(1/x). Use direct
  // approximation otherwise.
  double x = large_x ? (1.0 / abs_x) : abs_x;

  constexpr double kAlpha[] = {2.6667153866462208e-05, 3.0917513112462781e-03,
                               5.2574296781008604e-02, 3.0409318473444424e-01,
                               7.5365702534987022e-01, 8.2704055405494614e-01,
                               3.3004361289279920e-01};

  constexpr double kBeta[] = {2.7311202462436667e-04,
                              1.0899150928962708e-02,
                              1.1548932646420353e-01,
                              4.9716458728465573e-01,
                              1.0,
                              9.3705509168587852e-01,
                              3.3004361289279920e-01};

  double x2 = x * x;
  double p =
      (((((kAlpha[0] * x2 + kAlpha[1]) * x2 + kAlpha[2]) * x2 + kAlpha[3]) *
            x2 +
        kAlpha[4]) *
           x2 +
       kAlpha[5]) *
          x2 +
      kAlpha[6];

  double q =
      (((((kBeta[0] * x2 + kBeta[1]) * x2 + kBeta[2]) * x2 + kBeta[3]) * x2 +
        kBeta[4]) *
           x2 +
       kBeta[5]) *
          x2 +
      kBeta[6];

  double r = x * (p / q);
  double result = large_x ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
#if defined(__x86_64__)
Vec4d atan_v4f64(Vec4d x) { return VectorAtan(x); }
Vec8d atan_v8f64(Vec8d x) { return VectorAtan(x); }
#endif

extern "C" double __xla_internal_sin_f64(double x) { return std::sin(x); }
extern "C" double __xla_internal_cos_f64(double x) { return std::cos(x); }

// Double precision helper
template <typename VecT>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline VecT VectorSinCosF64(VecT x, bool is_cos) {
  using ElementT = decltype(VecT{}[0]);
  constexpr int kWidth = sizeof(VecT) / sizeof(ElementT);
  using IntVecT = typename internal::CorrespondingIntVector<VecT>::type;

  // 1. Check for slow path (huge inputs, inf, nan)
  bool need_slow_path = false;
  for (size_t i = 0; i < kWidth; ++i) {
    need_slow_path |=
        (std::isnan(x[i]) || std::isinf(x[i]) || std::abs(x[i]) > 1e14);
  }

  if (need_slow_path) {
    VecT result = x;
    for (size_t i = 0; i < kWidth; ++i) {
      result[i] =
          is_cos ? __xla_internal_cos_f64(x[i]) : __xla_internal_sin_f64(x[i]);
    }
    return result;
  }

  auto make_vec = [](double v) {
    VecT res;
    for (size_t i = 0; i < kWidth; ++i) {
      res[i] = v;
    }
    return res;
  };

  auto make_ivec = [](uint64_t v) {
    IntVecT res;
    for (size_t i = 0; i < kWidth; ++i) {
      res[i] = v;
    }
    return res;
  };

  // 2. Vectorized fast path
  VecT x_abs = BitwiseAbs(x);

  // Range reduction: x_abs = q * pi/2 + r
  const VecT cst_2oPI = make_vec(0.63661977236758134307553505349006);
  VecT qval = x_abs * cst_2oPI;

  IntVecT q_int;
  for (size_t i = 0; i < kWidth; ++i) {
    q_int[i] = static_cast<int64_t>(qval[i] + 0.5);
  }

  VecT q;
  for (size_t i = 0; i < kWidth; ++i) {
    q[i] = static_cast<double>(q_int[i]);
  }

  // Multi-stage FMA for Remainder
  const VecT cst_pio2_a = make_vec(-1.570796325802803);
  const VecT cst_pio2_b = make_vec(-9.920935184482005e-10);
  const VecT cst_pio2_c = make_vec(-6.123234014771656e-17);
  const VecT cst_pio2_d = make_vec(1.903488962019325e-25);

  VecT r = __builtin_elementwise_fma(q, cst_pio2_a, x_abs);
  r = __builtin_elementwise_fma(q, cst_pio2_b, r);
  r = __builtin_elementwise_fma(q, cst_pio2_c, r);
  r = __builtin_elementwise_fma(q, cst_pio2_d, r);

  // Core Padé approximations for r in [-pi/4, pi/4]
  VecT rr = r * r;

  // Cos Padé
  const VecT cn4 = make_vec(80737373);
  const VecT cn3 = make_vec(-13853547000);
  const VecT cn2 = make_vec(727718024880);
  const VecT cn1 = make_vec(-11275015752000);
  const VecT cn0 = make_vec(23594700729600);
  const VecT cd3 = make_vec(147173);
  const VecT cd2 = make_vec(39328920);
  const VecT cd1 = make_vec(5772800880);
  const VecT cd0 = make_vec(522334612800);

  VecT sc_num = rr * cn4 + cn3;
  sc_num = rr * sc_num + cn2;
  sc_num = rr * sc_num + cn1;
  sc_num = rr * sc_num + cn0;

  VecT sc_den = rr * cd3 + cd2;
  sc_den = rr * sc_den + cd1;
  sc_den = rr * sc_den + cd0;
  sc_den = rr * sc_den + cn0;

  VecT poly_cos = sc_num / sc_den;

  // Sin Padé
  const VecT sn4 = make_vec(4585922449);
  const VecT sn3 = make_vec(-1066023933480);
  const VecT sn2 = make_vec(83284044283440);
  const VecT sn1 = make_vec(-2303682236856000);
  const VecT sn0 = make_vec(15605159573203200);
  const VecT sd3 = make_vec(1029037);
  const VecT sd2 = make_vec(345207016);
  const VecT sd1 = make_vec(61570292784);
  const VecT sd0_inner = make_vec(6603948711360);
  const VecT sd0 = make_vec(346781323848960);
  const VecT cst_45 = make_vec(45);

  VecT ss_num = rr * sn4 + sn3;
  ss_num = rr * ss_num + sn2;
  ss_num = rr * ss_num + sn1;
  ss_num = rr * ss_num + sn0;

  VecT ss_den = rr * sd3 + sd2;
  ss_den = rr * ss_den + sd1;
  ss_den = rr * ss_den + sd0_inner;
  ss_den = rr * ss_den + sd0;

  VecT poly_sin = (r * ss_num) / (cst_45 * ss_den);

  // Quadrant logic
  IntVecT q_is_odd = q_int & make_ivec(1);
  IntVecT swap_mask = q_is_odd != make_ivec(0);

  VecT base_res = swap_mask ? (is_cos ? poly_sin : poly_cos)
                            : (is_cos ? poly_cos : poly_sin);

  // Sign logic
  IntVecT quad_for_sign = is_cos ? (q_int + make_ivec(1)) : q_int;
  IntVecT flip_sign = (quad_for_sign & make_ivec(2)) != make_ivec(0);

  constexpr uint64_t kSignBit = 0x8000000000000000ULL;
  IntVecT sign_mask = flip_sign & make_ivec(kSignBit);

  if (!is_cos) {
    IntVecT orig_sign = __builtin_bit_cast(IntVecT, x) & make_ivec(kSignBit);
    sign_mask = sign_mask ^ orig_sign;
  }

  VecT final_res = __builtin_bit_cast(
      VecT, __builtin_bit_cast(IntVecT, base_res) ^ sign_mask);

  // 3. Fast bypass for tiny inputs to guarantee 0 ULP and avoid underflow
  IntVecT tiny_mask = x_abs < make_vec(1e-9);
  VecT tiny_res = is_cos ? make_vec(1.0) : x;

  return tiny_mask ? tiny_res : final_res;
}

// Double precision Sine
double sin_f64(double x) { return VectorSinCosF64(Vec2d{x, 0.0}, false)[0]; }
Vec2d sin_v2f64(Vec2d x) { return VectorSinCosF64(x, false); }
#if defined(__x86_64__)
Vec4d sin_v4f64(Vec4d x) { return VectorSinCosF64(x, false); }
Vec8d sin_v8f64(Vec8d x) { return VectorSinCosF64(x, false); }
#endif

// Double precision Cosine
double cos_f64(double x) { return VectorSinCosF64(Vec2d{x, 0.0}, true)[0]; }
Vec2d cos_v2f64(Vec2d x) { return VectorSinCosF64(x, true); }
#if defined(__x86_64__)
Vec4d cos_v4f64(Vec4d x) { return VectorSinCosF64(x, true); }
Vec8d cos_v8f64(Vec8d x) { return VectorSinCosF64(x, true); }
#endif

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(vector_size) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
