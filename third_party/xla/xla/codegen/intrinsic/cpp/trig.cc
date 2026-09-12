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

#include "xla/codegen/intrinsic/cpp/trig.h"

#include <type_traits>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

namespace {

template <typename IntType, typename FloatType>
inline IntType FloatToInt(FloatType x) {
  if constexpr (std::is_fundamental_v<FloatType>) {
    return static_cast<IntType>(x);
  } else {
    return __builtin_convertvector(x, IntType);
  }
}

// Evaluates float32 sinh(x) to within 1 ULP by using odd symmetry
// sinh(-x) = -sinh(x) and splitting |x| at 1.0:
// - For |x| < 1, evaluates a minimax polynomial
//   sinh(|x|) = |x| + |x|^3 * P_small(|x|^2) to avoid cancellation in
//   exp(|x|) - exp(-|x|) near 0.
// - For |x| >= 1, uses Cody-Waite range reduction |x| = n * ln(2) + r to
//   compute 0.25 * exp(|x|) = 2^(n - 2) * exp(r) as a high part (`head`) plus
//   a low part (`tail`), computes `div_term = -0.25 * exp(-|x|) =
//   -0.0625 / (head + tail)`, and sums the smaller terms first in
//   `2 * (head + (tail + div_term))`.
template <typename T>
inline T SinhF32(T x) {
#pragma clang fp contract(fast)
  using IntType = typename internal::CorrespondingIntVector<T>::type;

  // 1. Absolute value and branch condition:
  // Split into |x| < 1 (small) and |x| >= 1 (large).
  T abs_x = BitwiseAbs(x);
  auto is_small = abs_x < Splat<T>(1.0f);

  // 2. Cody-Waite range reduction for |x| >= 1:
  // Clamp |x| at 89.5 so that n = round_nearest_even(|x| * log2(e)) <= 129.
  // This keeps (n + 125) <= 254 within normal float32 exponent range when
  // constructing scale = 2^(n - 2), while ensuring 2 * head (where
  // head = scale * (1 + r_hi) ~= 0.25 * exp(|x|) is the leading part of
  // 0.25 * exp(|x|) computed in step 3) overflows to +inf for |x| > ~89.415985.
  T clamped_x = abs_x < Splat<T>(89.5f) ? abs_x : Splat<T>(89.5f);
  T x_log2e = clamped_x * Splat<T>(0x1.715476p+0f);
  T n_float = __builtin_elementwise_rint(x_log2e);
  IntType n_int = FloatToInt<IntType>(n_float);

  // Split ln(2) into a 16-bit high part (ln2_hi = 0x1.62e4p-1f) and a low
  // part (ln2_lo = 0x1.7f7d1cp-20f). Because n <= 129 has at most 8 integer
  // bits and ln2_hi has 16 bits, n * ln2_hi and r_hi = |x| - n * ln2_hi are
  // bit-exact without FMA.
  T r_hi = abs_x - n_float * Splat<T>(0x1.62e4p-1f);
  T r_lo = -n_float * Splat<T>(0x1.7f7d1cp-20f);
  T r = r_hi + r_lo;
  T r2 = r * r;

  // 3. Large-|x| minimax polynomial for (exp(r) - 1 - r) / r^2 on
  // [-ln(2)/2, ln(2)/2].
  // Generated in Sollya:
  //   display = hexadecimal!;
  //   p_exp = fpminimax((expm1(x) - x) / x, [|1, 2, 3, 4, 5|], [|SG...|],
  //                     [-log(2)/2, log(2)/2], relative);
  //   supnorm(1 + x + x * p_exp, exp(x), [-log(2)/2, log(2)/2], relative,
  //           2^-40);
  //   // -> 0.95924801632e-8 (~0.0805 ULP in FP32)
  T p_exp = HornerPoly(r, 0x1.6ca992p-10f, 0x1.120abep-7f, 0x1.55556cp-5f,
                       0x1.5554dep-3f, 0x1p-1f);

  // Construct scale = 2^(n - 2) via integer exponent bitcast ((n + 125) << 23).
  // Split 0.25 * exp(|x|) = scale * exp(r) into head + tail:
  //   head = scale + scale * r_hi
  //   tail = scale * (r_lo + r^2 * P(r))
  // Then -0.25 * exp(-|x|) = -0.0625 / (head + tail). Summing the lower-order
  // terms (tail + div_term) before adding head and multiplying by 2.0
  // preserves 1-ULP precision without intermediate overflow.
  T scale = __builtin_bit_cast(T, (n_int + 125u) << 23);
  T head = scale + scale * r_hi;
  T tail = scale * (r_lo + r2 * p_exp);
  T half_h = head + tail;
  T div_term = Splat<T>(-0.0625f) / half_h;
  T large_sinh_result = Splat<T>(2.0f) * (head + (tail + div_term));

  // 4. Small-|x| odd minimax polynomial on [-1, 1] for (sinh(|x|) - |x|) /
  // |x|^3: sinh(|x|) = |x| + |x|^3 * (c3 + |x|^2 * (c5 + |x|^2 * (c7 + |x|^2 *
  // c9))).
  // Since the Taylor series sinh(x) = x + x^3/3! + x^5/5! + x^7/7! + x^9/9! +
  // ... has only odd powers of x and a linear coefficient of 1, we factor out
  // |x|^3 from the higher-order terms and evaluate a polynomial in |x|^2.
  // Computing |x| + |x|^3 * P(|x|^2) (rather than |x| * (1 + |x|^2 * P(|x|^2)))
  // avoids an extra rounding error on the leading |x| term when |x|^2 *
  // P(|x|^2) is small. Generated in Sollya:
  //   display = hexadecimal!;
  //   p_small = fpminimax(sinh(x), [|3, 5, 7, 9|], [|SG...|], [1b-20, 1],
  //                       floating, relative, x);
  //   supnorm(p_small, sinh(x), [1b-20, 1], relative, 2^-40);
  //   // -> 0.3558484444e-9 (~0.0030 ULP in FP32)
  T x2 = abs_x * abs_x;
  T p_small = HornerPoly(x2, 0x1.76771ep-19f, 0x1.a01c22p-13f, 0x1.1110eap-7f,
                         0x1.555556p-3f);
  T small_sinh_result = abs_x + (abs_x * x2) * p_small;

  // 5. Select branch and restore sign: sinh(-x) = -sinh(x).
  T res_mag = is_small ? small_sinh_result : large_sinh_result;
  return BitwiseCopysign(res_mag, x);
}

}  // namespace

float sinh_f32(float x) { return SinhF32(x); }
Vec4f sinh_v4f32(Vec4f x) { return SinhF32(x); }
Vec8f sinh_v8f32(Vec8f x) { return SinhF32(x); }
Vec16f sinh_v16f32(Vec16f x) { return SinhF32(x); }

}  // namespace xla::codegen
