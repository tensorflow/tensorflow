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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

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

// Evaluates float32 sinh(x) = (e^x - e^{-x}) / 2 to within 1 ULP using
// sinh(-x) = -sinh(x) and splitting |x| at 1.0:
//
// - For |x| < 1, avoids cancellation in e^{|x|} - e^{-|x|} via Taylor series:
//     sinh(|x|) = [(1 + |x| + |x|^2/2! + |x|^3/3! + ...) -
//                  (1 - |x| + |x|^2/2! - |x|^3/3! + ...)] / 2
//               = |x| + |x|^3/3! + |x|^5/5! + ...
//               = |x| + |x|^3 * (1/3! + |x|^2/5! + ...)
//              ~= |x| + |x|^3 * P_small(|x|^2)
//
// - For |x| >= 1, decomposes |x| = n * ln(2) + r_hi + r_lo (r = r_hi + r_lo)
//   and evaluates:
//     sinh(|x|) = 2 * (0.25 * e^{|x|} - 0.25 * e^{-|x|})
//     e^r       = 1 + r + r^2/2! + r^3/3! + ...
//              ~= 1 + r + r^2 * P_exp(r)
//               = (1 + r_hi) + (r_lo + r^2 * P_exp(r))
//     0.25 * e^{|x|} = 2^{n - 2} * e^r
//                   ~= head + tail
//       where head = 2^{n - 2} * (1 + r_hi)
//             tail = 2^{n - 2} * (r_lo + r^2 * P_exp(r))
//     -0.25 * e^{-|x|} = -0.25^2 / (0.25 * e^{|x|})
//                     ~= -0.0625 / (head + tail)
//     sinh(|x|) ~= 2 * ((head + tail) - 0.0625 / (head + tail))
//                = 2 * (head + (tail - 0.0625 / (head + tail)))
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

  // Construct scale = 2^(n - 2) via integer exponent bitcast ((n + 125) << 23)
  // and evaluate:
  //   head      = scale + scale * r_hi
  //   tail      = scale * (r_lo + r^2 * P_exp(r))
  //   half_h    = head + tail                  (~= 0.25 * e^{|x|})
  //   div_term  = -0.0625 / half_h             (~= -0.25 * e^{-|x|})
  //   sinh(|x|) ~= 2 * (head + (tail + div_term))
  T scale = __builtin_bit_cast(T, (n_int + 125u) << 23);
  T head = scale + scale * r_hi;
  T tail = scale * (r_lo + r2 * p_exp);
  T half_h = head + tail;
  T div_term = Splat<T>(-0.0625f) / half_h;
  T large_sinh_result = Splat<T>(2.0f) * (head + (tail + div_term));

  // 4. Small-|x| odd minimax polynomial on [-1, 1] for (sinh(|x|) - |x|) /
  // |x|^3:
  //   sinh(|x|) = |x| + |x|^3 * (c3 + |x|^2 * (c5 + |x|^2 * (c7 + |x|^2 * c9)))
  // Computing |x| + |x|^3 * P_small(|x|^2) (rather than
  // |x| * (1 + |x|^2 * P_small(|x|^2))) avoids an extra rounding error on the
  // leading |x| term when |x|^2 * P_small(|x|^2) is small. Generated in Sollya:
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

template <typename T>
inline T CoshF32(T x_in) {
  // Computes cosh(x) = (e^x + e^-x) / 2 in float32 within 1 ULP.
  // Because cosh(-x) = cosh(x), we work with |x| and split into two cases:
  //
  // - Small inputs (|x| < ln(2)):
  //   Let x2 = fl(|x| * |x|). We approximate cosh(|x|) with a degree-8
  //   polynomial:
  //     1 + (0.5 * x2 + x2^2 * P(x2))
  //   Evaluating 0.5 * x2 + x2^2 * P(x2) before adding 1.0 keeps the x2^2
  //   terms from being lost when added to 1.0.
  //
  // - Large inputs (|x| >= ln(2)):
  //   1. Write |x| = n * ln(2) + r, where n is the integer nearest to
  //      |x| / ln(2) and |r| <= ln(2) / 2. We subtract n * ln(2) in two pieces
  //      (ln2_hi and ln2_lo) so that |x| - n * ln2_hi does not round.
  //   2. Since e^|x| = 2^n * e^r, we compute 0.25 * e^|x| = 2^(n - 2) * e^r.
  //      We approximate e^r with 1 + r + r^2 * P_exp(r), split 2^(n - 2) * e^r
  //      into a main part (head) and a small correction (tail), and compute
  //      0.25 * e^-|x| as 0.0625 / (head + tail).
  //   3. Combine them as cosh(|x|) = 2 * (head + (tail + div_term)). Working
  //      with 0.25 * e^|x| and multiplying by 2 at the end prevents e^|x| from
  //      overflowing to infinity early when |x| is near 89.4 (where e^|x|
  //      exceeds FLT_MAX, but cosh(|x|) still fits in float32).
  using IntType = typename internal::CorrespondingIntVector<T>::type;

  T abs_x = BitwiseAbs(x_in);

  // Clamp abs_x to 89.5f so that inputs in the overflow range [89.4160, 89.5]
  // still evaluate to +Inf while avoiding intermediate overflow for huge
  // finite inputs. NaN takes the false branch and propagates through poly_cosh.
  T x = abs_x > Splat<T>(89.5f) ? Splat<T>(89.5f) : abs_x;

  // Range reduction: x = n * ln(2) + r, where n = round(x / ln(2)) and
  // |r| <= ln(2) / 2.
  // Round-to-nearest-even for non-negative x * log2(e) in [0, 129.12]: adding
  // 0x1.8p23f (1.5 * 2^23 = 0x4B400000) shifts the unit bit to bit 0 of the
  // float32 fraction, rounding to the nearest integer with ties to even.
  constexpr float kMagicBias = 0x1.8p23f;
  T biased = Fmuladd(x, Splat<T>(0x1.715476p+0f), Splat<T>(kMagicBias));
  T n_float = biased - Splat<T>(kMagicBias);
  IntType n = __builtin_bit_cast(IntType, biased) - 0x4B400000u;

  // Split ln(2) into a 16-bit high part (ln2_hi = 0x1.62e4p-1f) and a low
  // part (ln2_lo = 0x1.7f7d1cp-20f). Because n <= 129 has at most 8 integer
  // bits and ln2_hi has 16 bits, n * ln2_hi and r_hi = |x| - n * ln2_hi do not
  // round even without FMA.
  T r_hi = Fmuladd(-n_float, Splat<T>(0x1.62e4p-1f), x);
  T r_lo = -n_float * Splat<T>(0x1.7f7d1cp-20f);
  T r = r_hi + r_lo;

  T r2 = r * r;

  // Polynomial for (exp(r) - 1 - r) / r^2 on [-ln(2)/2, ln(2)/2]:
  T p_exp = HornerPoly(r, 0x1.6ca992p-10f, 0x1.120abep-7f, 0x1.55556cp-5f,
                       0x1.5554dep-3f, 0.5f);

  // Construct scale = 2^(n - 2) via integer exponent bitcast ((n + 125) << 23).
  // Split 0.25 * exp(|x|) = scale * exp(r) into head + tail:
  //   s_r_hi          = scale * r_hi (does not round because scale is a power
  //                     of 2)
  //   (head, head_lo) = FastTwoSum(scale, s_r_hi) (the rounded sum scale +
  //                     s_r_hi and the low bits dropped by that addition)
  //   tail            = head_lo + scale * (r_lo + r^2 * P(r))
  // Then +0.25 * exp(-|x|) = +0.0625 / (head + tail). Summing the smaller
  // terms (tail + div_term) before adding head and multiplying by 2.0 keeps
  // the error below 0.89 ULP without intermediate overflow.
  T scale = __builtin_bit_cast(T, (n + 125u) << 23);
  T s_r_hi = scale * r_hi;
  auto [head, head_lo] = FastTwoSum(scale, s_r_hi);
  T tail = Fmuladd(scale, Fmuladd(r2, p_exp, r_lo), head_lo);
  T half_h = head + tail;
  T div_term = Splat<T>(0.0625f) / half_h;
  T poly_cosh = Splat<T>(2.0f) * (head + (tail + div_term));

  // For |x| < ln(2) (~0.69314718f), use a degree-8 polynomial on [-ln(2),
  // ln(2)]:
  //   cosh(x) ~= 1 + 0.5 * x2 + x2^2 * (c4 + x2 * (c6 + x2 * c8))
  // Computing 0.5 * x2 + x2^2 * P(x2) before adding 1.0 avoids rounding the
  // 0.5 * x2 term against 1.0 before the x2^2 terms are added.
  T x2 = x * x;
  T p_small = HornerPoly(x2, 0x1.a9039p-16f, 0x1.6c023ap-10f, 0x1.55557p-5f);
  T x4 = x2 * x2;
  T small_cosh = Splat<T>(1.0f) + Fmuladd(x4, p_small, Splat<T>(0.5f) * x2);

  auto use_small = abs_x < Splat<T>(0.69314718f);
  return use_small ? small_cosh : poly_cosh;
}

}  // namespace

float sinh_f32(float x) { return SinhF32(x); }
Vec4f sinh_v4f32(Vec4f x) { return SinhF32(x); }
Vec8f sinh_v8f32(Vec8f x) { return SinhF32(x); }
Vec16f sinh_v16f32(Vec16f x) { return SinhF32(x); }

float cosh_f32(float x) { return CoshF32(x); }
Vec4f cosh_v4f32(Vec4f x) { return CoshF32(x); }
Vec8f cosh_v8f32(Vec8f x) { return CoshF32(x); }
Vec16f cosh_v16f32(Vec16f x) { return CoshF32(x); }

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
