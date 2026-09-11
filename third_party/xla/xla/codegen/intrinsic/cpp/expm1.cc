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

#if defined(__FLT16_MANT_DIG__) && defined(__has_attribute) &&    \
    __has_attribute(ext_vector_type) && defined(__has_builtin) && \
    __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/expm1.h"

#include <cstdint>
#include <limits>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {
namespace {

// Computes expm1(x) = e^x - 1 for float32 within 0.72 ULP across all inputs
// (assuming hardware FMA; without FMA, `@llvm.fmuladd` falls back to unfused
// multiply/add and accuracy degrades by a few ULPs).
// See https://en.cppreference.com/w/cpp/numeric/math/expm1 for background.
//
// Algorithm overview:
// 1. Range reduction: We write x = n * ln(2) + (r_hi + r_lo), where
//    n = round(x / ln(2)) and (r_hi, r_lo) = FastTwoSum(x - n * kLn2Hi,
//    -n * kLn2Lo) in [-ln(2)/2, ln(2)/2]. Then:
//      expm1(x) = e^x - 1 = 2^n * e^(r_hi + r_lo) - 1.
// 2. Polynomial approximation: On [-ln(2)/2, ln(2)/2], we approximate the
//    cubic-and-higher terms with a degree-4 Sollya minimax polynomial q(r).
//    Since |r_lo| <= 0.5 * ulp(r_hi), we can drop r_hi * r_lo, r_lo^2, and
//    higher-order r_lo terms:
//      e^(r_hi + r_lo) - 1
//        ~ r_hi + r_lo + (r_hi + r_lo)^2 / 2 + (r_hi + r_lo)^3 * q(r_hi + r_lo)
//        ~ r_hi + r_lo + r_hi^2 / 2 + r_hi^3 * q(r_hi)
//        = r_hi + tail
//    where tail = r_lo + 0.5 * r_hi^2 + r_hi^3 * q(r_hi).
// 3. Reconstruction: Substituting e^(r_hi + r_lo) ~ 1 + r_hi + tail into
//    expm1(x) = 2^n * e^(r_hi + r_lo) - 1 gives:
//      expm1(x) ~ 2^n * (1 + r_hi + tail) - 1
//               = (2^n - 1) + 2^n * (r_hi + tail).
//    We evaluate this by summing the two largest terms first
//    (`head = (2^n - 1) + 2^n * r_hi`) and accumulating the lower-order
//    rounding bits of `2^n - 1` and `head` into `tail` before the final
//    addition (see Steps 3 and 5).
template <typename VecT>
inline VecT Expm1F32Impl(VecT input_x) {
  constexpr float kLog2e = 0x1.715476p+0f;  // 1 / ln(2)
  constexpr float kHalf = 0.5f;
  constexpr float kOne = 1.0f;
  constexpr float kNegOne = -1.0f;

  // Step 1: Thresholds and input clamping.
  // For x > 88.72283935546875f (0x1.62e42ep+6f), expm1(x) overflows float32
  // to +infinity. For x < -18.0f, e^x < 2^-25.95, so expm1(x) rounds to -1.0f
  // in round-to-nearest-even.
  constexpr float kExpHi = 0x1.62e42ep+6f;
  constexpr float kExpLo = -18.0f;

  // We clamp x to [kExpLo, kExpHi] before computing n = round(x / ln(2)) so
  // that n stays within [-26, 128] and integer conversions never overflow
  // int32.
  VecT x_clamped = Clamp(input_x, kExpLo, kExpHi);

  // Step 2: Cody-Waite range reduction (x = n * ln(2) + r_hi + r_lo).
  // Compute n = round(x / ln(2)) using round-to-nearest-even so |r_hi| <=
  // ln(2)/2.
  VecT n = __builtin_elementwise_rint(x_clamped * kLog2e);

  // We split ln(2) into a 15-bit high part (kLn2Hi) and a 24-bit low part
  // (kLn2Lo) to represent ln(2) to ~39 bits of precision. Because ln(2) is
  // irrational, a single float32 constant only holds 24 bits of ln(2). When x
  // is close to n * ln(2), their leading bits cancel, so subtracting a single
  // 24-bit constant leaves a remainder missing the lower bits of ln(2).
  // Choosing kLn2Hi (0x1.62e4p-1f) with 9 trailing zero bits ensures that for
  // |n| <= 128 (at most 7 bits), the product n * kLn2Hi requires at most
  // 15 + 7 = 22 <= 24 significand bits and is exact in float32. By Sterbenz's
  // lemma, x - n * kLn2Hi is also exact, and FastTwoSum normalizes
  // (x - n * kLn2Hi) and (-n * kLn2Lo) into a non-overlapping pair
  // (r_hi, r_lo).
  constexpr float kLn2Hi = 0x1.62e4p-1f;
  constexpr float kLn2Lo = 0x1.7f7d1cp-20f;

  VecT neg_n = -n;
  auto [r_hi, r_lo] = FastTwoSum(Fmuladd(neg_n, Splat<VecT>(kLn2Hi), x_clamped),
                                 neg_n * kLn2Lo);

  // Step 3: Polynomial approximation for e^(r_hi + r_lo) - 1 on
  // [-ln(2)/2, ln(2)/2].
  //
  // We approximate q(r) = (e^r - 1 - r - 0.5 * r^2) / r^3 using a degree-4
  // Sollya minimax polynomial (so e^r - 1 is degree 7 overall):
  //   q(r) ~ C3 + C4 * r + C5 * r^2 + C6 * r^3 + C7 * r^4
  //
  // Sollya script to derive C3..C7 and certify the relative ULP bound:
  //   prec = 165!;
  //   dom = [-log(2)/2; log(2)/2];
  //   f = (expm1(x) - x - 0.5*x^2) / x^3;
  //   p = fpminimax(f, 4, [|SG...|], dom);
  //   display = hexadecimal!;
  //   print(p);
  //   display = decimal!;
  //   print("Max relative error (in 2^-24 ULPs):",
  //         sup(supnorm(x + 0.5*x^2 + x^3*p, expm1(x), dom, relative, 1b-60))
  //         * 2^24);
  // Certified supnorm relative error bound: 0.0197 ULPs (in units of 2^-24).
  VecT q = HornerPoly(r_hi,
                      0x1.a2ffc4p-13f,  // C7
                      0x1.6d0ffap-10f,  // C6
                      0x1.110fbp-7f,    // C5
                      0x1.55551ap-5f,   // C4
                      0x1.555556p-3f);  // C3

  // Because `|r_lo| <= 0.5 * ulp(r_hi)` from FastTwoSum, the cross-term
  // `r_hi * r_lo` and `r_lo^2` in `(r_hi + r_lo)^2 / 2` are `<= 2^-24 *
  // r_hi^2` and can be dropped:
  //   e^(r_hi + r_lo) - 1
  //     ~ r_hi + r_lo + (r_hi + r_lo)^2 / 2 + (r_hi + r_lo)^3 * q(r_hi + r_lo)
  //     ~ r_hi + r_lo + r_hi^2 / 2 + r_hi^3 * q(r_hi)
  //     = r_hi + (0.5 * r_hi^2 + (r_hi^3 * q(r_hi) + r_lo))
  //     = r_hi + tail.
  VecT r2 = r_hi * r_hi;
  VecT r_lo_adj = Fmuladd(r2, r_hi * q, r_lo);
  VecT half_r_hi = kHalf * r_hi;
  VecT tail = Fmuladd(half_r_hi, r_hi, r_lo_adj);

  // Step 4: Construct 2^n via IEEE-754 exponent bit manipulation.
  // We split n = n1 + n2 into two scale factors s1 = 2^n1 and s2 = 2^n2 with
  // n1 = min(n, 127) and n2 = n - n1.
  // Near the upper threshold x ~ 88.72, n reaches 128. Because the maximum
  // finite float32 exponent is 127, constructing 2^128 directly would overflow
  // to +infinity. Clamping n1 to 127 ensures both s1 = 2^n1 and s2 = 2^n2 stay
  // finite (with s2 = 1 almost everywhere, and s2 = 2 only when n == 128).
  constexpr int32_t kMaxExp = 127;
  constexpr int32_t kMantissaBits = 23;

  auto n_int = ToSignedInt(n);
  auto max_exp = Splat<decltype(n_int)>(kMaxExp);
  auto n1 = n_int < max_exp ? n_int : max_exp;
  auto n2 = n_int - n1;

  auto exp1_bits = (n1 + max_exp) << kMantissaBits;
  auto exp2_bits = (n2 + max_exp) << kMantissaBits;
  VecT s1 = __builtin_bit_cast(VecT, exp1_bits);
  VecT s2 = __builtin_bit_cast(VecT, exp2_bits);

  // Step 5: Reconstruct expm1(x) from 2^n = s1 * s2 (Step 4) and
  // e^(r_hi + r_lo) ~ 1 + r_hi + tail (Step 3), where
  // `tail = 0.5 * r_hi^2 + r_hi^3 * q(r_hi) + r_lo`:
  //   expm1(x) = 2^n * e^(r_hi + r_lo) - 1
  //            ~ s1 * s2 * (1 + r_hi + tail) - 1
  //            = ((s1 - 1/s2) + s1 * (r_hi + tail)) * s2
  //            ~ ((s1 - 1) + s1 * (r_hi + tail)) * s2
  //            = (((s1 - 1) + s1 * r_hi) + s1 * tail) * s2.
  // Note that for all valid n, s1 - 1/s2 rounds to the same float32 value as
  // s1 - 1 (when n <= 127, s2 = 1; when n == 128, s1 = 2^127 and both
  // 2^127 - 0.5 and 2^127 - 1 round to 2^127).
  //
  // First we compute `head = (s1 - 1) + s1 * r_hi`, summing the two largest
  // terms so that any leading cancellation (e.g., when s1 = 0.5, where
  // `s1 - 1 = -0.5` and `s1 * r_hi > 0`) happens before adding the smaller
  // `s1 * tail` term.
  //
  // Next, we compute the low-order bits discarded when forming `s1 - 1` and
  // `head`, and add them into `tail_total = s1 * tail + head_err` before
  // the final addition `head + tail_total`:
  // - `FastTwoSum(s1, -1)` recovers the `-1` lost when `s1 >= 2^25` (where
  //   float32 spacing is >= 2.0, so `s1 - 1` rounds to `s1`).
  // - Because `s1_minus_one` and `head` are close in magnitude, their
  //   difference `d = s1_minus_one - head` is exact. `Fmuladd(s1, r_hi, d)`
  //   computes `(s1 * r_hi + s1_minus_one) - head` before rounding—recovering
  //   the exact low-order bits that were rounded off when `head` was formed.
  // Keeping these low-order bits prevents double-rounding across power-of-two
  // spacing boundaries (e.g., at n = 24, where `head` temporarily rounds above
  // 2^24 into the 2.0-spacing range before negative `s1 * tail` brings the
  // result back below 2^24).
  auto [s1_minus_one, s1_m1_err] = FastTwoSum(s1, Splat<VecT>(-kOne));
  VecT head = Fmuladd(s1, r_hi, s1_minus_one);
  VecT head_err = Fmuladd(s1, r_hi, s1_minus_one - head) + s1_m1_err;
  VecT tail_total = Fmuladd(s1, tail, head_err);
  VecT unscaled_result = head + tail_total;
  VecT result = unscaled_result * s2;

  // Step 6: Handle special values and boundary conditions:
  // - If x == +/-0.0, return input_x directly to preserve the sign of -0.0.
  // - If x < kExpLo, return -1.0 (asymptote as x -> -infinity).
  // - If x > kExpHi, return +infinity.
  // - If x is NaN, `ToSignedInt` guards `n_int` to 0 (so `s1 = s2 = 1.0`),
  //   while `r_hi` and `tail` are quiet NaN, so `result` evaluates to quiet
  //   NaN and all ordered comparisons below evaluate to false.
  constexpr float kInf = std::numeric_limits<float>::infinity();
  result = input_x == 0.0f ? input_x : result;
  result = input_x < kExpLo ? Splat<VecT>(kNegOne) : result;
  result = input_x > kExpHi ? Splat<VecT>(kInf) : result;
  return result;
}

// Computes expm1(x) = e^x - 1 for float64 within 1 ULP across all inputs.
// Follows the same algorithm as Expm1F32Impl above (Cody-Waite range reduction,
// degree-12 Sollya minimax polynomial approximation on [-ln(2)/2, ln(2)/2],
// and two-step reconstruction preserving low-order bits).
template <typename VecT>
inline VecT Expm1F64Impl(VecT input_x) {
  constexpr double kLog2e = 0x1.71547652b82fep+0;  // 1 / ln(2)
  constexpr double kHalf = 0.5;
  constexpr double kOne = 1.0;
  constexpr double kNegOne = -1.0;

  // Step 1: Thresholds and input clamping.
  // For x > 709.78271289338397 (0x1.62e42fefa39efp+9), expm1(x) overflows
  // float64 to +infinity. For x < -0x1.2b708872320e2p+5 (~ -37.42), e^x <
  // 2^-54, so expm1(x) rounds to -1.0 in round-to-nearest-even.
  constexpr double kExpHi = 0x1.62e42fefa39efp+9;
  constexpr double kExpLo = -0x1.2b708872320e2p+5;

  // We clamp x to [kExpLo, kExpHi] before computing n = round(x / ln(2)) so
  // that n stays within [-55, 1024] and integer conversions never overflow
  // int64.
  VecT x_clamped = Clamp(input_x, kExpLo, kExpHi);

  // Step 2: Cody-Waite range reduction (x = n * ln(2) + r_hi + r_lo).
  // Compute n = round(x / ln(2)) using round-to-nearest-even so |r_hi| <=
  // ln(2)/2.
  VecT n = __builtin_elementwise_rint(x_clamped * kLog2e);

  // We split ln(2) into a 36-bit high part (kLn2Hi) and a 53-bit low part
  // (kLn2Lo). Since |n| <= 1024 takes at most 10 bits, n * kLn2Hi requires at
  // most 36 + 10 = 46 <= 53 bits and is exact in float64, making
  // x - n * kLn2Hi exact by Sterbenz's lemma.
  constexpr double kLn2Hi = 0x1.62e42fefa0000p-1;
  constexpr double kLn2Lo = 0x1.cf79abc9e3b3ap-40;

  VecT neg_n = -n;
  auto [r_hi, r_lo] = FastTwoSum(Fmuladd(neg_n, Splat<VecT>(kLn2Hi), x_clamped),
                                 neg_n * kLn2Lo);

  // Step 3: Polynomial approximation for e^(r_hi + r_lo) - 1 on
  // [-ln(2)/2, ln(2)/2].
  //
  // We approximate q(r) = (e^r - 1 - r - 0.5 * r^2) / r^3 using a degree-9
  // Sollya minimax polynomial (so e^r - 1 is degree 12 overall):
  //   q(r) ~ C3 + C4 * r + ... + C12 * r^9
  //
  // Sollya script to derive C3..C12 and certify the relative ULP bound:
  //   prec = 256!;
  //   dom = [-log(2)/2; log(2)/2];
  //   p = fpminimax(expm1(x), [|3, 4, 5, 6, 7, 8, 9, 10, 11, 12|], [|D...|],
  //                 dom, relative, floating, x + 0.5*x^2);
  //   display = hexadecimal!;
  //   print(p);
  //   display = decimal!;
  //   print("Max relative error (in 2^-53 ULPs):",
  //         sup(supnorm(p, expm1(x), dom, relative, 1b-80)) * 2^53);
  // Certified supnorm relative error bound: 0.0028 ULPs (in units of 2^-53).
  VecT q = HornerPoly(r_hi,
                      0x1.1f0413a6fc15ap-29,  // C12
                      0x1.af5f1ac764956p-26,  // C11
                      0x1.27e521cebe596p-22,  // C10
                      0x1.71ddf816ea1afp-19,  // C9
                      0x1.a01a01872c144p-16,  // C8
                      0x1.a01a01b001948p-13,  // C7
                      0x1.6c16c16c1c0d1p-10,  // C6
                      0x1.111111110f657p-7,   // C5
                      0x1.555555555554ap-5,   // C4
                      0x1.5555555555559p-3);  // C3

  VecT r2 = r_hi * r_hi;
  VecT r_lo_adj = Fmuladd(r2, r_hi * q, r_lo);
  VecT half_r_hi = kHalf * r_hi;
  VecT tail = Fmuladd(half_r_hi, r_hi, r_lo_adj);

  // Step 4: Construct 2^n via IEEE-754 exponent bit manipulation.
  // We split n = n1 + n2 into two scale factors s1 = 2^n1 and s2 = 2^n2 with
  // n1 = min(n, 1023) and n2 = n - n1.
  // Near the upper threshold x ~ 709.78, n reaches 1024. Because the maximum
  // finite float64 exponent is 1023, constructing 2^1024 directly would
  // overflow to +infinity. Clamping n1 to 1023 ensures both s1 = 2^n1 and
  // s2 = 2^n2 stay finite (with s2 = 1 almost everywhere, and s2 = 2 only when
  // n == 1024).
  constexpr int64_t kMaxExp = 1023;
  constexpr int64_t kMantissaBits = 52;

  auto n_int = ToSignedInt(n);
  auto max_exp = Splat<decltype(n_int)>(kMaxExp);
  auto n1 = n_int < max_exp ? n_int : max_exp;
  auto n2 = n_int - n1;

  auto exp1_bits = (n1 + max_exp) << kMantissaBits;
  auto exp2_bits = (n2 + max_exp) << kMantissaBits;
  VecT s1 = __builtin_bit_cast(VecT, exp1_bits);
  VecT s2 = __builtin_bit_cast(VecT, exp2_bits);

  // Step 5: Reconstruct expm1(x) from 2^n = s1 * s2 (Step 4) and
  // e^(r_hi + r_lo) ~ 1 + r_hi + tail (Step 3), where
  // `tail = 0.5 * r_hi^2 + r_hi^3 * q(r_hi) + r_lo`:
  //   expm1(x) = 2^n * e^(r_hi + r_lo) - 1
  //            ~ s1 * s2 * (1 + r_hi + tail) - 1
  //            = ((s1 - 1/s2) + s1 * (r_hi + tail)) * s2
  //            ~ ((s1 - 1) + s1 * (r_hi + tail)) * s2
  //            = (((s1 - 1) + s1 * r_hi) + s1 * tail) * s2.
  // Note that for all valid n, s1 - 1/s2 rounds to the same float64 value as
  // s1 - 1 (when n <= 1023, s2 = 1; when n == 1024, s1 = 2^1023 and both
  // 2^1023 - 0.5 and 2^1023 - 1 round to 2^1023).
  //
  // First we compute `head = (s1 - 1) + s1 * r_hi`, summing the two largest
  // terms so that any leading cancellation (e.g., when s1 = 0.5, where
  // `s1 - 1 = -0.5` and `s1 * r_hi > 0`) happens before adding the smaller
  // `s1 * tail` term.
  //
  // Next, we compute the low-order bits discarded when forming `s1 - 1` and
  // `head`, and add them into `tail_total = s1 * tail + head_err` before
  // the final addition `head + tail_total`:
  // - `FastTwoSum(s1, -1)` recovers the `-1` lost when `s1 >= 2^54` (where
  //   float64 spacing is >= 2.0, so `s1 - 1` rounds to `s1`).
  // - Because `s1_minus_one` and `head` are close in magnitude, their
  //   difference `d = s1_minus_one - head` is exact. `Fmuladd(s1, r_hi, d)`
  //   computes `(s1 * r_hi + s1_minus_one) - head` before rounding—recovering
  //   the exact low-order bits that were rounded off when `head` was formed.
  // Keeping these low-order bits prevents double-rounding across power-of-two
  // spacing boundaries (e.g., at n = 53, where `head` temporarily rounds above
  // 2^53 into the 2.0-spacing range before negative `s1 * tail` brings the
  // result back below 2^53).
  auto [s1_minus_one, s1_m1_err] = FastTwoSum(s1, Splat<VecT>(-kOne));
  VecT head = Fmuladd(s1, r_hi, s1_minus_one);
  VecT head_err = Fmuladd(s1, r_hi, s1_minus_one - head) + s1_m1_err;
  VecT tail_total = Fmuladd(s1, tail, head_err);
  VecT unscaled_result = head + tail_total;
  VecT result = unscaled_result * s2;

  // Step 6: Handle special values and boundary conditions:
  // - If x == +/-0.0, return input_x directly to preserve the sign of -0.0.
  // - If x < kExpLo, return -1.0 (asymptote as x -> -infinity).
  // - If x > kExpHi, return +infinity.
  // - If x is NaN, `ToSignedInt` guards `n_int` to 0 (so `s1 = s2 = 1.0`),
  //   while `r_hi` and `tail` are quiet NaN, so `result` evaluates to quiet
  //   NaN and all ordered comparisons below evaluate to false.
  constexpr double kInf = std::numeric_limits<double>::infinity();
  result = input_x == 0.0 ? input_x : result;
  result = input_x < kExpLo ? Splat<VecT>(kNegOne) : result;
  result = input_x > kExpHi ? Splat<VecT>(kInf) : result;
  return result;
}

}  // namespace

//===--------------------------------------------------------------------===//
// XLA entry points, renamed with asm in header file.
//===--------------------------------------------------------------------===//

// Single precision
float expm1_f32(float x) { return Expm1F32Impl(x); }
Vec4f expm1_v4f32(Vec4f x) { return Expm1F32Impl(x); }
Vec8f expm1_v8f32(Vec8f x) { return Expm1F32Impl(x); }
Vec16f expm1_v16f32(Vec16f x) { return Expm1F32Impl(x); }

// Double precision
double expm1_f64(double x) { return Expm1F64Impl(x); }
Vec4d expm1_v4f64(Vec4d x) { return Expm1F64Impl(x); }
Vec8d expm1_v8f64(Vec8d x) { return Expm1F64Impl(x); }

}  // namespace xla::codegen
#endif  // defined(__FLT16_MANT_DIG__) && defined(__has_attribute) && ...
