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

#include "xla/codegen/intrinsic/cpp/custom_cpp_intrinsics.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Every multiply-add below is `Fmuladd`, so targets without FMA get a separate
// multiply and add instead of an `fma` libcall. The Cody-Waite reductions are
// built from products that are exact without FMA, so they stay accurate on
// such targets; only the error terms of `TwoProd` in the polynomials are lost.

// Lanes with `|x|` at or above this use Payne-Hanek instead of Cody-Waite.
// Below them `n = round(|x| * 2/pi)` has at most 20 (f64) or 28 (f32) bits,
// which bounds the width of the pi/2 parts that `n` can multiply exactly.
constexpr double kF64FastThreshold = 0x1.0p+20;
constexpr double kF32FastThreshold = 0x1.0p+28;

// Adding 1.5 * 2^52 sets the ULP of `j` to 1.0 so round-to-nearest-even rounds
// `ax * (2/pi)` to an integer `n` and places `n mod 4` in the low 2 mantissa
// bits of `j`. Without FMA `n` can be off by one near a half-integer, which
// only lets |r| exceed pi/4 by a few ULPs.
constexpr double kMagic = 0x1.8p+52;
constexpr double kTwoOverPi = 0x1.45f306dc9c883p-1;
// pi/2 rounded to 53 bits and the next 53 bits, used by Payne-Hanek.
constexpr double kPiOver2Hi = 0x1.921fb54442d18p+0;
constexpr double kPiOver2Mid = 0x1.1a62633145c07p-54;

// pi/2 split for the f64 fast path. Each part is the remainder of pi/2 after
// the previous parts, rounded to nearest with at most 33, 32, 32 and 53 bits,
// which leaves 31, 32, 28 and 53 significant bits; the sum is within 2^-159 of
// pi/2. With |n| < 2^20 the products of n with the first three parts fit in 53
// bits and are exact.
constexpr double kF64PiOver2A = 0x1.921fb544p+0;
constexpr double kF64PiOver2B = 0x1.0b4611a6p-34;
constexpr double kF64PiOver2C = 0x1.3198a2ep-69;
constexpr double kF64PiOver2D = 0x1.b839a252049c1p-104;

// pi/2 split for the f32 fast path: `kPiOver2Hi` cut after 25 bits into
// 0x1.921fb5p+0 (25 bits) and 0x1.110b46p-26 (24 bits) so that their products
// with the 28-bit n are exact, followed by `kPiOver2Mid`.
constexpr double kF32PiOver2A = 0x1.921fb5p+0;
constexpr double kF32PiOver2B = 0x1.110b46p-26;

// Vectors with every |x| below 2^12 (so |n| < 2^12) use a cheaper 2-part
// split: pi/2 rounded to 41 bits, so that its product with n is exact, and the
// rounded remainder.
constexpr double kF32SmallThreshold = 0x1.0p+12;
constexpr double kF32SmallPiOver2A = 0x1.921fb54443p+0;
constexpr double kF32SmallPiOver2B = -0x1.73dcb3b399d74p-43;

// -1/6 and 1/24 as hi + lo pairs. The r^3 term of sin is up to 11% and the r^4
// term of cos up to 2% of the result, so a single rounded coefficient would
// cost a visible fraction of an ULP.
constexpr double kSinS1 = -0x1.5555555555555p-3;
constexpr double kSinS1Lo = -0x1.5555555555555p-57;
constexpr double kSinS2 = 0x1.1111111111111p-7;
constexpr double kSinS2Lo = 0x1.1111111111111p-63;
constexpr double kCosC1 = 0x1.5555555555555p-5;
constexpr double kCosC1Lo = 0x1.5555555555555p-59;

// (sin(r) - r + r^3/6 - r^5/120) / r^7 as a polynomial in r^2 on
// [-pi/4, pi/4].
template <typename T>
T EvalF64SinTailPoly(T r2) {
  return HornerPoly(r2, -0x1.2f49b46814157p-57, 0x1.952c77030ad4ap-49,
                    -0x1.ae7f3e733b81fp-41, 0x1.6124613a86d09p-33,
                    -0x1.ae64567f544e4p-26, 0x1.71de3a556c734p-19,
                    -0x1.a01a01a01a01ap-13);
}

// (cos(r) - 1 + r^2/2 - r^4/24) / r^6 as a polynomial in r^2 on [-pi/4, pi/4].
template <typename T>
T EvalF64CosTailPoly(T r2) {
  return HornerPoly(r2, 0x1.e542ba4020225p-62, -0x1.6827863b97d97p-53,
                    0x1.ae7f3e733b81fp-45, -0x1.93974a8c07c9dp-37,
                    0x1.1eed8eff8d898p-29, -0x1.27e4fb7789f5cp-22,
                    0x1.a01a01a01a01ap-16, -0x1.6c16c16c16c17p-10);
}

// Minimax fit of (sin(r) - r) / r^3 as a polynomial in r^2 on [-pi/4, pi/4].
// r + r^3 * P(r^2) has relative error below 2^-47.
template <typename T>
T EvalF32SinPoly(T r2) {
  return HornerPoly(r2, -0x1.a961a17edc631p-26, 0x1.71d76d1676743p-19,
                    -0x1.a019f93968c5bp-13, 0x1.1111110c87334p-7,
                    -0x1.5555555552239p-3);
}

// Minimax fit of (cos(r) - 1) / r^2 as a polynomial in r^2 on [-pi/4, pi/4].
// 1 + r^2 * P(r^2) has relative error below 2^-43.
template <typename T>
T EvalF32CosPoly(T r2) {
  return HornerPoly(r2, -0x1.23c97e6419bd1p-22, 0x1.a00eb9af46668p-16,
                    -0x1.6c16b348bc373p-10, 0x1.55555545c514ep-5,
                    -0x1.ffffffffe98afp-2);
}

// 21 words of 2/pi: word 0 is the integer part (0), and words 1..20 hold bits
// [1..1280] after the binary point of 2/pi (`sum_{k=1}^{20} W[k] * 2^{-64*k}`).
static constexpr uint64_t kTwoOverPiTable[21] = {
    0x0000000000000000ULL, 0xa2f9836e4e441529ULL, 0xfc2757d1f534ddc0ULL,
    0xdb6295993c439041ULL, 0xfe5163abdebbc561ULL, 0xb7246e3a424dd2e0ULL,
    0x06492eea09d1921cULL, 0xfe1deb1cb129a73eULL, 0xe88235f52ebb4484ULL,
    0xe99c7026b45f7e41ULL, 0x3991d639835339f4ULL, 0x9c845f8bbdf9283bULL,
    0x1ff897ffde05980fULL, 0xef2f118b5a0a6d1fULL, 0x6d367ecf27cb09b7ULL,
    0x4f463f669e5fea2dULL, 0x7527bac7ebe5f17bULL, 0x3d0739f78a5292eaULL,
    0x6bfb5fb11f8d5d08ULL, 0x56033046fc7b6babULL, 0xf0cfbc209af4361dULL,
};

struct PayneHanekResult {
  uint64_t q_raw;
  double r_hi;
  double r_lo;
};

// Cold-path Payne-Hanek range reduction for a finite double lane at or above
// the fast-path threshold (`ax >= 2^20`).
inline PayneHanekResult PayneHanekLaneF64(double ax) {
  using uint128_t = unsigned __int128;

  uint64_t bits = __builtin_bit_cast(uint64_t, ax);
  int exp = static_cast<int>((bits >> 52) & 0x7ff) - 1023;
  // Shift 53-bit significand left by 2 so that with e = exp + 10, the binary
  // point of the product lies at bit 64 of p2.
  uint64_t m = ((bits & ((1ULL << 52) - 1)) | (1ULL << 52)) << 2;
  int e = exp + 10;
  int idx = e >> 6;
  int shift = e & 63;

  uint64_t w0 = kTwoOverPiTable[idx];
  uint64_t w1 = kTwoOverPiTable[idx + 1];
  uint64_t w2 = kTwoOverPiTable[idx + 2];

  uint64_t hi = (shift == 0) ? w0 : ((w0 << shift) | (w1 >> (64 - shift)));
  uint64_t mid = (shift == 0) ? w1 : ((w1 << shift) | (w2 >> (64 - shift)));

  uint128_t p1 = static_cast<uint128_t>(m) * mid;
  uint128_t p2 = static_cast<uint128_t>(m) * hi + (p1 >> 64);
  uint128_t u128 = (p2 << 64) | static_cast<uint64_t>(p1);
  __int128 s128 = static_cast<__int128>(u128);
  int64_t i_hi = static_cast<int64_t>(s128 >> 76);

  // When fewer than 14 fractional bits cancel, the 2-word product (hi, mid)
  // already provides >= 90 bits of relative precision; otherwise fold in the
  // third word (lo) and low 24 bits (i_lo).
  int64_t top14 = i_hi >> 38;
  double f_hi, f_lo;
  if (__builtin_expect(top14 != 0 && top14 != -1 && top14 != -(1LL << 13), 1)) {
    int64_t i_mid = static_cast<int64_t>((u128 >> 24) & ((1ULL << 52) - 1));
    double d_hi = static_cast<double>(i_hi) * 0x1.0p-52;
    double d_mid = static_cast<double>(i_mid) * 0x1.0p-104;
    auto [fh, fl] = FastTwoSum(d_hi, d_mid);
    f_hi = fh;
    f_lo = fl;
  } else {
    uint64_t w3 = kTwoOverPiTable[idx + 3];
    uint64_t lo = (shift == 0) ? w2 : ((w2 << shift) | (w3 >> (64 - shift)));
    p1 += (static_cast<uint128_t>(m) * lo) >> 64;
    p2 = static_cast<uint128_t>(m) * hi + (p1 >> 64);
    u128 = (p2 << 64) | static_cast<uint64_t>(p1);
    s128 = static_cast<__int128>(u128);
    i_hi = static_cast<int64_t>(s128 >> 76);
    int64_t i_mid = static_cast<int64_t>((u128 >> 24) & ((1ULL << 52) - 1));
    int64_t i_lo = static_cast<int64_t>(u128 & ((1ULL << 24) - 1));
    double d_hi = static_cast<double>(i_hi) * 0x1.0p-52;
    double d_mid = static_cast<double>(i_mid) * 0x1.0p-104;
    double d_lo = static_cast<double>(i_lo) * 0x1.0p-128;
    auto [s1, e1] = FastTwoSum(d_mid, d_lo);
    auto [f0_hi, f0_lo] = FastTwoSum(d_hi, s1);
    auto [fh, fl] = FastTwoSum(f0_hi, f0_lo + e1);
    f_hi = fh;
    f_lo = fl;
  }

  uint64_t q_raw =
      static_cast<uint64_t>((p2 + (uint128_t{1} << 63)) >> 64) & 3ULL;
  auto [prod_hi, prod_err] = TwoProd(f_hi, kPiOver2Hi);
  double prod_lo =
      Fmuladd(f_lo, kPiOver2Hi, Fmuladd(f_hi, kPiOver2Mid, prod_err));
  auto [r_hi, r_lo] = FastTwoSum(prod_hi, prod_lo);
  return {q_raw, r_hi, r_lo};
}

template <typename T, typename IntT>
struct ReducedArgument {
  IntT q_raw;
  T r_hi;
  T r_lo;
};

// Returns `fast` with the Payne-Hanek reduction in every finite lane with
// `ax >= threshold`. Marked `noinline, cold` so `LinkIntoModule` /
// `PrepareForInlining` keeps the table lookup out of hot vector code while
// avoiding per-lane `noinline` calls and dynamic-index stack spills. The
// reduction is passed by value: passing the caller's variables by reference
// would make their addresses escape and force stack stores on the hot path.
template <typename T, typename IntT>
__attribute__((noinline, cold)) ReducedArgument<T, IntT> PayneHanekColdLoop(
    T ax, double threshold, ReducedArgument<T, IntT> fast) {
  for (size_t lane = 0; lane < NumLanes<T>; ++lane) {
    double ax_lane = GetLane(ax, lane);
    if (ax_lane >= threshold && ax_lane < kInf) {
      auto [q_lane, rh_lane, rl_lane] = PayneHanekLaneF64(ax_lane);
      SetLane(fast.q_raw, lane, q_lane);
      SetLane(fast.r_hi, lane, rh_lane);
      SetLane(fast.r_lo, lane, rl_lane);
    }
  }
  return fast;
}

// Picks sin or cos of the reduced argument by quadrant, applies the quadrant
// sign (and for sin the sign of `x`), and maps +-Inf/NaN inputs to NaN.
template <bool kIsCos, typename T, typename IntT>
inline T ApplyQuadrant(T x, T ax, IntT q_raw, T sin_val, T cos_val) {
  IntT q = (q_raw + Splat<IntT>(kIsCos ? 1 : 0)) & Splat<IntT>(3);
  T y = ((q & Splat<IntT>(1)) != Splat<IntT>(0)) ? cos_val : sin_val;

  // Bit 1 of `q` is the quadrant sign (`q in {2, 3}`); shift it into the
  // IEEE-754 sign bit 63.
  IntT sign_bit = (q & Splat<IntT>(2)) << 62;
  if constexpr (!kIsCos) {
    sign_bit ^= (__builtin_bit_cast(IntT, x) & Splat<IntT>(1ULL << 63));
  }
  y = __builtin_bit_cast(T, __builtin_bit_cast(IntT, y) ^ sign_bit);
  return (ax < Splat<T>(kInf)) ? y : (x - x);
}

template <typename To, typename From>
inline To ConvertLanes(From v) {
  if constexpr (std::is_arithmetic_v<From>) {
    return static_cast<To>(v);
  } else {
    return __builtin_convertvector(v, To);
  }
}

// f64 sin/cos with a double-word reduced argument. With |r| <= pi/4 the result
// is at least 0.7 * |r| (sin) or 0.7 (cos), so dropping the rounding error of a
// term that is a fraction f of the result costs up to about f / 0.7 ULP, and
// even a few hundredths of an ULP visibly increase the misrounding rate. The
// r^3 and r^5 terms of sin (f <= 0.11 and 0.0035) and the r^4 term of cos
// (f <= 0.022) are therefore accumulated as double-words, including the
// rounding of the powers of r and of the split coefficients. Terms of order r^6
// and above (f <= 0.0005) are plain polynomials added into the low word.
template <typename T, bool kIsCos>
inline T SinCosF64(T x) {
  using IntT = typename internal::CorrespondingIntVector<T>::type;

  T ax = BitwiseAbs(x);
  T j = Fmuladd(ax, Splat<T>(kTwoOverPi), Splat<T>(kMagic));
  T n = j - Splat<T>(kMagic);
  IntT q_raw = __builtin_bit_cast(IntT, j) & Splat<IntT>(3);
  // r = ax - n * pi/2 as a double-word with every rounding error kept. The
  // products n * A, n * B and n * C are exact. `ax - n * A` is exact because
  // both are multiples of 2^-53 (ax > 0.5 unless n = 0) and the difference is
  // below 1. `FastTwoSum` needs |a| >= |b| unless `a + b` is exact, and it is:
  // if |t1| < |n * B| then t1 - n * B is a multiple of 2^-65 below
  // 2 * 2^20 * 2^-33.9 = 2^-12.9, and if |s1| < |n * C| then s1 - n * C is a
  // multiple of 2^-96 below 2^-47.7, so both fit in 53 bits. What remains is
  // the rounding of e1 + e2 and of n * D (< 2^-83), far below the smallest
  // |r| of 2^-60.5 (at x = 0x1.6c6cbc45dc8dep+5) times 2^-53.
  T t1 = Fmuladd(-n, Splat<T>(kF64PiOver2A), ax);
  auto [s1, e1] = FastTwoSum(t1, -n * Splat<T>(kF64PiOver2B));
  auto [s2, e2] = FastTwoSum(s1, -n * Splat<T>(kF64PiOver2C));
  T lo = Fmuladd(-n, Splat<T>(kF64PiOver2D), e1 + e2);
  auto [r_hi, r_lo] = FastTwoSum(s2, lo);

  auto need_cold = (ax >= Splat<T>(kF64FastThreshold)) & (ax < Splat<T>(kInf));
  if (__builtin_expect(AnyLane(need_cold), 0)) {
    auto cold = PayneHanekColdLoop(ax, kF64FastThreshold,
                                   ReducedArgument<T, IntT>{q_raw, r_hi, r_lo});
    q_raw = cold.q_raw;
    r_hi = cold.r_hi;
    r_lo = cold.r_lo;
  }

  auto [r2, e_r2] = TwoProd(r_hi, r_hi);
  auto [r3, r3_err] = TwoProd(r2, r_hi);
  T e_r3 = Fmuladd(e_r2, r_hi, r3_err);
  auto [r4, r4_err] = TwoProd(r2, r2);
  T e_r4 = Fmuladd(Splat<T>(2.0) * r2, e_r2, r4_err);

  T w = Fmuladd(Splat<T>(-0.5), r2, Splat<T>(1.0));
  T w_lo = Fmuladd(Splat<T>(-0.5), e_r2,
                   Fmuladd(Splat<T>(-0.5), r2, Splat<T>(1.0) - w));
  auto [c4_hi, c4_err] = TwoProd(Splat<T>(kCosC1), r4);
  T c4_lo =
      Fmuladd(Splat<T>(kCosC1), e_r4, Fmuladd(Splat<T>(kCosC1Lo), r4, c4_err));
  auto [c_hi, c_lo] = FastTwoSum(w, c4_hi);

  auto [t3_hi, t3_err] = TwoProd(Splat<T>(kSinS1), r3);
  T t3_lo =
      Fmuladd(Splat<T>(kSinS1), e_r3, Fmuladd(Splat<T>(kSinS1Lo), r3, t3_err));
  auto [s_hi, s_lo] = FastTwoSum(r_hi, t3_hi);
  auto [r5, r5_err] = TwoProd(r3, r2);
  T e_r5 = Fmuladd(r3, e_r2, Fmuladd(e_r3, r2, r5_err));
  auto [t5_hi, t5_err] = TwoProd(Splat<T>(kSinS2), r5);
  T t5_lo =
      Fmuladd(Splat<T>(kSinS2), e_r5, Fmuladd(Splat<T>(kSinS2Lo), r5, t5_err));
  auto [s2_hi, s2_lo] = FastTwoSum(s_hi, t5_hi);
  // r_lo is scaled by d/dr sin(r) = cos(r) ~= c_hi and d/dr cos(r) = -sin(r)
  // ~= -s2_hi; the dropped higher-order terms are below 2^-60 relative.
  T sin_lo = Fmuladd(r5 * r2, EvalF64SinTailPoly(r2),
                     Fmuladd(r_lo, c_hi, (s_lo + s2_lo) + (t3_lo + t5_lo)));
  T sin_val = s2_hi + sin_lo;

  T cos_lo = Fmuladd(r4 * r2, EvalF64CosTailPoly(r2),
                     Fmuladd(-s2_hi, r_lo, (c_lo + w_lo) + c4_lo));
  T cos_val = c_hi + cos_lo;

  return ApplyQuadrant<kIsCos>(x, ax, q_raw, sin_val, cos_val);
}

// f32 sin/cos evaluated entirely in double with a single rounding to float at
// the end. An exhaustive scan shows |x mod pi/2| >= 2^-27.8 for every float in
// [pi/4, 2^28), and both reductions below give r with relative error <= 2^-53
// with or without FMA. With the polynomial errors above, the double result is
// within 2^-19 float ULP of the exact value.
//
// When every lane is below 2^12, n * kF32SmallPiOver2A is exact and so is
// `ax - n * A`: both are multiples of 2^-40 and the difference is below 1.
// n * B is below 2^-31.1 and rounds by < 2^-84, and the split is within 2^-96
// of pi/2.
//
// Otherwise the products of n with kF32PiOver2A and kF32PiOver2B are exact, and
// so are t and u: ax is a multiple of 2^-24 (ax > 0.5 unless n = 0), n * A of
// 2^-24 and n * B of 2^-49, and |u| < 1. Only n * kPiOver2Mid (< 2^-26.5,
// rounding error < 2^-80) and the last subtraction round, and the split is
// within 2^-109 of pi/2 (< 2^-81 after scaling by n).
template <typename T, bool kIsCos>
inline T SinCosF32(T x) {
  constexpr size_t kLanes = NumLanes<T>;
  using D =
      std::conditional_t<kLanes == 1, double,
                         double
                         __attribute__((vector_size(kLanes * sizeof(double))))>;
  using Q = std::conditional_t<kLanes == 1, uint64_t,
                               uint64_t __attribute__((
                                   vector_size(kLanes * sizeof(uint64_t))))>;

  D xd = ConvertLanes<D>(x);
  D ax = ConvertLanes<D>(BitwiseAbs(x));
  D j = Fmuladd(ax, Splat<D>(kTwoOverPi), Splat<D>(kMagic));
  D n = j - Splat<D>(kMagic);
  Q q_raw = __builtin_bit_cast(Q, j) & Splat<Q>(3);
  D r = Fmuladd(-n, Splat<D>(kF32SmallPiOver2B),
                Fmuladd(-n, Splat<D>(kF32SmallPiOver2A), ax));
  // NaN lanes compare false and stay on the short path; their result is NaN
  // either way.
  if (__builtin_expect(AnyLane(ax >= Splat<D>(kF32SmallThreshold)), 0)) {
    D t = Fmuladd(-n, Splat<D>(kF32PiOver2A), ax);
    D u = Fmuladd(-n, Splat<D>(kF32PiOver2B), t);
    r = Fmuladd(-n, Splat<D>(kPiOver2Mid), u);
    auto need_cold =
        (ax >= Splat<D>(kF32FastThreshold)) & (ax < Splat<D>(kInf));
    if (__builtin_expect(AnyLane(need_cold), 0)) {
      auto cold =
          PayneHanekColdLoop(ax, kF32FastThreshold,
                             ReducedArgument<D, Q>{q_raw, r, Splat<D>(0.0)});
      q_raw = cold.q_raw;
      r = cold.r_hi + cold.r_lo;
    }
  }

  D r2 = r * r;
  D sin_val = Fmuladd(r2 * r, EvalF32SinPoly(r2), r);
  D cos_val = Fmuladd(r2, EvalF32CosPoly(r2), Splat<D>(1.0));
  return ConvertLanes<T>(
      ApplyQuadrant<kIsCos>(xd, ax, q_raw, sin_val, cos_val));
}

}  // namespace

float sin_f32(float x) { return SinCosF32<float, false>(x); }
Vec4f sin_v4f32(Vec4f x) { return SinCosF32<Vec4f, false>(x); }
Vec8f sin_v8f32(Vec8f x) { return SinCosF32<Vec8f, false>(x); }
Vec16f sin_v16f32(Vec16f x) { return SinCosF32<Vec16f, false>(x); }
double sin_f64(double x) { return SinCosF64<double, false>(x); }
Vec2d sin_v2f64(Vec2d x) { return SinCosF64<Vec2d, false>(x); }
Vec4d sin_v4f64(Vec4d x) { return SinCosF64<Vec4d, false>(x); }
Vec8d sin_v8f64(Vec8d x) { return SinCosF64<Vec8d, false>(x); }

float cos_f32(float x) { return SinCosF32<float, true>(x); }
Vec4f cos_v4f32(Vec4f x) { return SinCosF32<Vec4f, true>(x); }
Vec8f cos_v8f32(Vec8f x) { return SinCosF32<Vec8f, true>(x); }
Vec16f cos_v16f32(Vec16f x) { return SinCosF32<Vec16f, true>(x); }
double cos_f64(double x) { return SinCosF64<double, true>(x); }
Vec2d cos_v2f64(Vec2d x) { return SinCosF64<Vec2d, true>(x); }
Vec4d cos_v4f64(Vec4d x) { return SinCosF64<Vec4d, true>(x); }
Vec8d cos_v8f64(Vec8d x) { return SinCosF64<Vec8d, true>(x); }

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
