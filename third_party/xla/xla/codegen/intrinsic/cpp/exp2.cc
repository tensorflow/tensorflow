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

#include "xla/codegen/intrinsic/cpp/exp2.h"

#include <cstdint>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {
namespace {

// Computes exp2(x) = 2^x for float32 and float64 (scalars and vectors).
//
// Algorithm overview:
//
// 1. Clamp x to [lo, hi]:
//    Inputs below `lo` underflow to 0 and inputs at or above `hi` overflow to
//    +infinity. For NaN inputs, the comparisons in `Clamp` evaluate to false,
//    so NaN passes through unchanged.
//
// 2. Split the clamped input into an integer and a fraction (xc = n + r):
//    We round `xc` to the nearest integer `n` so the remainder `r = xc - n`
//    lies in [-0.5, 0.5] and 2^xc = 2^n * 2^r.
//
//    Adding `kShift = 1.5 * 2^23` (for float32; `1.5 * 2^52` for float64)
//    shifts the binary point to bit 0 of the mantissa, forcing floating-point
//    addition to round `xc` to the nearest integer:
//      t = xc + kShift
//      n = t - kShift
//      r = xc - n
//    Because the unit place lands at bit 0 of `t`, the integer bit pattern of
//    `t` is `bits(kShift) + n`. This gives us both `n` as a float (to compute
//    `r`) and `n` as an integer in the low bits of `t` without float-to-integer
//    conversion instructions.
//
// 3. Approximate 2^r on [-0.5, 0.5] with a polynomial p(r):
//    Evaluated with Horner's method (`HornerPoly`, using fused multiply-add
//    where available):
//    - float32: degree 6 (within 1 ULP)
//    - float64: degree 11 (within 1 ULP)
//
// 4. Multiply p(r) by 2^n using exponent bit-shifting:
//    A normal floating-point power of two 2^k has mantissa 0 and biased
//    exponent `k + bias` (`bias = 127` for float32, `1023` for float64), so its
//    bit pattern is `(k + bias) << mantissa_bits`.
//
//    `n` can reach -151 (float32) or -1076 (float64) for subnormal outputs and
//    128 or 1024 at the overflow boundary, which is wider than a single normal
//    exponent can represent ([-126, 127] for float32, [-1022, 1023] for
//    float64). We split `n` into two halves `n1 = floor(n / 2)` and
//    `n2 = n - n1` and scale in two steps: `(p * 2^n1) * 2^n2`. Both halves
//    stay inside the normal exponent range, and the second multiply produces
//    subnormal outputs or +infinity at the ends of the range.
//
//    Since `bits(t) = bits(kShift) + n`, subtracting `bits(kShift) - 2 * bias`
//    gives `u = n + 2 * bias` (always positive after clamping), which splits
//    into the two biased exponents `e1 = u >> 1` and `e2 = u - e1`.

// Computes 2^x for float32 within 1 ULP across all 2^32 inputs (checked
// exhaustively against double-precision std::exp2), including subnormal
// outputs down to 2^-149.
//
// Sollya script for the degree-6 polynomial on [-0.5, 0.5]:
//   prec = 200!;
//   dom = [-0.5; 0.5];
//   p = fpminimax(2^x, [|1, 2, 3, 4, 5, 6|], [|SG...|], dom, relative,
//                 floating, 1);
template <typename T>
inline T Exp2F32Impl(T input_x) {
  using UintT = typename internal::CorrespondingIntVector<T>::type;

  T xc = Clamp(input_x, -151.0f, 128.0f);
  constexpr float kShift = 0x1.8p+23f;
  constexpr uint32_t kShiftMinusTwoBiases = 0x4b400000u - 254u;
  T t = xc + kShift;
  T n = t - kShift;
  T r = xc - n;

  T p = HornerPoly(r,
                   0x1.3fa206p-13f,  // c6
                   0x1.5f0b82p-10f,  // c5
                   0x1.3b30ap-7f,    // c4
                   0x1.c6af76p-5f,   // c3
                   0x1.ebfbd8p-3f,   // c2
                   0x1.62e43p-1f,    // c1
                   0x1p+0f);         // c0 = 1.0f

  UintT u = __builtin_bit_cast(UintT, t) - kShiftMinusTwoBiases;
  UintT e1 = u >> 1;
  UintT e2 = u - e1;
  T s1 = __builtin_bit_cast(T, e1 << 23);
  T s2 = __builtin_bit_cast(T, e2 << 23);
  return (p * s1) * s2;
}

// Computes 2^x for float64 within 1 ULP across the float64 domain, including
// subnormal outputs down to 2^-1074.
//
// Sollya script for the degree-11 polynomial on [-0.5, 0.5]:
//   prec = 256!;
//   dom = [-0.5; 0.5];
//   p = fpminimax(2^x, [|1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11|], [|D...|], dom,
//                 relative, floating, 1);
template <typename T>
inline T Exp2F64Impl(T input_x) {
  using UintT = typename internal::CorrespondingIntVector<T>::type;

  T xc = Clamp(input_x, -1076.0, 1024.0);
  constexpr double kShift = 0x1.8p+52;
  constexpr uint64_t kShiftMinusTwoBiases = 0x4338000000000000ULL - 2046ULL;
  T t = xc + kShift;
  T n = t - kShift;
  T r = xc - n;

  T p = HornerPoly(r,
                   0x1.e9731780488cp-32,   // c11
                   0x1.e61d4332035bap-28,  // c10
                   0x1.b52522bc20bcdp-24,  // c9
                   0x1.62bfc6596be6ap-20,  // c8
                   0x1.ffcbfc6685ee5p-17,  // c7
                   0x1.4309130f6cbdbp-13,  // c6
                   0x1.5d87fe789e96fp-10,  // c5
                   0x1.3b2ab6fb9fda6p-7,   // c4
                   0x1.c6b08d704a12cp-5,   // c3
                   0x1.ebfbdff82c5a6p-3,   // c2
                   0x1.62e42fefa39efp-1,   // c1
                   0x1p+0);                // c0 = 1.0

  UintT u = __builtin_bit_cast(UintT, t) - kShiftMinusTwoBiases;
  UintT e1 = u >> 1;
  UintT e2 = u - e1;
  T s1 = __builtin_bit_cast(T, e1 << 52);
  T s2 = __builtin_bit_cast(T, e2 << 52);
  return (p * s1) * s2;
}

}  // namespace

//===--------------------------------------------------------------------===//
// XLA entry points, renamed with asm in header file.
//===--------------------------------------------------------------------===//

// Single precision
float exp2_f32(float x) { return Exp2F32Impl(x); }
Vec4f exp2_v4f32(Vec4f x) { return Exp2F32Impl(x); }
Vec8f exp2_v8f32(Vec8f x) { return Exp2F32Impl(x); }
Vec16f exp2_v16f32(Vec16f x) { return Exp2F32Impl(x); }

// Double precision
double exp2_f64(double x) { return Exp2F64Impl(x); }
Vec2d exp2_v2f64(Vec2d x) { return Exp2F64Impl(x); }
Vec4d exp2_v4f64(Vec4d x) { return Exp2F64Impl(x); }
Vec8d exp2_v8f64(Vec8d x) { return Exp2F64Impl(x); }

}  // namespace xla::codegen
#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
