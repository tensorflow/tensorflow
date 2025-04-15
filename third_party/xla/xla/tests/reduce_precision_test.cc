/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdint>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace {

// Testcases in this file work as follows.
//
// for ty in {f16, bf16, f32, f64}:
//   for (operation_index, (e, m)) in \
//     enumerate(zip(ty_exponent_sizes, ty_mantissa_sizes)):
//
//     for testcase in ty_test_values:
//       let expected = testcase[0]
//       let input = testcase[operation_index]
//
//       CHECK that XLA-reduce-precision(
//         input, /*exp_bits=*/e, /*mantissa_bits=*/m) == expected
//
// Put into words:
//
//  - ty_{exponent,mantissa}_sizes tell us the different ways we will reduce the
//    precision of `ty`.
//
//  - ty_test_values is a 2D array of testcases, each of which is
//    len(ty_exponent_sizes) elements long.  The first element corresponds to
//    the input, and the j'th element corresponds to the expected output of
//    doing a reduce-precision with parameters ty_{exponent,mantissa}_sizes[j].
//
// You'll note above that testcase[0] is reused as both expected and input when
// operation_index == 0.  This implies that ty_{exponent,mantissa}_sizes[0] must
// be equal to `ty`'s exponent/mantissa size, making the reduce-precision op
// tested a nop.

// We want to test IEEE-f16 (a nop), cases that reduce just the
// mantissa/exponent, and a case that reduces both.
//
// We don't have a lot of tests here, relying instead on the coverage we have of
// f32 and f64.
//
// Note: The hypothetical float(3,7) type we're "converting" to would have:
//   max exp = 2^(3-1) - 1  =  3
//   min exp = -max_exp + 1 = -2
static const int f16_exponent_sizes[] = {5, 3, 3, 5};
static const int f16_mantissa_sizes[] = {10, 7, 10, 7};

// The F16VAL macro lets us write out the binary representation of an f16 in a
// more readable manner, separating out the exponent and mantissa.
#define F16VAL(EXPONENT, MANTISSA) ((EXPONENT << 10) + (MANTISSA))

static const uint16_t f16_test_values[][4] = {
    // True zero.
    {
        F16VAL(0b00000, 0b0000000000),  // 0.0
        F16VAL(0b00000, 0b0000000000),  // 0.0
        F16VAL(0b00000, 0b0000000000),  // 0.0
        F16VAL(0b00000, 0b0000000000),  // 0.0
    },
    // One.
    {
        F16VAL(0b01111, 0b0000000000),  // 1.0
        F16VAL(0b01111, 0b0000000000),  // 1.0
        F16VAL(0b01111, 0b0000000000),  // 1.0
        F16VAL(0b01111, 0b0000000000),  // 1.0
    },
    // Largest exponent that underflows to zero is -3, which is encoded as
    // -3 + 15 = 12
    {
        F16VAL(0b01100, 0b0000000000),  // 2^-3
        F16VAL(0b00000, 0b0000000000),  // 0
        F16VAL(0b00000, 0b0000000000),  // 0
        F16VAL(0b01100, 0b0000000000),  // 2^-3
    },
    // Smallest value that doesn't underflow to zero, due to mantissa rounding
    // up and incrementing the exponent out of the denormal range.
    {
        F16VAL(0b01100, 0b1111111100),  // 1020 * 2^-3
        F16VAL(0b01101, 0b0000000000),  // 2^-2
        F16VAL(0b00000, 0b0000000000),  // 0
        F16VAL(0b01101, 0b0000000000),  // 2^-2
    },
};

// We want to test bfloat16 (a nop), cases that reduce just the
// mantissa/exponent, and a case that reduces both.
//
// We don't have a lot of tests here, relying instead on the coverage we have of
// f32 and f64.
static const int bf16_exponent_sizes[] = {8, 5, 5, 8};
static const int bf16_mantissa_sizes[] = {7, 5, 7, 5};

// The BF16VAL macro lets us write out the binary representation of a bf16 in a
// more readable manner, separating out the exponent and mantissa.
#define BF16VAL(EXPONENT, MANTISSA) ((EXPONENT << 7) + (MANTISSA))

static const uint16_t bf16_test_values[][4] = {
    // True zero.
    {
        BF16VAL(0b00000, 0b0000000000),  // 0.0
        BF16VAL(0b00000, 0b0000000000),  // 0.0
        BF16VAL(0b00000, 0b0000000000),  // 0.0
        BF16VAL(0b00000, 0b0000000000),  // 0.0
    },
    // One.
    {
        BF16VAL(0b01111111, 0b0000000),  // 1.0
        BF16VAL(0b01111111, 0b0000000),  // 1.0
        BF16VAL(0b01111111, 0b0000000),  // 1.0
        BF16VAL(0b01111111, 0b0000000),  // 1.0
    },
    // Largest exponent that underflows to zero.
    {
        BF16VAL(0b01110000, 0b0000000),  // 3.05176e-05
        BF16VAL(0b00000000, 0b0000000),  // 0.0
        BF16VAL(0b00000000, 0b0000000),  // 0.0
        BF16VAL(0b01110000, 0b0000000)   // 3.05176e-05
    },
    // Smallest value that doesn't underflow to zero, due to mantissa rounding
    // up and incrementing the exponent out of the denormal range.
    {
        BF16VAL(0b01110000, 0b1111110),  // 6.05583e-05
        BF16VAL(0b01110001, 0b0000000),  // 6.10352e-05
        BF16VAL(0b00000000, 0b0000000),  // 0.0
        BF16VAL(0b01110001, 0b0000000),  // 6.10352e-05
    },
};

// We want to test IEEE-f32 (a no-op), IEEE-f16, and exponent-reduction-only and
// mantissa-reduction-only variants of IEEE-f16.
static const int f32_exponent_sizes[] = {8, 5, 5, 8};
static const int f32_mantissa_sizes[] = {23, 10, 23, 10};

// The F32VAL macro allows us to write out the binary representation of the
// input and expected values in a more readable manner.  The mantissa bits
// are separated into the "high" bits (retained with reduction to IEEE-f16)
// and the "low" bits (truncated with reduction to IEEE-f16).
#define F32VAL(EXPONENT, HIGH_MANTISSA, LOW_MANTISSA) \
  ((EXPONENT << 23) + (HIGH_MANTISSA << 13) + (LOW_MANTISSA))

static const uint32_t f32_test_values[][4] = {
    // True zero.
    {
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000)   // 0.0
    },
    // Largest exponent that underflows to zero.
    {
        F32VAL(0b01110000, 0b0000000000, 0b0000000000000),  // 3.05176e-05
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b01110000, 0b0000000000, 0b0000000000000)   // 3.05176e-05
    },
    // Largest value that rounds to a denormal and thus clamps to zero.
    {
        F32VAL(0b01110000, 0b1111111111, 0b0111111111111),  // 6.10203e-05
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b01110000, 0b1111111111, 0b0000000000000)   // 6.10054e-05
    },
    // Smallest value that doesn't underflow to zero, due to mantissa rounding
    // up and incrementing the exponent out of the denormal range.
    {
        F32VAL(0b01110000, 0b1111111111, 0b1000000000000),  // 6.10203e-05
        F32VAL(0b01110001, 0b0000000000, 0b0000000000000),  // 6.10352e-05
        F32VAL(0b00000000, 0b0000000000, 0b0000000000000),  // 0.0
        F32VAL(0b01110001, 0b0000000000, 0b0000000000000)   // 6.10352e-05
    },
    // Smallest value that doesn't underflow to zero even without mantissa
    // rounding.
    {
        F32VAL(0b01110001, 0b0000000000, 0b0000000000000),  // 6.10352e-05
        F32VAL(0b01110001, 0b0000000000, 0b0000000000000),  // 6.10352e-05
        F32VAL(0b01110001, 0b0000000000, 0b0000000000000),  // 6.10352e-05
        F32VAL(0b01110001, 0b0000000000, 0b0000000000000)   // 6.10352e-05
    },
    // One (to make sure bias-handling is done correctly).
    {
        F32VAL(0b01111111, 0b0000000000, 0b0000000000000),  // 1.0
        F32VAL(0b01111111, 0b0000000000, 0b0000000000000),  // 1.0
        F32VAL(0b01111111, 0b0000000000, 0b0000000000000),  // 1.0
        F32VAL(0b01111111, 0b0000000000, 0b0000000000000)   // 1.0
    },
    // Values in a space where ties round down due to ties-to-even:
    //   Value with highest mantissa that rounds down.
    {
        F32VAL(0b01111111, 0b0000000000, 0b1000000000000),  // 1.00049
        F32VAL(0b01111111, 0b0000000000, 0b0000000000000),  // 1.0
        F32VAL(0b01111111, 0b0000000000, 0b1000000000000),  // 1.00049
        F32VAL(0b01111111, 0b0000000000, 0b0000000000000)   // 1.0
    },
    //   Value with lowest mantissa that rounds up.
    {
        F32VAL(0b01111111, 0b0000000000, 0b1000000000001),  // 1.00049
        F32VAL(0b01111111, 0b0000000001, 0b0000000000000),  // 1.00098
        F32VAL(0b01111111, 0b0000000000, 0b1000000000001),  // 1.00049
        F32VAL(0b01111111, 0b0000000001, 0b0000000000000)   // 1.00098
    },
    // Values in a space where ties round up due to ties-to-even:
    //   Value with highest mantissa that rounds down.
    {
        F32VAL(0b01111111, 0b0000000001, 0b0111111111111),  // 1.00146
        F32VAL(0b01111111, 0b0000000001, 0b0000000000000),  // 1.00098
        F32VAL(0b01111111, 0b0000000001, 0b0111111111111),  // 1.00146
        F32VAL(0b01111111, 0b0000000001, 0b0000000000000)   // 1.00098
    },
    //   Value with a mantissa that rounds up.
    {
        F32VAL(0b01111111, 0b0000000001, 0b1000000000000),  // 1.00146
        F32VAL(0b01111111, 0b0000000010, 0b0000000000000),  // 1.00195
        F32VAL(0b01111111, 0b0000000001, 0b1000000000000),  // 1.00146
        F32VAL(0b01111111, 0b0000000010, 0b0000000000000)   // 1.00195
    },
    // Largest value that does not overflow to infinity.
    {
        F32VAL(0b10001110, 0b1111111111, 0b0111111111111),  // 65520.0
        F32VAL(0b10001110, 0b1111111111, 0b0000000000000),  // 65504.0
        F32VAL(0b10001110, 0b1111111111, 0b0111111111111),  // 65520.0
        F32VAL(0b10001110, 0b1111111111, 0b0000000000000)   // 65504.0
    },
    // Smallest value that overflows to infinity due to mantissa rounding up.
    {
        F32VAL(0b10001110, 0b1111111111, 0b1000000000000),  // 65520.0
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b10001110, 0b1111111111, 0b1000000000000),  // 65520.0
        F32VAL(0b10001111, 0b0000000000, 0b0000000000000)   // 65536.0
    },
    // Smallest value that overflows to infinity, without mantissa rounding.
    {
        F32VAL(0b10001111, 0b0000000000, 0b0000000000000),  // 65536.0
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b10001111, 0b0000000000, 0b0000000000000)   // 65536.0
    },
    // Smallest value that overflows to infinity due to mantissa rounding up,
    // even when exponent bits aren't reduced.
    {
        F32VAL(0b11111110, 0b1111111111, 0b1000000000000),  // 3.40199e+38
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000)   // Inf
    },
    // True infinity.
    {
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000),  // Inf
        F32VAL(0b11111111, 0b0000000000, 0b0000000000000)   // Inf
    },
    // NAN with a 1 in the preserved bits.
    {
        F32VAL(0b11111111, 0b1000000000, 0b0000000000000),  // NaN
        F32VAL(0b11111111, 0b1000000000, 0b0000000000000),  // NaN
        F32VAL(0b11111111, 0b1000000000, 0b0000000000000),  // NaN
        F32VAL(0b11111111, 0b1000000000, 0b0000000000000)   // NaN
    },
    // NAN with a 1 in the truncated bits.
    {
        F32VAL(0b11111111, 0b0000000000, 0b0000000000001),  // NaN
        F32VAL(0b11111111, 0b0000000000, 0b0000000000001),  // NaN
        F32VAL(0b11111111, 0b0000000000, 0b0000000000001),  // NaN
        F32VAL(0b11111111, 0b0000000000, 0b0000000000001)   // NaN
    },
    // NAN with all ones, causing rounding overflow.
    {
        F32VAL(0b11111111, 0b1111111111, 0b1111111111111),  // NaN
        F32VAL(0b11111111, 0b1111111111, 0b1111111111111),  // NaN
        F32VAL(0b11111111, 0b1111111111, 0b1111111111111),  // NaN
        F32VAL(0b11111111, 0b1111111111, 0b1111111111111)   // NaN
    }};

// F64VAL is like F32VAL but for doubles.
//
// Here the "high" mantissa bits are those retained with reduction to IEEE-f32
// (the first 23 bits), and the "low" bits are those truncated with reduction to
// IEEE-f32 (the remaining 29 bits).
#define F64VAL(EXPONENT, HIGH_MANTISSA, LOW_MANTISSA)             \
  ((uint64_t{EXPONENT} << 52) + (uint64_t{HIGH_MANTISSA} << 29) + \
   uint64_t{LOW_MANTISSA})

// We want to test IEEE-f64 (a no-op), IEEE-f32, and exponent-reduction-only and
// mantissa-reduction-only variants of IEEE-f32.
static const int f64_exponent_sizes[] = {11, 8, 8, 11};
static const int f64_mantissa_sizes[] = {52, 23, 52, 23};

static const uint64_t f64_test_values[][4] = {
    // True zero.
    {
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
    },
    // Largest exponent that underflows to zero, namely -127 (encoded as
    // -127 + 1023).
    {
        F64VAL(0b01110000000, 0x000000, 0x00000000),  // 5.8774717541114375e-39
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b01110000000, 0x000000, 0x00000000),  // 5.8774717541114375e-39
    },
    // Largest value that rounds to a denormal and thus clamps to zero.
    {
        F64VAL(0b01110000000, 0x7FFFFF, 0x0FFFFFFF),  // 1.1754943157898258e-38
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b01110000000, 0x7FFFFF, 0x00000000),  // 1.1754942807573643e-38
    },
    // Smallest value that doesn't underflow to zero, due to mantissa rounding
    // up and incrementing the exponent out of the denormal range.
    {
        F64VAL(0b01110000000, 0x7FFFFF, 0x10000000),  // 1.1754943157898259e-38
        F64VAL(0b01110000001, 0x000000, 0x00000000),  // 1.1754943508222875e-38
        F64VAL(0b00000000000, 0x000000, 0x00000000),  // 0.0
        F64VAL(0b01110000001, 0x000000, 0x00000000)   // 1.1754943508222875e-38
    },
    // Smallest value that doesn't underflow to zero even without mantissa
    // rounding.
    {
        F64VAL(0b01110000001, 0x000000, 0x00000000),  // 1.1754943508222875e-38
        F64VAL(0b01110000001, 0x000000, 0x00000000),  // 1.1754943508222875e-38
        F64VAL(0b01110000001, 0x000000, 0x00000000),  // 1.1754943508222875e-38
        F64VAL(0b01110000001, 0x000000, 0x00000000)   // 1.1754943508222875e-38
    },
    // One (to make sure bias-handling is done correctly).
    {
        F64VAL(0b01111111111, 0x000000, 0x00000000),  // 1.0
        F64VAL(0b01111111111, 0x000000, 0x00000000),  // 1.0
        F64VAL(0b01111111111, 0x000000, 0x00000000),  // 1.0
        F64VAL(0b01111111111, 0x000000, 0x00000000)   // 1.0
    },
    // Values in a space where ties round down due to ties-to-even:
    //   Value with highest mantissa that rounds down.
    {
        F64VAL(0b01111111111, 0x000000, 0x10000000),  // 1.0000000596046448
        F64VAL(0b01111111111, 0x000000, 0x00000000),  // 1.0
        F64VAL(0b01111111111, 0x000000, 0x10000000),  // 1.0000000596046448
        F64VAL(0b01111111111, 0x000000, 0x00000000)   // 1.0
    },
    //   Value with lowest mantissa that rounds up.
    {
        F64VAL(0b01111111111, 0x000000, 0x10000001),  // 1.000000059604645
        F64VAL(0b01111111111, 0x000001, 0x00000000),  // 1.0000001192092896
        F64VAL(0b01111111111, 0x000000, 0x10000001),  // 1.000000059604645
        F64VAL(0b01111111111, 0x000001, 0x00000000)   // 1.0000001192092896
    },
    // Values in a space where ties round up due to ties-to-even:
    //   Value with highest mantissa that rounds down.
    {
        F64VAL(0b01111111111, 0x000001, 0x0fffffff),  // 1.0000001788139341
        F64VAL(0b01111111111, 0x000001, 0x00000000),  // 1.0000001192092896
        F64VAL(0b01111111111, 0x000001, 0x0fffffff),  // 1.0000001788139341
        F64VAL(0b01111111111, 0x000001, 0x00000000)   // 1.0000001192092896
    },
    //   Value with a mantissa that rounds up.
    {
        F64VAL(0b01111111111, 0x000001, 0x10000000),  // 1.0000001788139343
        F64VAL(0b01111111111, 0x000002, 0x00000000),  // 1.0000002384185791
        F64VAL(0b01111111111, 0x000001, 0x10000000),  // 1.0000001788139343
        F64VAL(0b01111111111, 0x000002, 0x00000000),  // 1.0000002384185791
    },
    // Largest value that does not overflow to infinity.
    {
        F64VAL(0b10001111110, 0x7fffff, 0x0fffffff),  // 3.4028235677973362e+38
        F64VAL(0b10001111110, 0x7fffff, 0x00000000),  // 3.4028234663852886e+38
        F64VAL(0b10001111110, 0x7fffff, 0x0fffffff),  // 3.4028235677973362e+38
        F64VAL(0b10001111110, 0x7fffff, 0x00000000),  // 3.4028234663852886e+38
    },
    // Smallest value that overflows to infinity due to mantissa rounding up.
    {
        F64VAL(0b10001111110, 0x7fffff, 0x10000000),  // 3.4028235677973366e+38
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b10001111110, 0x7fffff, 0x10000000),  // 3.4028235677973366e+38
        F64VAL(0b10001111111, 0x000000, 0x00000000)   // 3.4028236692093846e+38
    },
    // Smallest value that overflows to infinity, without mantissa rounding.
    {
        F64VAL(0b10001111111, 0x000000, 0x00000000),  // 3.4028236692093846e+38
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b10001111111, 0x000000, 0x00000000)   // 3.4028236692093846e+38
    },
    // Smallest value that overflows to infinity due to mantissa rounding up,
    // even when exponent bits aren't reduced.
    {
        F64VAL(0b11111111110, 0x7fffff, 0x10000000),  // 1.7976930812868855e+308
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b11111111111, 0x000000, 0x00000000)   // Inf
    },
    // True infinity.
    {
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
        F64VAL(0b11111111111, 0x000000, 0x00000000),  // Inf
    },
    // NAN with a 1 in the preserved bits.
    {
        F64VAL(0b11111111111, 0x800000, 0x00000000),  // -0
        F64VAL(0b11111111111, 0x800000, 0x00000000),  // -0
        F64VAL(0b11111111111, 0x800000, 0x00000000),  // -0
        F64VAL(0b11111111111, 0x800000, 0x00000000),  // -0
    },
    // NAN with a 1 in the truncated bits.
    {
        F64VAL(0b11111111111, 0x000000, 0x00000001),  // NaN
        F64VAL(0b11111111111, 0x000000, 0x00000001),  // NaN
        F64VAL(0b11111111111, 0x000000, 0x00000001),  // NaN
        F64VAL(0b11111111111, 0x000000, 0x00000001),  // NaN
    },
    // NAN with all ones, causing rounding overflow.
    {
        F64VAL(0b11111111111, 0x7fffff, 0x1fffffff),  // NaN
        F64VAL(0b11111111111, 0x7fffff, 0x1fffffff),  // NaN
        F64VAL(0b11111111111, 0x7fffff, 0x1fffffff),  // NaN
        F64VAL(0b11111111111, 0x7fffff, 0x1fffffff),  // NaN
    },
};

class ReducedPrecisionAccuracyTest
    : public ClientLibraryTestRunnerMixin<HloTestBase>,
      public ::testing::WithParamInterface<int> {
 protected:
  template <typename Fp, typename Uint, int kNumTestcases, int kNumInputs>
  void DoIt(int exponent_bits, int mantissa_bits,
            const Uint (&test_values)[kNumInputs][kNumTestcases],
            int operation_index);
};

XLA_TEST_P(ReducedPrecisionAccuracyTest, ReducePrecisionHalf) {
  int operation_index = GetParam();
  DoIt<Eigen::half, uint16_t>(f16_exponent_sizes[operation_index],
                              f16_mantissa_sizes[operation_index],
                              f16_test_values, operation_index);
}

XLA_TEST_P(ReducedPrecisionAccuracyTest, ReducePrecisionBfloat16) {
  int operation_index = GetParam();
  DoIt<bfloat16, uint16_t>(bf16_exponent_sizes[operation_index],
                           bf16_mantissa_sizes[operation_index],
                           bf16_test_values, operation_index);
}

XLA_TEST_P(ReducedPrecisionAccuracyTest, ReducePrecisionFloat) {
  int operation_index = GetParam();
  DoIt<float, uint32_t>(f32_exponent_sizes[operation_index],
                        f32_mantissa_sizes[operation_index], f32_test_values,
                        operation_index);
}

XLA_TEST_P(ReducedPrecisionAccuracyTest,
           DISABLED_ON_TPU(ReducePrecisionDouble)) {
  int operation_index = GetParam();
  DoIt<double, uint64_t>(f64_exponent_sizes[operation_index],
                         f64_mantissa_sizes[operation_index], f64_test_values,
                         operation_index);
}

template <typename Fp, typename Uint, int kNumTestcases, int kNumInputs>
void ReducedPrecisionAccuracyTest::DoIt(
    int exponent_bits, int mantissa_bits,
    const Uint (&test_values)[kNumInputs][kNumTestcases], int operation_index) {
  SCOPED_TRACE(absl::StrFormat("operation_index %d", operation_index));
  SCOPED_TRACE(absl::StrFormat("%d exponent bits, %d mantissa bits",
                               exponent_bits, mantissa_bits));

  std::vector<Fp> input_values;
  std::vector<Fp> expected_values;

  const Uint sign_bit = Uint{1} << (sizeof(Fp) * 8 - 1);
  for (const auto& test_value : test_values) {
    // Add positive values.
    input_values.push_back(absl::bit_cast<Fp>(test_value[0]));
    // Add negative values.  We do this in the bitwise representation so as to
    // avoid problems with NaN handling.
    input_values.push_back(absl::bit_cast<Fp, Uint>(test_value[0] ^ sign_bit));
  }

  XlaBuilder builder(TestName());

  Literal a_literal = LiteralUtil::CreateR1<Fp>({input_values});
  auto a = Parameter(&builder, 0, a_literal.shape(), "a");

  ReducePrecision(a, exponent_bits, mantissa_bits);

  ComputeAndCompare(&builder, {&a_literal});
}

INSTANTIATE_TEST_CASE_P(ReducedPrecisionAccuracyTest,
                        ReducedPrecisionAccuracyTest, ::testing::Range(0, 4));

}  // namespace
}  // namespace xla
