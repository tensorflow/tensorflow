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

#ifndef XLA_EF57_H_
#define XLA_EF57_H_

#include <cmath>
#include <utility>

#include "absl/types/span.h"

namespace xla {

// Utility function to split a double-precision float (F64) into a pair of F32s.
// For a p-bit number, and a splitting point (p/2) <= s <= (p - 1), the
// algorithm produces a (p - s)-bit value 'hi' and a non-overlapping (s - 1)-bit
// value 'lo'. See Theorem 4 in [1] (attributed to Dekker) or [2] for the
// original theorem by Dekker.
//
// For double-precision F64s, which contain a 53 bit mantissa (52 of them
// explicit), we can represent the most significant 49 digits as the unevaluated
// sum of two single-precision floats 'hi' and 'lo'. The 'hi' float stores the
// most significant 24 bits and the sign bit of 'lo' together with its mantissa
// store the remaining 25 bits. The exponent of the resulting representation is
// still restricted to 8 bits of F32.
//
// References:
// [1] A. Thall, Extended-Precision Floating-Point Numbers for GPU Computation,
//     SIGGRAPH Research Posters, 2006.
//     (http://andrewthall.org/papers/df64_qf128.pdf)
// [2] T. J. Dekker, A floating point technique for extending the available
//     precision, Numerische Mathematik, vol. 18, pp. 224–242, 1971.
inline std::pair<float, float> SplitF64ToF32(double x) {
  const float x_f32 = static_cast<float>(x);

  const bool result_is_finite = std::isfinite(x_f32);

  // The high float is simply the double rounded to the nearest float. Because
  // we are rounding to nearest with ties to even, the error introduced in
  // rounding is less than half an ULP in the high ULP.
  const float hi = x_f32;
  // We can compute the low term using Sterbenz' lemma: If a and b are two
  // positive floating point numbers and a/2 ≤ b ≤ 2a, then their difference can
  // be computed exactly.
  // Note: the difference is computed exactly but is rounded to the nearest
  // float which will introduce additional error.
  const float lo = static_cast<float>(x - static_cast<double>(hi));
  return std::make_pair(hi, result_is_finite ? lo : 0.0f);
}
void ConvertF64ToEf57(absl::Span<const double> input, absl::Span<float> output);

}  // namespace xla

#endif  // XLA_EF57_H_
