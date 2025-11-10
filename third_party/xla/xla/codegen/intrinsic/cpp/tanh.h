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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_TANH_H_
#define XLA_CODEGEN_INTRINSIC_CPP_TANH_H_

#include <array>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla {
namespace codegen {

// WARNING: This file exists right now purely as a proof-of-concept showing how
// to hand-code portable llvm ir intrinsics using C++.

template <typename T>
T FastTanhf(T x) {
  T abs_x = BitwiseAbs<T>(x);
  T large_result = BitwiseCopysign<T>(T{1.0f}, x);
  auto is_large = abs_x >= T{20.0f};

  // For small inputs, tanh(x) is approximately x.
  constexpr float kSmallValueThreshold = 0.0004f;
  auto is_small = abs_x < kSmallValueThreshold;

  constexpr float kPlusClamp = 7.99881172180175781f;
  T clamped_x = Clamp(x, -kPlusClamp, kPlusClamp);

  static constexpr std::array<float, 7> kNumeratorCoeffs = {
      -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
      5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
      4.89352455891786e-03f};
  static constexpr std::array<float, 4> kDenominatorCoeffs = {
      1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
      4.89352518554385e-03f};

  // Evaluate the polynomial using Horner's method.
  T x2 = clamped_x * clamped_x;
  T numerator = T{kNumeratorCoeffs[0]};
  for (int i = 1; i < 7; ++i) {
    numerator = numerator * x2 + T{kNumeratorCoeffs[i]};
  }
  numerator = clamped_x * numerator;

  T denominator = T{kDenominatorCoeffs[0]};
  for (int i = 1; i < 4; ++i) {
    denominator = denominator * x2 + T{kDenominatorCoeffs[i]};
  }

  T poly_result = numerator / denominator;

  // The ternary operator `?:` compiles to a branchless `select` instruction.
  T small_or_poly = is_small ? x : poly_result;
  return is_large ? large_result : small_or_poly;
}

}  // namespace codegen
}  // namespace xla

#endif  // XLA_CODEGEN_INTRINSIC_CPP_TANH_H_
