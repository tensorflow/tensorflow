/* Copyright 2024 The OpenXLA Authors.

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

// This file is generated using functional_algorithms tool (0.3.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.

#ifndef XLA_HLO_BUILDER_LIB_MATH_IMPL_H_
#define XLA_HLO_BUILDER_LIB_MATH_IMPL_H_

#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"

namespace xla {
namespace math_impl {
// NOLINTBEGIN(whitespace/line_length)
// clang-format off

// Arcus sine on complex input.
//
//     Here we well use a modified version of the [Hull et
//     al]((https://dl.acm.org/doi/10.1145/275323.275324) algorithm with
//     a reduced number of approximation regions.
//
//     Hull et al define complex arcus sine as
//
//       arcsin(x + I*y) = arcsin(x/a) + sign(y; x) * I * log(a + sqrt(a*a-1))
//
//     where
//
//       x and y are real and imaginary parts of the input to arcsin, and
//       I is imaginary unit,
//       a = (hypot(x+1, y) + hypot(x-1, y))/2,
//       sign(y; x) = 1 when y >= 0 and abs(x) <= 1, otherwise -1.
//
//     x and y are assumed to be non-negative as the arcus sine on other
//     quadrants of the complex plane are defined by
//
//       arcsin(-z) == -arcsin(z)
//       arcsin(conj(z)) == conj(arcsin(z))
//
//     where z = x + I*y.
//
//     Hull et al split the first quadrant into 11 regions in total, each
//     region using a different approximation of the arcus sine
//     function. It turns out that when considering the evaluation of
//     arcus sine real and imaginary parts separately, the 11 regions can
//     be reduced to 3 regions for the real part, and to 4 regions for
//     the imaginary part. This reduction of the approximation regions
//     constitutes the modification of the Hull et al algorithm that is
//     implemented below and it is advantageous for functional
//     implementations as there will be less branches. The modified Hull
//     et al algorithm is validated against the original Hull algorithm
//     implemented in MPMath.
//
//     Per Hull et al Sec. "Analyzing Errors", in the following we'll use
//     symbol ~ (tilde) to denote "approximately equal" relation with the
//     following meaning:
//
//       A ~ B  iff  A = B * (1 + s * eps)
//
//     where s * eps is a small multiple of eps that quantification
//     depends on the particular case of error analysis.
//     To put it simply, A ~ B means that the numerical values of A and B
//     within the given floating point system are equal or very
//     close. So, from the numerical evaluation point of view it does not
//     matter which of the expressions, A or B, to use as the numerical
//     results will be the same.
//
//     We define:
//       safe_min = sqrt(<smallest normal value>) * 4
//       safe_max = sqrt(<largest finite value>) / 8
//
//     Real part
//     ---------
//     In general, the real part of arcus sine input can be expressed as
//     follows:
//
//       arcsin(x / a) = arctan((x/a) / sqrt(1 - (x/a)**2))
//                     = arctan(x / sqrt(a**2 - x**2))
//                     = arctan2(x, sqrt(a**2 - x**2))              Eq. 1
//                     = arctan2(x, sqrt((a + x) * (a - x)))        Eq. 2
//
//     for which the following approximations will be used (in the
//     missing derivation cases, see Hull et al paper for details):
//
//     - Hull et al Case 5:
//       For x > safe_max and any y, we have
//         x + 1 ~ x - 1 ~ x
//       so that
//         a ~ hypot(x, y)
//       For y > safe_max and x < safe_max, we have
//         hypot(x + 1, y) ~ hypot(x - 1, y) ~ hypot(x, y) ~ a.
//       Combining these together gives: if max(x, y) > safe_max then
//         a**2 ~ hypot(x, y)**2 ~ x**2 + y**2
//       and Eq. 1 becomes
//         arcsin(x / a) ~ arctan2(x, y)
//
//     - Hull et al Safe region: for max(x, y) < safe_max, we have (see
//       `a - x` approximation in Hull et al Fig. 2):
//
//       If x <= 1 then
//         arcsin(x / a) ~ arctan2(x, sqrt(0.5 * (a + x) * (y * y / (hypot(x + 1, y) + x + 1) + hypot(x - 1, y) - x - 1)))
//       else
//         arcsin(x / a) ~ arctan2(x, y * sqrt(0.5 * (a + x) * (1 / (hypot(x + 1, y) + x + 1) + 1 / (hypot(x - 1, y) + x - 1))))
//
//     Imaginary part
//     --------------
//     In general, the unsigned imaginary part of arcus sine input can be
//     expressed as follows:
//
//       log(a + sqrt(a*a-1)) = log(a + sqrt((a + 1) * (a - 1)))
//                            = log1p(a - 1 + sqrt((a + 1) * (a - 1)))   # Eq.3
//
//     for which the following approximations will be used (for the
//     derivation, see Hull et al paper):
//
//     - modified Hull et al Case 5: for y > safe_max_opt we have
//         log(a + sqrt(a*a-1)) ~ log(2) + log(y) + 0.5 * log1p((x / y) * (x / y))
//       where using
//         safe_max_opt = safe_max * 1e-6 if x < safe_max * 1e12 else safe_max * 1e2
//       will expand the approximation region to capture also the Hull et
//       Case 4 (x is large but less that eps * y) that does not have
//       log1p term but under the Case 4 conditions, log(y) +
//       0.5*log1p(...) ~ log(y).
//
//     - Hull et al Case 1 & 2: for 0 <= y < safe_min and x < 1, we have
//         log(a + sqrt(a*a-1)) ~ y / sqrt((a - 1) * (a + 1))
//       where
//         a - 1 ~ -(x + 1) * (x - 1) / (a + 1)
//
//     - Hull et al Safe region. See the approximation of `a -
//       1` in Hull et al Fig. 2 for Eq. 3:
//         log(a + sqrt(a*a-1)) ~ log1p(a - 1 + sqrt((a + 1) * (a - 1)))
//       where
//         a - 1 ~ 0.5 * y * y / (hypot(x + 1, y) + x + 1) + 0.5 * (hypot(x - 1, y) + x - 1)        if x >= 1
//         a - 1 ~ 0.5 * y * y * (1 / (hypot(x + 1, y) + x + 1) + 1 / (hypot(x - 1, y) - x - 1))    if x < 1 and a < 1.5
//         a - 1 ~ a - 1                                                                            otherwise
//
//     Different from Hull et al, we don't handle Cases 3 and 6 because
//     these only minimize the number of operations which may be
//     advantageous for procedural implementations but for functional
//     implementations these would just increase the number of branches
//     with no gain in accuracy.
//
//
template <typename FloatType>
XlaOp AsinComplex(XlaOp z) {
  XlaOp signed_x = Real(z);
  XlaOp x = Abs(signed_x);
  XlaOp signed_y = Imag(z);
  XlaOp y = Abs(signed_y);
  FloatType safe_max_ =
      (std::sqrt(std::numeric_limits<FloatType>::max())) / (8);
  XlaOp safe_max = ScalarLike(signed_x, safe_max_);
  XlaOp one = ScalarLike(signed_x, 1);
  XlaOp half = ScalarLike(signed_x, 0.5);
  XlaOp xp1 = Add(x, one);
  XlaOp abs_xp1 = Abs(xp1);
  XlaOp _hypot_1_mx = Max(abs_xp1, y);
  XlaOp mn = Min(abs_xp1, y);
  FloatType two_ = 2;
  XlaOp sqrt_two = ScalarLike(signed_x, std::sqrt(two_));
  XlaOp _hypot_1_r = Square(Div(mn, _hypot_1_mx));
  XlaOp sqa = Sqrt(Add(one, _hypot_1_r));
  XlaOp zero = ScalarLike(signed_x, 0);
  XlaOp two = ScalarLike(signed_x, two_);
  XlaOp r =
      Select(Eq(_hypot_1_mx, mn), Mul(sqrt_two, _hypot_1_mx),
             Select(And(Eq(sqa, one), Gt(_hypot_1_r, zero)),
                    Add(_hypot_1_mx, Div(Mul(_hypot_1_mx, _hypot_1_r), two)),
                    Mul(_hypot_1_mx, sqa)));
  XlaOp xm1 = Sub(x, one);
  XlaOp abs_xm1 = Abs(xm1);
  XlaOp _hypot_2_mx = Max(abs_xm1, y);
  XlaOp _hypot_2_mn = Min(abs_xm1, y);
  XlaOp _hypot_2_r = Square(Div(_hypot_2_mn, _hypot_2_mx));
  XlaOp _hypot_2_sqa = Sqrt(Add(one, _hypot_2_r));
  XlaOp s =
      Select(Eq(_hypot_2_mx, _hypot_2_mn), Mul(sqrt_two, _hypot_2_mx),
             Select(And(Eq(_hypot_2_sqa, one), Gt(_hypot_2_r, zero)),
                    Add(_hypot_2_mx, Div(Mul(_hypot_2_mx, _hypot_2_r), two)),
                    Mul(_hypot_2_mx, _hypot_2_sqa)));
  XlaOp a = Mul(half, Add(r, s));
  XlaOp half_apx = Mul(half, Add(a, x));
  XlaOp yy = Mul(y, y);
  XlaOp rpxp1 = Add(r, xp1);
  XlaOp smxm1 = Sub(s, xm1);
  XlaOp spxm1 = Add(s, xm1);
  XlaOp real = Atan2(
      signed_x,
      Select(Ge(Max(x, y), safe_max), y,
             Select(Le(x, one), Sqrt(Mul(half_apx, Add(Div(yy, rpxp1), smxm1))),
                    Mul(y, Sqrt(Add(Div(half_apx, rpxp1),
                                    Div(half_apx, spxm1)))))));
  XlaOp safe_max_opt =
      Select(Lt(x, ScalarLike(signed_x, (safe_max_) * (1000000000000.0))),
             ScalarLike(signed_x, (safe_max_) * (1e-06)),
             ScalarLike(signed_x, (safe_max_) * (100.0)));
  XlaOp y_gt_safe_max_opt = Ge(y, safe_max_opt);
  XlaOp mx = Select(y_gt_safe_max_opt, y, x);
  XlaOp xoy = Select(
      And(y_gt_safe_max_opt,
          Not(Eq(y, ScalarLike(signed_y,
                               std::numeric_limits<FloatType>::infinity())))),
      Div(x, y), zero);
  XlaOp logical_and_lt_y_safe_min_lt_x_one = And(
      Lt(y,
         ScalarLike(signed_x,
                    (std::sqrt(std::numeric_limits<FloatType>::min())) * (4))),
      Lt(x, one));
  XlaOp ap1 = Add(a, one);
  XlaOp half_yy = Mul(half, yy);
  XlaOp divide_half_yy_rpxp1 = Div(half_yy, rpxp1);
  XlaOp x_ge_1_or_not = Select(
      Ge(x, one), Add(divide_half_yy_rpxp1, Mul(half, spxm1)),
      Select(Le(a, ScalarLike(signed_x, 1.5)),
             Add(divide_half_yy_rpxp1, Div(half_yy, smxm1)), Sub(a, one)));
  XlaOp am1 = Select(logical_and_lt_y_safe_min_lt_x_one,
                     Neg(Div(Mul(xp1, xm1), ap1)), x_ge_1_or_not);
  XlaOp sq = Sqrt(Mul(am1, ap1));
  XlaOp imag = Select(Ge(mx, Select(y_gt_safe_max_opt, safe_max_opt, safe_max)),
                      Add(Add(ScalarLike(signed_x, std::log(two_)), Log(mx)),
                          Mul(half, Log1p(Mul(xoy, xoy)))),
                      Select(logical_and_lt_y_safe_min_lt_x_one, Div(y, sq),
                             Log1p(Add(am1, sq))));
  return Complex(real, Select(Lt(signed_y, zero), Neg(imag), imag));
}

// Arcus sine on real input:
//
//     arcsin(x) = 2 * arctan2(x, (1 + sqrt(1 - x * x)))
//
template <typename FloatType>
XlaOp AsinReal(XlaOp x) {
  XlaOp one = ScalarLike(x, 1);
  return Mul(ScalarLike(x, 2),
             Atan2(x, Add(one, Sqrt(Mul(Sub(one, x), Add(one, x))))));
}

// clang-format on
// NOLINTEND(whitespace/line_length)
}  // namespace math_impl
}  // namespace xla

#endif  // XLA_HLO_BUILDER_LIB_MATH_IMPL_H_
