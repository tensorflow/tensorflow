/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_LIB_MATH_MATH_UTIL_H_
#define TENSORFLOW_TSL_LIB_MATH_MATH_UTIL_H_

#include <limits>

#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/types.h"

namespace tsl {

class MathUtil {
 public:
  // ----------------------------------------------------------------------
  // CeilOfRatio<IntegralType>
  // FloorOfRatio<IntegralType>
  //   Returns the ceil (resp. floor) of the ratio of two integers.
  //
  //  * IntegralType: any integral type, whether signed or not.
  //  * numerator: any integer: positive, negative, or zero.
  //  * denominator: a non-zero integer, positive or negative.
  //
  // This implementation is correct, meaning there is never any precision loss,
  // and there is never an overflow. However, if the type is signed, having
  // numerator == MathLimits<IntegralType>::kMin and denominator == -1 is not a
  // valid input, because kMin has a greater absolute value than kMax.
  //
  // Input validity is DCHECKed. When not in debug mode, invalid inputs raise
  // SIGFPE.
  //
  // This method has been designed and tested so that it should always be
  // preferred to alternatives. Indeed, there exist popular recipes to compute
  // the result, such as casting to double, but they are in general incorrect.
  // In cases where an alternative technique is correct, performance measurement
  // showed the provided implementation is faster.
  template <typename IntegralType>
  static IntegralType CeilOfRatio(IntegralType numerator,
                                  IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
  }
  template <typename IntegralType>
  static IntegralType FloorOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
  }

  template <typename IntegralType, bool ceil>
  static IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                         IntegralType denominator);

  template <typename IntegralType>
  static IntegralType GCD(IntegralType x, IntegralType y);

  // ----------------------------------------------------------------------
  // IPow<T>
  //   Computes the result of raising a number to a non-negative integral power.
  //
  //  * T: An integral type, floating-point type, or user-defined type for which
  //    operator*= is defined.
  //  * base: the base "v" of the operation
  //  * exp: the exponent "i" of the operation; must be non-negative.
  //
  // Computes v^i, in a way that is faster than std::pow (which supports
  // arbitrary real exponents).
  //
  // When T is a floating point type, this has the same semantics as std::pow,
  // but it is much faster. When T is an integral type, computations are
  // performed in the value domain of T, and overflow semantics are those of T.
  //
  // Input validity is DCHECKed.
  template <typename T>
  static T IPow(T base, int exp);
};

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64 into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template <typename IntegralType, bool ceil>
IntegralType MathUtil::CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  DCHECK_NE(0, denominator) << "Division by zero is not supported.";

  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  if (ceil) {  // Compile-time condition: not an actual branching
    // When rounded_toward_zero is negative, then an adjustment is never needed:
    // the real ratio is negative, and so rounded toward zero is the ceil.
    // When rounded_toward_zero is non-negative, an adjustment is needed if the
    // sign of the difference numerator - intermediate_product is the same as
    // the sign of the denominator.
    //
    //
    // Using a bool and then a static_cast to IntegralType is not strictly
    // necessary, but it makes the code clear, and anyway the compiler should
    // get rid of it.
    const bool needs_adjustment =
        (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
         (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    // Floor case: symmetrical to the previous one
    const bool needs_adjustment =
        (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}

template <typename IntegralType>
IntegralType MathUtil::GCD(IntegralType x, IntegralType y) {
  static_assert(std::is_unsigned<IntegralType>::value,
                "signed GCD not supported!");
  while (y != 0) {
    IntegralType r = x % y;
    x = y;
    y = r;
  }
  return x;
}

// ---- IPow ----
// Implemented with the squared exponentiation method (a.k.a. double-and-add).
//
// Note that "exp >>= 1" is faster than "exp /= 2" on at least one platform.
template <typename T>
T MathUtil::IPow(T base, int exp) {
  DCHECK_GE(exp, 0);
  for (T result(1);; base *= base) {
    if ((exp & 1) != 0) result *= base;
    exp >>= 1;
    if (exp == 0) return result;
  }
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_MATH_MATH_UTIL_H_
