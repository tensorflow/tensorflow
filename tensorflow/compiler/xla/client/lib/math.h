/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_MATH_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_MATH_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

// Determines whether operand is +/-inf or nan.
//
// Raises an error if called on integral or complex values.
XlaOp IsPosInf(XlaOp operand);
XlaOp IsNegInf(XlaOp operand);
XlaOp IsInf(XlaOp operand);
XlaOp IsNan(XlaOp operand);

// Returns the next number after 'from' in the direction of 'to' the same way
// std::nextafter(from, to) would.
XlaOp NextAfter(XlaOp from, XlaOp to);

// Computes the square root of 'operand'.
XlaOp Sqrt(XlaOp operand);

// Computes the reciprocal of the square root of 'operand'.
XlaOp Rsqrt(XlaOp operand);

// Computes the square of 'operand'.
XlaOp Square(XlaOp operand);

// Computes the reciprocal of 'operand'.
XlaOp Reciprocal(XlaOp operand);

// Evaluates a polynomial given coefficients and 'x'.
// N.B. Coefficients should be supplied in decreasing order.
XlaOp EvaluatePolynomial(XlaOp x, absl::Span<const float> coefficients);

// Computes an approximation of the error function complement (1 - erf(x)).
XlaOp Erfc(XlaOp x);

// Computes an approximation of the error function.
XlaOp Erf(XlaOp x);

// Computes an approximation of the inverse of the error function.
XlaOp ErfInv(XlaOp x);

// Computes an approximation of the lgamma function.
XlaOp Lgamma(XlaOp input);

// Computes an approximation of the digamma function.
XlaOp Digamma(XlaOp input);

// Rounds the given number to even when the number is equidistant between two
// integers.
XlaOp RoundToEven(XlaOp x);

// Trigonometric functions

// Computes the arc cosine of 'x'.
XlaOp Acos(XlaOp x);

// Computes the arc sine of 'x'.
XlaOp Asin(XlaOp x);

// Computes the arc tangent of 'x'.
XlaOp Atan(XlaOp x);

// Computes the tangent of 'x'.
XlaOp Tan(XlaOp x);

// Hyperbolic trigonometric functions

// Computes the inverse hyperbolic cosine of 'x'.
XlaOp Acosh(XlaOp x);

// Computes the inverse hyperbolic sine of 'x'.
XlaOp Asinh(XlaOp x);

// Computes the inverse hyperbolic tangent of 'x'.
XlaOp Atanh(XlaOp x);

// Computes the hyperbolic cosine of 'x'.
XlaOp Cosh(XlaOp x);

// Computes the hyperbolic sine of 'x'.
XlaOp Sinh(XlaOp x);

// Applies a complex conjugation operation if 'a' is complex and 'conjugate'
// is true, otherwise returns its argument.
xla::XlaOp MaybeConjugate(xla::XlaOp x, bool conjugate);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_MATH_H_
