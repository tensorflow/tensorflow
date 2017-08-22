/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"

namespace tensorflow {
namespace ops {
namespace {

// Logical operations have no gradients.
REGISTER_NO_GRADIENT_OP("Less");
REGISTER_NO_GRADIENT_OP("LessEqual");
REGISTER_NO_GRADIENT_OP("Greater");
REGISTER_NO_GRADIENT_OP("GreaterEqual");
REGISTER_NO_GRADIENT_OP("Equal");
REGISTER_NO_GRADIENT_OP("ApproximateEqual");
REGISTER_NO_GRADIENT_OP("NotEqual");
REGISTER_NO_GRADIENT_OP("LogicalAnd");
REGISTER_NO_GRADIENT_OP("LogicalOr");
REGISTER_NO_GRADIENT_OP("LogicalNot");

// Conjugate helper function returns the conjugate of an Output if it
// is complex valued.
Output ConjugateHelper(const Scope& scope, const Output& out) {
  DataType dtype = out.type();
  if (dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128) {
    return Conj(scope, out);
  } else {
    return out;
  }
}

// TODO(andydavis) Add control dependencies to gradient functions (as needed).

Status AbsGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // dx = dy * sign(x)
  grad_outputs->push_back(Mul(scope, grad_inputs[0], Sign(scope, op.input(0))));
  return scope.status();
}
REGISTER_GRADIENT_OP("Abs", AbsGrad);

Status NegGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // dx = -dy;
  grad_outputs->push_back(Neg(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Neg", NegGrad);

Status InvGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // Use the built-in operator.
  grad_outputs->push_back(
      internal::ReciprocalGrad(scope, op.output(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Inv", InvGrad);
REGISTER_GRADIENT_OP("Reciprocal", InvGrad);

Status SquareGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  // dy/dx = (2 * x)
  auto two = Cast(scope, Const(scope, 2), op.input(0).type());
  auto dydx = Mul(scope, two, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Square", SquareGrad);

Status SqrtGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // Use the built-in operator.
  grad_outputs->push_back(
      internal::SqrtGrad(scope, op.output(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sqrt", SqrtGrad);

Status RsqrtGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // Use the built-in operator.
  grad_outputs->push_back(
      internal::RsqrtGrad(scope, op.output(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Rsqrt", RsqrtGrad);

Status ExpGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // dy/dx = exp(x) = y
  // grad(x) = grad(y) * conj(dy/dx)
  //         = grad(y) * conj(y)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, op.output(0))));
  return scope.status();
}
REGISTER_GRADIENT_OP("Exp", ExpGrad);

Status Expm1Grad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = expm1(x)
  // dy/dx = exp(x)
  auto dydx = Exp(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Expm1", Expm1Grad);

Status LogGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // y = log(x)
  // dy/dx = 1 / x
  auto dydx = Reciprocal(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Log", LogGrad);

Status Log1pGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = log1p(x)
  // dy/dx = 1 / (1 + x)
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Add(scope, one, op.input(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Log1p", Log1pGrad);

Status SinhGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // y = sinh(x)
  // dy/dx = cosh(x)
  auto dydx = Cosh(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sinh", SinhGrad);

Status CoshGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // y = cosh(x)
  // dy/dx = sinh(x)
  auto dydx = Sinh(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Cosh", CoshGrad);

Status TanhGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // Use the built-in operator.
  // Note that the built-in operator does not return the conjugate of
  // the gradient.
  auto grad = grad_inputs[0];
  // Optimization to avoid calculating conj(y) until the gradient is
  // evaluated.
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto y = ConjugateHelper(grad_scope, op.output(0));
  grad_outputs->push_back(internal::TanhGrad(scope, y, grad));
  return scope.status();
}
REGISTER_GRADIENT_OP("Tanh", TanhGrad);

Status AsinhGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = asinh(x)
  // dy/dx = 1 / cosh(y)
  auto dydx = Reciprocal(scope, Cosh(scope, op.output(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Asinh", AsinhGrad);

Status AcoshGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = acosh(x)
  // dy/dx = 1 / sinh(y)
  auto dydx = Reciprocal(scope, Sinh(scope, op.output(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Acosh", AcoshGrad);

Status AtanhGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = atanh(x)
  // dy/dx = 1 / (1 - x^2)
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Sub(scope, one, Square(scope, op.input(0))));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Atanh", AtanhGrad);

Status SigmoidGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  // Use the built-in operator.
  // Note that the built-in operator does not return the conjugate of
  // the gradient.
  auto grad = grad_inputs[0];
  // Optimization to avoid calculating conj(y) until the gradient is
  // evaluated.
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto y = ConjugateHelper(grad_scope, op.output(0));
  grad_outputs->push_back(internal::SigmoidGrad(scope, y, grad));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sigmoid", SigmoidGrad);

Status SignGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  auto shape = Shape(scope, op.input(0));
  auto zero = Cast(scope, Const(scope, 0.0), op.input(0).type());
  auto dx = Fill(scope, shape, zero);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Sign", SignGrad);

Status SinGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // y = sin(x)
  // dy/dx = cos(x)
  auto dydx = Cos(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sin", SinGrad);

Status CosGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // y = cos(x)
  // dy/dx = -sin(x)
  auto dydx = Neg(scope, Sin(scope, op.input(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Cos", CosGrad);

Status AsinGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // y = asin(x)
  // dy/dx = 1 / sqrt(1 - x^2)
  auto x2 = Square(scope, op.input(0));
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Sqrt(scope, Sub(scope, one, x2)));
  // grad(x) = grad(y) * conj(dy/dx)
  auto dx = Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Asin", AsinGrad);

Status AcosGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // y = acos(x)
  // dy/dx = - 1 / (1 - x * x)^1/2
  // dx = dy * (- 1 / (1 - x * x)^1/2)
  auto x2 = Square(scope, op.input(0));
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Neg(scope, Reciprocal(scope, Sqrt(scope, Sub(scope, one, x2))));
  auto dx = Mul(scope, grad_inputs[0], dydx);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Acos", AcosGrad);

Status TanGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // y = tan(x)
  // dy/dx = sec(x)^2 = 1 / cos(x)^2
  auto dydx = Square(scope, Reciprocal(scope, Cos(scope, op.input(0))));
  // grad(x) = grad(y) * conj(dy/dx)
  auto dx = Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Tan", TanGrad);

Status AtanGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // y = arctan(x)
  // dy/dx = 1 / (1 + x^2)
  // dx = dy * (1 / (1 + x^2)
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Add(scope, one, Square(scope, op.input(0))));
  auto dx = Mul(scope, grad_inputs[0], dydx);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Atan", AtanGrad);

// BinaryGradCommon handles the setup for binary ops that broadcast
// their inputs.
Status BinaryGradCommon(const Scope& scope, const Operation& op,
                        std::vector<Output>* grad_outputs, const Output& gx_1,
                        const Output& gx_2) {
  auto sx_1 = Shape(scope, op.input(0));
  auto sx_2 = Shape(scope, op.input(1));
  auto rx = internal::BroadcastGradientArgs(scope, sx_1, sx_2);
  auto dx_1 = Reshape(scope, Sum(scope, gx_1, rx.r0), sx_1);
  auto dx_2 = Reshape(scope, Sum(scope, gx_2, rx.r1), sx_2);
  grad_outputs->push_back(dx_1);
  grad_outputs->push_back(dx_2);
  return scope.status();
}

Status AddGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // y = x_1 + x_2
  // dy/dx_1 = dy/dx_2 = 1
  auto gx_1 = Identity(scope, grad_inputs[0]);
  auto gx_2 = Identity(scope, grad_inputs[0]);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Add", AddGrad);

Status SubGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  // y = x_1 - x_2
  // dy/dx_1 = 1
  // dy/dx_2 = -1
  auto gx_1 = Identity(scope, grad_inputs[0]);
  auto gx_2 = Neg(scope, grad_inputs[0]);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Sub", SubGrad);

Status MulGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 * x_2
  // dy/dx_1 = x_2
  // dy/dx_2 = x_1
  auto gx_1 = Mul(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0], x_1);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Mul", MulGrad);

Status DivGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 / x_2
  // dy/dx_1 = 1/x_2
  // dy/dx_2 = -x_1/x_2^2
  auto gx_1 = Div(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0],
                  Div(scope, Div(scope, Neg(scope, x_1), x_2), x_2));
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Div", DivGrad);

Status RealDivGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 / x_2
  // dy/dx_1 = 1/x_2
  // dy/dx_2 = -x_1/x_2^2
  auto gx_1 = RealDiv(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0],
                  RealDiv(scope, RealDiv(scope, Neg(scope, x_1), x_2), x_2));
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("RealDiv", RealDivGrad);

Status SquaredDifferenceGrad(const Scope& scope, const Operation& op,
                             const std::vector<Output>& grad_inputs,
                             std::vector<Output>* grad_outputs) {
  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = (x_1 - x_2)^2
  // dy/dx_1 = 2 * (x_1 - x_2)
  // dy/dx_2 = -2 * (x_1 - x_2)
  auto two = Cast(scope, Const(scope, 2), grad_inputs[0].type());
  auto gx_1 = Mul(scope, grad_inputs[0], Mul(scope, two, Sub(scope, x_1, x_2)));
  auto gx_2 = Neg(scope, gx_1);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("SquaredDifference", SquaredDifferenceGrad);

Status AddNGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // AddN doesn't support broadcasting, so all the inputs must be the
  // same shape.
  // Note:
  // dy/dx_k = d(x_1 + x_2 + ... + x_n)/dx_k = 1 for all x_k
  // hence dx_k = dy for all x_k
  // So the gradient for AddN just transfers the incoming gradient to
  // all outgoing gradients.
  auto incoming = Identity(scope, grad_inputs[0]);
  for (int32 i = 0; i < op.num_inputs(); ++i) {
    grad_outputs->push_back(incoming);
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("AddN", AddNGrad);

// MaximumMinimumGradCommon adds shared ops to calculate gradients for
// the binary Maximum and Minimum ops.
Status MaximumMinimumGradCommon(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs,
                                const Output& comparator) {
  // comparator is a boolean tensor, with
  // y = x_1 at points where comparator is true, and x_2 otherwise
  // Therefore
  // dy/dx_1 = 1 where comparator is true, and 0 otherwise.
  // dy/dx_2 = 0 where comparator is true, and 1 otherwise.
  auto grad = grad_inputs[0];
  auto zeros = ZerosLike(scope, grad);
  auto gx_1 = Where3(scope, comparator, grad, zeros);
  auto gx_2 = Where3(scope, LogicalNot(scope, comparator), grad, zeros);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}

Status MaximumGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  auto comparator = GreaterEqual(scope, op.input(0), op.input(1));
  return MaximumMinimumGradCommon(scope, op, grad_inputs, grad_outputs,
                                  comparator);
}
REGISTER_GRADIENT_OP("Maximum", MaximumGrad);

Status MinimumGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  auto comparator = LessEqual(scope, op.input(0), op.input(1));
  return MaximumMinimumGradCommon(scope, op, grad_inputs, grad_outputs,
                                  comparator);
}
REGISTER_GRADIENT_OP("Minimum", MinimumGrad);

Status RealGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  auto zero = Cast(scope, Const(scope, 0.0), op.output(0).type());
  auto dx = Complex(scope, grad_inputs[0], zero);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Real", RealGrad);

Status ImagGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  auto zero = Cast(scope, Const(scope, 0.0), op.output(0).type());
  auto dx = Complex(scope, zero, grad_inputs[0]);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Imag", ImagGrad);

Status AngleGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = Angle(x)
  // dx = -dy / (Im(x) + iRe(x)) = -dy * z
  auto re = Real(scope, op.input(0));
  auto im = Imag(scope, op.input(0));
  auto z_inv = Reciprocal(scope, Complex(scope, im, re));
  auto zero = Cast(scope, Const(scope, 0), grad_inputs[0].type());
  auto grad = Complex(scope, grad_inputs[0], zero);
  auto dx = Neg(scope, Mul(scope, grad, z_inv));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Angle", AngleGrad);

Status ConjGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Conj(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Conj", ConjGrad);

// MatMulGrad helper function used to compute two MatMul operations
// based on input matrix transposition combinations.
Status MatMulGradHelper(const Scope& scope, const bool is_batch,
                        const Output& x0, const bool adj_x0, const Output& x1,
                        const bool adj_x1, const Output& y0, const bool adj_y0,
                        const Output& y1, const bool adj_y1,
                        std::vector<Output>* grad_outputs) {
  if (is_batch == false) {
    auto dx =
        MatMul(scope, x0, x1, MatMul::TransposeA(adj_x0).TransposeB(adj_x1));
    grad_outputs->push_back(dx);
    auto dy =
        MatMul(scope, y0, y1, MatMul::TransposeA(adj_y0).TransposeB(adj_y1));
    grad_outputs->push_back(dy);
  } else {
    auto dx =
        BatchMatMul(scope, x0, x1, BatchMatMul::AdjX(adj_x0).AdjY(adj_x1));
    grad_outputs->push_back(dx);
    auto dy =
        BatchMatMul(scope, y0, y1, BatchMatMul::AdjX(adj_y0).AdjY(adj_y1));
    grad_outputs->push_back(dy);
  }
  return scope.status();
}

// MatMulGrad common used to read and check node attr state, and determine
// proper MatMul products for gradients based on input matrix transposition
// combinations.
// TODO(andydavis) Re-use this function for BatchMatMulGrad.
Status MatMulGradCommon(const Scope& scope, const Operation& op,
                        const bool is_batch,
                        const std::vector<Output>& grad_inputs,
                        const string& attr_adj_x, const string& attr_adj_y,
                        std::vector<Output>* grad_outputs) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.output(0).node()->attrs(), "T", &dtype));
  if (dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128) {
    return errors::Unimplemented(
        "MatMul gradient for complex data type is not supported yet.");
  }

  bool ta;
  bool tb;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), attr_adj_x, &ta));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), attr_adj_y, &tb));

  if (!ta && !tb) {
    return MatMulGradHelper(scope, is_batch, grad_inputs[0], false, op.input(1),
                            true, op.input(0), true, grad_inputs[0], false,
                            grad_outputs);
  } else if (!ta && tb) {
    return MatMulGradHelper(scope, is_batch, grad_inputs[0], false, op.input(1),
                            false, grad_inputs[0], true, op.input(0), false,
                            grad_outputs);
  } else if (ta && !tb) {
    return MatMulGradHelper(scope, is_batch, op.input(1), false, grad_inputs[0],
                            true, op.input(0), false, grad_inputs[0], false,
                            grad_outputs);
  }
  return MatMulGradHelper(scope, is_batch, op.input(1), true, grad_inputs[0],
                          true, grad_inputs[0], true, op.input(0), true,
                          grad_outputs);
}

Status MatMulGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  return MatMulGradCommon(scope, op, false, grad_inputs, "transpose_a",
                          "transpose_b", grad_outputs);
}
REGISTER_GRADIENT_OP("MatMul", MatMulGrad);

Status BatchMatMulGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  return MatMulGradCommon(scope, op, true, grad_inputs, "adj_x", "adj_y",
                          grad_outputs);
}
REGISTER_GRADIENT_OP("BatchMatMul", BatchMatMulGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
