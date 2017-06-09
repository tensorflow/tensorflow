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

#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"

namespace tensorflow {
namespace ops {
namespace {

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
  // dy/dx = -1/x^2 = -y^2
  auto dydx = Neg(scope, Square(scope, op.output(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
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
  // y = sqrt(x)
  // dy/dx =  0.5 * (1 / sqrt(x)) = 0.5 * (1 / y)
  auto y_inv = Reciprocal(scope, op.output(0));
  auto half = Cast(scope, Const(scope, 0.5), op.input(0).type());
  auto dydx = Mul(scope, half, y_inv);
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sqrt", SqrtGrad);

Status RsqrtGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  // y = 1/x^1/2 = x^-1/2
  // dy/dx = -1/2 * x^-3/2 = -1/2 * x^-1/2 * x^-1 = -1/2 * y * x^-1
  auto x_inv = Reciprocal(scope, op.input(0));
  auto y = op.output(0);
  auto neghalf = Cast(scope, Const(scope, -0.5), op.input(0).type());
  auto a = Mul(scope, neghalf, x_inv);
  auto dydx = Mul(scope, a, y);
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
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
  // y = tanh(x)
  // dy/dx = 1 - (tanh(x))^2 = 1 - y^2
  auto y2 = Square(scope, op.output(0));
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Sub(scope, one, y2);
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Tanh", TanhGrad);

Status SigmoidGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  // y = 1 / (1 + exp(-x))
  // dy/dx = y * (1 - y)
  auto y = op.output(0);
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Mul(scope, y, Sub(scope, one, y));
  // dx = dy * y * (1 - y)
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
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
