/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Native XLA implementations of simple unary Ops

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

#define XLAJIT_MAKE_UNARY(NAME, COMPUTATION)                           \
  class NAME##Op : public XlaOpKernel {                                \
   public:                                                             \
    explicit NAME##Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {} \
    void Compile(XlaOpKernelContext* ctx) {                            \
      xla::XlaBuilder* b = ctx->builder();                             \
      (void)b;                                                         \
      xla::XlaOp x = ctx->Input(0);                                    \
      xla::XlaOp y = COMPUTATION;                                      \
      ctx->SetOutput(0, y);                                            \
    }                                                                  \
  };                                                                   \
  REGISTER_XLA_OP(Name(#NAME), NAME##Op);

XLAJIT_MAKE_UNARY(ComplexAbs, xla::Abs(x));

XLAJIT_MAKE_UNARY(Angle, xla::Atan2(xla::Imag(x), xla::Real(x)));

XLAJIT_MAKE_UNARY(Conj, xla::Conj(x));

// Return x if x>0, otherwise -x.
XLAJIT_MAKE_UNARY(Abs, xla::Abs(x));
XLAJIT_MAKE_UNARY(Acos, xla::Acos(x));
XLAJIT_MAKE_UNARY(Acosh, xla::Acosh(x));
XLAJIT_MAKE_UNARY(Asin, xla::Asin(x))
XLAJIT_MAKE_UNARY(Asinh, xla::Asinh(x));
XLAJIT_MAKE_UNARY(Atan, xla::Atan(x));
XLAJIT_MAKE_UNARY(Atanh, xla::Atanh(x));
XLAJIT_MAKE_UNARY(Ceil, xla::Ceil(x));
XLAJIT_MAKE_UNARY(Cos, xla::Cos(x));
XLAJIT_MAKE_UNARY(Cosh, xla::Cosh(x));
XLAJIT_MAKE_UNARY(Sin, xla::Sin(x));
XLAJIT_MAKE_UNARY(Exp, xla::Exp(x));
XLAJIT_MAKE_UNARY(Expm1, xla::Expm1(x));
XLAJIT_MAKE_UNARY(Floor, xla::Floor(x));
XLAJIT_MAKE_UNARY(IsFinite, xla::IsFinite(x));
XLAJIT_MAKE_UNARY(IsInf, xla::IsInf(x));
XLAJIT_MAKE_UNARY(IsNan, xla::IsNan(x));
// Return 1/x
XLAJIT_MAKE_UNARY(Inv, xla::ScalarLike(x, 1.0) / x);
XLAJIT_MAKE_UNARY(Reciprocal, xla::ScalarLike(x, 1.0) / x);
XLAJIT_MAKE_UNARY(Log, xla::Log(x));
XLAJIT_MAKE_UNARY(Log1p, xla::Log1p(x));

XLAJIT_MAKE_UNARY(Invert, xla::Not(x));
XLAJIT_MAKE_UNARY(LogicalNot, xla::Not(x));
XLAJIT_MAKE_UNARY(Neg, -x);

XLAJIT_MAKE_UNARY(Rint, xla::RoundToEven(x));
XLAJIT_MAKE_UNARY(Round, xla::RoundToEven(x));

XLAJIT_MAKE_UNARY(Rsqrt, xla::Rsqrt(x));

// Expresses sigmoid as a rescaled tanh: sigmoid(x) == (tanh(x/2) + 1) / 2.
xla::XlaOp Sigmoid(xla::XlaOp x) {
  auto half = xla::ScalarLike(x, 0.5);
  return half + half * xla::Tanh(half * x);
}
XLAJIT_MAKE_UNARY(Sigmoid, Sigmoid(x));

// Returns 0 if x is 0, -1 if x < 0 and 1 if x > 0.
XLAJIT_MAKE_UNARY(Sign, xla::Sign(x));
XLAJIT_MAKE_UNARY(Sinh, xla::Sinh(x));

// softplus(x) = log(1 + exp(x))
//
// This is not numerically stable when x is large, it can easily overflow.
// However, we can compute it as LogSumExp(x, 0):
//   max(x, 0) + log(exp(x - max(x, 0)) + exp(0 - max(x, 0)))
//
// This is equivalent to:
//   max(x, 0) + log1p(exp(-abs(x)))
XLAJIT_MAKE_UNARY(Softplus, xla::Max(x, xla::ScalarLike(x, 0.0)) +
                                xla::Log1p(xla::Exp(-xla::Abs(x))));

// softsign(x) = x / (abs(x) + 1)
XLAJIT_MAKE_UNARY(Softsign, x / (xla::Abs(x) + xla::ScalarLike(x, 1.0)));
XLAJIT_MAKE_UNARY(Sqrt, xla::Sqrt(x));
XLAJIT_MAKE_UNARY(Square, x* x);
XLAJIT_MAKE_UNARY(Tan, xla::Tan(x));
XLAJIT_MAKE_UNARY(Tanh, xla::Tanh(x));

XLAJIT_MAKE_UNARY(Real, xla::Real(x));
XLAJIT_MAKE_UNARY(Imag, xla::Imag(x));

#undef XLAJIT_MAKE_UNARY

// Erf/Erfc.  For x in (-1, 1), the erf approximation is used; erfc polynomial
// is used outside of this range.
class ErfOp : public XlaOpKernel {
 public:
  explicit ErfOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp x = ctx->Input(0);
    xla::XlaOp one = xla::ScalarLike(x, 1.0);
    auto y =
        xla::Select(xla::Gt(xla::Abs(x), one), one - xla::Erfc(x), xla::Erf(x));
    ctx->SetOutput(0, y);
  }
};
REGISTER_XLA_OP(Name("Erf"), ErfOp);

class ErfcOp : public XlaOpKernel {
 public:
  explicit ErfcOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp x = ctx->Input(0);
    xla::XlaOp one = xla::ScalarLike(x, 1.0);
    auto y =
        xla::Select(xla::Lt(xla::Abs(x), one), one - xla::Erf(x), xla::Erfc(x));
    ctx->SetOutput(0, y);
  }
};
REGISTER_XLA_OP(Name("Erfc"), ErfcOp);

class LgammaOp : public XlaOpKernel {
 public:
  explicit LgammaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Calculate lgamma using the Lanczos approximation
  // (https://en.wikipedia.org/wiki/Lanczos_approximation).
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::PrimitiveType input_type = ctx->input_xla_type(0);

    if (input_type == xla::F16 || input_type == xla::BF16) {
      // The approximation works better with at least 32-bits of accuracy.
      xla::XlaOp input_f32 = xla::ConvertElementType(input, xla::F32);
      xla::XlaOp result_f32 = xla::Lgamma(input_f32);
      xla::XlaOp result_x16 = xla::ConvertElementType(result_f32, input_type);
      ctx->SetOutput(0, result_x16);
    } else {
      xla::XlaOp result = xla::Lgamma(input);
      ctx->SetOutput(0, result);
    }
  }
};  // namespace
REGISTER_XLA_OP(Name("Lgamma"), LgammaOp);

class DigammaOp : public XlaOpKernel {
 public:
  explicit DigammaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Calculate lgamma using the Lanczos approximation
  // (https://en.wikipedia.org/wiki/Lanczos_approximation).
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::PrimitiveType input_type = ctx->input_xla_type(0);

    if (input_type == xla::F16 || input_type == xla::BF16) {
      // The approximation works better with at least 32-bits of accuracy.
      xla::XlaOp input_f32 = xla::ConvertElementType(input, xla::F32);
      xla::XlaOp result_f32 = xla::Digamma(input_f32);
      xla::XlaOp result_x16 = xla::ConvertElementType(result_f32, input_type);
      ctx->SetOutput(0, result_x16);
    } else {
      xla::XlaOp result = xla::Digamma(input);
      ctx->SetOutput(0, result);
    }
  }
};  // namespace
REGISTER_XLA_OP(Name("Digamma"), DigammaOp);

}  // namespace
}  // namespace tensorflow
