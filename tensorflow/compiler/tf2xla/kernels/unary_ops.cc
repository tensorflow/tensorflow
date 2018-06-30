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
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
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

// acos(x) = 2 * atan(sqrt(1 - x^2) / (1 + x))
XLAJIT_MAKE_UNARY(
    Acos,
    xla::Mul(XlaHelpers::FloatLiteral(b, input_type(0), 2.0),
             xla::Atan2(xla::Pow(xla::Sub(XlaHelpers::One(b, input_type(0)),
                                          xla::Mul(x, x)),
                                 XlaHelpers::FloatLiteral(b, input_type(0),
                                                          0.5)),
                        xla::Add(XlaHelpers::One(b, input_type(0)), x))));

// acosh(x) = log(x + sqrt(x^2 - 1))
//          = log(x + sqrt((x+1)*(x-1)))
XLAJIT_MAKE_UNARY(
    Acosh,
    xla::Log(xla::Add(
        x, xla::Pow(xla::Mul(xla::Add(x, XlaHelpers::One(b, input_type(0))),
                             xla::Sub(x, XlaHelpers::One(b, input_type(0)))),
                    XlaHelpers::FloatLiteral(b, input_type(0), 0.5)))));

// asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
XLAJIT_MAKE_UNARY(
    Asin,
    xla::Mul(
        XlaHelpers::FloatLiteral(b, input_type(0), 2.0),
        xla::Atan2(x,
                   xla::Add(XlaHelpers::One(b, input_type(0)),
                            xla::Pow(xla::Sub(XlaHelpers::One(b, input_type(0)),
                                              xla::Mul(x, x)),
                                     XlaHelpers::FloatLiteral(b, input_type(0),
                                                              0.5))))));

// asinh(x) = log(x + sqrt(x^2 + 1))
XLAJIT_MAKE_UNARY(
    Asinh,
    xla::Log(xla::Add(
        x, xla::Pow(xla::Add(xla::Mul(x, x), XlaHelpers::One(b, input_type(0))),
                    XlaHelpers::FloatLiteral(b, input_type(0), 0.5)))));

XLAJIT_MAKE_UNARY(Atan, xla::Atan2(x, XlaHelpers::One(b, input_type(0))));

// atanh(x) = 0.5 * log((1 + x) / (1 - x))
XLAJIT_MAKE_UNARY(
    Atanh,
    xla::Mul(xla::Log(xla::Div(xla::Add(XlaHelpers::One(b, input_type(0)), x),
                               xla::Sub(XlaHelpers::One(b, input_type(0)), x))),
             XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Ceil, xla::Ceil(x));
XLAJIT_MAKE_UNARY(Cos, xla::Cos(x));
XLAJIT_MAKE_UNARY(Cosh,
                  xla::Mul(xla::Add(xla::Exp(x), xla::Exp(xla::Neg(x))),
                           XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Sin, xla::Sin(x));
XLAJIT_MAKE_UNARY(Exp, xla::Exp(x));

XLAJIT_MAKE_UNARY(Expm1, xla::Expm1(x));

XLAJIT_MAKE_UNARY(Floor, xla::Floor(x));
XLAJIT_MAKE_UNARY(IsFinite, xla::IsFinite(x));
XLAJIT_MAKE_UNARY(IsInf, xla::Eq(xla::Abs(x),
                                 XlaHelpers::FloatLiteral(
                                     b, input_type(0),
                                     std::numeric_limits<double>::infinity())));
XLAJIT_MAKE_UNARY(IsNan, xla::Ne(x, x));
// Return 1/x
XLAJIT_MAKE_UNARY(Inv, xla::Div(XlaHelpers::One(b, input_type(0)), x));
XLAJIT_MAKE_UNARY(Reciprocal, xla::Div(XlaHelpers::One(b, input_type(0)), x));
XLAJIT_MAKE_UNARY(Log, xla::Log(x));

XLAJIT_MAKE_UNARY(Log1p, xla::Log1p(x));

XLAJIT_MAKE_UNARY(Invert, xla::Not(x));
XLAJIT_MAKE_UNARY(LogicalNot, xla::Not(x));
XLAJIT_MAKE_UNARY(Neg, xla::Neg(x));

// Implements Banker's rounding: numbers that are equidistant between two
// integers are rounded towards even.
static xla::XlaOp Round(xla::XlaBuilder* b, DataType dtype,
                        const xla::XlaOp& x) {
  auto half = XlaHelpers::FloatLiteral(b, dtype, 0.5);
  auto one = XlaHelpers::FloatLiteral(b, dtype, 1.0);
  auto two = XlaHelpers::FloatLiteral(b, dtype, 2.0);

  auto round_val = xla::Floor(x);
  auto fraction = xla::Sub(x, round_val);
  auto nearest_even_int =
      xla::Sub(round_val, xla::Mul(two, xla::Floor(xla::Mul(half, x))));
  auto is_odd = xla::Eq(nearest_even_int, one);
  return xla::Select(xla::Or(xla::Gt(fraction, half),
                             xla::And(xla::Eq(fraction, half), is_odd)),
                     xla::Add(round_val, one), round_val);
}

XLAJIT_MAKE_UNARY(Rint, Round(b, input_type(0), x));
XLAJIT_MAKE_UNARY(Round, Round(b, input_type(0), x));

XLAJIT_MAKE_UNARY(Rsqrt, xla::Pow(x, XlaHelpers::FloatLiteral(b, input_type(0),
                                                              -0.5)));

// Expresses sigmoid as a rescaled tanh: sigmoid(x) == (tanh(x/2) + 1) / 2.
static xla::XlaOp Sigmoid(xla::XlaBuilder* b, DataType dtype,
                          const xla::XlaOp& x) {
  auto half = XlaHelpers::FloatLiteral(b, dtype, 0.5);
  return xla::Add(half, xla::Mul(half, xla::Tanh(xla::Mul(half, x))));
}
XLAJIT_MAKE_UNARY(Sigmoid, Sigmoid(b, input_type(0), x));

// Returns 0 if x is 0, -1 if x < 0 and 1 if x > 0.
XLAJIT_MAKE_UNARY(Sign, xla::Sign(x));
XLAJIT_MAKE_UNARY(Sinh,
                  xla::Mul(xla::Sub(xla::Exp(x), xla::Exp(xla::Neg(x))),
                           XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));

// softplus(x) = log(1 + exp(x))
//
// This is not numerically stable when x is large, it can easily overflow.
// However, we can compute it as LogSumExp(x, 0):
//   max(x, 0) + log(exp(x - max(x, 0)) + exp(0 - max(x, 0)))
//
// This is equivalent to:
//   max(x, 0) + log1p(exp(-abs(x)))
XLAJIT_MAKE_UNARY(Softplus,
                  xla::Add(xla::Max(x, XlaHelpers::Zero(b, input_type(0))),
                           xla::Log1p(xla::Exp(xla::Neg(xla::Abs(x))))));

// softsign(x) = x / (abs(x) + 1)
XLAJIT_MAKE_UNARY(Softsign,
                  xla::Div(x, xla::Add(xla::Abs(x),
                                       XlaHelpers::One(b, input_type(0)))));
XLAJIT_MAKE_UNARY(Sqrt,
                  xla::Pow(x, XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Square, xla::Mul(x, x));
XLAJIT_MAKE_UNARY(Tan, xla::Div(xla::Sin(x), xla::Cos(x)));
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
    xla::XlaBuilder* b = ctx->builder();
    xla::PrimitiveType primitive_type;
    xla::XlaOp one = XlaHelpers::One(b, input_type(0));
    xla::XlaOp x = ctx->Input(0);
    xla::XlaOp abs_x = xla::Abs(x);

    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(input_type(0), &primitive_type));

    auto y =
        xla::Select(xla::Gt(abs_x, one), xla::Sub(one, Erfc(x, primitive_type)),
                    Erf(x, primitive_type));
    ctx->SetOutput(0, y);
  }
};
REGISTER_XLA_OP(Name("Erf"), ErfOp);

class ErfcOp : public XlaOpKernel {
 public:
  explicit ErfcOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp one = XlaHelpers::One(b, input_type(0));
    xla::XlaOp x = ctx->Input(0);
    xla::XlaOp abs_x = xla::Abs(x);

    xla::PrimitiveType primitive_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(input_type(0), &primitive_type));

    auto y =
        xla::Select(xla::Lt(abs_x, one), xla::Sub(one, Erf(x, primitive_type)),
                    Erfc(x, primitive_type));
    ctx->SetOutput(0, y);
  }
};
REGISTER_XLA_OP(Name("Erfc"), ErfcOp);

}  // namespace
}  // namespace tensorflow
