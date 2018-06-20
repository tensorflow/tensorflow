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

// A subclass of a TlaUnaryOp must build the lambda computation that
// describes the scalar->scalar function to apply to each element of
// the input.
#define XLAJIT_MAKE_UNARY(NAME, COMPUTATION)                           \
  class NAME##Op : public XlaOpKernel {                                \
   public:                                                             \
    explicit NAME##Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {} \
    void Compile(XlaOpKernelContext* ctx) {                            \
      xla::XlaBuilder* b = ctx->builder();                             \
      xla::XlaOp x = ctx->Input(0);                                    \
      xla::XlaOp y = COMPUTATION;                                      \
      ctx->SetOutput(0, y);                                            \
    }                                                                  \
  };                                                                   \
  REGISTER_XLA_OP(Name(#NAME), NAME##Op);

XLAJIT_MAKE_UNARY(ComplexAbs, b->Abs(x));

XLAJIT_MAKE_UNARY(Angle, b->Atan2(b->Imag(x), b->Real(x)));

XLAJIT_MAKE_UNARY(Conj, b->Conj(x));

// Return x if x>0, otherwise -x.
XLAJIT_MAKE_UNARY(Abs, b->Abs(x));

// acos(x) = 2 * atan(sqrt(1 - x^2) / (1 + x))
XLAJIT_MAKE_UNARY(
    Acos,
    b->Mul(XlaHelpers::FloatLiteral(b, input_type(0), 2.0),
           b->Atan2(b->Pow(b->Sub(XlaHelpers::One(b, input_type(0)),
                                  b->Mul(x, x)),
                           XlaHelpers::FloatLiteral(b, input_type(0), 0.5)),
                    b->Add(XlaHelpers::One(b, input_type(0)), x))));

// acosh(x) = log(x + sqrt(x^2 - 1))
//          = log(x + sqrt((x+1)*(x-1)))
XLAJIT_MAKE_UNARY(
    Acosh,
    b->Log(b->Add(x,
                  b->Pow(b->Mul(b->Add(x, XlaHelpers::One(b, input_type(0))),
                                b->Sub(x, XlaHelpers::One(b, input_type(0)))),
                         XlaHelpers::FloatLiteral(b, input_type(0), 0.5)))));

// asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
XLAJIT_MAKE_UNARY(
    Asin,
    b->Mul(XlaHelpers::FloatLiteral(b, input_type(0), 2.0),
           b->Atan2(x, b->Add(XlaHelpers::One(b, input_type(0)),
                              b->Pow(b->Sub(XlaHelpers::One(b, input_type(0)),
                                            b->Mul(x, x)),
                                     XlaHelpers::FloatLiteral(b, input_type(0),
                                                              0.5))))));

// asinh(x) = log(x + sqrt(x^2 + 1))
XLAJIT_MAKE_UNARY(
    Asinh,
    b->Log(b->Add(x, b->Pow(b->Add(b->Mul(x, x),
                                   XlaHelpers::One(b, input_type(0))),
                            XlaHelpers::FloatLiteral(b, input_type(0), 0.5)))));

XLAJIT_MAKE_UNARY(Atan, b->Atan2(x, XlaHelpers::One(b, input_type(0))));

// atanh(x) = 0.5 * log((1 + x) / (1 - x))
XLAJIT_MAKE_UNARY(
    Atanh, b->Mul(b->Log(b->Div(b->Add(XlaHelpers::One(b, input_type(0)), x),
                                b->Sub(XlaHelpers::One(b, input_type(0)), x))),
                  XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Ceil, b->Ceil(x));
XLAJIT_MAKE_UNARY(Cos, b->Cos(x));
XLAJIT_MAKE_UNARY(Cosh,
                  b->Mul(b->Add(b->Exp(x), b->Exp(b->Neg(x))),
                         XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Sin, b->Sin(x));
XLAJIT_MAKE_UNARY(Exp, b->Exp(x));

XLAJIT_MAKE_UNARY(Expm1, b->Expm1(x));

XLAJIT_MAKE_UNARY(Floor, b->Floor(x));
XLAJIT_MAKE_UNARY(IsFinite, b->IsFinite(x));
XLAJIT_MAKE_UNARY(IsInf, b->Eq(b->Abs(x),
                               XlaHelpers::FloatLiteral(
                                   b, input_type(0),
                                   std::numeric_limits<double>::infinity())));
XLAJIT_MAKE_UNARY(IsNan, b->Ne(x, x));
// Return 1/x
XLAJIT_MAKE_UNARY(Inv, b->Div(XlaHelpers::One(b, input_type(0)), x));
XLAJIT_MAKE_UNARY(Reciprocal, b->Div(XlaHelpers::One(b, input_type(0)), x));
XLAJIT_MAKE_UNARY(Log, b->Log(x));

XLAJIT_MAKE_UNARY(Log1p, b->Log1p(x));

XLAJIT_MAKE_UNARY(Invert, b->Not(x));
XLAJIT_MAKE_UNARY(LogicalNot, b->Not(x));
XLAJIT_MAKE_UNARY(Neg, b->Neg(x));

// Implements Banker's rounding: numbers that are equidistant between two
// integers are rounded towards even.
static xla::XlaOp Round(xla::XlaBuilder* b, DataType dtype,
                        const xla::XlaOp& x) {
  auto half = XlaHelpers::FloatLiteral(b, dtype, 0.5);
  auto one = XlaHelpers::FloatLiteral(b, dtype, 1.0);
  auto two = XlaHelpers::FloatLiteral(b, dtype, 2.0);

  auto round_val = b->Floor(x);
  auto fraction = b->Sub(x, round_val);
  auto nearest_even_int =
      b->Sub(round_val, b->Mul(two, b->Floor(b->Mul(half, x))));
  auto is_odd = b->Eq(nearest_even_int, one);
  return b->Select(
      b->Or(b->Gt(fraction, half), b->And(b->Eq(fraction, half), is_odd)),
      b->Add(round_val, one), round_val);
}

XLAJIT_MAKE_UNARY(Rint, Round(b, input_type(0), x));
XLAJIT_MAKE_UNARY(Round, Round(b, input_type(0), x));

XLAJIT_MAKE_UNARY(Rsqrt,
                  b->Pow(x, XlaHelpers::FloatLiteral(b, input_type(0), -0.5)));

// Expresses sigmoid as a rescaled tanh: sigmoid(x) == (tanh(x/2) + 1) / 2.
static xla::XlaOp Sigmoid(xla::XlaBuilder* b, DataType dtype,
                          const xla::XlaOp& x) {
  auto half = XlaHelpers::FloatLiteral(b, dtype, 0.5);
  return b->Add(half, b->Mul(half, b->Tanh(b->Mul(half, x))));
}
XLAJIT_MAKE_UNARY(Sigmoid, Sigmoid(b, input_type(0), x));

// Returns 0 if x is 0, -1 if x < 0 and 1 if x > 0.
XLAJIT_MAKE_UNARY(Sign, b->Sign(x));
XLAJIT_MAKE_UNARY(Sinh,
                  b->Mul(b->Sub(b->Exp(x), b->Exp(b->Neg(x))),
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
                  b->Add(b->Max(x, XlaHelpers::Zero(b, input_type(0))),
                         b->Log1p(b->Exp(b->Neg(b->Abs(x))))));

// softsign(x) = x / (abs(x) + 1)
XLAJIT_MAKE_UNARY(Softsign,
                  b->Div(x,
                         b->Add(b->Abs(x), XlaHelpers::One(b, input_type(0)))));
XLAJIT_MAKE_UNARY(Sqrt,
                  b->Pow(x, XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Square, b->Mul(x, x));
XLAJIT_MAKE_UNARY(Tan, b->Div(b->Sin(x), b->Cos(x)));
XLAJIT_MAKE_UNARY(Tanh, b->Tanh(x));

XLAJIT_MAKE_UNARY(Real, b->Real(x));
XLAJIT_MAKE_UNARY(Imag, b->Imag(x));

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
    xla::XlaOp abs_x = b->Abs(x);

    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(input_type(0), &primitive_type));

    auto y = b->Select(b->Gt(abs_x, one),
                       b->Sub(one, ComputeErfc(b, x, primitive_type)),
                       ComputeErf(b, x, primitive_type));
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
    xla::XlaOp abs_x = b->Abs(x);

    xla::PrimitiveType primitive_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(input_type(0), &primitive_type));

    auto y = b->Select(b->Lt(abs_x, one),
                       b->Sub(one, ComputeErf(b, x, primitive_type)),
                       ComputeErfc(b, x, primitive_type));
    ctx->SetOutput(0, y);
  }
};
REGISTER_XLA_OP(Name("Erfc"), ErfcOp);

}  // namespace
}  // namespace tensorflow
