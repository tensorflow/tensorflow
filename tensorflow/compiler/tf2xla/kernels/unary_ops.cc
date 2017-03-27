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
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

// A subclass of a TlaUnaryOp must build the lambda computation that
// describes the scalar->scalar function to apply to each element of
// the input.
#define XLAJIT_MAKE_UNARY(Name, COMPUTATION)                           \
  class Name##Op : public XlaOpKernel {                                \
   public:                                                             \
    explicit Name##Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {} \
    void Compile(XlaOpKernelContext* ctx) {                            \
      xla::ComputationBuilder* b = ctx->builder();                     \
      xla::ComputationDataHandle x = ctx->Input(0);                    \
      xla::ComputationDataHandle y = COMPUTATION;                      \
      ctx->SetOutput(0, y);                                            \
    }                                                                  \
  };                                                                   \
  REGISTER_XLA_OP(#Name, Name##Op);

// Return x if x>0, otherwise -x.
XLAJIT_MAKE_UNARY(Abs, b->Abs(x));
XLAJIT_MAKE_UNARY(Ceil, b->Ceil(x));
XLAJIT_MAKE_UNARY(Exp, b->Exp(x));
XLAJIT_MAKE_UNARY(Floor, b->Floor(x));
// Returns 0 if x is 0, -1 if x < 0 and 1 if x > 0.
XLAJIT_MAKE_UNARY(Sign, b->Sign(x));
// Return 1/x
XLAJIT_MAKE_UNARY(Inv, b->Div(XlaHelpers::One(b, input_type(0)), x));
XLAJIT_MAKE_UNARY(Reciprocal, b->Div(XlaHelpers::One(b, input_type(0)), x));
XLAJIT_MAKE_UNARY(Log, b->Log(x));

// TODO(b/34703906): use a more accurate implementation of log1p.
XLAJIT_MAKE_UNARY(Log1p, b->Log(b->Add(XlaHelpers::One(b, input_type(0)), x)));

XLAJIT_MAKE_UNARY(LogicalNot, b->LogicalNot(x));
XLAJIT_MAKE_UNARY(Neg, b->Neg(x));

// Implements Banker's rounding: numbers that are equidistant between two
// integers are rounded towards even.
static xla::ComputationDataHandle Round(xla::ComputationBuilder* b,
                                        DataType dtype,
                                        const xla::ComputationDataHandle& x) {
  auto half = XlaHelpers::FloatLiteral(b, dtype, 0.5);
  auto one = XlaHelpers::FloatLiteral(b, dtype, 1.0);
  auto two = XlaHelpers::FloatLiteral(b, dtype, 2.0);

  auto round_val = b->Floor(x);
  auto fraction = b->Sub(x, round_val);
  auto nearest_even_int =
      b->Sub(round_val, b->Mul(two, b->Floor(b->Mul(half, x))));
  auto is_odd = b->Eq(nearest_even_int, one);
  return b->Select(b->LogicalOr(b->Gt(fraction, half),
                                b->LogicalAnd(b->Eq(fraction, half), is_odd)),
                   b->Add(round_val, one), round_val);
}
XLAJIT_MAKE_UNARY(Round, Round(b, input_type(0), x));

XLAJIT_MAKE_UNARY(Rsqrt,
                  b->Pow(x, XlaHelpers::FloatLiteral(b, input_type(0), -0.5)));
XLAJIT_MAKE_UNARY(Sigmoid,
                  b->Map({x}, *ctx->GetOrCreateSigmoid(input_type(0))));
XLAJIT_MAKE_UNARY(Softplus,
                  b->Log(b->Add(b->Exp(x), XlaHelpers::One(b, input_type(0)))));
XLAJIT_MAKE_UNARY(Sqrt,
                  b->Pow(x, XlaHelpers::FloatLiteral(b, input_type(0), 0.5)));
XLAJIT_MAKE_UNARY(Square, b->Mul(x, x));
XLAJIT_MAKE_UNARY(Tanh, b->Tanh(x));

#undef XLAJIT_MAKE_UNARY

}  // namespace
}  // namespace tensorflow
