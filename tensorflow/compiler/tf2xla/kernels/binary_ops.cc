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

// Native XLA implementations of simple binary Ops

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

// A subclass of a XlaBinaryOp must build the computation that
// describes the (tensor,tensor)->tensor function to apply to each element of
// the input.
#define XLA_MAKE_BINARY(NAME, HLO)                                      \
  class NAME##Op : public XlaBinaryOp {                                 \
   public:                                                              \
    explicit NAME##Op(OpKernelConstruction* ctx) : XlaBinaryOp(ctx) {}  \
    xla::XlaOp Computation(                                             \
        XlaOpKernelContext* ctx, const xla::XlaOp& lhs,                 \
        const gtl::ArraySlice<int64>& lhs_shape, const xla::XlaOp& rhs, \
        const gtl::ArraySlice<int64>& rhs_shape,                        \
        const BCast& broadcast_helper,                                  \
        const std::vector<int64>& extend_dimensions) override {         \
      xla::XlaBuilder* b = ctx->builder();                              \
      return HLO;                                                       \
    }                                                                   \
  };                                                                    \
  REGISTER_XLA_OP(Name(#NAME), NAME##Op)

XLA_MAKE_BINARY(Add, b->Add(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Sub, b->Sub(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Mul, b->Mul(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Div, b->Div(lhs, rhs, extend_dimensions));

XLA_MAKE_BINARY(Atan2, b->Atan2(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Complex, b->Complex(lhs, rhs, extend_dimensions));

// Implementation of FloorDiv. Pseudo-code:
// if ((x < 0) != (y < 0)) {
//   T abs_x = std::abs(x);
//   T abs_y = std::abs(y);
//   return -(abs_x + abs_y - 1) / abs_y;
// } else {
//   return x / y;
// }
static xla::XlaOp FloorDivImpl(xla::XlaBuilder* b, DataType dtype, xla::XlaOp x,
                               xla::XlaOp y, const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(b, x, y, broadcast_helper);
  auto zero = XlaHelpers::Zero(b, dtype);
  auto one = XlaHelpers::One(b, dtype);
  auto different_sign = b->Ne(b->Lt(x, zero), b->Lt(y, zero));
  auto abs_x = b->Abs(x);
  auto abs_y = b->Abs(y);
  auto t = b->Neg(b->Sub(b->Add(abs_x, abs_y), one));
  auto result = b->Select(different_sign, b->Div(t, abs_y), b->Div(x, y));
  if (DataTypeIsFloating(dtype)) {
    result = b->Floor(result);
  }
  return result;
}
XLA_MAKE_BINARY(FloorDiv,
                FloorDivImpl(b, input_type(0), lhs, rhs, broadcast_helper));

// Implementation of FloorMod. Pseudo-code:
// T trunc_mod = std::fmod(x, y);
// return (x < T(0)) == (y < T(0)) ? trunc_mod : std::fmod(trunc_mod + y, y);
static xla::XlaOp FloorModImpl(xla::XlaBuilder* b, DataType dtype, xla::XlaOp x,
                               xla::XlaOp y, const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(b, x, y, broadcast_helper);
  auto zero = XlaHelpers::Zero(b, dtype);
  auto same_sign = b->Eq(b->Lt(x, zero), b->Lt(y, zero));
  auto trunc_mod = b->Rem(x, y);
  return b->Select(same_sign, trunc_mod, b->Rem(b->Add(trunc_mod, y), y));
}
XLA_MAKE_BINARY(FloorMod,
                FloorModImpl(b, input_type(0), lhs, rhs, broadcast_helper));

XLA_MAKE_BINARY(BitwiseAnd, b->And(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(BitwiseOr, b->Or(lhs, rhs, extend_dimensions));

XLA_MAKE_BINARY(LeftShift, b->ShiftLeft(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(RightShift,
                (DataTypeIsUnsigned(ctx->input_type(0))
                     ? b->ShiftRightLogical(lhs, rhs, extend_dimensions)
                     : b->ShiftRightArithmetic(lhs, rhs, extend_dimensions)));

XLA_MAKE_BINARY(LogicalAnd, b->And(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(LogicalOr, b->Or(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Mod, b->Rem(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Maximum, b->Max(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Minimum, b->Min(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(RealDiv, b->Div(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(ReciprocalGrad, b->Neg(b->Mul(rhs, b->Mul(lhs, lhs))));
XLA_MAKE_BINARY(
    RsqrtGrad,
    b->Mul(b->Pow(lhs, XlaHelpers::IntegerLiteral(b, input_type(0), 3)),
           b->Div(rhs, XlaHelpers::IntegerLiteral(b, input_type(0), -2)),
           extend_dimensions));
XLA_MAKE_BINARY(SqrtGrad,
                b->Div(b->Mul(rhs,
                              XlaHelpers::FloatLiteral(b, input_type(0), 0.5)),
                       lhs, extend_dimensions));

static xla::XlaOp Square(xla::XlaBuilder* builder, const xla::XlaOp& x) {
  return builder->Mul(x, x);
}

XLA_MAKE_BINARY(SquaredDifference,
                Square(b, b->Sub(lhs, rhs, extend_dimensions)));

XLA_MAKE_BINARY(TruncateDiv, b->Div(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(TruncateMod, b->Rem(lhs, rhs, extend_dimensions));

// Comparison ops
XLA_MAKE_BINARY(Equal, b->Eq(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(NotEqual, b->Ne(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Greater, b->Gt(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(GreaterEqual, b->Ge(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(Less, b->Lt(lhs, rhs, extend_dimensions));
XLA_MAKE_BINARY(LessEqual, b->Le(lhs, rhs, extend_dimensions));

// Non-linear ops
XLA_MAKE_BINARY(SigmoidGrad,
                b->Mul(b->Mul(rhs, lhs),
                       b->Sub(XlaHelpers::One(b, input_type(0)), lhs)));

XLA_MAKE_BINARY(SoftplusGrad,
                b->Div(lhs, b->Add(b->Exp(b->Neg(rhs)),
                                   XlaHelpers::One(b, input_type(1)))));

// softsigngrad(gradients, features) = gradients / (1 + abs(features)) ** 2
XLA_MAKE_BINARY(SoftsignGrad,
                b->Div(lhs, Square(b, b->Add(XlaHelpers::One(b, input_type(0)),
                                             b->Abs(rhs)))));

XLA_MAKE_BINARY(TanhGrad, b->Mul(rhs, b->Sub(XlaHelpers::One(b, input_type(0)),
                                             b->Mul(lhs, lhs))));

XLA_MAKE_BINARY(Pow, b->Pow(lhs, rhs, extend_dimensions));

#undef XLA_MAKE_BINARY

class ApproximateEqualOp : public XlaOpKernel {
 public:
  explicit ApproximateEqualOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tolerance", &tolerance_));
  }

  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto abs = b->Abs(b->Sub(ctx->Input(0), ctx->Input(1)));
    auto abs_shape = b->GetShape(abs);
    OP_REQUIRES_OK(ctx, abs_shape.status());
    auto abs_type = abs_shape.ValueOrDie().element_type();
    auto result = b->Lt(
        abs, b->ConvertElementType(b->ConstantR0<float>(tolerance_), abs_type));
    ctx->SetOutput(0, result);
  }

 private:
  float tolerance_;
};
REGISTER_XLA_OP(Name("ApproximateEqual"), ApproximateEqualOp);

}  // namespace
}  // namespace tensorflow
