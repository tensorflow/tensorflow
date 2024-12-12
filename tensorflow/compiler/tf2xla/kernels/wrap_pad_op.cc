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

#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace tensorflow {
namespace {

class WrapPadOp : public XlaOpKernel {
 public:
  explicit WrapPadOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  absl::StatusOr<xla::XlaOp> DoWrapPad(const xla::XlaOp t,
                                       const xla::Shape& original_shape,
                                       const xla::LiteralSlice& pad_literal,
                                       xla::XlaBuilder* b) {
    xla::XlaOp accum = t;
    for (int64_t dimno = original_shape.rank() - 1; dimno >= 0; --dimno) {
      int64_t lhs_padding = pad_literal.Get<int64_t>({dimno, 0});
      int64_t rhs_padding = pad_literal.Get<int64_t>({dimno, 1});
      int64_t dim_size = original_shape.dimensions(dimno);

      // Padding amounts on each side must be less than the size of the
      // original shape.
      TF_RET_CHECK(lhs_padding >= 0 && lhs_padding < dim_size);
      TF_RET_CHECK(rhs_padding >= 0 && rhs_padding < dim_size);

      auto lhs_pad =
          xla::SliceInDim(accum, dim_size - lhs_padding, dim_size, 1, dimno);
      auto rhs_pad = xla::SliceInDim(accum, 0, rhs_padding, 1, dimno);
      accum = xla::ConcatInDim(b, {lhs_pad, accum, rhs_pad}, dimno);
    }
    return accum;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");

    const int dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(pad_shape) && pad_shape.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                pad_shape.DebugString()));
    OP_REQUIRES(
        ctx, dims == pad_shape.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            pad_shape.DebugString(), " ", input_shape.DebugString()));

    // Evaluate the 'padding' constant input, reshaping to a matrix.
    xla::Literal pad_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal("paddings", &pad_literal));

    xla::XlaBuilder* b = ctx->builder();
    auto in0 = ctx->Input("input");
    absl::StatusOr<xla::Shape> in0_shape = b->GetShape(in0);
    OP_REQUIRES(ctx, in0_shape.ok(), in0_shape.status());
    absl::StatusOr<xla::XlaOp> accum_status =
        DoWrapPad(in0, in0_shape.value(), pad_literal, b);

    OP_REQUIRES_OK(ctx, accum_status.status());

    ctx->SetOutput(0, accum_status.value());
  }

 private:
  WrapPadOp(const WrapPadOp&) = delete;
  void operator=(const WrapPadOp&) = delete;
};

REGISTER_XLA_OP(Name("WrapPad").CompileTimeConstantInput("paddings"),
                WrapPadOp);

class WrapPadGradOp : public XlaOpKernel {
 public:
  explicit WrapPadGradOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  absl::StatusOr<xla::XlaOp> DoWrapPadGrad(const xla::XlaOp t,
                                           const xla::Shape& original_shape,
                                           const xla::LiteralSlice& pad_literal,
                                           xla::XlaBuilder* b) {
    xla::XlaOp grad = t;
    for (int64_t dimno = original_shape.rank() - 1; dimno >= 0; --dimno) {
      int64_t lhs_padding = pad_literal.Get<int64_t>({dimno, 0});
      int64_t rhs_padding = pad_literal.Get<int64_t>({dimno, 1});
      int64_t dim_size = original_shape.dimensions(dimno);
      int64_t result_dim_size = dim_size - lhs_padding - rhs_padding;

      // Padding amounts on each side must be less than the size of the
      // original shape.
      TF_RET_CHECK(lhs_padding >= 0 && lhs_padding < dim_size);
      TF_RET_CHECK(rhs_padding >= 0 && rhs_padding < dim_size);

      xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dimno);
      xla::XlaOp padded_lhs_pad =
          xla::PadInDim(lhs_pad, xla::ScalarLike(lhs_pad, 0), dimno,
                        /*pad_lo=*/result_dim_size - lhs_padding,
                        /*pad_hi=*/0);

      xla::XlaOp rhs_pad =
          xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dimno);
      xla::XlaOp padded_rhs_pad =
          xla::PadInDim(rhs_pad, xla::ScalarLike(rhs_pad, 0), dimno,
                        /*pad_lo=*/0,
                        /*pad_hi=*/result_dim_size - rhs_padding);

      xla::XlaOp grad_core =
          xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dimno);

      grad = padded_lhs_pad + grad_core + padded_rhs_pad;
    }
    return grad;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");

    const int dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(pad_shape) && pad_shape.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                pad_shape.DebugString()));
    OP_REQUIRES(
        ctx, dims == pad_shape.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            pad_shape.DebugString(), " ", input_shape.DebugString()));

    // Evaluate the 'padding' constant input, reshaping to a matrix.
    xla::Literal pad_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal("paddings", &pad_literal));

    xla::XlaBuilder* b = ctx->builder();
    auto in0 = ctx->Input("input");
    absl::StatusOr<xla::Shape> in0_shape = b->GetShape(in0);
    OP_REQUIRES(ctx, in0_shape.ok(), in0_shape.status());
    absl::StatusOr<xla::XlaOp> accum_status =
        DoWrapPadGrad(in0, in0_shape.value(), pad_literal, b);

    OP_REQUIRES_OK(ctx, accum_status.status());

    ctx->SetOutput(0, accum_status.value());
  }

 private:
  WrapPadGradOp(const WrapPadGradOp&) = delete;
  void operator=(const WrapPadGradOp&) = delete;
};

REGISTER_XLA_OP(Name("WrapPadGrad").CompileTimeConstantInput("paddings"),
                WrapPadGradOp);

}  // namespace
}  // namespace tensorflow
