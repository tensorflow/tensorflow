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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/util/mirror_pad_mode.h"

namespace tensorflow {
namespace {

class MirrorPadOp : public XlaOpKernel {
 public:
  explicit MirrorPadOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  StatusOr<xla::XlaOp> DoMirrorPad(const xla::XlaOp t,
                                   const xla::Shape& original_shape,
                                   const xla::LiteralSlice& pad_literal,
                                   const MirrorPadMode mode,
                                   xla::XlaBuilder* b) {
    // The difference in the semantics of REFLECT and SYMMETRIC is that REFLECT
    // will not mirror the border values while symmetric does.
    // e.g. input is [1, 2, 3] and paddings is [0, 2], then the output is:
    // - [1, 2, 3, 2, 1] in reflect mode
    // - [1, 2, 3, 3, 2] in symmetric mode.
    int64_t excluded_edges = mode == MirrorPadMode::REFLECT ? 1 : 0;
    xla::XlaOp accum = t;
    for (int64_t dimno = original_shape.rank() - 1; dimno >= 0; --dimno) {
      auto t_rev = xla::Rev(accum, {dimno});
      int64_t lhs_padding = pad_literal.Get<int64_t>({dimno, 0});
      int64_t rhs_padding = pad_literal.Get<int64_t>({dimno, 1});
      int64_t dim_size = original_shape.dimensions(dimno);

      // Padding amounts on each side must be no more than the size of the
      // original shape.
      TF_RET_CHECK(lhs_padding >= 0 &&
                   lhs_padding <= dim_size - excluded_edges);
      TF_RET_CHECK(rhs_padding >= 0 &&
                   rhs_padding <= dim_size - excluded_edges);

      auto lhs_pad =
          xla::SliceInDim(t_rev, dim_size - excluded_edges - lhs_padding,
                          dim_size - excluded_edges, 1, dimno);
      auto rhs_pad = xla::SliceInDim(t_rev, excluded_edges,
                                     excluded_edges + rhs_padding, 1, dimno);
      accum = xla::ConcatInDim(b, {lhs_pad, accum, rhs_pad}, dimno);
    }
    return accum;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");

    MirrorPadMode mode;
    OP_REQUIRES_OK(ctx, GetNodeAttr(def(), "mode", &mode));
    OP_REQUIRES(
        ctx, mode == MirrorPadMode::REFLECT || mode == MirrorPadMode::SYMMETRIC,
        xla::Unimplemented("Unsupported MirrorPad mode. Only SYMMETRIC and "
                           "REFLECT modes are currently supported"));

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
    StatusOr<xla::Shape> in0_shape = b->GetShape(in0);
    OP_REQUIRES(ctx, in0_shape.ok(), in0_shape.status());
    StatusOr<xla::XlaOp> accum_status =
        DoMirrorPad(in0, in0_shape.value(), pad_literal, mode, b);

    OP_REQUIRES_OK(ctx, accum_status.status());

    ctx->SetOutput(0, accum_status.value());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MirrorPadOp);
};

REGISTER_XLA_OP(Name("MirrorPad").CompileTimeConstantInput("paddings"),
                MirrorPadOp);

class MirrorPadGradOp : public XlaOpKernel {
 public:
  explicit MirrorPadGradOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  StatusOr<xla::XlaOp> DoMirrorPadGrad(const xla::XlaOp t,
                                       const xla::Shape& original_shape,
                                       const xla::LiteralSlice& pad_literal,
                                       const MirrorPadMode mode,
                                       xla::XlaBuilder* b) {
    // The difference in the semantics of REFLECT and SYMMETRIC is that REFLECT
    // will not mirror the border values while symmetric does.
    // e.g. input is [1, 2, 3] and paddings is [0, 2], then the output is:
    // - [1, 2, 3, 2, 1] in reflect mode
    // - [1, 2, 3, 3, 2] in symmetric mode.
    int64_t excluded_edges = mode == MirrorPadMode::REFLECT ? 1 : 0;
    xla::XlaOp grad = t;
    for (int64_t dimno = original_shape.rank() - 1; dimno >= 0; --dimno) {
      int64_t lhs_padding = pad_literal.Get<int64_t>({dimno, 0});
      int64_t rhs_padding = pad_literal.Get<int64_t>({dimno, 1});
      int64_t dim_size = original_shape.dimensions(dimno);
      int64_t result_dim_size = dim_size - lhs_padding - rhs_padding;

      // Padding amounts on each side must be no more than the size of the
      // original shape.
      TF_RET_CHECK(lhs_padding >= 0 &&
                   lhs_padding <= dim_size - excluded_edges);
      TF_RET_CHECK(rhs_padding >= 0 &&
                   rhs_padding <= dim_size - excluded_edges);

      xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dimno);
      xla::XlaOp reverse_lhs_pad = xla::Rev(lhs_pad, {dimno});
      xla::XlaOp padded_lhs_pad = xla::PadInDim(
          reverse_lhs_pad, xla::ScalarLike(reverse_lhs_pad, 0), dimno,
          /*pad_lo=*/excluded_edges,
          /*pad_hi=*/result_dim_size - lhs_padding - excluded_edges);

      xla::XlaOp rhs_pad =
          xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dimno);
      xla::XlaOp reverse_rhs_pad = xla::Rev(rhs_pad, {dimno});
      xla::XlaOp padded_rhs_pad = xla::PadInDim(
          reverse_rhs_pad, xla::ScalarLike(reverse_rhs_pad, 0), dimno,
          /*pad_lo=*/result_dim_size - rhs_padding - excluded_edges,
          /*pad_hi=*/excluded_edges);

      xla::XlaOp grad_core =
          xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dimno);

      grad = padded_lhs_pad + grad_core + padded_rhs_pad;
    }
    return grad;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");

    MirrorPadMode mode;
    OP_REQUIRES_OK(ctx, GetNodeAttr(def(), "mode", &mode));
    OP_REQUIRES(
        ctx, mode == MirrorPadMode::REFLECT || mode == MirrorPadMode::SYMMETRIC,
        xla::Unimplemented("Unsupported MirrorPadGrad mode. Only SYMMETRIC and "
                           "REFLECT modes are currently supported"));

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
    StatusOr<xla::Shape> in0_shape = b->GetShape(in0);
    OP_REQUIRES(ctx, in0_shape.ok(), in0_shape.status());
    StatusOr<xla::XlaOp> accum_status =
        DoMirrorPadGrad(in0, in0_shape.value(), pad_literal, mode, b);

    OP_REQUIRES_OK(ctx, accum_status.status());

    ctx->SetOutput(0, accum_status.value());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MirrorPadGradOp);
};

REGISTER_XLA_OP(Name("MirrorPadGrad").CompileTimeConstantInput("paddings"),
                MirrorPadGradOp);

}  // namespace
}  // namespace tensorflow
