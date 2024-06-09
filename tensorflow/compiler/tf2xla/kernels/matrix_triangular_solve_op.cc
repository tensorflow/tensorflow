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

#include <tuple>
#include <utility>

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {
namespace {

class MatrixTriangularSolveOp : public XlaOpKernel {
 public:
  explicit MatrixTriangularSolveOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint", &adjoint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape lhs_shape = ctx->InputShape(0);
    const TensorShape rhs_shape = ctx->InputShape(1);

    // By TensorFlow conventions the inputs may not have the same
    // shapes, in which case they will be automatically broadcast if
    // possible before mapping. Use the standard TensorFlow helper to
    // compute valid broadcast shapes, but rely below on XLA to
    // automatically perform the broadcast assuming its valid shapes are
    // a superset of TensorFlow's valid shapes.
    MatMulBCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape));
    if (!bcast.IsValid()) {
      ctx->SetStatus(errors::InvalidArgument(
          "Incompatible shapes: ", lhs_shape.DebugString(), " vs. ",
          rhs_shape.DebugString()));
      return;
    }

    auto lhs_size = lhs_shape.dims();
    OP_REQUIRES(
        ctx,
        lhs_shape.dim_size(lhs_size - 1) == lhs_shape.dim_size(lhs_size - 2),
        errors::InvalidArgument("The coefficient matrix must be square in "
                                "the inner-most two dimensions: ",
                                lhs_shape.DebugString()));

    xla::XlaOp a = ctx->Input(0);
    xla::XlaOp b = ctx->Input(1);
    std::tie(a, b) = Broadcast(a, lhs_shape, b, rhs_shape, bcast);
    auto result = xla::TriangularSolve(
        a, b, /*left_side=*/true,
        /*lower=*/lower_, /*unit_diagonal=*/false,
        /*transpose_a=*/
        adjoint_ ? xla::TriangularSolveOptions::ADJOINT
                 : xla::TriangularSolveOptions::NO_TRANSPOSE);
    ctx->SetOutput(0, result);
  }

 private:
  static std::pair<xla::XlaOp, xla::XlaOp> Broadcast(
      xla::XlaOp lhs, const TensorShape& lhs_shape, xla::XlaOp rhs,
      const TensorShape& rhs_shape, const MatMulBCast& broadcast_helper);
  bool lower_;
  bool adjoint_;
};

/* static */ std::pair<xla::XlaOp, xla::XlaOp>
MatrixTriangularSolveOp::Broadcast(xla::XlaOp lhs, const TensorShape& lhs_shape,
                                   xla::XlaOp rhs, const TensorShape& rhs_shape,
                                   const MatMulBCast& broadcast_helper) {
  // Get the batch shape.
  int64_t m = lhs_shape.dim_size(lhs_shape.dims() - 1);
  int64_t n = rhs_shape.dim_size(rhs_shape.dims() - 1);

  TensorShape lhs_broadcast_shape(broadcast_helper.output_batch_shape());
  lhs_broadcast_shape.AddDim(m);
  lhs_broadcast_shape.AddDim(m);
  auto lhs_output = BroadcastTo(lhs, lhs_broadcast_shape.dim_sizes());
  if (!lhs_output.ok()) {
    xla::XlaOp error = lhs.builder()->ReportError(lhs_output.status());
    return {error, error};
  }

  TensorShape rhs_broadcast_shape(broadcast_helper.output_batch_shape());
  rhs_broadcast_shape.AddDim(m);
  rhs_broadcast_shape.AddDim(n);
  auto rhs_output = BroadcastTo(rhs, rhs_broadcast_shape.dim_sizes());
  if (!rhs_output.ok()) {
    xla::XlaOp error = rhs.builder()->ReportError(rhs_output.status());
    return {error, error};
  }
  return {lhs_output.value(), rhs_output.value()};
}

REGISTER_XLA_OP(Name("MatrixTriangularSolve"), MatrixTriangularSolveOp);

}  // namespace
}  // namespace tensorflow
