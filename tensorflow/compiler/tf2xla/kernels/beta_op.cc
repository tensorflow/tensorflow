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

#include <limits>

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/loops.h"
#include "xla/client/lib/math.h"
#include "xla/client/xla_builder.h"
#include "xla/status_macros.h"

namespace tensorflow {
namespace {

class BetaincOp : public XlaOpKernel {
 public:
  explicit BetaincOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape& a_shape = ctx->InputShape(0);
    const TensorShape& b_shape = ctx->InputShape(1);
    const TensorShape& x_shape = ctx->InputShape(2);
    if (a_shape.dims() > 0 && b_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == b_shape,
                  errors::InvalidArgument(
                      "Shapes of a and b are inconsistent: ",
                      a_shape.DebugString(), " vs. ", b_shape.DebugString()));
    }
    if (a_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of a and x are inconsistent: ",
                      a_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }
    if (b_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, b_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of b and x are inconsistent: ",
                      b_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }

    TensorShape merged_shape(a_shape);
    if (b_shape.dims() > 0) merged_shape = b_shape;
    if (x_shape.dims() > 0) merged_shape = x_shape;

    auto builder = ctx->builder();
    auto result =
        builder->ReportErrorOrReturn([&]() -> absl::StatusOr<xla::XlaOp> {
          TF_ASSIGN_OR_RETURN(
              auto a, BroadcastTo(ctx->Input(0), merged_shape.dim_sizes()));
          TF_ASSIGN_OR_RETURN(
              auto b, BroadcastTo(ctx->Input(1), merged_shape.dim_sizes()));
          TF_ASSIGN_OR_RETURN(
              auto x, BroadcastTo(ctx->Input(2), merged_shape.dim_sizes()));
          return xla::RegularizedIncompleteBeta(a, b, x);
        });
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("Betainc"), BetaincOp);

}  // namespace
}  // namespace tensorflow
