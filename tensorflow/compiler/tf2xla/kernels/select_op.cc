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

#include <numeric>

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

class SelectOp : public XlaOpKernel {
 public:
  explicit SelectOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape cond_shape = ctx->InputShape(0);
    const TensorShape then_shape = ctx->InputShape(1);
    const TensorShape else_shape = ctx->InputShape(2);

    OP_REQUIRES(
        ctx, then_shape.IsSameSize(else_shape),
        errors::InvalidArgument(
            "'then' and 'else' must have the same size.  but received: ",
            then_shape.DebugString(), " vs. ", else_shape.DebugString()));

    auto cond_handle = ctx->Input(0);
    auto then_handle = ctx->Input(1);
    auto else_handle = ctx->Input(2);

    bool broadcasting = !cond_shape.IsSameSize(then_shape);
    bool cond_is_scalar = TensorShapeUtils::IsScalar(cond_shape);
    if (broadcasting && !cond_is_scalar) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(cond_shape),
                  errors::InvalidArgument(
                      "'cond' must be a scalar or a vector, but saw shape: ",
                      cond_shape.DebugString()));
      OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(then_shape),
                  errors::InvalidArgument(
                      "'then' must be at least a vector, but saw shape: ",
                      then_shape.DebugString()));
      OP_REQUIRES(ctx, then_shape.dim_size(0) == cond_shape.num_elements(),
                  errors::InvalidArgument("Number of batches of 'then' must "
                                          "match size of 'cond', but saw: ",
                                          then_shape.dim_size(0), " vs. ",
                                          cond_shape.num_elements()));

      // Broadcast into the dimensions on the right.
      std::vector<int64_t> broadcast_dimensions(cond_shape.dims());
      absl::c_iota(broadcast_dimensions, 0);
      cond_handle = xla::BroadcastInDim(cond_handle, then_shape.dim_sizes(),
                                        broadcast_dimensions);
    }
    ctx->SetOutput(0, xla::Select(cond_handle, then_handle, else_handle));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SelectOp);
};

REGISTER_XLA_OP(Name("Select"), MlirXlaOpKernel);

class SelectOpV2 : public XlaOpKernel {
 public:
  explicit SelectOpV2(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape cond_shape = ctx->InputShape(0);
    const TensorShape then_shape = ctx->InputShape(1);
    const TensorShape else_shape = ctx->InputShape(2);

    // Compute the output shape from the broadcast of the two data inputs, with
    // the broadcast of the conditional.
    // Then Broadcast all three inputs to the output shape and emit a select.

    BCast bcast_then_else(BCast::FromShape(then_shape),
                          BCast::FromShape(else_shape),
                          /*fewer_dims_optimization=*/false);
    if (!bcast_then_else.IsValid()) {
      ctx->SetStatus(errors::InvalidArgument(
          "Incompatible shapes: ", then_shape.DebugString(), " vs. ",
          else_shape.DebugString()));
      return;
    }
    BCast bcast(bcast_then_else.output_shape(), BCast::FromShape(cond_shape),
                /*fewer_dims_optimization=*/false);
    if (!bcast.IsValid()) {
      ctx->SetStatus(errors::InvalidArgument(
          "Incompatible shapes: ",
          BCast::ToShape(bcast_then_else.output_shape()).DebugString(), " vs. ",
          cond_shape.DebugString()));
      return;
    }

    auto bcasted_cond = BroadcastTo(ctx->Input(0), bcast.output_shape());
    OP_REQUIRES_OK(ctx, bcasted_cond.status());
    auto cond_handle = bcasted_cond.value();

    auto bcasted_then = BroadcastTo(ctx->Input(1), bcast.output_shape());
    OP_REQUIRES_OK(ctx, bcasted_then.status());
    auto then_handle = bcasted_then.value();

    auto bcasted_else = BroadcastTo(ctx->Input(2), bcast.output_shape());
    OP_REQUIRES_OK(ctx, bcasted_else.status());
    auto else_handle = bcasted_else.value();

    ctx->SetOutput(0, xla::Select(cond_handle, then_handle, else_handle));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SelectOpV2);
};

REGISTER_XLA_OP(Name("SelectV2"), SelectOpV2);

}  // namespace
}  // namespace tensorflow
