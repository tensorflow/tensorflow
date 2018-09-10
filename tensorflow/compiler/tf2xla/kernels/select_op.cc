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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/kernels/bounds_check.h"

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

      // TODO(phawkins): broadcasting on the right seems pretty awkward in
      // XLA. It seems we have to broadcast on the left and then Reshape
      // to get the dimensions in the right order.
      const auto dim_sizes = then_shape.dim_sizes();
      absl::Span<const int64> bdims = dim_sizes;
      bdims.remove_prefix(1);
      cond_handle = xla::Broadcast(cond_handle, bdims);

      std::vector<int64> dim_order(then_shape.dims());
      dim_order[0] = then_shape.dims() - 1;
      std::iota(dim_order.begin() + 1, dim_order.end(), 0);
      cond_handle = xla::Transpose(cond_handle, dim_order);
    }
    ctx->SetOutput(0, xla::Select(cond_handle, then_handle, else_handle));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SelectOp);
};

REGISTER_XLA_OP(Name("Select"), SelectOp);

}  // namespace
}  // namespace tensorflow
