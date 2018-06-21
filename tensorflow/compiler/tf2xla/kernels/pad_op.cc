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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace {

class PadOp : public XlaOpKernel {
 public:
  explicit PadOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape pad_shape = ctx->InputShape(1);
    const int dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(pad_shape) && pad_shape.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                pad_shape.DebugString()));
    const int fixed_dims =
        (allow_legacy_scalars() && dims == 0 && pad_shape.dim_size(0) == 1)
            ? 1
            : dims;
    OP_REQUIRES(
        ctx, fixed_dims == pad_shape.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            pad_shape.DebugString(), " ", input_shape.DebugString()));

    if (fixed_dims == 0) {
      // Tensor is rank 0. Return it unchanged.
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }

    // Evaluate the 'padding' constant input, reshaping to a matrix.
    xla::Literal pad_literal;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputReshaped(1, {fixed_dims, 2}, &pad_literal));

    xla::PaddingConfig config;
    for (int i = 0; i < fixed_dims; ++i) {
      auto* dim = config.add_dimensions();
      int before = pad_literal.Get<int32>({i, 0});
      int after = pad_literal.Get<int32>({i, 1});
      OP_REQUIRES(ctx, before >= 0 && after >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before, " ", after));
      dim->set_edge_padding_low(before);
      dim->set_edge_padding_high(after);
    }

    // PadV2 added a "constant_values" input that indicates the pad value.
    xla::XlaOp constant_values;
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->InputShape(2)),
                  errors::InvalidArgument("constant_values must be a scalar."));
      ctx->SetOutput(0,
                     ctx->builder()->Pad(ctx->Input(0), ctx->Input(2), config));
    } else {
      auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
      ctx->SetOutput(0, ctx->builder()->Pad(ctx->Input(0), zero, config));
    }
  }
};

REGISTER_XLA_OP(Name("Pad").CompileTimeConstInput("paddings"), PadOp);
REGISTER_XLA_OP(Name("PadV2").CompileTimeConstInput("paddings"), PadOp);

}  // namespace
}  // namespace tensorflow
