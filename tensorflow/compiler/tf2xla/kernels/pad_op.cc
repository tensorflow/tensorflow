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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class PadOp : public XlaOpKernel {
 public:
  explicit PadOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

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

    xla::XlaOp input = ctx->Input("input");
    if (dims == 0) {
      // Tensor is rank 0. Return it unchanged.
      ctx->SetOutput(0, input);
      return;
    }

    xla::Literal pad_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal("paddings", &pad_literal));

    xla::PaddingConfig config;
    for (int i = 0; i < dims; ++i) {
      auto* dim = config.add_dimensions();
      int before = pad_literal.Get<int64>({i, 0});
      int after = pad_literal.Get<int64>({i, 1});
      OP_REQUIRES(ctx, before >= 0 && after >= 0,
                  errors::InvalidArgument(
                      "Paddings must be non-negative: ", before, " ", after));
      dim->set_edge_padding_low(before);
      dim->set_edge_padding_high(after);
    }

    // PadV2 added a "constant_values" input that indicates the pad value.
    xla::XlaOp constant_values;
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(ctx->InputShape("constant_values")),
          errors::InvalidArgument("constant_values must be a scalar."));
      ctx->SetOutput(0, xla::Pad(input, ctx->Input("constant_values"), config));
    } else {
      auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
      ctx->SetOutput(0, xla::Pad(input, zero, config));
    }
  }
};

REGISTER_XLA_OP(Name("Pad").CompileTimeConstantInput("paddings"), PadOp);
REGISTER_XLA_OP(Name("PadV2").CompileTimeConstantInput("paddings"), PadOp);

}  // namespace
}  // namespace tensorflow
