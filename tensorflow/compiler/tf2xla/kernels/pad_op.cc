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
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
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
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(
                            "paddings", &pad_literal,
                            xla::ValueInferenceMode::kUpperBound));

    xla::Literal padding_dynamism_literal;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamism("paddings", &padding_dynamism_literal));

    xla::PaddingConfig config;
    for (int i = 0; i < dims; ++i) {
      auto* dim = config.add_dimensions();
      int before = pad_literal.Get<int64_t>({i, 0});
      int after = pad_literal.Get<int64_t>({i, 1});
      OP_REQUIRES(ctx, before >= 0 && after >= 0,
                  errors::InvalidArgument(
                      "Paddings must be non-negative: ", before, " ", after));
      dim->set_edge_padding_low(before);
      dim->set_edge_padding_high(after);
    }

    // PadV2 added a "constant_values" input that indicates the pad value.
    xla::XlaOp constant_values;
    xla::XlaOp pad;
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(ctx->InputShape("constant_values")),
          errors::InvalidArgument("constant_values must be a scalar."));
      pad = xla::Pad(input, ctx->Input("constant_values"), config);
    } else {
      auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
      pad = xla::Pad(input, zero, config);
    }

    for (int i = 0; i < dims; ++i) {
      bool low_pad_is_dynamic = padding_dynamism_literal.Get<bool>({i, 0});

      OP_REQUIRES(
          ctx, !low_pad_is_dynamic,
          errors::InvalidArgument("low_pad in Pad op has to be static."));
      bool high_pad_is_dynamic = padding_dynamism_literal.Get<bool>({i, 1});
      if (high_pad_is_dynamic) {
        // When we have
        // pad_width = MAX_WIDTH - size(t)
        // op = pad(t, /*high_pad=*/pad_width)
        // The bound of the result size should be MAX_WIDTH, instead of
        // `bound(t) + bound(pad_width)`
        //
        // We do this by analyzing the expression
        // size(op) = size(t) + MAX_WIDTH - size(t)
        // and leave value inference to analyze it.
        xla::XlaOp high_pad_size =
            xla::Slice(ctx->Input("paddings"), {i, 1}, {i + 1, 2}, {1, 1});
        high_pad_size = xla::Reshape(high_pad_size, {});
        high_pad_size = xla::ConvertElementType(high_pad_size, xla::S32);
        // Low pad has to be static.
        xla::XlaOp low_pad_size = xla::ConstantR0<int32>(
            ctx->builder(), pad_literal.Get<int64_t>({i, 0}));
        xla::XlaOp input_size = xla::GetDimensionSize(input, i);
        xla::XlaOp total_size = low_pad_size + input_size + high_pad_size;
        auto size_upper_bound_status_or =
            ctx->value_inference().AnalyzeConstant(
                total_size, xla::ValueInferenceMode::kUpperBound);
        OP_REQUIRES_OK(ctx, size_upper_bound_status_or.status());
        auto size_upper_bound =
            size_upper_bound_status_or.ValueOrDie().Get<int32>({});
        OP_REQUIRES(
            ctx, size_upper_bound.has_value(),
            errors::InvalidArgument(
                "Failed to infer upperbound of total size after padding."));
        // If we know a tighter upperbound, trim the output with the new
        // upperbound.
        pad = xla::SliceInDim(pad, 0, size_upper_bound.value(), 1, i);
        pad = xla::SetDimensionSize(pad, total_size, i);
      }
    }
    ctx->SetOutput(0, pad);
  }
};

REGISTER_XLA_OP(Name("Pad").CompileTimeConstantInput("paddings"), PadOp);
REGISTER_XLA_OP(Name("PadV2").CompileTimeConstantInput("paddings"), PadOp);

}  // namespace
}  // namespace tensorflow
