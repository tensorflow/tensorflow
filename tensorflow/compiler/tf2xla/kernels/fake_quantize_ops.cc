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

#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

// Gymnastics with nudged zero point is to ensure that the real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
void CpuNudge(const float min, const float max, const float quant_min,
              const float quant_max, float* nudged_min, float* nudged_max,
              float* scale) {
  *scale = (max - min) / (quant_max - quant_min);

  const float zero_point_from_min = quant_min - min / *scale;
  float nudged_zero_point;
  if (zero_point_from_min <= quant_min) {
    nudged_zero_point = quant_min;
  } else if (zero_point_from_min >= quant_max) {
    nudged_zero_point = quant_max;
  } else {
    nudged_zero_point = std::round(zero_point_from_min);
  }

  *nudged_min = (quant_min - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max - nudged_zero_point) * (*scale);
}

// An XLA version of CpuNudge().
void XlaNudge(xla::XlaBuilder* b, const DataType data_type,
              const xla::XlaOp min, const xla::XlaOp max,
              const float quant_min_value, const float quant_max_value,
              xla::XlaOp* nudged_min, xla::XlaOp* nudged_max,
              xla::XlaOp* scale) {
  *scale = xla::Div(xla::Sub(max, min),
                    XlaHelpers::FloatLiteral(
                        b, data_type, quant_max_value - quant_min_value));
  xla::XlaOp quant_min =
      XlaHelpers::FloatLiteral(b, data_type, quant_min_value);
  xla::XlaOp zero_point_from_min = xla::Sub(quant_min, xla::Div(min, *scale));
  xla::XlaOp quant_max =
      XlaHelpers::FloatLiteral(b, data_type, quant_max_value);
  xla::XlaOp nudged_zero_point =
      xla::Select(xla::Le(zero_point_from_min, quant_min), quant_min,
                  xla::Select(xla::Ge(zero_point_from_min, quant_max),
                              quant_max, xla::Round(zero_point_from_min)));
  *nudged_min = xla::Mul(xla::Sub(quant_min, nudged_zero_point), *scale);
  *nudged_max = xla::Mul(xla::Sub(quant_max, nudged_zero_point), *scale);
}

xla::XlaOp Quantize(xla::XlaBuilder* b, const xla::XlaOp input,
                    const DataType data_type, const xla::XlaOp nudged_input_min,
                    const xla::XlaOp nudged_input_max,
                    const xla::XlaOp input_scale) {
  xla::XlaOp one = XlaHelpers::FloatLiteral(b, data_type, 1.0f);
  xla::XlaOp inv_scale = xla::Div(one, input_scale);
  xla::XlaOp half = XlaHelpers::FloatLiteral(b, data_type, 0.5f);

  xla::XlaOp clamped = xla::Clamp(nudged_input_min, input, nudged_input_max);
  xla::XlaOp clamped_shifted = xla::Sub(clamped, nudged_input_min);
  xla::XlaOp rounded =
      xla::Floor(xla::Add(xla::Mul(clamped_shifted, inv_scale), half));
  return xla::Add(xla::Mul(rounded, input_scale), nudged_input_min);
}

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxArgs"), MlirXlaOpKernel);

class FakeQuantWithMinMaxArgsGradOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxArgsGradOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    int num_bits;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(ctx, num_bits >= 2 && num_bits <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits));
    bool narrow_range;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    const float quant_min = narrow_range ? 1 : 0;
    const float quant_max = (1 << num_bits) - 1;

    float input_min, input_max, scale;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min", &input_min));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max", &input_max));
    CpuNudge(input_min, input_max, quant_min, quant_max, &nudged_input_min_,
             &nudged_input_max_, &scale);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::XlaOp input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp nudged_input_min =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_min_);
    xla::XlaOp nudged_input_max =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_max_);

    xla::XlaOp between_nudged_min_max = xla::And(
        xla::Le(nudged_input_min, input), xla::Le(input, nudged_input_max));
    xla::XlaOp zeroes = xla::Broadcast(XlaHelpers::Zero(b, data_type),
                                       gradient_shape.dim_sizes());
    xla::XlaOp output = xla::Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output);
  }

 private:
  float nudged_input_min_;
  float nudged_input_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxArgsGradient"),
                FakeQuantWithMinMaxArgsGradOp);

class FakeQuantWithMinMaxVarsOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ >= 2 && num_bits_ <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    quant_min_ = narrow_range_ ? 1 : 0;
    quant_max_ = (1 << num_bits_) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);
    xla::XlaOp input_min = ctx->Input(1);
    xla::XlaOp input_max = ctx->Input(2);

    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp output = Quantize(b, input, data_type, nudged_input_min,
                                 nudged_input_max, input_scale);
    ctx->SetOutput(0, output);
  }

 private:
  int num_bits_;
  bool narrow_range_;
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVars"), FakeQuantWithMinMaxVarsOp);

class FakeQuantWithMinMaxVarsGradOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsGradOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    int num_bits;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(ctx, num_bits >= 2 && num_bits <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits));
    bool narrow_range;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::XlaOp input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);
    const DataType accumulation_type =
        XlaHelpers::SumAccumulationType(data_type);
    xla::XlaOp input_min = ctx->Input(2);
    xla::XlaOp input_max = ctx->Input(3);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp between_nudged_min_max = xla::And(
        xla::Le(nudged_input_min, input), xla::Le(input, nudged_input_max));
    xla::XlaOp zero = XlaHelpers::Zero(b, data_type);
    xla::XlaOp zeroes = xla::Broadcast(zero, gradient_shape.dim_sizes());
    xla::XlaOp output0 = xla::Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output0);

    xla::XlaOp below_min = xla::Lt(input, nudged_input_min);
    xla::XlaOp select1 = xla::Select(below_min, gradient, zeroes);
    xla::XlaOp reduce1 = xla::ReduceAll(
        XlaHelpers::ConvertElementType(select1, accumulation_type),
        XlaHelpers::Zero(b, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type));
    xla::XlaOp output1 = XlaHelpers::ConvertElementType(reduce1, data_type);
    ctx->SetOutput(1, output1);

    xla::XlaOp above_max = xla::Gt(input, nudged_input_max);
    xla::XlaOp select2 = xla::Select(above_max, gradient, zeroes);
    xla::XlaOp reduce2 = xla::ReduceAll(
        XlaHelpers::ConvertElementType(select2, accumulation_type),
        XlaHelpers::Zero(b, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type));
    xla::XlaOp output2 = XlaHelpers::ConvertElementType(reduce2, data_type);
    ctx->SetOutput(2, output2);
  }

 private:
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsGradient"),
                FakeQuantWithMinMaxVarsGradOp);

class FakeQuantWithMinMaxVarsPerChannelOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ >= 2 && num_bits_ <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    quant_min_ = narrow_range_ ? 1 : 0;
    quant_max_ = (1 << num_bits_) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp input = ctx->Input(0);

    const DataType data_type = ctx->input_type(0);
    xla::XlaOp input_min = ctx->Input(1);
    xla::XlaOp input_max = ctx->Input(2);

    xla::Shape input_shape = b->GetShape(input).value();
    absl::Span<const int64_t> input_dimensions = input_shape.dimensions();
    auto convert_to_input_shape = [&](const xla::XlaOp op) {
      return xla::BroadcastInDim(op, input_dimensions,
                                 {input_shape.rank() - 1});
    };
    input_min = convert_to_input_shape(input_min);
    input_max = convert_to_input_shape(input_max);

    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp output = Quantize(b, input, data_type, nudged_input_min,
                                 nudged_input_max, input_scale);
    ctx->SetOutput(0, output);
  }

 private:
  int num_bits_;
  bool narrow_range_;
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsPerChannel"),
                FakeQuantWithMinMaxVarsPerChannelOp);

class FakeQuantWithMinMaxVarsPerChannelGradOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelGradOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    int num_bits;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(ctx, num_bits >= 2 && num_bits <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits));
    bool narrow_range;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::XlaOp input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);
    const DataType accumulation_type =
        XlaHelpers::SumAccumulationType(data_type);
    xla::XlaOp input_min = ctx->Input(2);
    xla::XlaOp input_max = ctx->Input(3);

    xla::XlaBuilder* b = ctx->builder();
    xla::Shape input_shape = b->GetShape(input).value();
    absl::Span<const int64_t> input_dimensions = input_shape.dimensions();

    std::vector<int64_t> reduce_axes;
    for (int64_t i = 0; i + 1 < input_shape.rank(); ++i) {
      reduce_axes.push_back(i);
    }

    auto convert_to_input_shape = [&](const xla::XlaOp op) {
      return xla::BroadcastInDim(op, input_dimensions,
                                 {input_shape.rank() - 1});
    };
    input_min = convert_to_input_shape(input_min);
    input_max = convert_to_input_shape(input_max);

    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp between_nudged_min_max = xla::And(
        xla::Le(nudged_input_min, input), xla::Le(input, nudged_input_max));
    xla::XlaOp zero = XlaHelpers::Zero(b, data_type);
    xla::XlaOp zeroes = xla::Broadcast(zero, gradient_shape.dim_sizes());
    xla::XlaOp output0 = xla::Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output0);

    xla::XlaOp below_min = xla::Lt(input, nudged_input_min);
    xla::XlaOp select1 = xla::Select(below_min, gradient, zeroes);
    xla::XlaOp reduce1 =
        xla::Reduce(XlaHelpers::ConvertElementType(select1, accumulation_type),
                    XlaHelpers::Zero(b, accumulation_type),
                    *ctx->GetOrCreateAdd(accumulation_type), reduce_axes);
    xla::XlaOp output1 = XlaHelpers::ConvertElementType(reduce1, data_type);
    ctx->SetOutput(1, output1);

    xla::XlaOp above_max = xla::Gt(input, nudged_input_max);
    xla::XlaOp select2 = xla::Select(above_max, gradient, zeroes);
    xla::XlaOp reduce2 =
        xla::Reduce(XlaHelpers::ConvertElementType(select2, accumulation_type),
                    XlaHelpers::Zero(b, accumulation_type),
                    *ctx->GetOrCreateAdd(accumulation_type), reduce_axes);
    xla::XlaOp output2 = XlaHelpers::ConvertElementType(reduce2, data_type);
    ctx->SetOutput(2, output2);
  }

 private:
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsPerChannelGradient"),
                FakeQuantWithMinMaxVarsPerChannelGradOp);

}  // namespace
}  // namespace tensorflow
