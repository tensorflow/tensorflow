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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
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
void XlaNudge(xla::ComputationBuilder* b, const DataType data_type,
              const xla::ComputationDataHandle& min,
              const xla::ComputationDataHandle& max,
              const float quant_min_value, const float quant_max_value,
              xla::ComputationDataHandle* nudged_min,
              xla::ComputationDataHandle* nudged_max,
              xla::ComputationDataHandle* scale) {
  *scale = b->Div(b->Sub(max, min),
                  XlaHelpers::FloatLiteral(b, data_type,
                                           quant_max_value - quant_min_value));
  xla::ComputationDataHandle quant_min =
      XlaHelpers::FloatLiteral(b, data_type, quant_min_value);
  xla::ComputationDataHandle zero_point_from_min =
      b->Sub(quant_min, b->Div(min, *scale));
  xla::ComputationDataHandle quant_max =
      XlaHelpers::FloatLiteral(b, data_type, quant_max_value);
  xla::ComputationDataHandle nudged_zero_point =
      b->Select(b->Le(zero_point_from_min, quant_min), quant_min,
                b->Select(b->Ge(zero_point_from_min, quant_max), quant_max,
                          b->Round(zero_point_from_min)));
  *nudged_min = b->Mul(b->Sub(quant_min, nudged_zero_point), *scale);
  *nudged_max = b->Mul(b->Sub(quant_max, nudged_zero_point), *scale);
}

xla::ComputationDataHandle Quantize(
    xla::ComputationBuilder* b, const xla::ComputationDataHandle& input,
    const DataType data_type,
    const xla::ComputationDataHandle& nudged_input_min,
    const xla::ComputationDataHandle& nudged_input_max,
    const xla::ComputationDataHandle& input_scale) {
  xla::ComputationDataHandle one = XlaHelpers::FloatLiteral(b, data_type, 1.0f);
  xla::ComputationDataHandle inv_scale = b->Div(one, input_scale);
  xla::ComputationDataHandle half =
      XlaHelpers::FloatLiteral(b, data_type, 0.5f);

  xla::ComputationDataHandle clamped =
      b->Clamp(nudged_input_min, input, nudged_input_max);
  xla::ComputationDataHandle clamped_shifted =
      b->Sub(clamped, nudged_input_min);
  xla::ComputationDataHandle rounded =
      b->Floor(b->Add(b->Mul(clamped_shifted, inv_scale), half));
  return b->Add(b->Mul(rounded, input_scale), nudged_input_min);
}

class FakeQuantWithMinMaxArgsOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxArgsOp(OpKernelConstruction* ctx)
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

    float input_min, input_max;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min", &input_min));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max", &input_max));
    CpuNudge(input_min, input_max, quant_min_, quant_max_, &nudged_input_min_,
             &nudged_input_max_, &input_scale_);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle nudged_input_min =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_min_);
    xla::ComputationDataHandle nudged_input_max =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_max_);
    xla::ComputationDataHandle input_scale =
        XlaHelpers::FloatLiteral(b, data_type, input_scale_);
    xla::ComputationDataHandle output = Quantize(
        b, input, data_type, nudged_input_min, nudged_input_max, input_scale);
    ctx->SetOutput(0, output);
  }

 private:
  float quant_min_;
  float quant_max_;
  float nudged_input_min_;
  float nudged_input_max_;
  float input_scale_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxArgs"), FakeQuantWithMinMaxArgsOp);

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
    xla::ComputationDataHandle gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::ComputationDataHandle input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle nudged_input_min =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_min_);
    xla::ComputationDataHandle nudged_input_max =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_max_);

    xla::ComputationDataHandle between_nudged_min_max =
        b->And(b->Le(nudged_input_min, input), b->Le(input, nudged_input_max));
    xla::ComputationDataHandle zeroes = b->Broadcast(
        XlaHelpers::Zero(b, data_type), gradient_shape.dim_sizes());
    xla::ComputationDataHandle output =
        b->Select(between_nudged_min_max, gradient, zeroes);
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
    xla::ComputationDataHandle input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);
    xla::ComputationDataHandle input_min = ctx->Input(1);
    xla::ComputationDataHandle input_max = ctx->Input(2);

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::ComputationDataHandle output = Quantize(
        b, input, data_type, nudged_input_min, nudged_input_max, input_scale);
    ctx->SetOutput(0, output);
  }

 private:
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
    xla::ComputationDataHandle gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::ComputationDataHandle input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);
    xla::ComputationDataHandle input_min = ctx->Input(2);
    xla::ComputationDataHandle input_max = ctx->Input(3);

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::ComputationDataHandle between_nudged_min_max =
        b->And(b->Le(nudged_input_min, input), b->Le(input, nudged_input_max));
    xla::ComputationDataHandle zero = XlaHelpers::Zero(b, data_type);
    xla::ComputationDataHandle zeroes =
        b->Broadcast(zero, gradient_shape.dim_sizes());
    xla::ComputationDataHandle output0 =
        b->Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output0);

    xla::ComputationDataHandle below_min = b->Lt(input, nudged_input_min);
    xla::ComputationDataHandle output1 =
        b->ReduceAll(b->Select(below_min, gradient, zeroes), zero,
                     *ctx->GetOrCreateAdd(data_type));
    ctx->SetOutput(1, output1);

    xla::ComputationDataHandle above_max = b->Gt(input, nudged_input_max);
    xla::ComputationDataHandle output2 =
        b->ReduceAll(b->Select(above_max, gradient, zeroes), zero,
                     *ctx->GetOrCreateAdd(data_type));
    ctx->SetOutput(2, output2);
  }

 private:
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsGradient"),
                FakeQuantWithMinMaxVarsGradOp);

}  // namespace
}  // namespace tensorflow
