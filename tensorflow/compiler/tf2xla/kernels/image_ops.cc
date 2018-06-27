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

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"

namespace tensorflow {
namespace {

// Converts 'input' from RGB format to HSV format.
// 'shape' is the shape of the red/green/blue tensors.
std::array<xla::XlaOp, 3> RGBToHSV(XlaOpKernelContext* ctx, xla::XlaBuilder* b,
                                   const std::array<xla::XlaOp, 3>& rgb,
                                   DataType dtype, const TensorShape& shape) {
  auto zero = XlaHelpers::Zero(b, dtype);
  auto one = XlaHelpers::One(b, dtype);

  auto red = rgb[0];
  auto green = rgb[1];
  auto blue = rgb[2];
  auto value = xla::Max(xla::Max(red, green), blue);
  auto minimum = xla::Min(xla::Min(red, green), blue);
  auto range = xla::Sub(value, minimum);

  auto zeros = xla::Broadcast(zero, shape.dim_sizes());
  auto saturation =
      xla::Select(xla::Gt(value, zero), xla::Div(range, value), zeros);

  auto norm = xla::Div(XlaHelpers::FloatLiteral(b, dtype, 1.0 / 6.0), range);

  auto hue =
      xla::Select(xla::Eq(green, value),
                  xla::Add(xla::Mul(norm, xla::Sub(blue, red)),
                           XlaHelpers::FloatLiteral(b, dtype, 2.0 / 6.0)),
                  xla::Add(xla::Mul(norm, xla::Sub(red, green)),
                           XlaHelpers::FloatLiteral(b, dtype, 4.0 / 6.0)));
  hue = xla::Select(xla::Eq(red, value), xla::Mul(norm, xla::Sub(green, blue)),
                    hue);
  hue = xla::Select(xla::Gt(range, zero), hue, zeros);
  hue = xla::Select(xla::Lt(hue, zero), xla::Add(hue, one), hue);
  return {hue, saturation, value};
}

// Converts 'input' from HSV format to RGB format.
std::array<xla::XlaOp, 3> HSVToRGB(xla::XlaBuilder* b,
                                   const std::array<xla::XlaOp, 3>& hsv,
                                   DataType dtype) {
  xla::XlaOp hue = hsv[0];
  xla::XlaOp saturation = hsv[1];
  xla::XlaOp value = hsv[2];
  auto zero = XlaHelpers::Zero(b, dtype);
  auto one = XlaHelpers::FloatLiteral(b, dtype, 1.0);
  auto two = XlaHelpers::FloatLiteral(b, dtype, 2.0);
  auto three = XlaHelpers::FloatLiteral(b, dtype, 3.0);
  auto four = XlaHelpers::FloatLiteral(b, dtype, 4.0);
  auto six = XlaHelpers::FloatLiteral(b, dtype, 6.0);

  auto dh = xla::Mul(hue, six);
  auto dr = xla::Clamp(zero, xla::Sub(xla::Abs(xla::Sub(dh, three)), one), one);
  auto dg = xla::Clamp(zero, xla::Sub(two, xla::Abs(xla::Sub(dh, two))), one);
  auto db = xla::Clamp(zero, xla::Sub(two, xla::Abs(xla::Sub(dh, four))), one);
  auto one_minus_s = xla::Sub(one, saturation);

  auto red = xla::Mul(xla::Add(one_minus_s, xla::Mul(saturation, dr)), value);
  auto green = xla::Mul(xla::Add(one_minus_s, xla::Mul(saturation, dg)), value);
  auto blue = xla::Mul(xla::Add(one_minus_s, xla::Mul(saturation, db)), value);
  return {red, green, blue};
}

class RGBToHSVOp : public XlaOpKernel {
 public:
  explicit RGBToHSVOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input_shape.DebugString()));
    int channel_dim = input_shape.dims() - 1;
    int64 channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::FailedPrecondition("input must have 3 channels but input has ",
                                   channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);

    xla::XlaOp red =
        b->SliceInDim(input, /*start_index=*/0, /*limit_index=*/1, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp green =
        b->SliceInDim(input, /*start_index=*/1, /*limit_index=*/2, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp blue =
        b->SliceInDim(input, /*start_index=*/2, /*limit_index=*/3, /*stride=*/1,
                      /*dimno=*/channel_dim);
    TensorShape channel_shape = input_shape;
    channel_shape.set_dim(channel_dim, 1);
    auto hsv = RGBToHSV(context, b, {red, green, blue}, context->input_type(0),
                        channel_shape);

    context->SetOutput(0, xla::ConcatInDim(b, hsv, channel_dim));
  }
};
REGISTER_XLA_OP(Name("RGBToHSV"), RGBToHSVOp);

class HSVToRGBOp : public XlaOpKernel {
 public:
  explicit HSVToRGBOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input_shape.DebugString()));
    int channel_dim = input_shape.dims() - 1;
    int64 channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::FailedPrecondition("input must have 3 channels but input has ",
                                   channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);
    xla::XlaOp hue =
        b->SliceInDim(input, /*start_index=*/0, /*limit_index=*/1, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp saturation =
        b->SliceInDim(input, /*start_index=*/1, /*limit_index=*/2, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp value =
        b->SliceInDim(input, /*start_index=*/2, /*limit_index=*/3, /*stride=*/1,
                      /*dimno=*/channel_dim);

    auto rgb = HSVToRGB(context->builder(), {hue, saturation, value},
                        context->input_type(0));

    context->SetOutput(0, xla::ConcatInDim(b, rgb, channel_dim));
  }
};
REGISTER_XLA_OP(Name("HSVToRGB"), HSVToRGBOp);

class AdjustContrastOpV2 : public XlaOpKernel {
 public:
  explicit AdjustContrastOpV2(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape& input_shape = context->InputShape(0);
    const TensorShape& factor_shape = context->InputShape(1);
    OP_REQUIRES(context, input_shape.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input_shape.DebugString()));
    int height_dim = input_shape.dims() - 3;
    int width_dim = input_shape.dims() - 2;
    int channel_dim = input_shape.dims() - 1;
    const int64 height = input_shape.dim_size(height_dim);
    const int64 width = input_shape.dim_size(width_dim);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor_shape),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor_shape.DebugString()));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);
    xla::XlaOp factor = context->Input(1);

    DataType type = context->input_type(0);

    const DataType accumulation_type = XlaHelpers::SumAccumulationType(type);
    auto converted =
        XlaHelpers::ConvertElementType(b, input, accumulation_type);
    auto reduce = xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                              *context->GetOrCreateAdd(accumulation_type),
                              {height_dim, width_dim});
    auto output = XlaHelpers::ConvertElementType(b, reduce, type);
    output =
        xla::Div(output, XlaHelpers::FloatLiteral(b, type, height * width));

    std::vector<int64> broadcast_dims(input_shape.dims() - 2);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    broadcast_dims.back() = channel_dim;
    output =
        xla::Add(xla::Mul(input, factor),
                 xla::Mul(output, xla::Sub(XlaHelpers::One(b, type), factor)),
                 broadcast_dims);
    context->SetOutput(0, output);
  }
};
REGISTER_XLA_OP(Name("AdjustContrastv2"), AdjustContrastOpV2);

class AdjustSaturationOp : public XlaOpKernel {
 public:
  explicit AdjustSaturationOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape& input_shape = context->InputShape(0);
    const TensorShape& scale_shape = context->InputShape(1);
    OP_REQUIRES(context, input_shape.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input_shape.DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale_shape),
                errors::InvalidArgument("scale must be scalar: ",
                                        scale_shape.DebugString()));
    const int channel_dim = input_shape.dims() - 1;
    const int64 channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);
    xla::XlaOp scale = context->Input(1);

    DataType type = context->input_type(0);

    xla::XlaOp red =
        b->SliceInDim(input, /*start_index=*/0, /*limit_index=*/1, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp green =
        b->SliceInDim(input, /*start_index=*/1, /*limit_index=*/2, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp blue =
        b->SliceInDim(input, /*start_index=*/2, /*limit_index=*/3, /*stride=*/1,
                      /*dimno=*/channel_dim);
    TensorShape channel_shape = input_shape;
    channel_shape.set_dim(channel_dim, 1);
    auto hsv = RGBToHSV(context, b, {red, green, blue}, context->input_type(0),
                        channel_shape);

    hsv[1] = xla::Clamp(XlaHelpers::Zero(b, type), xla::Mul(hsv[1], scale),
                        XlaHelpers::One(b, type));

    auto rgb = HSVToRGB(context->builder(), hsv, context->input_type(0));

    context->SetOutput(0, xla::ConcatInDim(b, rgb, channel_dim));
  }
};
REGISTER_XLA_OP(Name("AdjustSaturation"), AdjustSaturationOp);

class AdjustHueOp : public XlaOpKernel {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape& input_shape = context->InputShape(0);
    const TensorShape& delta_shape = context->InputShape(1);
    OP_REQUIRES(context, input_shape.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input_shape.DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta_shape),
                errors::InvalidArgument("delta must be scalar: ",
                                        delta_shape.DebugString()));
    const int channel_dim = input_shape.dims() - 1;
    const int64 channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);
    xla::XlaOp delta = context->Input(1);

    DataType type = context->input_type(0);

    xla::XlaOp red =
        b->SliceInDim(input, /*start_index=*/0, /*limit_index=*/1, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp green =
        b->SliceInDim(input, /*start_index=*/1, /*limit_index=*/2, /*stride=*/1,
                      /*dimno=*/channel_dim);
    xla::XlaOp blue =
        b->SliceInDim(input, /*start_index=*/2, /*limit_index=*/3, /*stride=*/1,
                      /*dimno=*/channel_dim);
    TensorShape channel_shape = input_shape;
    channel_shape.set_dim(channel_dim, 1);
    auto hsv = RGBToHSV(context, b, {red, green, blue}, context->input_type(0),
                        channel_shape);

    auto zero = XlaHelpers::Zero(b, type);
    auto one = XlaHelpers::One(b, type);

    auto& hue = hsv[0];
    hue = xla::Rem(xla::Add(hsv[0], delta), one);
    hue =
        xla::Select(xla::Lt(hue, zero), xla::Rem(xla::Add(one, hue), one), hue);

    auto rgb = HSVToRGB(context->builder(), hsv, context->input_type(0));

    context->SetOutput(0, xla::ConcatInDim(b, rgb, channel_dim));
  }
};
REGISTER_XLA_OP(Name("AdjustHue"), AdjustHueOp);

}  // namespace
}  // namespace tensorflow
