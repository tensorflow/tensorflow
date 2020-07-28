/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

enum Interpolation { INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR };
enum Mode { FILL_REFLECT, FILL_WRAP, FILL_CONSTANT, FILL_NEAREST };

xla::XlaOp MapCoordinate(const xla::XlaOp& out, const int64_t length,
                         const Mode& mode) {
  xla::XlaOp in = out;

  if (mode == Mode::FILL_REFLECT) {
    const xla::XlaOp boundary = xla::ScalarLike(in, length * 2);
    const xla::XlaOp zero = xla::ScalarLike(in, 0.0);
    const xla::XlaOp one = xla::ScalarLike(in, 1.0);
    in = in - boundary * xla::ConvertElementType(
                             xla::ConvertElementType(in / boundary, xla::S32),
                             xla::F32);
    in = xla::Select(xla::Lt(in, zero), in + boundary, in);
    in = xla::Select(xla::Gt(in, xla::ScalarLike(in, length - 1)),
                     boundary - one - in, in);
    return xla::Clamp(xla::ScalarLike(in, 0.0), in,
                      xla::ScalarLike(in, length - 1));
  } else if (mode == Mode::FILL_WRAP) {
    const xla::XlaOp boundary = xla::ScalarLike(in, length);
    const xla::XlaOp zero = xla::ScalarLike(in, 0.0);
    in = in - boundary * xla::ConvertElementType(
                             xla::ConvertElementType(in / boundary, xla::S32),
                             xla::F32);
    in = xla::Select(xla::Lt(in, zero), in + boundary, in);
    return xla::Clamp(xla::ScalarLike(in, 0.0), in,
                      xla::ScalarLike(in, length - 1));
  } else if (mode == Mode::FILL_CONSTANT) {
    return in;
  } else if (mode == Mode::FILL_NEAREST) {
    return xla::Clamp(xla::ScalarLike(out, 0.0), out,
                      xla::ScalarLike(out, length - 1));
  }

  return in;
}

class ImageProjectiveTransformOp : public XlaOpKernel {
 public:
  explicit ImageProjectiveTransformOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    std::string interpolation_str;
    OP_REQUIRES_OK(context,
                   context->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "NEAREST") {
      interpolation_ = Interpolation::INTERPOLATION_NEAREST;
    } else if (interpolation_str == "BILINEAR") {
      interpolation_ = Interpolation::INTERPOLATION_BILINEAR;
    } else {
      LOG(ERROR) << "Invalid interpolation " << interpolation_str
                 << ". Supported types: NEAREST, BILINEAR";
    }
    std::string mode_str;
    OP_REQUIRES_OK(context, context->GetAttr("fill_mode", &mode_str));
    if (mode_str == "REFLECT") {
      fill_mode_ = Mode::FILL_REFLECT;
    } else if (mode_str == "WRAP") {
      fill_mode_ = Mode::FILL_WRAP;
    } else if (mode_str == "CONSTANT") {
      fill_mode_ = Mode::FILL_CONSTANT;
    } else if (mode_str == "NEAREST") {
      fill_mode_ = Mode::FILL_NEAREST;
    } else {
      LOG(ERROR) << "Invalid mode " << mode_str
                 << ". Supported types: REFLECT, WRAP, CONSTANT, NEAREST";
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Validate images and transforms shape.
    const TensorShape& images_shape = ctx->InputShape(0);
    const TensorShape& transforms_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, images_shape.dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));
    OP_REQUIRES(ctx,
                (TensorShapeUtils::IsMatrix(transforms_shape) &&
                 (transforms_shape.dim_size(0) == images_shape.dim_size(0) ||
                  transforms_shape.dim_size(0) == 1) &&
                 transforms_shape.dim_size(1) == 8),
                errors::InvalidArgument(
                    "Input transform should be num_images x 8 or 1 x 8"));

    // Validate output shape.
    TensorShape output_shape{images_shape.dim_size(1),
                             images_shape.dim_size(2)};
    ctx->ConstantInputAsShape(2, &output_shape);
    OP_REQUIRES(ctx, output_shape.dims() == 2,
                errors::InvalidArgument("output shape must be 1-dimensional",
                                        output_shape.DebugString()));
    OP_REQUIRES(ctx,
                output_shape.dim_size(0) > 0 && output_shape.dim_size(1) > 0,
                errors::InvalidArgument("output dimensions must be positive"));
    output_shape.InsertDim(0, images_shape.dim_size(0));
    output_shape.InsertDim(3, images_shape.dim_size(3));

    // Extract fill_value.
    double f;
    ctx->ConstantInputAsFloatScalar(3, &f);
    xla::XlaOp fill_value = xla::ScalarLike(ctx->Input(0), f);

    // Extract transforms.
    std::vector<xla::XlaOp> transforms(8);
    for (int i = 0; i < 8; ++i) {
      xla::XlaOp t =
          xla::Slice(ctx->Input(1), {0, i},
                     {transforms_shape.dim_size(0), i + 1}, /*strides=*/{1, 1});
      if (transforms_shape.dim_size(0) == 1) {
        // Case of shape [1, 8].
        t = xla::Reshape(t, /*new_sizes=*/{});
      } else {
        // Case of shape [b, 8].
        t = xla::Reshape(t, /*new_sizes=*/{transforms_shape.dim_size(0)});
        t = xla::BroadcastInDim(
            t,
            {output_shape.dim_size(0), output_shape.dim_size(1),
             output_shape.dim_size(2), output_shape.dim_size(3), 1},
            {0});
      }
      transforms[i] = t;
    }

    xla::XlaOp output = DoProjectiveTransform(
        ctx, ctx->Input(0), transforms, fill_value, images_shape, output_shape);
    ctx->SetOutput(0, output);
    return;
  }

 private:
  Interpolation interpolation_;
  Mode fill_mode_;

  xla::XlaOp DoProjectiveTransform(XlaOpKernelContext* ctx,
                                   const xla::XlaOp& input,
                                   const std::vector<xla::XlaOp>& transforms,
                                   const xla::XlaOp& fill_value,
                                   const TensorShape& input_tensor_shape,
                                   const TensorShape& output_tensor_shape) {
    xla::XlaBuilder* builder = ctx->builder();

    const int64_t batch_size = input_tensor_shape.dim_size(0);
    const int64_t num_channels = input_tensor_shape.dim_size(3);
    const int64_t input_height = input_tensor_shape.dim_size(1);
    const int64_t input_width = input_tensor_shape.dim_size(2);
    const int64_t output_height = output_tensor_shape.dim_size(1);
    const int64_t output_width = output_tensor_shape.dim_size(2);

    // Manipulate each dim's indices of shape
    // [batch_size, output_height, output_width, num_channels, 1].
    xla::XlaOp batch_indices = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(xla::S32, {batch_size, output_height,
                                             output_width, num_channels, 1}),
        /*iota_dimension=*/0);
    xla::XlaOp output_y_indices = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(xla::F32, {batch_size, output_height,
                                             output_width, num_channels, 1}),
        /*iota_dimension=*/1);
    xla::XlaOp output_x_indices = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(xla::F32, {batch_size, output_height,
                                             output_width, num_channels, 1}),
        /*iota_dimension=*/2);
    xla::XlaOp channel_indices = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(xla::S32, {batch_size, output_height,
                                             output_width, num_channels, 1}),
        /*iota_dimension=*/3);

    xla::XlaOp projection = transforms[6] * output_x_indices +
                            transforms[7] * output_y_indices +
                            xla::One(builder, xla::F32);
    xla::XlaOp zero_projection =
        xla::Eq(projection, xla::Zero(builder, xla::F32));
    projection =
        xla::Select(zero_projection, xla::One(builder, xla::F32), projection);
    zero_projection = xla::Reshape(
        zero_projection,
        /*new_sizes=*/{batch_size, output_height, output_width, num_channels});

    xla::XlaOp input_x_indices =
        (transforms[0] * output_x_indices + transforms[1] * output_y_indices +
         transforms[2]) /
        projection;
    xla::XlaOp input_y_indices =
        (transforms[3] * output_x_indices + transforms[4] * output_y_indices +
         transforms[5]) /
        projection;

    input_x_indices = MapCoordinate(input_x_indices, input_width, fill_mode_);
    input_y_indices = MapCoordinate(input_y_indices, input_height, fill_mode_);

    auto ReshapeToOutputShape = [&](xla::XlaOp input) {
      return xla::Reshape(input, /*new_sizes=*/{batch_size, output_height,
                                                output_width, num_channels});
    };

    // Lambda function that gathers pixels with given indices.
    auto GatherWithFillValue = [&](xla::XlaOp y_indices, xla::XlaOp x_indices,
                                   xla::XlaOp fill_value) {
      y_indices = xla::ConvertElementType(y_indices, xla::S32);
      x_indices = xla::ConvertElementType(x_indices, xla::S32);
      const xla::XlaOp zero = xla::ScalarLike(x_indices, 0);
      const xla::XlaOp input_height_minus_one =
          xla::ScalarLike(y_indices, input_tensor_shape.dim_size(1) - 1);
      const xla::XlaOp input_width_minus_one =
          xla::ScalarLike(x_indices, input_tensor_shape.dim_size(2) - 1);

      xla::XlaOp in_bound = ReshapeToOutputShape(xla::And(
          xla::Ge(x_indices, zero), xla::Le(x_indices, input_width_minus_one),
          xla::Ge(y_indices, zero),
          xla::Le(y_indices, input_height_minus_one)));

      // Clamp indices within boundary.
      x_indices = xla::Clamp(zero, x_indices, input_width_minus_one);
      y_indices = xla::Clamp(zero, y_indices, input_height_minus_one);

      const xla::XlaOp indices = xla::ConcatInDim(
          builder, {batch_indices, y_indices, x_indices, channel_indices},
          /*dimension=*/4);

      xla::XlaOp output;
      XlaGather(input, input_tensor_shape, indices,
                {batch_size, output_height, output_width, num_channels, 4},
                /*axis=*/0, /*indices_are_nd=*/true, ctx->input_type(0),
                DT_INT32, builder, &output);

      // Fill out of boundary pixels with fill_value.
      output = xla::Select(in_bound, output, fill_value);
      return output;
    };

    xla::XlaOp output;
    if (interpolation_ == Interpolation::INTERPOLATION_NEAREST) {
      // Round coordinate for nearest neighbor interpolation.
      input_x_indices = xla::RoundToEven(input_x_indices);
      input_y_indices = xla::RoundToEven(input_y_indices);
      output =
          GatherWithFillValue(input_y_indices, input_x_indices, fill_value);
    } else if (interpolation_ == Interpolation::INTERPOLATION_BILINEAR) {
      const xla::XlaOp one = xla::ScalarLike(input_x_indices, 1.0);
      const xla::XlaOp floor_x_indices = xla::Floor(input_x_indices);
      const xla::XlaOp floor_y_indices = xla::Floor(input_y_indices);
      const xla::XlaOp ceil_x_indices = floor_x_indices + one;
      const xla::XlaOp ceil_y_indices = floor_y_indices + one;
      const xla::XlaOp floor_y_values =
          ReshapeToOutputShape(ceil_x_indices - input_x_indices) *
              xla::ConvertElementType(
                  GatherWithFillValue(floor_y_indices, floor_x_indices,
                                      fill_value),
                  xla::F32) +
          ReshapeToOutputShape(input_x_indices - floor_x_indices) *
              xla::ConvertElementType(
                  GatherWithFillValue(floor_y_indices, ceil_x_indices,
                                      fill_value),
                  xla::F32);
      const xla::XlaOp ceil_y_values =
          ReshapeToOutputShape(ceil_x_indices - input_x_indices) *
              xla::ConvertElementType(
                  GatherWithFillValue(ceil_y_indices, floor_x_indices,
                                      fill_value),
                  xla::F32) +
          ReshapeToOutputShape(input_x_indices - floor_x_indices) *
              xla::ConvertElementType(
                  GatherWithFillValue(ceil_y_indices, ceil_x_indices,
                                      fill_value),
                  xla::F32);
      output = xla::ConvertElementType(
          ReshapeToOutputShape(ceil_y_indices - input_y_indices) *
                  floor_y_values +
              ReshapeToOutputShape(input_y_indices - floor_y_indices) *
                  ceil_y_values,
          ctx->input_xla_type(0));
    }

    // Fill pixels of zero projection with fill_value.
    output = xla::Select(zero_projection, fill_value, output);
    return output;
  }
};

REGISTER_XLA_OP(Name("ImageProjectiveTransformV3")
                    .CompileTimeConstantInput("output_shape")
                    .CompileTimeConstantInput("fill_value"),
                ImageProjectiveTransformOp);

}  // namespace
}  // namespace tensorflow
