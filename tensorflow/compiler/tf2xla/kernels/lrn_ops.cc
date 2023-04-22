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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

// Local response normalization
class LRNOp : public XlaOpKernel {
 public:
  explicit LRNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("depth_radius", &depth_radius_));

    // TODO(phawkins): handle non-float types for attributes.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape in_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, in_shape.dims() == 4,
                errors::InvalidArgument("in must be 4-dimensional"));

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);

    // sqr_sum[a, b, c, d] =
    //    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    // output = input / (bias + alpha * sqr_sum) ** beta

    // We use a window of depth_radius_ * 2 + 1, to account for the current
    // element and a depth_radius_ on either side.
    auto accumulation_type = XlaHelpers::SumAccumulationType(input_type(0));
    auto converted = XlaHelpers::ConvertElementType(input, accumulation_type);
    auto squared = xla::Mul(converted, converted);
    auto reduce = xla::ReduceWindow(
        squared, XlaHelpers::Zero(builder, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);
    auto sqr_sum = XlaHelpers::ConvertElementType(reduce, input_type(0));

    auto scale = xla::Pow(
        xla::Add(xla::ConstantR0<float>(builder, bias_),
                 xla::Mul(xla::ConstantR0<float>(builder, alpha_), sqr_sum)),
        xla::ConstantR0<float>(builder, -beta_));

    ctx->SetOutput(0, xla::Mul(input, scale));
  }

 private:
  int64 depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_XLA_OP(Name("LRN"), LRNOp);

class LRNGradOp : public XlaOpKernel {
 public:
  explicit LRNGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("depth_radius", &depth_radius_));

    // TODO(phawkins): handle non-float types for attributes.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape in_grads_shape = ctx->InputShape(0);
    const TensorShape in_image_shape = ctx->InputShape(1);
    const TensorShape out_image_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx, in_grads_shape.dims() == 4 && in_image_shape.dims() == 4,
                errors::InvalidArgument("inputs must be 4-dimensional"));
    const int64 batch = in_grads_shape.dim_size(0);
    const int64 rows = in_grads_shape.dim_size(1);
    const int64 cols = in_grads_shape.dim_size(2);
    const int64 depth = in_grads_shape.dim_size(3);
    OP_REQUIRES(
        ctx, in_image_shape.dim_size(0) == batch &&
                 in_image_shape.dim_size(1) == rows &&
                 in_image_shape.dim_size(2) == cols &&
                 in_image_shape.dim_size(3) == depth &&
                 out_image_shape.dim_size(0) == batch &&
                 out_image_shape.dim_size(1) == rows &&
                 out_image_shape.dim_size(2) == cols &&
                 out_image_shape.dim_size(3) == depth,
        errors::InvalidArgument(
            "input_grads, input_image, and out_image should have the same "
            "shape"));

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp in_grads = ctx->Input(0);
    xla::XlaOp in_image = ctx->Input(1);
    xla::XlaOp out_image = ctx->Input(2);

    // This code is ported from tensorflow/core/kernels/lrn_op.cc. In Python
    // pseudo-code, the Eigen code does this for each spatial position:
    // grads = [0.0] * depth
    // for j in range(depth):
    //   depth_begin = max(0, j - depth_radius)
    //   depth_end = min(depth, j + depth_radius + 1)
    //
    //   norm = 0
    //   for k in range(depth_begin, depth_end):
    //     norm += in_image[k] * in_image[k]
    //   norm = alpha * norm + bias
    //
    //   for k in range(depth_begin, depth_end):
    //     dyi = -2.0 * alpha * beta * in_image[k] * out_image[j] / norm
    //     if k == j:
    //       dyi += norm ** (-beta)
    //     dyi *= out_grads[j]
    //     grads[k] += dyi

    auto accumulation_type = XlaHelpers::SumAccumulationType(input_type(0));
    auto converted =
        XlaHelpers::ConvertElementType(in_image, accumulation_type);
    auto squared = xla::Mul(converted, converted);
    auto reduce = xla::ReduceWindow(
        squared, XlaHelpers::Zero(builder, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);
    auto sqr_sum = XlaHelpers::ConvertElementType(reduce, input_type(0));

    auto norm =
        xla::Add(xla::ConstantR0<float>(builder, bias_),
                 xla::Mul(xla::ConstantR0<float>(builder, alpha_), sqr_sum));

    auto dy = xla::Mul(
        xla::Mul(xla::ConstantR0<float>(builder, -2.0f * alpha_ * beta_),
                 xla::Div(out_image, norm)),
        in_grads);

    auto converted_dy = XlaHelpers::ConvertElementType(dy, accumulation_type);
    auto dy_reduce = xla::ReduceWindow(
        converted_dy, XlaHelpers::Zero(builder, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);
    auto dy_reduced = XlaHelpers::ConvertElementType(dy_reduce, input_type(0));

    xla::XlaOp gradients = xla::Add(
        xla::Mul(in_image, dy_reduced),
        xla::Mul(in_grads,
                 xla::Pow(norm, xla::ConstantR0<float>(builder, -beta_))));

    ctx->SetOutput(0, gradients);
  }

 private:
  int64 depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_XLA_OP(Name("LRNGrad"), LRNGradOp);

}  // anonymous namespace
}  // namespace tensorflow
