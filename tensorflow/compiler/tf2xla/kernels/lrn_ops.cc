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

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
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

    xla::ComputationBuilder* builder = ctx->builder();
    xla::ComputationDataHandle input = ctx->Input(0);

    // sqr_sum[a, b, c, d] =
    //    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    // output = input / (bias + alpha * sqr_sum) ** beta

    // We use a window of depth_radius_ * 2 + 1, to account for the current
    // element and a depth_radius_ on either side.
    auto squared = builder->Mul(input, input);
    auto sqr_sum = builder->ReduceWindow(
        squared, XlaHelpers::Zero(builder, input_type(0)),
        *ctx->GetOrCreateAdd(input_type(0)),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);

    auto scale = builder->Pow(
        builder->Add(builder->ConstantR0<float>(bias_),
                     builder->Mul(builder->ConstantR0<float>(alpha_), sqr_sum)),
        builder->ConstantR0<float>(-beta_));

    ctx->SetOutput(0, builder->Mul(input, scale));
  }

 private:
  int64 depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_XLA_OP("LRN", LRNOp);

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

    xla::ComputationBuilder* builder = ctx->builder();
    xla::ComputationDataHandle in_grads = ctx->Input(0);
    xla::ComputationDataHandle in_image = ctx->Input(1);
    xla::ComputationDataHandle out_image = ctx->Input(2);

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

    auto squared = builder->Mul(in_image, in_image);
    auto sqr_sum = builder->ReduceWindow(
        squared, XlaHelpers::Zero(builder, input_type(0)),
        *ctx->GetOrCreateAdd(input_type(0)),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);

    auto norm =
        builder->Add(builder->ConstantR0<float>(bias_),
                     builder->Mul(builder->ConstantR0<float>(alpha_), sqr_sum));

    auto dy = builder->Mul(
        builder->Mul(builder->ConstantR0<float>(-2.0f * alpha_ * beta_),
                     builder->Div(out_image, norm)),
        in_grads);

    auto dy_reduced = builder->ReduceWindow(
        dy, XlaHelpers::Zero(builder, input_type(0)),
        *ctx->GetOrCreateAdd(input_type(0)),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);

    xla::ComputationDataHandle gradients = builder->Add(
        builder->Mul(in_image, dy_reduced),
        builder->Mul(in_grads,
                     builder->Pow(norm, builder->ConstantR0<float>(-beta_))));

    ctx->SetOutput(0, gradients);
  }

 private:
  int64 depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_XLA_OP("LRNGrad", LRNGradOp);

}  // anonymous namespace
}  // namespace tensorflow
