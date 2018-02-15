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

// XLA implementation of BatchNorm operations.
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class FusedBatchNormOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(
        ctx, FormatFromString(data_format_str, &data_format_),
        errors::InvalidArgument("Invalid data format: ", data_format_str));
    OP_REQUIRES(ctx,
                (data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW),
                errors::InvalidArgument(
                    "Unsupported data format ", ToString(data_format_),
                    "; supported formats are NHWC and NCHW"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::PrimitiveType scale_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(1), &scale_type));

    xla::ComputationBuilder* builder = ctx->builder();

    xla::ComputationDataHandle input = ctx->Input(0);
    TensorShape input_shape = ctx->InputShape(0);

    int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);

    // TODO(b/69928690): support mixed precision in the XLA batch normalization
    // operators. As a workaround, cast everything to the statistics type (which
    // may be more precise than the input type).
    input = builder->ConvertElementType(input, scale_type);

    if (is_training_) {
      xla::ComputationDataHandle output = builder->BatchNormTraining(
          input, ctx->Input(1), ctx->Input(2), epsilon_, feature_index);

      // In training mode, outputs the normalized value as well as the
      // calculated mean and variance.
      ctx->SetOutput(0, builder->ConvertElementType(
                            builder->GetTupleElement(output, 0), input_type));
      ctx->SetOutput(1, builder->GetTupleElement(output, 1));
      ctx->SetOutput(2, builder->GetTupleElement(output, 2));

      // Output 3 and 4 for "FusedBatchNorm" are currently marked as "reserved
      // space 1 & 2". They are used to pass the per-batch mean and
      // variance to the gradient. Here we maintain the same behavior by setting
      // them to the mean and variance calculated by BatchNormTraining.
      ctx->SetOutput(3, builder->GetTupleElement(output, 1));
      ctx->SetOutput(4, builder->GetTupleElement(output, 2));
    } else {
      xla::ComputationDataHandle output = builder->BatchNormInference(
          input, ctx->Input(1), ctx->Input(2), ctx->Input(3), ctx->Input(4),
          epsilon_, feature_index);
      ctx->SetOutput(0, builder->ConvertElementType(output, input_type));
      // Directly send input to output as mean and variance in inference mode.
      ctx->SetOutput(1, ctx->Input(3));
      ctx->SetOutput(2, ctx->Input(4));
      ctx->SetOutput(3, ctx->Input(3));
      ctx->SetOutput(4, ctx->Input(4));
    }
  }

 private:
  float epsilon_;
  TensorFormat data_format_;
  bool is_training_;
};

REGISTER_XLA_OP(Name("FusedBatchNorm"), FusedBatchNormOp);
REGISTER_XLA_OP(Name("FusedBatchNormV2"), FusedBatchNormOp);

class FusedBatchNormGradOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(
        ctx, FormatFromString(data_format_str, &data_format_),
        errors::InvalidArgument("Invalid data format: ", data_format_str));
    OP_REQUIRES(ctx,
                (data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW),
                errors::InvalidArgument(
                    "Unsupported data format ", ToString(data_format_),
                    "; supported formats are NHWC and NCHW"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    auto grad_backprop = ctx->Input(0);
    auto activations = ctx->Input(1);
    auto scale = ctx->Input(2);
    auto mean = ctx->Input(3);
    auto var = ctx->Input(4);

    TensorShape input_shape = ctx->InputShape(0);
    int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);

    DataType input_dtype = ctx->input_type(0);
    DataType scale_dtype = ctx->input_type(2);
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_dtype, &input_type));
    xla::PrimitiveType scale_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(scale_dtype, &scale_type));

    // TODO(b/69928690): support mixed precision in the XLA batch normalization
    // operators. For now, cast everything to the statistics type (which
    // may be more precise than the input type).
    grad_backprop = b->ConvertElementType(grad_backprop, scale_type);
    activations = b->ConvertElementType(activations, scale_type);

    xla::ComputationDataHandle x_backprop;
    xla::ComputationDataHandle scale_backprop;
    xla::ComputationDataHandle offset_backprop;
    if (is_training_) {
      xla::ComputationDataHandle output =
          b->BatchNormGrad(activations, scale, mean, var, grad_backprop,
                           epsilon_, feature_index);

      x_backprop = b->GetTupleElement(output, 0);
      scale_backprop = b->GetTupleElement(output, 1);
      offset_backprop = b->GetTupleElement(output, 2);
    } else {
      // Reduce over all dimensions except the feature dim.
      std::vector<int64> reduction_dims(input_shape.dims() - 1);
      std::iota(reduction_dims.begin(), reduction_dims.begin() + feature_index,
                0);
      std::iota(reduction_dims.begin() + feature_index, reduction_dims.end(),
                feature_index + 1);
      // offset_backprop  = sum(y_backprop)
      // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var +
      // epsilon))
      // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))
      offset_backprop =
          b->Reduce(grad_backprop, XlaHelpers::Zero(b, scale_dtype),
                    *ctx->GetOrCreateAdd(scale_dtype), reduction_dims);

      // scratch1 = rsqrt(pop_var + epsilon)
      auto neg_half = XlaHelpers::FloatLiteral(b, scale_dtype, -0.5);
      auto scratch1 =
          b->Pow(b->Add(var, b->ConstantR0<float>(epsilon_)), neg_half);

      // scratch2 = sum(y_backprop * (x - mean))
      auto scratch2 = b->Reduce(
          b->Mul(grad_backprop, b->Sub(activations, mean, {feature_index})),
          XlaHelpers::Zero(b, scale_dtype), *ctx->GetOrCreateAdd(scale_dtype),
          reduction_dims);

      x_backprop =
          b->Mul(grad_backprop, b->Mul(scratch1, scale), {feature_index});
      scale_backprop = b->Mul(scratch1, scratch2);
    }

    ctx->SetOutput(0, b->ConvertElementType(x_backprop, input_type));
    ctx->SetOutput(1, scale_backprop);
    ctx->SetOutput(2, offset_backprop);
    ctx->SetConstantOutput(3, Tensor(scale_dtype, {}));
    ctx->SetConstantOutput(4, Tensor(scale_dtype, {}));
  }

 private:
  TensorFormat data_format_;
  float epsilon_;
  bool is_training_;
};

REGISTER_XLA_OP(Name("FusedBatchNormGrad"), FusedBatchNormGradOp);
REGISTER_XLA_OP(Name("FusedBatchNormGradV2"), FusedBatchNormGradOp);

}  // namespace
}  // namespace tensorflow
