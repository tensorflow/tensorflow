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
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class FusedBatchNormOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    TensorFormat tensor_format;
    if (ctx->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(ctx, FormatFromString(data_format, &tensor_format),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          ctx, (tensor_format == FORMAT_NHWC || tensor_format == FORMAT_NCHW),
          errors::InvalidArgument("Not supported format"));
      feature_index_ = GetTensorFeatureDimIndex(/*num_dims=*/4, tensor_format);
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    if (is_training_) {
      xla::ComputationDataHandle output = ctx->builder()->BatchNormTraining(
          ctx->Input(0), ctx->Input(1), ctx->Input(2), epsilon_,
          feature_index_);

      // In training mode, outputs the normalized value as well as the
      // calculated mean and variance.
      for (int i = 0; i < 3; i++) {
        ctx->SetOutput(i, ctx->builder()->GetTupleElement(output, i));
      }
      // Output 3 and 4 for "FusedBatchNorm" are currently marked as "reserved
      // space 1 & 2". They are used to pass the per-batch mean and
      // variance to the gradient. Here we maintain the same behavior by setting
      // them to the mean and variance calculated by BatchNormTraining.
      ctx->SetOutput(3, ctx->builder()->GetTupleElement(output, 1));
      ctx->SetOutput(4, ctx->builder()->GetTupleElement(output, 2));
    } else {
      xla::ComputationDataHandle output = ctx->builder()->BatchNormInference(
          ctx->Input(0), ctx->Input(1), ctx->Input(2), ctx->Input(3),
          ctx->Input(4), epsilon_, feature_index_);
      ctx->SetOutput(0, output);
      // Directly send input to output as mean and variance in inference mode.
      ctx->SetOutput(1, ctx->Input(3));
      ctx->SetOutput(2, ctx->Input(4));
      ctx->SetOutput(3, ctx->Input(3));
      ctx->SetOutput(4, ctx->Input(4));
    }
  }

 private:
  float epsilon_;
  int64 feature_index_;
  bool is_training_;
};

REGISTER_XLA_OP(Name("FusedBatchNorm"), FusedBatchNormOp);

class FusedBatchNormGradOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    bool is_training;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
    CHECK(is_training) << "FusedBatchNormGradOp with is_training=False cannot "
                          "be used with XLA for now!";
    TensorFormat tensor_format;
    if (ctx->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(ctx, FormatFromString(data_format, &tensor_format),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          ctx, (tensor_format == FORMAT_NHWC || tensor_format == FORMAT_NCHW),
          errors::InvalidArgument("Not supported format"));
      feature_index_ = GetTensorFeatureDimIndex(4, tensor_format);
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto grad_output = ctx->Input(0);
    auto activation = ctx->Input(1);
    auto scale = ctx->Input(2);
    auto mean = ctx->Input(3);
    auto var = ctx->Input(4);
    xla::ComputationDataHandle output = ctx->builder()->BatchNormGrad(
        activation, scale, mean, var, grad_output, epsilon_, feature_index_);

    for (int i = 0; i < 3; i++) {
      ctx->SetOutput(i, ctx->builder()->GetTupleElement(output, i));
    }
    ctx->SetOutput(3, ctx->builder()->GetTupleElement(output, 1));
    ctx->SetOutput(4, ctx->builder()->GetTupleElement(output, 2));
  }

 private:
  float epsilon_;
  int64 feature_index_;
};

REGISTER_XLA_OP(Name("FusedBatchNormGrad"), FusedBatchNormGradOp);

}  // namespace
}  // namespace tensorflow
