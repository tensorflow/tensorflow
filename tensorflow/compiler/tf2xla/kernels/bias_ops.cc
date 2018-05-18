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

#include <numeric>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class BiasOp : public XlaOpKernel {
 public:
  explicit BiasOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format;
    if (ctx->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape bias_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias_shape),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias_shape.DebugString()));
    int feature_dim = (data_format_ == FORMAT_NHWC) ? input_shape.dims() - 1
                                                    : input_shape.dims() - 3;
    OP_REQUIRES(
        ctx, feature_dim >= 0,
        errors::InvalidArgument("Input tensor does not have enough dimensions "
                                "to contain the feature dimension"));
    OP_REQUIRES(
        ctx, bias_shape.dim_size(0) == input_shape.dim_size(feature_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias_shape.DebugString(), " vs. ", input_shape.DebugString()));

    xla::XlaOp result =
        ctx->builder()->Add(ctx->Input(0), ctx->Input(1), {feature_dim});
    ctx->SetOutput(0, result);
  }

 private:
  TensorFormat data_format_;
};

REGISTER_XLA_OP(Name("BiasAdd"), BiasOp);
REGISTER_XLA_OP(Name("BiasAddV1"), BiasOp);

class BiasAddGradOp : public XlaOpKernel {
 public:
  explicit BiasAddGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format;
    if (ctx->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape out_backprop_shape = ctx->InputShape(0);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(out_backprop_shape),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        out_backprop_shape.DebugString()));

    int feature_dim = (data_format_ == FORMAT_NHWC)
                          ? out_backprop_shape.dims() - 1
                          : out_backprop_shape.dims() - 3;
    OP_REQUIRES(
        ctx, feature_dim >= 0,
        errors::InvalidArgument("Input tensor does not have enough dimensions "
                                "to contain the feature dimension"));

    std::vector<int64> reduce_dims(out_backprop_shape.dims() - 1);
    std::iota(reduce_dims.begin(), reduce_dims.begin() + feature_dim, 0);
    std::iota(reduce_dims.begin() + feature_dim, reduce_dims.end(),
              feature_dim + 1);
    xla::XlaBuilder* const b = ctx->builder();
    const DataType accumulation_type =
        XlaHelpers::SumAccumulationType(input_type(0));
    auto converted =
        XlaHelpers::ConvertElementType(b, ctx->Input(0), accumulation_type);
    auto reduce =
        b->Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), reduce_dims);
    ctx->SetOutput(0, XlaHelpers::ConvertElementType(b, reduce, input_type(0)));
  }

 private:
  TensorFormat data_format_;
};

REGISTER_XLA_OP(Name("BiasAddGrad"), BiasAddGradOp);

}  // namespace
}  // namespace tensorflow
