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

// XLA-specific Ops for softmax.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace {

class SoftmaxOp : public XlaOpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    log_ = str_util::StartsWith(type_string(), "Log");
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape logits_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(logits_shape),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_shape.DebugString()));

    // Major dimensions are batch dimensions, minor dimension is the class
    // dimension.
    std::vector<int64> batch_dims(logits_shape.dims() - 1);
    std::iota(batch_dims.begin(), batch_dims.end(), 0);
    const int kClassDim = logits_shape.dims() - 1;

    const DataType type = input_type(0);
    const xla::PrimitiveType xla_type = ctx->input_xla_type(0);
    auto logits = ctx->Input(0);

    xla::XlaBuilder* const b = ctx->builder();
    const xla::XlaComputation& max_func = *ctx->GetOrCreateMax(type);

    // Find the max in each batch, resulting in a tensor of shape [batch]
    auto logits_max =
        xla::Reduce(logits, xla::MinValue(b, xla_type), max_func, {kClassDim});
    // Subtract the max in batch b from every element in batch b. Broadcasts
    // along the batch dimension.
    auto shifted_logits = xla::Sub(logits, logits_max, batch_dims);
    auto exp_shifted = xla::Exp(shifted_logits);
    const DataType accumulation_type = XlaHelpers::SumAccumulationType(type);
    xla::PrimitiveType xla_accumulation_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(accumulation_type,
                                                &xla_accumulation_type));
    auto converted =
        xla::ConvertElementType(exp_shifted, xla_accumulation_type);
    auto reduce =
        xla::Reduce(converted, xla::Zero(b, xla_accumulation_type),
                    *ctx->GetOrCreateAdd(accumulation_type), {kClassDim});
    auto sum = XlaHelpers::ConvertElementType(b, reduce, type);
    auto softmax =
        log_
            // softmax = shifted_logits - log(sum(exp(shifted_logits)))
            ? xla::Sub(shifted_logits, xla::Log(sum), batch_dims)
            // softmax = exp(shifted_logits) / sum(exp(shifted_logits))
            : xla::Div(exp_shifted, sum, batch_dims);
    ctx->SetOutput(0, softmax);
  }

 private:
  bool log_;
};

REGISTER_XLA_OP(Name("Softmax"), SoftmaxOp);
REGISTER_XLA_OP(Name("LogSoftmax"), SoftmaxOp);

std::pair<xla::XlaOp, xla::XlaOp> CrossEntropyWithLogits(
    XlaOpKernelContext* ctx, DataType type, xla::PrimitiveType xla_type,
    xla::XlaOp logits, xla::XlaOp labels) {
  const xla::XlaComputation& max_func = *ctx->GetOrCreateMax(type);

  const int kBatchDim = 0;
  const int kClassDim = 1;

  xla::XlaBuilder* b = ctx->builder();
  // Find the max in each batch, resulting in a tensor of shape [batch]
  auto logits_max =
      xla::Reduce(logits, xla::MinValue(b, xla_type), max_func, {kClassDim});

  // Subtract the max in batch b from every element in batch b.
  // Broadcasts along the batch dimension.
  auto shifted_logits = xla::Sub(logits, logits_max, {kBatchDim});

  // exp(logits - max_logits)
  auto exp_shifted_logits = xla::Exp(shifted_logits);

  // sum_{class} (exp(logits - max_logits))
  const DataType accumulation_type = XlaHelpers::SumAccumulationType(type);
  auto converted =
      XlaHelpers::ConvertElementType(b, exp_shifted_logits, accumulation_type);
  auto reduce =
      xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), {kClassDim});
  auto sum_exp = XlaHelpers::ConvertElementType(b, reduce, type);

  // log(sum(exp(logits - max_logits)))
  auto log_sum_exp = xla::Log(sum_exp);

  // sum(-labels *
  //    ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
  // along classes
  // (The subtraction broadcasts along the batch dimension.)
  auto sub = xla::Sub(shifted_logits, log_sum_exp, {kBatchDim});
  auto mul = xla::Mul(xla::Neg(labels), sub);
  auto sum =
      xla::Reduce(XlaHelpers::ConvertElementType(b, mul, accumulation_type),
                  XlaHelpers::Zero(b, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), {kClassDim});
  auto loss = XlaHelpers::ConvertElementType(b, sum, type);

  // backprop: prob - labels, where
  //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
  //     (where the division broadcasts along the batch dimension)
  xla::XlaOp backprop =
      xla::Sub(xla::Div(exp_shifted_logits, sum_exp, {kBatchDim}), labels);
  return {loss, backprop};
}

class SoftmaxXentWithLogitsOp : public XlaOpKernel {
 public:
  explicit SoftmaxXentWithLogitsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape logits_shape = ctx->InputShape(0);
    const TensorShape labels_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, logits_shape.IsSameSize(labels_shape),
                errors::InvalidArgument(
                    "logits and labels must be same size: logits_size=",
                    logits_shape.DebugString(),
                    " labels_size=", labels_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits must be 2-dimensional"));
    // As we already tested that both inputs have the same shape no need to
    // check that "labels" is a matrix too.

    const DataType type = input_type(0);
    const xla::PrimitiveType xla_type = ctx->input_xla_type(0);
    auto logits = ctx->Input(0);
    auto labels = ctx->Input(1);

    xla::XlaOp loss, backprop;
    std::tie(loss, backprop) =
        CrossEntropyWithLogits(ctx, type, xla_type, logits, labels);
    ctx->SetOutput(0, loss);
    ctx->SetOutput(1, backprop);
  }
};

REGISTER_XLA_OP(Name("SoftmaxCrossEntropyWithLogits"), SoftmaxXentWithLogitsOp);

class SparseSoftmaxXentWithLogitsOp : public XlaOpKernel {
 public:
  explicit SparseSoftmaxXentWithLogitsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape logits_shape = ctx->InputShape(0);
    const TensorShape labels_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_shape),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels_shape.DebugString()));
    OP_REQUIRES(ctx, logits_shape.dim_size(0) == labels_shape.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits_shape.DebugString(), " and labels shape ",
                    labels_shape.DebugString()));
    OP_REQUIRES(ctx, logits_shape.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits_shape.DebugString()));

    int64 batch_size = logits_shape.dim_size(0);
    int64 depth = logits_shape.dim_size(1);

    const DataType logits_type = input_type(0);
    const xla::PrimitiveType xla_logits_type = ctx->input_xla_type(0);
    const DataType indices_type = input_type(1);

    xla::XlaOp indices = ctx->Input(1);

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp labels;
    OP_REQUIRES_OK(ctx,
                   XlaHelpers::OneHot(
                       builder, depth, /*axis=*/1, input_type(1), labels_shape,
                       indices, XlaHelpers::One(builder, logits_type),
                       XlaHelpers::Zero(builder, logits_type), &labels));

    // If any of the indices are out of range, we must populate the labels with
    // NaNs to obey the interface contract of
    // tf.nn.sparse_softmax_cross_entropy_with_logits.
    // Builds a vector of {batch_size} that is 0 if the index is in range, or
    // NaN otherwise; then add that vector to the labels to force out-of-range
    // values to NaNs.
    xla::XlaOp nan_or_zero = xla::Select(
        xla::And(xla::Le(XlaHelpers::Zero(builder, indices_type), indices),
                 xla::Lt(indices, XlaHelpers::IntegerLiteral(
                                      builder, indices_type, depth))),
        xla::Broadcast(XlaHelpers::Zero(builder, logits_type), {batch_size}),
        xla::Broadcast(XlaHelpers::FloatLiteral(builder, logits_type, NAN),
                       {batch_size}));
    labels = xla::Add(labels, nan_or_zero, {0});

    xla::XlaOp loss, backprop;
    std::tie(loss, backprop) = CrossEntropyWithLogits(
        ctx, logits_type, xla_logits_type, ctx->Input(0), labels);
    ctx->SetOutput(0, loss);
    ctx->SetOutput(1, backprop);
  }
};

REGISTER_XLA_OP(Name("SparseSoftmaxCrossEntropyWithLogits"),
                SparseSoftmaxXentWithLogitsOp);

}  // namespace
}  // namespace tensorflow
