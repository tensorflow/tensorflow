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

#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/comparison_util.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

// Builds a LowerBound or UpperBound op, the distinction lying in
// comparison_direction: GT => LowerBoundOp, GE => UpperBoundOp.
// Note that this is an O(MN) algorithm: all entries in each sorted_inputs row
// are considered, and their sorted nature is not fully exploited.
void BuildLowerUpperBoundOp(XlaOpKernelContext* ctx, DataType out_dtype,
                            xla::ComparisonDirection comparison_direction) {
  const TensorShape sorted_inputs_shape = ctx->InputShape("sorted_inputs");
  const TensorShape values_shape = ctx->InputShape("values");
  const xla::XlaOp sorted_inputs = ctx->Input("sorted_inputs");
  const xla::XlaOp values = ctx->Input("values");

  // We are assuming both inputs are 2D, which they will be given the current
  // implementation of tf.searchsorted.
  OP_REQUIRES(ctx, sorted_inputs_shape.dims() == 2,
              absl::FailedPreconditionError("sorted_inputs must be 2D"));
  OP_REQUIRES(ctx, values_shape.dims() == 2,
              absl::FailedPreconditionError("values must be 2D"));

  // Add a new inner dimension to values, to allow broadcasting along the inner
  // dimension of sorted_sequence.
  auto new_values_shape = values_shape;
  new_values_shape.InsertDim(/* d */ 2, /* size */ 1);
  auto values_reshaped = xla::Reshape(values, new_values_shape.dim_sizes());

  // Add a new penultimate dimension to sorted_inputs, to allow broadcasting of
  // sorted_sequence entries for each value.
  auto new_sorted_inputs_shape = sorted_inputs_shape;
  new_sorted_inputs_shape.InsertDim(/* d */ 1, /* size */ 1);
  auto sorted_inputs_reshaped =
      xla::Reshape(sorted_inputs, new_sorted_inputs_shape.dim_sizes());

  // We are relying on broadcasting to compare each value against each entry in
  // the associated sorted_inputs row.
  // The reshapes above leave the tensors with equal rank of 3, so broadcast
  // dimensions are not explicitly specified.
  // Use explicit NaN-aware logic: NaN is treated as the largest value.
  bool is_fp = tensorflow::DataTypeIsFloating(ctx->InputType("sorted_inputs"));
  xla::XlaOp element_is_nan;
  xla::XlaOp val_is_nan;
  if (is_fp) {
    element_is_nan = xla::IsNan(sorted_inputs_reshaped);
    val_is_nan = xla::IsNan(values_reshaped);
  } else {
    element_is_nan = xla::ConstantR0<bool>(ctx->builder(), false);
    val_is_nan = xla::ConstantR0<bool>(ctx->builder(), false);
  }

  xla::XlaBuilder* builder = ctx->builder();
  const DataType accumulation_type = XlaHelpers::SumAccumulationType(out_dtype);
  xla::XlaOp zero = XlaHelpers::Zero(builder, accumulation_type);

  // Non-NaN mask for sorted_inputs elements: [batch, 1, inputs_size]
  auto non_nan_mask = xla::Not(element_is_nan);

  // Standard comparisons with NaN elements masked out.
  // When element is NaN, comparison is false (NaN is largest, no value is >/>= NaN).
  auto standard_gt = xla::Gt(values_reshaped, sorted_inputs_reshaped);
  auto standard_ge = xla::Ge(values_reshaped, sorted_inputs_reshaped);
  auto standard_gt_masked = xla::And(standard_gt, non_nan_mask);
  auto standard_ge_masked = xla::And(standard_ge, non_nan_mask);

  // Reduce standard comparisons over inputs dimension -> [batch, values_size, 1]
  auto standard_gt_int = XlaHelpers::ConvertElementType(standard_gt_masked, accumulation_type);
  auto standard_ge_int = XlaHelpers::ConvertElementType(standard_ge_masked, accumulation_type);
  auto standard_gt_count =
      xla::Reduce(standard_gt_int, zero, *ctx->GetOrCreateAdd(accumulation_type), {2});
  auto standard_ge_count =
      xla::Reduce(standard_ge_int, zero, *ctx->GetOrCreateAdd(accumulation_type), {2});

  // Special counts for NaN values:
  // lower_bound (GT): NaN goes after all non-NaN elements -> count non-NaN elements
  // upper_bound (GE): NaN goes after ALL elements (including NaN) -> count all elements
  auto non_nan_int = XlaHelpers::ConvertElementType(non_nan_mask, accumulation_type);
  auto non_nan_count =
      xla::Reduce(non_nan_int, zero, *ctx->GetOrCreateAdd(accumulation_type), {2});
  int64_t inputs_size = sorted_inputs_shape.dim_size(1);
  auto total_count = xla::ConstantR0<int64_t>(builder, inputs_size);
  total_count = XlaHelpers::ConvertElementType(total_count, accumulation_type);
  // Broadcast total_count to [batch, values_size, 1] to match standard_*_count
  auto shape_or = builder->GetShape(standard_gt_count);
  OP_REQUIRES_OK(ctx, shape_or.status());
  total_count = xla::Broadcast(total_count, shape_or.value().dimensions());

  // Select based on whether value is NaN: [batch, values_size, 1]
  xla::XlaOp result_count;
  if (comparison_direction == xla::ComparisonDirection::kGt) {
    // LowerBound: if value is NaN use non_nan_count, else use standard_gt_count
    result_count = xla::Select(val_is_nan, non_nan_count, standard_gt_count);
  } else {
    // UpperBound: if value is NaN use total_count, else use standard_ge_count
    result_count = xla::Select(val_is_nan, total_count, standard_ge_count);
  }

  ctx->SetOutput(0, result_count);
}

class LowerBoundOp : public XlaOpKernel {
 public:
  explicit LowerBoundOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    BuildLowerUpperBoundOp(ctx, out_dtype_, xla::ComparisonDirection::kGt);
  }

 private:
  DataType out_dtype_;
};

REGISTER_XLA_OP(Name("LowerBound"), LowerBoundOp);

class UpperBoundOp : public XlaOpKernel {
 public:
  explicit UpperBoundOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    BuildLowerUpperBoundOp(ctx, out_dtype_, xla::ComparisonDirection::kGe);
  }

 private:
  DataType out_dtype_;
};

REGISTER_XLA_OP(Name("UpperBound"), UpperBoundOp);

}  // namespace
}  // namespace tensorflow
