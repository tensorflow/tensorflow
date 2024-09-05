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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/comparison_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
              errors::FailedPrecondition("sorted_inputs must be 2D"));
  OP_REQUIRES(ctx, values_shape.dims() == 2,
              errors::FailedPrecondition("values must be 2D"));

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
  auto comparison = xla::Compare(values_reshaped, sorted_inputs_reshaped, {},
                                 comparison_direction);

  const DataType accumulation_type = XlaHelpers::SumAccumulationType(out_dtype);

  // Convert boolean comparison results to integers so we can sum them.
  auto comparison_int =
      XlaHelpers::ConvertElementType(comparison, accumulation_type);

  // Sum the comparison results over the inner dimension to find the index for
  // each value.
  xla::XlaBuilder* builder = ctx->builder();
  auto reduced =
      xla::Reduce(comparison_int, XlaHelpers::Zero(builder, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), {2});

  ctx->SetOutput(0, reduced);
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
