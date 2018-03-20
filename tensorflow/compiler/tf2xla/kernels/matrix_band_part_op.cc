/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class MatrixBandPartOp : public XlaOpKernel {
 public:
  explicit MatrixBandPartOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input_shape.DebugString()));

    const TensorShape num_lower_in_shape = context->InputShape(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_lower_in_shape),
                errors::InvalidArgument("num_lower must be scalar, got shape ",
                                        num_lower_in_shape.DebugString()));

    const TensorShape num_upper_in_shape = context->InputShape(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_upper_in_shape),
                errors::InvalidArgument("num_upper must be scalar, got shape ",
                                        num_upper_in_shape.DebugString()));

    xla::ComputationBuilder* builder = context->builder();
    xla::ComputationDataHandle input = context->Input(0);
    xla::ComputationDataHandle num_lower = context->Input(1);
    xla::ComputationDataHandle num_upper = context->Input(2);
    DataType input_type = context->input_type(0);
    DataType index_type = context->input_type(1);

    TensorShape batch_shape = input_shape;
    batch_shape.RemoveLastDims(2);
    const int64 m = input_shape.dim_size(input_shape.dims() - 2);
    const int64 n = input_shape.dim_size(input_shape.dims() - 1);

    // Compute 'offset', which is how many diagonals we are above/below the
    // diagonal.
    xla::ComputationDataHandle iota_m;
    OP_REQUIRES_OK(context, XlaHelpers::Iota(builder, index_type, m, &iota_m));

    xla::ComputationDataHandle iota_n;
    OP_REQUIRES_OK(context, XlaHelpers::Iota(builder, index_type, n, &iota_n));

    auto offset = builder->Sub(builder->Broadcast(iota_n, {m}), iota_m,
                               /*broadcast_dimensions=*/{0});

    // If num_lower or num_upper are negative, include all lower/upper
    // diagonals.
    auto zero_index = XlaHelpers::Zero(builder, index_type);
    num_lower = builder->Select(
        builder->Lt(num_lower, zero_index),
        XlaHelpers::IntegerLiteral(builder, index_type, m), num_lower);
    num_upper = builder->Select(
        builder->Lt(num_upper, zero_index),
        XlaHelpers::IntegerLiteral(builder, index_type, n), num_upper);

    auto indicator = builder->And(builder->Le(builder->Neg(num_lower), offset),
                                  builder->Le(offset, num_upper));
    indicator = builder->Broadcast(indicator, batch_shape.dim_sizes());

    auto zero_input = XlaHelpers::Zero(builder, input_type);
    auto output = builder->Select(
        indicator, input,
        builder->Broadcast(zero_input, input_shape.dim_sizes()));

    context->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixBandPartOp);
};
REGISTER_XLA_OP(Name("MatrixBandPart"), MatrixBandPartOp);

}  // namespace
}  // namespace tensorflow
