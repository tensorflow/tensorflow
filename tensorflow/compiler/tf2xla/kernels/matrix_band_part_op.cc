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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
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

    xla::XlaBuilder* builder = context->builder();
    xla::XlaOp input = context->Input(0);
    xla::XlaOp num_lower = context->Input(1);
    xla::XlaOp num_upper = context->Input(2);
    DataType input_type = context->input_type(0);
    DataType index_type = context->input_type(1);
    xla::PrimitiveType index_xla_type = context->input_xla_type(1);

    TensorShape batch_shape = input_shape;
    batch_shape.RemoveLastDims(2);
    const int64 m = input_shape.dim_size(input_shape.dims() - 2);
    const int64 n = input_shape.dim_size(input_shape.dims() - 1);

    // Compute 'offset', which is how many diagonals we are above/below the
    // diagonal.
    xla::Shape iota_shape = xla::ShapeUtil::MakeShape(index_xla_type, {m, n});
    xla::XlaOp iota_m = xla::Iota(builder, iota_shape, /*iota_dimension=*/0);
    xla::XlaOp iota_n = xla::Iota(builder, iota_shape, /*iota_dimension=*/1);

    auto offset = xla::Sub(iota_n, iota_m);

    // If num_lower or num_upper are negative, include all lower/upper
    // diagonals.
    auto zero_index = XlaHelpers::Zero(builder, index_type);
    num_lower = xla::Select(xla::Lt(num_lower, zero_index),
                            XlaHelpers::IntegerLiteral(builder, index_type, m),
                            num_lower);
    num_upper = xla::Select(xla::Lt(num_upper, zero_index),
                            XlaHelpers::IntegerLiteral(builder, index_type, n),
                            num_upper);

    auto indicator = xla::And(xla::Le(xla::Neg(num_lower), offset),
                              xla::Le(offset, num_upper));
    indicator = xla::Broadcast(indicator, batch_shape.dim_sizes());

    auto zero_input = XlaHelpers::Zero(builder, input_type);
    auto output = xla::Select(
        indicator, input, xla::Broadcast(zero_input, input_shape.dim_sizes()));

    context->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixBandPartOp);
};
REGISTER_XLA_OP(Name("MatrixBandPart"), MatrixBandPartOp);

}  // namespace
}  // namespace tensorflow
