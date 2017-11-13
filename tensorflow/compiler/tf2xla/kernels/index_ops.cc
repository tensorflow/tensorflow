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

// Native XLA implementations of indexing ops.

#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {
XlaArgMinMaxOp::XlaArgMinMaxOp(OpKernelConstruction* ctx, bool is_min)
    : XlaOpKernel(ctx), is_min_(is_min) {}

void XlaArgMinMaxOp::Compile(XlaOpKernelContext* ctx) {
  const TensorShape input_shape = ctx->InputShape(0);
  const TensorShape dimension_shape = ctx->InputShape(1);

  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dimension_shape),
              errors::InvalidArgument(
                  "dim must be a scalar, but received tensor of shape: ",
                  dimension_shape.DebugString()));

  int64 dim;
  OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &dim));

  const int input_dims = input_shape.dims();
  const int axis = dim < 0 ? dim + input_dims : dim;

  OP_REQUIRES(
      ctx, axis >= 0 && axis < input_dims,
      errors::InvalidArgument("Expected dimension in the range [", -input_dims,
                              ", ", input_dims, "), but got ", dim));
  const int64 axis_size = input_shape.dim_size(axis);
  OP_REQUIRES(
      ctx, axis_size > 0,
      errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                              input_shape.DebugString()));

  DataType index_type = output_type(0);
  xla::PrimitiveType xla_input_type;
  OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(0), &xla_input_type));
  xla::PrimitiveType xla_index_type;
  OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(index_type, &xla_index_type));

  xla::ComputationBuilder* b = ctx->builder();
  xla::ComputationDataHandle input = ctx->Input(0);

  xla::ComputationDataHandle init_value;
  const xla::Computation* reducer;
  if (is_min_) {
    init_value = XlaHelpers::MaxValue(b, input_type(0));
    reducer = ctx->GetOrCreateMin(input_type(0));
  } else {
    init_value = XlaHelpers::MinValue(b, input_type(0));
    reducer = ctx->GetOrCreateMax(input_type(0));
  }
  xla::ComputationDataHandle input_max =
      b->Reduce(input, init_value, *reducer, /*dimensions_to_reduce=*/{axis});
  std::vector<int64> broadcast_dims(input_dims - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
  // Compute a mask that has 1s for elements equal to the maximum.
  xla::ComputationDataHandle mask = b->ConvertElementType(
      b->Eq(input, input_max, broadcast_dims), xla_index_type);

  // Multiply by the vector [0, 1, 2, ...] to convert each 1 into its index.
  // TODO(phawkins): add a bitwise And operator to HLO, use a bitwise and
  // instead of a multiplication here.
  xla::ComputationDataHandle iota;
  OP_REQUIRES_OK(ctx, XlaHelpers::Iota(b, index_type, axis_size, &iota));
  xla::ComputationDataHandle product =
      b->Mul(mask, iota, /*broadcast_dimensions=*/{axis});

  // If there are multiple maximum elements, choose the one with the highest
  // index.
  xla::ComputationDataHandle output =
      b->Reduce(product, XlaHelpers::MinValue(b, index_type),
                *ctx->GetOrCreateMax(index_type),
                /*dimensions_to_reduce=*/{axis});

  ctx->SetOutput(0, output);
}

XlaArgMaxOp::XlaArgMaxOp(OpKernelConstruction* ctx)
    : XlaArgMinMaxOp(ctx, /*is_min=*/false) {}
REGISTER_XLA_OP(Name("ArgMax").Device(DEVICE_GPU_XLA_JIT), XlaArgMaxOp);

namespace {

class XlaArgMinOp : public XlaArgMinMaxOp {
 public:
  explicit XlaArgMinOp(OpKernelConstruction* ctx);
};
XlaArgMinOp::XlaArgMinOp(OpKernelConstruction* ctx)
    : XlaArgMinMaxOp(ctx, /*is_min=*/true) {}
REGISTER_XLA_OP(Name("ArgMin"), XlaArgMinOp);

}  // namespace
}  // namespace tensorflow
