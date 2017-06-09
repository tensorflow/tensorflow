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

// XLA-specific reduction Ops.

#include "tensorflow/compiler/tf2xla/kernels/reduction_ops.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {

XlaReductionOp::XlaReductionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  const DataType dt = BaseType(input_type(0));
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({dt, DT_INT32}, {dt}));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
}

// Return the base case for the reduction. Defaults to zero.
xla::ComputationDataHandle XlaReductionOp::InitialValue(
    xla::ComputationBuilder* builder) {
  return XlaHelpers::Zero(builder, input_type(0));
}

// Unless BuildFinalizer is overridden the reduction has no
// finalizer.
xla::ComputationDataHandle XlaReductionOp::BuildFinalizer(
    xla::ComputationBuilder* builder,
    const xla::ComputationDataHandle& reduce_output,
    int64 num_elements_reduced) {
  return reduce_output;
}

void XlaReductionOp::Compile(XlaOpKernelContext* ctx) {
  const TensorShape data_shape = ctx->InputShape(0);
  const TensorShape axes_tensor_shape = ctx->InputShape(1);
  VLOG(1) << "ReductionOp: " << ctx->op_kernel().name();

  if (axes_tensor_shape.num_elements() == 0) {
    // The reduction axes is an empty vector, which means there are no
    // axes to reduce so just pass the input directly through to the
    // output.
    ctx->SetOutput(0, ctx->Input(0));
    return;
  }

  // Evaluate the constant, reshaping to a 1-vector if it is a scalar.
  xla::Literal axes_literal;
  OP_REQUIRES_OK(ctx,
                 ctx->ConstantInputReshaped(
                     1, {axes_tensor_shape.num_elements()}, &axes_literal));

  VLOG(1) << "data shape: " << data_shape.DebugString();
  VLOG(1) << "axes      : " << xla::LiteralUtil::ToString(axes_literal);

  gtl::InlinedVector<bool, 4> bitmap(data_shape.dims(), false);
  std::vector<int64> xla_axes;
  int64 num_elements_reduced = 1LL;
  for (int64 i = 0; i < axes_tensor_shape.num_elements(); ++i) {
    int32 index = xla::LiteralUtil::Get<int>(axes_literal, {i});
    OP_REQUIRES(ctx,
                !(index < -data_shape.dims() || index >= data_shape.dims()),
                errors::InvalidArgument("Invalid reduction dimension (", index,
                                        " for input with ", data_shape.dims(),
                                        " dimension(s)"));
    index = (index + data_shape.dims()) % data_shape.dims();
    bitmap[index] = true;
    xla_axes.push_back(index);
    num_elements_reduced *= data_shape.dim_size(index);
  }

  std::vector<int64> final_shape;
  for (int i = 0; i < data_shape.dims(); ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      int64 dim = data_shape.dim_size(i);
      final_shape.push_back(dim);
    } else if (keep_dims_) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      final_shape.push_back(1);
    }
  }

  string desc = ctx->op_kernel().name();

  // Call virtual method to get the initial value.
  const xla::ComputationDataHandle initial = InitialValue(ctx->builder());
  // Construct the builder for the reduction lambda.
  xla::ComputationBuilder r(ctx->builder()->client(),
                            strings::StrCat(desc, "-reduction"));
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(input_type(0), &type));
  // Make two scalar parameters of the desired type for the lambda.
  xla::ComputationDataHandle rx =
      r.Parameter(0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::ComputationDataHandle ry =
      r.Parameter(1, xla::ShapeUtil::MakeShape(type, {}), "y");

  auto data = ctx->Input(0);

  // Call virtual method to build the reduction lambda.
  BuildReducer(&r, rx, ry);
  xla::Computation reduction_computation = r.Build().ConsumeValueOrDie();
  xla::ComputationDataHandle reduce =
      ctx->builder()->Reduce(data, initial, reduction_computation, xla_axes);

  xla::ComputationDataHandle finalized =
      BuildFinalizer(ctx->builder(), reduce, num_elements_reduced);

  xla::ComputationDataHandle result;
  if (keep_dims_) {
    result = ctx->builder()->Reshape(finalized, final_shape);
  } else {
    result = finalized;
  }
  ctx->SetOutput(0, result);
}

}  // namespace tensorflow
