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

#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/kernels/reduction_ops.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/status.h"

namespace tensorflow {

XlaReductionOp::XlaReductionOp(OpKernelConstruction* ctx,
                               DataType reduction_type)
    : XlaOpKernel(ctx), reduction_type_(reduction_type) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  OP_REQUIRES_OK(
      ctx, DataTypeToPrimitiveType(reduction_type_, &xla_reduction_type_));
}

// The default finalizer converts the results back into the input type. This can
// be overridden.
xla::XlaOp XlaReductionOp::BuildFinalizer(
    xla::XlaBuilder* /*builder*/, const xla::XlaOp& /*input*/,
    const xla::XlaOp& reduce_output,
    const std::vector<int64_t>& /*dimensions_to_reduce*/) {
  return XlaHelpers::ConvertElementType(reduce_output, input_type(0));
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

  OP_REQUIRES(ctx, axes_tensor_shape.dims() <= 1,
              errors::InvalidArgument(
                  "Expected scalar or vector as index argument, got ",
                  axes_tensor_shape.DebugString()));

  // Evaluate the constant, reshaping to a 1-vector if it is a scalar.
  std::vector<int64_t> axes;
  xla::Literal axes_literal;
  OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector(1, &axes));

  VLOG(1) << "data shape: " << data_shape.DebugString();
  VLOG(1) << "axes      : " << absl::StrJoin(axes, ",");

  absl::InlinedVector<bool, 4> bitmap(data_shape.dims(), false);
  std::vector<int64_t> xla_axes;
  auto num_elements = axes_tensor_shape.num_elements();
  xla_axes.reserve(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    int64_t index = axes[i];
    OP_REQUIRES(ctx,
                !(index < -data_shape.dims() || index >= data_shape.dims()),
                errors::InvalidArgument("Invalid reduction dimension (", index,
                                        " for input with ", data_shape.dims(),
                                        " dimension(s)"));
    index = (index + data_shape.dims()) % data_shape.dims();
    OP_REQUIRES(
        ctx, !bitmap[index],
        errors::InvalidArgument(
            "Invalid reduction arguments: Axes contains duplicate dimension: ",
            index));
    bitmap[index] = true;
    xla_axes.push_back(index);
  }

  std::vector<int64_t> final_shape;
  for (int i = 0; i < data_shape.dims(); ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      int64_t dim = data_shape.dim_size(i);
      final_shape.push_back(dim);
    } else if (keep_dims_) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      final_shape.push_back(1);
    }
  }

  string desc = ctx->op_kernel().name();

  xla::XlaBuilder* const b = ctx->builder();
  // Construct the builder for the reduction lambda.
  xla::XlaBuilder r(absl::StrCat(desc, "-reduction"));
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(reduction_type_, &type));

  auto data = xla::ConvertElementType(ctx->Input(0), type);
  // Call virtual method to get the initial value.
  auto initial = xla::ConvertElementType(InitialValue(b), type);
  // Make two scalar parameters of the desired type for the lambda.
  auto rx = xla::Parameter(&r, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  auto ry = xla::Parameter(&r, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  // Call virtual method to build the reduction lambda.
  BuildReducer(&r, rx, ry);
  xla::XlaComputation reduction_computation = r.Build().value();

  auto reduce = xla::Reduce(data, initial, reduction_computation, xla_axes);
  auto finalized = BuildFinalizer(b, data, reduce, xla_axes);
  auto result = keep_dims_ ? xla::Reshape(finalized, final_shape) : finalized;
  ctx->SetOutput(0, result);
}

}  // namespace tensorflow
