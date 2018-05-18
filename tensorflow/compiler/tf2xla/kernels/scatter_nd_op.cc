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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Check whether updates.shape = indices.shape[:batch_dim] +
// buffer_shape[num_index_dims:]
Status ValidateUpdateShape(const TensorShape& buffer_shape,
                           const TensorShape& indices_shape,
                           const TensorShape& updates_shape) {
  if (indices_shape.dims() < 1) {
    return errors::InvalidArgument(
        "indices shape must have >= 1 dimension; got ",
        indices_shape.DebugString());
  }

  const int64 num_index_dims = indices_shape.dim_size(indices_shape.dims() - 1);
  const int64 batch_dim = indices_shape.dims() - 1;

  auto shape_err = [&]() {
    return errors::InvalidArgument(
        "Must have updates.shape = indices.shape[:batch_dim] + ",
        "buffer_shape[num_index_dims:], got updates.shape: ",
        updates_shape.DebugString(),
        ", indices.shape: ", indices_shape.DebugString(),
        ", buffer_shape: ", buffer_shape.DebugString(),
        ", num_index_dims: ", num_index_dims, ", and batch_dim: ", batch_dim);
  };

  if (updates_shape.dims() < batch_dim) return shape_err();
  if (buffer_shape.dims() <
      num_index_dims + (updates_shape.dims() - batch_dim)) {
    return shape_err();
  }
  if (updates_shape.dims() !=
      batch_dim + buffer_shape.dims() - num_index_dims) {
    return shape_err();
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates_shape.dim_size(d) != indices_shape.dim_size(d)) {
      return shape_err();
    }
  }
  for (int d = 0; d < updates_shape.dims() - batch_dim; ++d) {
    if (updates_shape.dim_size(d + batch_dim) !=
        buffer_shape.dim_size(d + num_index_dims)) {
      return shape_err();
    }
  }
  return Status::OK();
}

class ScatterNdOp : public XlaOpKernel {
 public:
  explicit ScatterNdOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    DataType dtype = context->input_type(1);

    TensorShape indices_shape = context->InputShape(0);
    TensorShape updates_shape = context->InputShape(1);

    TensorShape buffer_shape;
    OP_REQUIRES_OK(context, context->ConstantInputAsShape(2, &buffer_shape));

    OP_REQUIRES(
        context, TensorShapeUtils::IsVectorOrHigher(buffer_shape),
        errors::InvalidArgument("Output must be at least 1-D, ",
                                "got shape: ", buffer_shape.DebugString()));

    OP_REQUIRES(
        context,
        buffer_shape.num_elements() > 0 || (indices_shape.num_elements() == 0 &&
                                            updates_shape.num_elements() == 0),
        errors::InvalidArgument(
            "Indices and updates specified for empty output. indices shape: ",
            indices_shape.DebugString()));

    OP_REQUIRES_OK(context, ValidateUpdateShape(buffer_shape, indices_shape,
                                                updates_shape));

    xla::XlaBuilder* builder = context->builder();
    auto buffer = builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                                     buffer_shape.dim_sizes());
    auto indices = context->Input(0);
    auto updates = context->Input(1);
    auto result =
        XlaScatter(buffer, updates, indices,
                   /*indices_are_vectors=*/true, /*combiner=*/{}, builder);
    OP_REQUIRES_OK(context, result.status());
    context->SetOutput(0, result.ValueOrDie());
  }
};

REGISTER_XLA_OP(Name("ScatterNd").CompileTimeConstInput("shape"), ScatterNdOp);

}  // namespace
}  // namespace tensorflow
