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

// XLA-specific BatchMatMul Op.
// The current implementation simply unrolls the computation along the batch
// dimension.
// TODO(dominikg,phawkins): Use a real batched matmul instead of unrolling.

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class BatchMatMulOp : public XlaOpKernel {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &adj_y_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape y_shape = ctx->InputShape(1);

    // Check that both tensors have the same number of dimensions. There must be
    // at least two (the batch dimensions can be empty).
    OP_REQUIRES(ctx, x_shape.dims() == y_shape.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        x_shape.DebugString(), " vs. ",
                                        y_shape.DebugString()));
    const int ndims = x_shape.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ", ndims));

    // The batch dimensions must be equal and the matrix dimensions must be
    // valid.
    std::vector<int64> dimensions;
    int batch_count = 1;
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(
          ctx, x_shape.dim_size(i) == y_shape.dim_size(i),
          errors::InvalidArgument("In[0].dim(", i, ") and In[1].dim(", i,
                                  ") must be the same: ", x_shape.DebugString(),
                                  " vs ", y_shape.DebugString()));
      dimensions.push_back(x_shape.dim_size(i));
      batch_count *= x_shape.dim_size(i);
    }

    int x_inner_dim = adj_x_ ? (ndims - 2) : (ndims - 1);
    int y_inner_dim = adj_y_ ? (ndims - 1) : (ndims - 2);
    OP_REQUIRES(
        ctx, x_shape.dim_size(x_inner_dim) == y_shape.dim_size(y_inner_dim),
        errors::InvalidArgument(
            "In[0] mismatch In[1] shape: ", x_shape.dim_size(x_inner_dim),
            " vs. ", y_shape.dim_size(y_inner_dim), ": ", x_shape.DebugString(),
            " ", y_shape.DebugString(), " ", adj_x_, " ", adj_y_));

    int x_outer_dim = adj_x_ ? (ndims - 1) : (ndims - 2);
    int y_outer_dim = adj_y_ ? (ndims - 2) : (ndims - 1);
    dimensions.push_back(x_shape.dim_size(x_outer_dim));
    dimensions.push_back(y_shape.dim_size(y_outer_dim));

    xla::ComputationBuilder* builder = ctx->builder();

    xla::ComputationDataHandle x_handle = ctx->Input(0);
    if (BaseType(input_type(0)) == DT_COMPLEX64 && adj_x_) {
      x_handle = builder->Conj(x_handle);
    }
    xla::ComputationDataHandle y_handle = ctx->Input(1);
    if (BaseType(input_type(1)) == DT_COMPLEX64 && adj_y_) {
      y_handle = builder->Conj(y_handle);
    }

    // Reshape input tensors into 3D tensors by flattening the batch
    // dimensions. This makes it easier to unroll the batch dimension.
    auto x_flat =
        builder->Reshape(x_handle, {batch_count, x_shape.dim_size(ndims - 2),
                                    x_shape.dim_size(ndims - 1)});
    auto y_flat =
        builder->Reshape(y_handle, {batch_count, y_shape.dim_size(ndims - 2),
                                    y_shape.dim_size(ndims - 1)});

    // Slice batches into individual matrices and multiply them.
    std::vector<xla::ComputationDataHandle> out_slices;
    for (int i = 0; i < batch_count; ++i) {
      // Slice off individual matrices and reshape to 2D tensors.
      auto x_slice = builder->Slice(
          x_flat, {i, 0, 0},
          {i + 1, x_shape.dim_size(ndims - 2), x_shape.dim_size(ndims - 1)},
          {1, 1, 1});
      x_slice = builder->Reshape(
          x_slice, {x_shape.dim_size(ndims - 2), x_shape.dim_size(ndims - 1)});
      auto y_slice = builder->Slice(
          y_flat, {i, 0, 0},
          {i + 1, y_shape.dim_size(ndims - 2), y_shape.dim_size(ndims - 1)},
          {1, 1, 1});
      y_slice = builder->Reshape(
          y_slice, {y_shape.dim_size(ndims - 2), y_shape.dim_size(ndims - 1)});

      // Transpose if needed.
      auto lhs = adj_x_ ? builder->Transpose(x_slice, {1, 0}) : x_slice;
      auto rhs = adj_y_ ? builder->Transpose(y_slice, {1, 0}) : y_slice;

      // Multiply matrices and add an outer singleton dimension to the output
      // so we can concatenate along the flattened batch dimension later.
      auto out = builder->Dot(lhs, rhs);
      out = builder->Reshape(out,
                             {1, dimensions[ndims - 2], dimensions[ndims - 1]});
      out_slices.push_back(out);
    }

    // Concatenate output slices and reshape to original number of dimensions.
    xla::ComputationDataHandle data;
    if (out_slices.empty()) {
      // It is illegal to pass an empty list to ConcatInDim.
      // The batch count is empty, so both inputs must have zero elements.
      // Arbitrarily use the left input as the argument to Reshape().
      data = x_handle;
    } else {
      data = builder->ConcatInDim(out_slices, 0);
    }
    data = builder->Reshape(data, dimensions);

    ctx->SetOutput(0, data);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

REGISTER_XLA_OP(Name("BatchMatMul"), BatchMatMulOp);

}  // namespace
}  // namespace tensorflow
