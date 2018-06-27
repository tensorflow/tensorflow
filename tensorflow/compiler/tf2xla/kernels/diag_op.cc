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

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Create a diagonal / batch diagonal matrix with 'input' on the diagonal.
xla::StatusOr<xla::XlaOp> CreateDiagonal(
    const xla::XlaOp& input, int64 last_dim_size,
    tensorflow::gtl::ArraySlice<int64> other_dims, XlaOpKernelContext* ctx,
    xla::XlaBuilder* builder) {
  // Create two matrices that have the following forms, and compare them:
  //
  // [[0, 0, 0, 0]            [[0, 1, 2, 3]
  //  [1, 1, 1, 1]             [0, 1, 2, 3]
  //  [2, 2, 2, 2]             [0, 1, 2, 3]
  //  [3, 3, 3, 3]]            [0, 1, 2, 3]]
  //
  // This produces a predicate matrix of the right size, with "true" on the
  // diagonal.
  xla::XlaOp iota;
  TF_RETURN_IF_ERROR(
      XlaHelpers::Iota(builder, DataType::DT_INT32, last_dim_size, &iota));
  xla::XlaOp iota_broadcast = xla::Broadcast(iota, {last_dim_size});
  xla::XlaOp mask = xla::Eq(iota_broadcast, iota, {0});

  // If this is a batched diagonal, broadcast the mask across the other
  // dimensions.
  if (!other_dims.empty()) {
    mask = xla::Broadcast(mask, other_dims);
  }

  // Broadcast the input, and then use the mask computed above to select the
  // diagonal:
  // e.g, in 2D:
  //         [[t, f, f]    [[1, 1, 1]    [[0, 0, 0]      [[1, 0, 0]
  // select(  [f, t, f]  ,  [4, 4, 4]  ,  [0, 0, 0]  ) =  [0, 4, 0]
  //          [f, f, t]]    [9, 9, 9]]    [0, 0, 0]]      [0, 0, 9]]
  //
  // Broadcasting the input is less-than-trivial, since we need to broadcast
  // into a "middle" dimension. We can do this with a reshape + implicit
  // broadcast.
  // TODO(b/30112114): Replace with in-dim broadcast when those are supported.
  std::vector<int64> broadcast_dims(other_dims.begin(), other_dims.end());
  broadcast_dims.push_back(1LL);
  broadcast_dims.push_back(last_dim_size);
  xla::XlaOp input_broadcast = xla::Reshape(input, broadcast_dims);

  broadcast_dims[broadcast_dims.size() - 2] = last_dim_size;
  xla::PrimitiveType element_type;
  TF_RETURN_IF_ERROR(
      DataTypeToPrimitiveType(ctx->input_type(0), &element_type));
  auto broadcast_shape =
      xla::ShapeUtil::MakeShape(element_type, broadcast_dims);
  xla::XlaOp zeros = Zeros(builder, broadcast_shape);

  input_broadcast = xla::Add(input_broadcast, zeros);
  return xla::Select(mask, input_broadcast, zeros);
}

class DiagOp : public XlaOpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("Diag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);

    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp input = ctx->Input(0);

    // Picture:
    // tf.diag([1, 2, 3, 4]) ==> [[1, 0, 0, 0]
    //                            [0, 2, 0, 0]
    //                            [0, 0, 3, 0]
    //                            [0, 0, 0, 4]]

    // Flattens the input to 1D.
    int64 size = input_shape.num_elements();
    input = xla::Reshape(input, {size});

    // Create an R2 with the R1 diagonal.
    auto diag_or_status =
        CreateDiagonal(input, size, /*other_dims=*/{}, ctx, builder);
    OP_REQUIRES_OK(ctx, diag_or_status.status());
    xla::XlaOp diag = diag_or_status.ValueOrDie();

    // Reshapes to the final shape.
    std::vector<int64> new_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), new_dims.begin());
    std::copy(dims.begin(), dims.end(), new_dims.begin() + dims.size());
    diag = xla::Reshape(diag, new_dims);

    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("Diag"), DiagOp);

class DiagPartOp : public XlaOpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    const TensorShape input_shape = ctx->InputShape(0);
    auto dims = input_shape.dim_sizes();

    int num_dims = dims.size();
    const int out_dims = num_dims / 2;

    OP_REQUIRES(ctx, 2 <= num_dims,
                errors::InvalidArgument("Expected 2 <= dims, got shape ",
                                        input_shape.DebugString()));
    OP_REQUIRES(ctx, num_dims % 2 == 0,
                errors::InvalidArgument("The input tensor must have even rank; "
                                        "got shape ",
                                        input_shape.DebugString()));
    int64 new_size = 1;
    std::vector<int64> new_dims;
    for (int i = 0; i < out_dims; i++) {
      OP_REQUIRES(
          ctx, dims[i] == dims[i + out_dims],
          errors::InvalidArgument("Invalid shape ", input_shape.DebugString(),
                                  ": dimensions ", i, " and ", i + out_dims,
                                  " do not match."));
      new_size *= dims[i];
      new_dims.push_back(dims[i]);
    }

    xla::XlaOp diag = ctx->Input(0);

    // TODO(b/30878775): use Slice with strides when supported, in place of
    // the Pad -> Reshape -> Slice.

    // Picture:
    // [[1, 0, 0, 0]  pad and reshape to [[1, 0, 0, 0, 0],
    //  [0, 2, 0, 0]  =================>  [2, 0, 0, 0, 0],
    //  [0, 0, 3, 0]                      [3, 0, 0, 0, 0],
    //  [0, 0, 0, 4]]                     [4, 0, 0, 0, 0]]
    // and then slice out the first column.

    // Flattens the input to 1D.
    int64 size = input_shape.num_elements();
    diag = xla::Reshape(diag, {size});

    // Adds padding after the last element of 'new_size'.
    xla::PaddingConfig config;
    auto* dim = config.add_dimensions();
    dim->set_edge_padding_high(new_size);
    auto zero = XlaHelpers::Zero(builder, input_type(0));
    diag = xla::Pad(diag, zero, config);

    // Reshapes so the diagonal is now in the first column.
    diag = xla::Reshape(diag, {new_size, new_size + 1});

    // Slices out the first column and reshapes to the final shape.
    diag = xla::Slice(diag, {0, 0}, {new_size, 1}, {1, 1});
    diag = xla::Reshape(diag, new_dims);

    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("DiagPart"), DiagPartOp);

class MatrixDiagOp : public XlaOpKernel {
 public:
  explicit MatrixDiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("MatrixDiag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);

    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp diag = ctx->Input(0);

    int last_dim = dims.size() - 1;
    int64 last_dim_size = input_shape.dim_size(last_dim);
    tensorflow::gtl::ArraySlice<int64> other_dims(dims);
    other_dims.pop_back();

    auto diag_or_status =
        CreateDiagonal(diag, last_dim_size, other_dims, ctx, builder);
    OP_REQUIRES_OK(ctx, diag_or_status.status());
    diag = diag_or_status.ValueOrDie();
    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("MatrixDiag"), MatrixDiagOp);

class MatrixDiagPartOp : public XlaOpKernel {
 public:
  explicit MatrixDiagPartOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    const TensorShape input_shape = ctx->InputShape(0);
    auto dims = input_shape.dim_sizes();

    OP_REQUIRES(ctx, 2 <= dims.size(),
                errors::InvalidArgument("Expected 2 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp diag = ctx->Input(0);

    int last_dim = dims.size() - 1;
    int64 last_dim_size = dims[last_dim];

    // The smaller of the last two dimension sizes.
    int64 smaller_dim_size = std::min(dims[last_dim - 1], dims[last_dim]);

    // TODO(b/30878775): use Slice with strides when supported, in place of
    // the Pad -> Reshape -> Slice.

    // Picture: for each 2D matrix in the tensor's last two dimensions:
    // [[1, 0, 0, 0]  pad and reshape to [[1, 0, 0, 0, 0],
    //  [0, 2, 0, 0]  =================>  [2, 0, 0, 0, 0],
    //  [0, 0, 3, 0]]                     [3, 0, 0, 0, 0],
    // and then slice out the first column.
    //
    // Another example, with tall and narrow input.
    // [[1, 0]  pad and reshape to [[1, 0, 0],
    //  [0, 2]  =================>  [2, 0, 0]]
    //  [0, 0]
    //  [0, 0]]

    // Collapses the last two dimensions.
    std::vector<int64> flattened_dims(dims.begin(), dims.end() - 1);
    flattened_dims.back() *= dims.back();
    diag = xla::Reshape(diag, flattened_dims);

    // Slices or pads the last dimension to 'target_size'.
    int64 actual_size = flattened_dims.back();
    int64 target_size = smaller_dim_size * (last_dim_size + 1);
    if (actual_size < target_size) {
      xla::PaddingConfig config =
          xla::MakeNoPaddingConfig(flattened_dims.size());
      auto* dim = config.mutable_dimensions(flattened_dims.size() - 1);
      dim->set_edge_padding_high(target_size - actual_size);
      auto zero = XlaHelpers::Zero(builder, input_type(0));
      diag = xla::Pad(diag, zero, config);
    } else if (actual_size > target_size) {
      std::vector<int64> start(flattened_dims.size(), 0);
      std::vector<int64> limits(flattened_dims.begin(), flattened_dims.end());
      std::vector<int64> strides(flattened_dims.size(), 1);
      limits[flattened_dims.size() - 1] = target_size;
      diag = xla::Slice(diag, start, limits, strides);
    }

    // Reshape so the target values are in the first position of the last
    // dimension.
    std::vector<int64> unflattened_dims(dims.begin(), dims.end());
    dims[last_dim - 1] = smaller_dim_size;
    dims[last_dim] = last_dim_size + 1;
    diag = xla::Reshape(diag, dims);

    // Slices out the first column and reshapes to the final shape.
    std::vector<int64> start(dims.size(), 0);
    std::vector<int64> limits(dims.begin(), dims.end());
    std::vector<int64> strides(dims.size(), 1);
    limits[last_dim] = 1;
    diag = xla::Slice(diag, start, limits, strides);

    // Collapses away the last dimension.
    dims.pop_back();
    diag = xla::Reshape(diag, dims);

    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("MatrixDiagPart"), MatrixDiagPartOp);

}  // namespace
}  // namespace tensorflow
