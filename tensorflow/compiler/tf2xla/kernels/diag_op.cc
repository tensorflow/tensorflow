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
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Create a diagonal / batch diagonal matrix with 'input' on the diagonal.
xla::XlaOp CreateDiagonal(xla::XlaOp input, int64 last_dim_size,
                          gtl::ArraySlice<int64> other_dims,
                          xla::PrimitiveType element_type) {
  xla::XlaBuilder* builder = input.builder();
  // Create two matrices that have the following forms, and compare them:
  //
  // [[0, 0, 0, 0]            [[0, 1, 2, 3]
  //  [1, 1, 1, 1]             [0, 1, 2, 3]
  //  [2, 2, 2, 2]             [0, 1, 2, 3]
  //  [3, 3, 3, 3]]            [0, 1, 2, 3]]
  //
  // This produces a predicate matrix of the right size, with "true" on the
  // diagonal.
  xla::XlaOp iota = xla::Iota(builder, xla::S32, last_dim_size);
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
  auto broadcast_shape =
      xla::ShapeUtil::MakeShape(element_type, broadcast_dims);
  xla::XlaOp zeros = xla::Zeros(builder, broadcast_shape);

  input_broadcast = xla::Add(input_broadcast, zeros);
  return xla::Select(mask, input_broadcast, zeros);
}

class DiagOp : public XlaOpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
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
    xla::XlaOp diag =
        CreateDiagonal(input, size, /*other_dims=*/{}, ctx->input_xla_type(0));

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

    xla::XlaOp input = ctx->Input(0);

    xla::XlaOp output = xla::Reshape(
        xla::GetMatrixDiagonal(xla::Reshape(input, {new_size, new_size})),
        new_dims);

    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("DiagPart"), DiagPartOp);

class MatrixDiagOp : public XlaOpKernel {
 public:
  explicit MatrixDiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("MatrixDiag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);

    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));


    int last_dim = dims.size() - 1;
    int64 last_dim_size = input_shape.dim_size(last_dim);
    tensorflow::gtl::ArraySlice<int64> other_dims(dims);
    other_dims.pop_back();

    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp diag = CreateDiagonal(input, last_dim_size, other_dims,
                                     ctx->input_xla_type(0));
    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("MatrixDiag"), MatrixDiagOp);

class MatrixDiagPartOp : public XlaOpKernel {
 public:
  explicit MatrixDiagPartOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    auto dims = input_shape.dim_sizes();

    OP_REQUIRES(ctx, 2 <= dims.size(),
                errors::InvalidArgument("Expected 2 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp input = ctx->Input(0);
    ctx->SetOutput(0, xla::GetMatrixDiagonal(input));
  }
};

REGISTER_XLA_OP(Name("MatrixDiagPart"), MatrixDiagPartOp);

}  // namespace
}  // namespace tensorflow
