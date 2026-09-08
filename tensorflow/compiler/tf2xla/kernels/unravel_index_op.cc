/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific UnravelIndex Op.
//
// This mirrors the CPU kernel in
// tensorflow/core/kernels/unravel_index_op.cc: given a flat `indices` tensor
// (scalar or vector) and a 1-D `dims` tensor, computes the row-major
// multi-dimensional coordinates for each flat index.
//
// `dims` is required to be a compile-time constant (the same requirement
// Reshape's `shape` input and similar tf2xla kernels already place on their
// shape-describing input), which lets the per-dimension strides be computed
// on the host and lowered to a small constant + elementwise div/mod
// computation instead of needing a dynamic loop.
//
// Unlike the CPU kernel, this lowering does not reproduce the CPU kernel's
// runtime "index is out of bound as with dims" check: `indices` is a
// general XlaOp that is not necessarily known at compile time, and emitting
// a device-side bounds check that raises is not something this op's XLA
// lowering does elsewhere in this file's sibling kernels (e.g. Gather also
// does not). Out-of-range indices produce an unspecified (wrapped) result
// rather than an error, which should be called out in the op's docs if this
// lands, the same way other index-consuming XLA kernels' behavior on
// invalid input is documented as unspecified rather than validated.

#include <cstdint>
#include <limits>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class UnravelIndexOp : public XlaOpKernel {
 public:
  explicit UnravelIndexOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape indices_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(indices_shape) ||
                    TensorShapeUtils::IsScalar(indices_shape),
                errors::InvalidArgument(
                    "The indices can only be scalar or vector, got \"",
                    indices_shape.DebugString(), "\""));

    const TensorShape dims_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(dims_shape),
                errors::InvalidArgument("The dims can only be 1-D, got \"",
                                        dims_shape.DebugString(), "\""));

    std::vector<int64_t> dims;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &dims));
    const int64_t rank = dims.size();

    int64_t product = 1;
    for (int64_t i = 0; i < rank; ++i) {
      OP_REQUIRES(ctx, dims[i] > 0,
                  errors::InvalidArgument(
                      "Input dims cannot be zero or negative, got dim = ",
                      dims[i], " at index ", i));
      OP_REQUIRES(ctx, product <= std::numeric_limits<int64_t>::max() / dims[i],
                  errors::InvalidArgument(
                      "Input dims product is causing integer overflow"));
      product *= dims[i];
    }

    // trailing[i] = product(dims[i+1:]), i.e. the stride to divide a flat
    // index by to get the coordinate along dimension i (before taking it
    // mod dims[i]). trailing[rank - 1] is 1.
    std::vector<int64_t> trailing(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) {
      trailing[i] = trailing[i + 1] * dims[i + 1];
    }

    xla::XlaBuilder* b = ctx->builder();
    const xla::PrimitiveType index_type = ctx->input_xla_type(0);
    xla::XlaOp indices = ctx->Input(0);

    xla::XlaOp trailing_const = xla::ConvertElementType(
        xla::ConstantR1<int64_t>(b, trailing), index_type);
    xla::XlaOp dims_const =
        xla::ConvertElementType(xla::ConstantR1<int64_t>(b, dims), index_type);

    xla::XlaOp output;
    if (TensorShapeUtils::IsScalar(indices_shape)) {
      // Output shape [rank]: broadcast the scalar index to [rank], then
      // (index / trailing[i]) % dims[i] elementwise against the [rank]
      // constants built above.
      xla::XlaOp indices_bcast = xla::Broadcast(indices, {rank});
      output = xla::Rem(xla::Div(indices_bcast, trailing_const), dims_const);
    } else {
      // Output shape [rank, num_indices]: broadcast indices (shape
      // [num_indices]) by prepending the rank dimension, and broadcast the
      // [rank] stride/dims constants along a new trailing indices
      // dimension via BroadcastInDim, then combine elementwise.
      const int64_t num_indices = indices_shape.num_elements();
      OP_REQUIRES(ctx, num_indices > 0,
                  errors::InvalidArgument("received empty tensor indices: ",
                                          indices_shape.DebugString()));
      xla::XlaOp indices_bcast = xla::Broadcast(indices, {rank});
      xla::XlaOp trailing_bcast =
          xla::BroadcastInDim(trailing_const, {rank, num_indices}, {0});
      xla::XlaOp dims_bcast =
          xla::BroadcastInDim(dims_const, {rank, num_indices}, {0});
      output = xla::Rem(xla::Div(indices_bcast, trailing_bcast), dims_bcast);
    }

    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("UnravelIndex").CompileTimeConstantInput("dims"),
                UnravelIndexOp);

}  // namespace
}  // namespace tensorflow
