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

// XLA-specific base classes for Unary and Binary Ops.

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

void XlaBinaryOp::Compile(XlaOpKernelContext* ctx) {
  const TensorShape lhs_shape = ctx->InputShape(0);
  const TensorShape rhs_shape = ctx->InputShape(1);

  // By TensorFlow conventions the inputs may not have the same
  // shapes, in which case they will be automatically broadcast if
  // possible before mapping. Use the standard TensorFlow helper to
  // compute valid broadcast shapes, but rely below on XLA to
  // automatically perform the broadcast assuming its valid shapes are
  // a superset of TensorFlow's valid shapes.
  BCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape),
              /*fewer_dims_optimization=*/false);
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument("Incompatible shapes: ",
                                           lhs_shape.DebugString(), " vs. ",
                                           rhs_shape.DebugString()));
    return;
  }
  TensorShape bcast_shape = BCast::ToShape(bcast.output_shape());

  // Fetch the expressions containing the input tensors.
  auto lhs_handle = ctx->Input(0);
  auto rhs_handle = ctx->Input(1);

  // If the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a
  // prefix. In other words to pad a 5-vector to a 3-dimensional
  // tensor it is reshaped to have shape [1,1,5]. XLA's automatic
  // broadcast code is able to broadcast from lower to higher rank,
  // but doesn't assume you want to pad as a prefix of the dimensions,
  // and instead needs to be told which dimensions of the higher rank
  // tensor to match to the lower rank tensor. In this example it
  // would be dimensions [2]. If we were matching a matrix against a
  // 4-D tensor the dimensions to match would be [2,3],
  // etc. extend_dimension encodes the general case.
  std::vector<int64> extend_dimension;
  int max_rank = std::max(lhs_shape.dims(), rhs_shape.dims());
  int min_rank = std::min(lhs_shape.dims(), rhs_shape.dims());
  if (min_rank != max_rank) {
    for (int i = 0; i < min_rank; ++i) {
      // Match the lower rank tensor along the larger-numbered
      // dimensions of the higher rank tensor.
      extend_dimension.push_back(max_rank - min_rank + i);
    }
  }

  // Call virtual method to emit the computation.
  xla::XlaOp output =
      Computation(ctx, lhs_handle, lhs_shape.dim_sizes(), rhs_handle,
                  rhs_shape.dim_sizes(), bcast, extend_dimension);

  // The TensorFlow helper computed the post-broadcast shape in
  // output_shape: we rely on subclassed Computations to implement the
  // same broadcast semantics.
  ctx->SetOutput(0, output);
}

/* static */ std::pair<xla::XlaOp, xla::XlaOp> XlaBinaryOp::Broadcast(
    xla::XlaOp lhs, xla::XlaOp rhs, const BCast& broadcast_helper) {
  auto lhs_output = BroadcastTo(lhs, broadcast_helper.output_shape());
  if (!lhs_output.ok()) {
    xla::XlaOp error = lhs.builder()->ReportError(lhs_output.status());
    return {error, error};
  }
  auto rhs_output = BroadcastTo(rhs, broadcast_helper.output_shape());
  if (!rhs_output.ok()) {
    xla::XlaOp error = rhs.builder()->ReportError(rhs_output.status());
    return {error, error};
  }
  return {lhs_output.ValueOrDie(), rhs_output.ValueOrDie()};
}

}  // namespace tensorflow
