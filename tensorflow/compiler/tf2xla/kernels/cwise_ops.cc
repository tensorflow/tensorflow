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

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

void XlaBinaryOp::Compile(XlaOpKernelContext* ctx) {
  TensorShape lhs_shape = ctx->InputShape(0);
  TensorShape rhs_shape = ctx->InputShape(1);
  xla::Shape lhs_xla_shape = ctx->InputXlaShape(0).value();
  xla::Shape rhs_xla_shape = ctx->InputXlaShape(1).value();
  // Fetch the expressions containing the input tensors.
  auto lhs_handle = ctx->Input(0);
  auto rhs_handle = ctx->Input(1);
  if (lhs_shape.dims() == rhs_shape.dims()) {
    auto reconcile_tensor_mismatched_dims = [ctx](
                                                xla::XlaOp lhs, xla::XlaOp rhs,
                                                const xla::Shape& lhs_xla_shape,
                                                const xla::Shape& rhs_xla_shape,
                                                TensorShape* lhs_tensor_shape) {
      // Find out mismatched dimensions that are non-broadcastable.
      // Reconcile the
      // difference by slicing the bigger dimension.
      for (int64_t i = 0; i < lhs_xla_shape.rank(); ++i) {
        if (lhs_xla_shape.is_dynamic_dimension(i)) {
          if (!rhs_xla_shape.is_dynamic_dimension(i) &&
              lhs_xla_shape.dimensions(i) > rhs_xla_shape.dimensions(i) &&
              rhs_xla_shape.dimensions(i) != 1) {
            // e.g., :
            // lhs = [..., <=N, ...]
            // rhs = [..., 2  , ...]
            // Slice N into 2.
            // Size 1 dim doesn't need slice as the other side is
            // broadcastable.
            auto size = xla::GetDimensionSize(lhs, i);
            lhs = xla::SliceInDim(lhs, 0, rhs_xla_shape.dimensions(i), 1,
                                  /*dimno=*/i);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            // Propagate dynamic dimension.
            lhs = xla::SetDimensionSize(lhs, size, i);
          }
          if (rhs_xla_shape.is_dynamic_dimension(i) &&
              lhs_xla_shape.dimensions(i) < rhs_xla_shape.dimensions(i) &&
              rhs_xla_shape.dimensions(i) != 1 &&
              lhs_xla_shape.dimensions(i) != 1) {
            // e.g., :
            // lhs = [..., <=M, ...]
            // rhs = [..., <=N  , ...]
            // where M < N
            //
            // In this case we pad M into N to make the bounds the same.
            // Note that we can't slice N into M because M could be a
            // dynamic size 1 dim that's meant to be broadcasted to N.
            auto size = xla::GetDimensionSize(lhs, i);
            int64_t diff =
                rhs_xla_shape.dimensions(i) - lhs_xla_shape.dimensions(i);
            lhs = xla::PadInDim(
                lhs, xla::Zero(ctx->builder(), lhs_xla_shape.element_type()), i,
                0, diff);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            // Propagate dynamic dimension.
            lhs = xla::SetDimensionSize(lhs, size, i);
          }
          if (lhs_xla_shape.dimensions(i) == 1 &&
              rhs_xla_shape.dimensions(i) != 1) {
            // lhs = [..., <=1, ...]
            // rhs = [...,   N, ...] or [..., <=N, ...]
            // where N != 1.
            //
            // In this case we will need to broadcast this dimension to N.
            // If the dynamic size is 0, the result size is zero.
            // If the dynamic size is 1, the result size is N.
            //
            // However, XLA only does degenerate broadcasts for non-dynamic
            // dimensions of size 1.

            // Get the original size.
            auto size = xla::GetDimensionSize(lhs, i);

            // Remove the dynamic dimension.
            lhs = xla::RemoveDynamicDimension(lhs, i);

            // Broadcast the dimension to N.
            std::vector<int64_t> dimensions(lhs_xla_shape.dimensions().begin(),
                                            lhs_xla_shape.dimensions().end());
            dimensions[i] = rhs_xla_shape.dimensions(i);
            std::vector<int64_t> broadcast_dimensions(lhs_xla_shape.rank());
            absl::c_iota(broadcast_dimensions, 0);
            lhs = xla::BroadcastInDim(lhs, dimensions, broadcast_dimensions);

            xla::XlaOp rhs_size;
            if (rhs_xla_shape.is_dynamic_dimension(i)) {
              rhs_size = xla::GetDimensionSize(rhs, i);
            } else {
              rhs_size = xla::ConstantR0<int32_t>(lhs.builder(),
                                                  rhs_xla_shape.dimensions(i));
            }

            // The original size is 0 or 1, so we can multiply it by the RHS
            // size to get the size of the resulting broadcast.
            size = xla::Mul(size, rhs_size);

            // Set the resulting dimension size.
            lhs = xla::SetDimensionSize(lhs, size, i);

            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
          }
        }
      }
      return lhs;
    };
    lhs_handle = reconcile_tensor_mismatched_dims(
        lhs_handle, rhs_handle, lhs_xla_shape, rhs_xla_shape, &lhs_shape);
    rhs_handle = reconcile_tensor_mismatched_dims(
        rhs_handle, lhs_handle, rhs_xla_shape, lhs_xla_shape, &rhs_shape);
  }
  // By TensorFlow conventions the inputs may not have the same
  // shapes, in which case they will be automatically broadcast if
  // possible before mapping. Use the standard TensorFlow helper to
  // compute valid broadcast shapes, but rely below on XLA to
  // automatically perform the broadcast assuming its valid shapes are
  // a superset of TensorFlow's valid shapes.
  BCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape),
              /*fewer_dims_optimization=*/false);
  if (!bcast.IsValid()) {
    ctx->SetStatus(absl::InvalidArgumentError(
        absl::StrCat("Incompatible shapes: ", lhs_shape.DebugString(), " vs. ",
                     rhs_shape.DebugString())));
    return;
  }

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
  std::vector<int64_t> extend_dimension;
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
  return {lhs_output.value(), rhs_output.value()};
}

}  // namespace tensorflow
