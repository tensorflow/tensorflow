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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_CWISE_OPS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_CWISE_OPS_H_

#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Coefficient-wise binary operations. Each binary Op expects two
// inputs that can be broadcast to the same shape. The base class
// contains pure virtual methods to override: description is a textual
// description of the operation; and Computation adds the
// implementation of the operation to a xla::XlaBuilder. For most
// arithmetic Ops XLA handles the broadcasting automatically given the input
// tensors.
class XlaBinaryOp : public XlaOpKernel {
 public:
  explicit XlaBinaryOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    const DataType lhs = BaseType(input_type(0));
    const DataType rhs = BaseType(input_type(1));
    OP_REQUIRES(ctx, lhs == rhs,
                errors::InvalidArgument("Input types of binary op must match"));
  }
  ~XlaBinaryOp() override = default;

  // Implement the (tensor,tensor)->tensor lambda that should be
  // applied to the inputs. The desired computation should be added to
  // 'tc->builder()' and '(lhs,rhs)' are the function's inputs and
  // (lhs_shape,rhs_shape) are their respective
  // shapes. 'broadcast_helper' contains metadata about the shapes of
  // the inputs and the dimensions that need to be broadcast, which
  // may be useful for Ops that can't use standard XLA automatic
  // broadcasting. 'extend_dimension' is non-empty if lhs and rhs have
  // different ranks, and indicates which dimensions of the
  // higher-rank input should be matched when broadcasting the
  // lower-rank input. See comment below and the documentation on broadcasting
  // in the XLA documentation.
  virtual xla::XlaOp Computation(
      XlaOpKernelContext* ctx, const xla::XlaOp& lhs,
      const absl::Span<const int64_t>& lhs_shape, const xla::XlaOp& rhs,
      const absl::Span<const int64_t>& rhs_shape, const BCast& broadcast_helper,
      const std::vector<int64_t>& extend_dimensions) = 0;

  void Compile(XlaOpKernelContext* ctx) override;

  // Helper function that performs the broadcasting described by
  // 'broadcast_helper', yielding arguments 'lhs' and 'rhs' that have the same
  // shape.
  static std::pair<xla::XlaOp, xla::XlaOp> Broadcast(
      xla::XlaOp lhs, xla::XlaOp rhs, const BCast& broadcast_helper);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_CWISE_OPS_H_
