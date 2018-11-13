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

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class XlaBroadcastHelperOp : public XlaOpKernel {
 public:
  explicit XlaBroadcastHelperOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaOp lhs = context->Input(0);
    xla::XlaOp rhs = context->Input(1);
    const TensorShape lhs_shape = context->InputShape(0);
    const TensorShape rhs_shape = context->InputShape(1);

    const bool broadcast_lhs = lhs_shape.dims() < rhs_shape.dims();
    const TensorShape* min_rank_shape = broadcast_lhs ? &lhs_shape : &rhs_shape;
    const TensorShape* max_rank_shape = broadcast_lhs ? &rhs_shape : &lhs_shape;

    std::vector<int64> broadcast_dims;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("broadcast_dims",
                                                              &broadcast_dims));
    if (broadcast_dims.empty()) {
      OP_REQUIRES(
          context,
          lhs_shape.dims() == rhs_shape.dims() || lhs_shape.dims() == 0 ||
              rhs_shape.dims() == 0,
          errors::InvalidArgument(
              "If broadcast_dims is empty, both "
              "arguments must have equal rank; "
              "argument shapes, or at least one argument must be a scalar: ",
              lhs_shape.DebugString(), " and ", rhs_shape.DebugString()));
      context->SetOutput(0, lhs);
      context->SetOutput(1, rhs);
      return;
    }

    OP_REQUIRES(
        context, broadcast_dims.size() == min_rank_shape->dims(),
        errors::InvalidArgument(
            "broadcast_dims must have size equal to the smaller argument rank; "
            "broadcast_dims: [",
            absl::StrJoin(broadcast_dims, ","), "]; argument shapes: ",
            lhs_shape.DebugString(), " and ", rhs_shape.DebugString()));
    std::vector<int64> sorted_broadcast_dims = broadcast_dims;
    absl::c_sort(sorted_broadcast_dims);
    std::set<int64> dims_set(broadcast_dims.begin(), broadcast_dims.end());
    OP_REQUIRES(context,
                dims_set.size() == broadcast_dims.size() &&
                    broadcast_dims == sorted_broadcast_dims,
                errors::InvalidArgument(
                    "Duplicate or nonmonotonic dimension in broadcast_dims; "
                    "broadcast_dims: [",
                    absl::StrJoin(broadcast_dims, ","), "]"));

    std::vector<int64> broadcast_shape(max_rank_shape->dims(), 1LL);
    for (int i = 0; i < broadcast_dims.size(); ++i) {
      const int dim = broadcast_dims[i];
      OP_REQUIRES(
          context, dim >= 0 && dim < broadcast_shape.size(),
          errors::InvalidArgument(
              "Invalid broadcast dimension (", dim, "); broadcast_dims: [",
              absl::StrJoin(broadcast_dims, ","), "]; argument shapes: ",
              lhs_shape.DebugString(), " and ", rhs_shape.DebugString()));
      broadcast_shape[dim] = min_rank_shape->dim_size(i);
    }
    xla::PrimitiveType type = context->input_xla_type(0);
    xla::Shape broadcast_xla_shape =
        xla::ShapeUtil::MakeShape(type, broadcast_shape);
    if (broadcast_lhs) {
      lhs = xla::BroadcastInDim(lhs, broadcast_xla_shape, broadcast_dims);
    } else {
      rhs = xla::BroadcastInDim(rhs, broadcast_xla_shape, broadcast_dims);
    }
    context->SetOutput(0, lhs);
    context->SetOutput(1, rhs);
  }

 private:
  xla::DotDimensionNumbers dnums_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaBroadcastHelperOp);
};

REGISTER_XLA_OP(
    Name("XlaBroadcastHelper").CompileTimeConstantInput("broadcast_dims"),
    XlaBroadcastHelperOp);

}  // namespace
}  // namespace tensorflow
