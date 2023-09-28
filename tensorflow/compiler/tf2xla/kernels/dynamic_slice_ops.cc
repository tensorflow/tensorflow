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

#include <algorithm>

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

absl::InlinedVector<xla::XlaOp, 4> SliceVector(xla::XlaOp input, int64_t rank) {
  absl::InlinedVector<xla::XlaOp, 4> scalar_indices;
  scalar_indices.reserve(rank);
  for (int i = 0; i < rank; i++)
    scalar_indices.push_back(
        xla::Reshape(xla::Slice(input, {i}, {i + 1}, {1}), {}));
  return scalar_indices;
}

class DynamicUpdateSliceOp : public XlaOpKernel {
 public:
  explicit DynamicUpdateSliceOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    DataType index_type = ctx->InputType("indices");
    CHECK(index_type == DT_INT32 || index_type == DT_INT64);

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape update_shape = ctx->InputShape("update");
    const TensorShape index_shape = ctx->InputShape("indices");

    int64_t rank = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(index_shape) &&
            index_shape.num_elements() == rank,
        errors::InvalidArgument("index must be a vector with length equal to "
                                "the number of input dimensions"));
    OP_REQUIRES(
        ctx, rank == update_shape.dims(),
        errors::InvalidArgument("input and update must have the same rank,"
                                " input shape is ",
                                input_shape.DebugString(), "; update shape is ",
                                update_shape.DebugString()));

    xla::XlaOp indices = ctx->Input("indices");
    xla::XlaOp result = xla::DynamicUpdateSlice(
        ctx->Input("input"), ctx->Input("update"), SliceVector(indices, rank));
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("XlaDynamicUpdateSlice"), DynamicUpdateSliceOp);

REGISTER_XLA_OP(
    Name("XlaDynamicSlice").CompileTimeConstantInput("size_indices"),
    MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
