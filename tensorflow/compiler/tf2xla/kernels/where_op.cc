/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

class WhereOp : public XlaOpKernel {
 public:
  explicit WhereOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp condition = ctx->Input(0);
    StatusOr<xla::Shape> input_shape = ctx->builder()->GetShape(condition);
    OP_REQUIRES_OK(ctx, input_shape.status());
    auto iota_shape = input_shape.ValueOrDie();
    iota_shape.set_element_type(xla::S32);

    int64_t flattened_size = xla::Product(iota_shape.dimensions());
    xla::XlaOp reshaped_condition = xla::Reshape(condition, {flattened_size});
    xla::XlaOp zeros = xla::ZerosLike(reshaped_condition);
    xla::XlaOp compared = xla::Ne(reshaped_condition, zeros);

    std::vector<xla::XlaOp> to_sort = {compared};
    std::vector<xla::PrimitiveType> types_to_sort = {xla::PRED};
    // Generate iota for each dimension, which after combining becomes
    // indices of each element.
    for (int64_t axis = 0; axis < iota_shape.rank(); ++axis) {
      xla::XlaOp iota = xla::Iota(ctx->builder(), iota_shape, axis);
      xla::XlaOp reshaped = xla::Reshape(iota, {flattened_size});
      to_sort.push_back(reshaped);
      types_to_sort.push_back(xla::S32);
    }

    xla::XlaOp sorted = xla::Sort(
        to_sort, xla::CreateScalarGtComputation(types_to_sort, ctx->builder()),
        /*dimension=*/0,
        /*is_stable=*/true);
    std::vector<xla::XlaOp> to_concat;
    for (int64_t i = 0; i < iota_shape.rank(); ++i) {
      xla::XlaOp index_single_dim = xla::GetTupleElement(sorted, i + 1);
      to_concat.push_back(xla::Reshape(index_single_dim, {flattened_size, 1}));
    }

    xla::XlaOp result = xla::ConcatInDim(ctx->builder(), to_concat, 1);
    result = xla::ConvertElementType(result, ctx->output_xla_type(0));

    // Dynamic padder will handle the dynamic dimension.
    xla::XlaOp compared_int = xla::ConvertElementType(compared, xla::S32);
    xla::XlaOp length = xla::ReduceAll(
        compared_int, xla::Zero(ctx->builder(), xla::S32),
        xla::CreateScalarAddComputation(xla::S32, ctx->builder()));
    StatusOr<xla::XlaOp> rebounded_result = xla::SetDimensionSizeWithRebound(
        &ctx->value_inference(), result, length, 0);
    if (rebounded_result.ok()) {
      ctx->SetOutput(0, *rebounded_result);
      return;
    }
    // TODO(b/207187072): Remove special handling once dynamic reshape can also
    // be handled.
    xla::XlaOp bounded_result = xla::SetDimensionSize(result, length, 0);
    ctx->SetOutput(0, bounded_result);
  }
};

REGISTER_XLA_OP(Name("Where"), WhereOp);

}  // namespace
}  // namespace tensorflow
