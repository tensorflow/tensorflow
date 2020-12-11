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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

class UniqueOp : public XlaOpKernel {
 public:
  explicit UniqueOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  // We use a two level loop algorithm to calculate unique.
  //
  // i = 0
  // output_size = 0
  // output_indices = broadcast(0, {input_size})
  // while (i < input_size) {
  //   search_result_index = output_size
  //   j = 0
  //   while (j < output_size) {
  //     if(input[j]==input[i]) {
  //       search_result_index = j
  //     }
  //     ++j
  //   }
  //   input[search_result_index] = input[i]
  //   output_indices[i] = search_result_index
  //   if (search_result_index == output_size) {
  //     // Not found
  //     output_size ++;
  //   }
  //   i ++;
  // }
  //
  // The algorithm is then functionalized into xla whiles.  Outer-scoped
  // variables are captured as inputs and outputs to the while loop.
  // Conditionals are rewritten into xla select for simplicity.
  xla::XlaComputation BuildInnerLoopCond(XlaOpKernelContext* ctx,
                                         xla::Shape inner_loop_shape) {
    std::unique_ptr<xla::XlaBuilder> builder =
        ctx->builder()->CreateSubBuilder("inner_loop_cond");
    auto param = xla::Parameter(builder.get(), 0, inner_loop_shape, "param");
    auto j = xla::GetTupleElement(param, 2);
    auto output_element_size = xla::GetTupleElement(param, 3);
    xla::Lt(j, output_element_size);
    return builder->Build().ConsumeValueOrDie();
  }

  xla::XlaComputation BuildInnerLoopBody(XlaOpKernelContext* ctx,
                                         xla::Shape inner_loop_shape,
                                         xla::Shape single_element_shape) {
    std::unique_ptr<xla::XlaBuilder> builder =
        ctx->builder()->CreateSubBuilder("inner_loop_body");
    auto param = xla::Parameter(builder.get(), 0, inner_loop_shape, "param");
    auto input = xla::GetTupleElement(param, 0);
    auto target = xla::GetTupleElement(param, 1);
    auto j = xla::GetTupleElement(param, 2);
    auto output_element_size = xla::GetTupleElement(param, 3);
    auto output_index = xla::GetTupleElement(param, 4);
    auto input_elem = xla::DynamicSlice(input, {j}, {1});
    auto input_elem_scalar = xla::Reshape(single_element_shape, input_elem);
    auto eq = xla::Eq(input_elem_scalar, target);
    auto select = xla::Select(eq, j, output_index);
    auto next_j = xla::Add(j, xla::One(builder.get(), xla::S32));
    xla::Tuple(builder.get(),
               {input, target, next_j, output_element_size, select});
    return builder->Build().ConsumeValueOrDie();
  }

  xla::XlaComputation BuildOuterLoopCond(XlaOpKernelContext* ctx,
                                         xla::Shape outer_loop_shape,
                                         int64 list_size) {
    std::unique_ptr<xla::XlaBuilder> builder =
        ctx->builder()->CreateSubBuilder("outer_loop_body");
    auto param =
        xla::Parameter(builder.get(), 0, outer_loop_shape, "outer_loop_param");
    auto i = xla::GetTupleElement(param, 2);
    auto bound = xla::ConstantR0<int32>(builder.get(), list_size);
    xla::Lt(i, bound);
    return builder->Build().ConsumeValueOrDie();
  }

  xla::XlaComputation BuildOuterLoopBody(
      XlaOpKernelContext* ctx, xla::Shape outer_loop_shape,
      xla::Shape single_element_shape, const xla::XlaComputation& inner_cond,
      const xla::XlaComputation& inner_body) {
    std::unique_ptr<xla::XlaBuilder> builder =
        ctx->builder()->CreateSubBuilder("outer_loop_body");
    auto param = xla::Parameter(builder.get(), 0, outer_loop_shape, "param");
    auto input = xla::GetTupleElement(param, 0);
    auto indices = xla::GetTupleElement(param, 1);
    auto i = xla::GetTupleElement(param, 2);
    auto output_element_size = xla::GetTupleElement(param, 3);
    auto zero = xla::Zero(builder.get(), xla::S32);
    auto target = xla::DynamicSlice(input, {i}, {1});
    auto target_scalar = xla::Reshape(single_element_shape, target);
    auto inner_loop_param = xla::Tuple(
        builder.get(),
        {input, target_scalar, zero, output_element_size, output_element_size});
    auto inner_loop = xla::While(inner_cond, inner_body, inner_loop_param);
    auto output_index = xla::GetTupleElement(inner_loop, 4);
    auto one = xla::One(builder.get(), xla::S32);
    auto update_output_element_size =
        xla::Select(xla::Eq(output_index, output_element_size),
                    xla::Add(output_element_size, one), output_element_size);
    auto update_input = xla::DynamicUpdateSlice(input, target, {output_index});
    auto update_indices =
        xla::DynamicUpdateSlice(indices, xla::Reshape(output_index, {1}), {i});
    xla::Tuple(builder.get(), {update_input, update_indices, xla::Add(i, one),
                               update_output_element_size});
    return builder->Build().ConsumeValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::StatusOr<xla::Shape> input_shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto input_shape = input_shape_or.ValueOrDie();
    xla::Shape single_index_shape = xla::ShapeUtil::MakeScalarShape(xla::S32);
    xla::Shape single_element_shape =
        xla::ShapeUtil::MakeScalarShape(input_shape.element_type());
    OP_REQUIRES(ctx, input_shape.rank() == 1,
                xla::InvalidArgument("Input to UniqueOp must be rank-1: %s",
                                     input_shape.ToString()));
    int64 list_size = input_shape.dimensions()[0];
    auto indices_shape =
        xla::ShapeUtil::ChangeElementType(input_shape, xla::S32);
    auto outer_loop_shape = xla::ShapeUtil::MakeTupleShape(
        {input_shape, indices_shape, single_index_shape, single_index_shape});
    auto inner_loop_shape = xla::ShapeUtil::MakeTupleShape(
        {input_shape, single_element_shape, single_index_shape,
         single_index_shape, single_index_shape});
    xla::XlaComputation inner_loop_cond =
        BuildInnerLoopCond(ctx, inner_loop_shape);
    xla::XlaComputation inner_loop_body =
        BuildInnerLoopBody(ctx, inner_loop_shape, single_element_shape);
    xla::XlaComputation outer_loop_cond =
        BuildOuterLoopCond(ctx, outer_loop_shape, list_size);
    xla::XlaComputation outer_loop_body =
        BuildOuterLoopBody(ctx, outer_loop_shape, single_element_shape,
                           inner_loop_cond, inner_loop_body);
    auto zero = xla::Zero(ctx->builder(), xla::S32);
    auto init_indices = xla::Broadcast(zero, {list_size});
    auto init = xla::Tuple(ctx->builder(), {input, init_indices, zero, zero});
    auto outer_while = xla::While(outer_loop_cond, outer_loop_body, init);
    auto output = xla::GetTupleElement(outer_while, 0);
    auto output_indices = xla::GetTupleElement(outer_while, 1);
    auto output_size = xla::GetTupleElement(outer_while, 3);
    auto output_dynamic = xla::SetDimensionSize(output, output_size, 0);
    ctx->SetOutput(0, output_dynamic);
    ctx->SetOutput(1, output_indices);
  }
};

REGISTER_XLA_OP(Name("Unique").Device(DEVICE_TPU_XLA_JIT), UniqueOp);

}  // namespace
}  // namespace tensorflow
