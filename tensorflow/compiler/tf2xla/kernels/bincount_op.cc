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

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace tensorflow {
namespace {

// TODO: This is only a dummy kernel
class BincountOp : public XlaOpKernel {
 public:
  explicit BincountOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    DataType dtype;
  }
  
  void Compile(XlaOpKernelContext* ctx) override {
  // Dumb implementation for the simplest test case
    xla::XlaOp input = ctx->Input(0);
    int64_t output_size;
    ctx->ConstantInputAsIntScalar("size",&output_size);
    StatusOr<xla::Shape> input_shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto input_shape = input_shape_or.ValueOrDie();
    int64_t size = input_shape.dimensions(0);
    auto counter_shape = xla::ShapeUtil::MakeShape(xla::S32, {});
    const xla::Shape data_shape = xla::ShapeUtil::MakeShape(xla::S32, {size});

    const xla::Shape output_shape = xla::ShapeUtil::MakeShape(xla::S32, {output_size});

    auto loop_shape = xla::ShapeUtil::MakeTupleShape(
        {counter_shape, data_shape, output_shape});

    // Create a computation for the condition
    xla::XlaComputation condition;
    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("condition");
      auto param = xla::Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);
      xla::Gt(xla::ConstantR0<int32_t>(builder.get(), size), counter);
      condition = builder->Build().ConsumeValueOrDie();
    }

    // Create a computation for the body
    xla::XlaComputation body;
    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("body");
      auto param = Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);
      auto data_stack = xla::GetTupleElement(param, 1);
      auto accum_stack = xla::GetTupleElement(param, 2);
      auto data = xla::DynamicSlice(data_stack, {counter}, {1});
      auto accum = xla::DynamicSlice(accum_stack, {data}, {1});
      accum = accum + xla::One(builder.get(), xla::S32);
      accum_stack = xla::DynamicUpdateSlice(
          accum_stack, xla::Reshape(accum, {1}), {data});
      counter = counter + xla::One(builder.get(), xla::S32);
      xla::Tuple(builder.get(), {counter, data_stack, accum_stack});
      body = builder->Build().ConsumeValueOrDie();
    }
    // Create a While node with computations for the condition and the body.
    auto zero = xla::Zero(ctx->builder(), xla::S32);
    auto zero_broadcast = xla::Broadcast(zero, {output_size});
    auto init = xla::Tuple(ctx->builder(), {zero, input, zero_broadcast});
    auto result = xla::While(condition, body, init);
    auto output = xla::GetTupleElement(result,2);
    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("Bincount").CompileTimeConstantInput("size"), BincountOp);

}  // namespace
}  // namespace tensorflow
