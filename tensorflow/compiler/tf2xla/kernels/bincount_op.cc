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
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace tensorflow {
namespace {

// TODO: This is only a dummy kernel
class DenseBincountOp : public XlaOpKernel {
  private:
  bool binary_output_;
 public:
  explicit DenseBincountOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }
  
  void Compile(XlaOpKernelContext* ctx) override {
  // Dumb implementation for the simplest test case
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp weights = ctx->Input(2);    
    StatusOr<xla::Shape> weights_shape_or = ctx->builder()->GetShape(weights);
    OP_REQUIRES_OK(ctx, weights_shape_or.status());
    auto weights_shape = weights_shape_or.ValueOrDie();
    auto weights_size = weights_shape.dimensions(0);
    auto input_xla_type = ctx->input_xla_type(0);
    xla::PrimitiveType dtype;
    bool has_weight;
    if (weights_size){
      has_weight = true;
      dtype = ctx->input_xla_type(2);
    }
    else {
      has_weight = false;
      dtype = input_xla_type;
    }
    int64_t output_size;
    ctx->ConstantInputAsIntScalar("size", &output_size);
    StatusOr<xla::Shape> input_shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto input_shape = input_shape_or.ValueOrDie();
    auto size = input_shape.dimensions(0);
    auto dim = 1;
    auto rank = input_shape.rank();
    auto counter_shape = xla::ShapeUtil::MakeShape(xla::S64, {});
    const xla::Shape data_shape = xla::ShapeUtil::MakeShape(input_xla_type, {input_shape.dimensions()});

    xla::Shape output_shape = xla::ShapeUtil::MakeShape(dtype, {output_size});
    if (rank == 2) {
      output_shape = xla::ShapeUtil::MakeShape(dtype, {size, output_size});
      dim = input_shape.dimensions(1);
    }

    auto loop_shape = xla::ShapeUtil::MakeTupleShape(
        {counter_shape, data_shape, output_shape, weights_shape});

    // Create a computation for the condition
    xla::XlaComputation condition;
    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("condition");
      auto param = xla::Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);
      xla::Gt(xla::ConstantR0<int64_t>(builder.get(), size*dim), counter);
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
      auto weights = xla::GetTupleElement(param, 3);
      auto accum_shape = xla::ShapeUtil::MakeShape(dtype, {});

      if (rank == 1) {
        auto data = xla::DynamicSlice(data_stack, {counter}, {1});
        auto accum = xla::DynamicSlice(accum_stack, {data}, {1});
        accum = xla::Reshape(accum, {0}, {});
        auto data_scalar = xla::Reshape(data, {0}, {});

        auto condition_shape = xla::ShapeUtil::MakeTupleShape(
          {counter_shape, counter_shape, accum_shape, output_shape, weights_shape});

        xla::XlaComputation update;
        { 
          std::unique_ptr<xla::XlaBuilder> true_builder =
              ctx->builder()->CreateSubBuilder("update");
          auto param = Parameter(true_builder.get(), 0, condition_shape, "param");
          auto data_scalar = xla::GetTupleElement(param, 0);
          auto counter = xla::GetTupleElement(param, 1);
          auto accum = xla::GetTupleElement(param, 2);
          auto accum_stack  = xla::GetTupleElement(param, 3);
          auto weights = xla::GetTupleElement(param, 4);

          if (binary_output_){
            accum = xla::One(true_builder.get(), dtype);
          }
          else if (! has_weight) {
            accum = accum + xla::One(true_builder.get(), dtype);
          }
          else {
            auto weight = xla::DynamicSlice(weights, {counter}, {1});
            weight = xla::Reshape(weight, {0}, {});
            accum = accum + weight;
          }
          accum_stack = xla::DynamicUpdateSlice(
              accum_stack, xla::Reshape(accum, {1}), {data_scalar});
          xla::Tuple(true_builder.get(), {accum, accum_stack});
          update = true_builder->Build().ValueOrDie();
        }

        xla::XlaComputation no_update;
        {
          std::unique_ptr<xla::XlaBuilder> false_builder =
              ctx->builder()->CreateSubBuilder("no_update");
          auto param = Parameter(false_builder.get(), 0, condition_shape, "param");
          auto data = xla::GetTupleElement(param, 0);
          auto count = xla::GetTupleElement(param, 1);
          auto accum = xla::GetTupleElement(param, 2);
          auto accum_stack  = xla::GetTupleElement(param, 3);
          xla::Tuple(false_builder.get(), {accum, accum_stack});
          no_update = false_builder->Build().ValueOrDie();
        }

        auto output_size_xla = xla::ConstantR0<int64_t>(builder.get(), output_size);
        data_scalar = xla::ConvertElementType(data_scalar, xla::S64);
        auto pred = xla::Lt(data_scalar, output_size_xla);
        auto tuple = xla::Tuple(builder.get(), {data_scalar, counter, accum, accum_stack, weights});
        auto cond = xla::Conditional(pred, tuple, update, tuple, no_update);
        accum = xla::GetTupleElement(cond, 0);
        accum_stack = xla::GetTupleElement(cond, 1);

      }
      else {

        auto condition_shape = xla::ShapeUtil::MakeTupleShape(
          {counter_shape, counter_shape, counter_shape, output_shape, 
           accum_shape, weights_shape});

        auto dim_xla = xla::ConstantR0<int64_t>(builder.get(), dim);
        auto idx_1 = xla::Div(counter, dim_xla);
        auto idx_2 = counter % dim_xla;
        auto data = xla::DynamicSlice(data_stack, {idx_1, idx_2}, {1, 1});
        auto data_scalar = xla::Reshape(data, {0,1}, {});
        data_scalar = xla::ConvertElementType(data_scalar, xla::S64);
        auto accum = xla::DynamicSlice(accum_stack, {idx_1, data_scalar}, {1, 1});
        accum = xla::Reshape(accum, {0,1}, {});
        xla::XlaComputation update;
        {
          std::unique_ptr<xla::XlaBuilder> true_builder =
              builder->CreateSubBuilder("update_rank2");
          auto param = Parameter(true_builder.get(), 0, condition_shape, "param");
          
          auto data_scalar = xla::GetTupleElement(param, 0);
          auto idx_1 = xla::GetTupleElement(param, 1);
          auto idx_2 = xla::GetTupleElement(param, 2);
          auto accum_stack  = xla::GetTupleElement(param, 3);
          auto accum = xla::GetTupleElement(param, 4);
          auto weights = xla::GetTupleElement(param, 5);

          if (binary_output_){
            accum = xla::One(true_builder.get(), dtype);
          }
          else if (! has_weight) {
            accum = accum + xla::One(true_builder.get(), dtype);
          }
          else {
            auto weight = xla::DynamicSlice(weights, {idx_1, idx_2}, {1, 1});
            auto weigth_scalar = xla::Reshape(weight, {0,1}, {});
            accum = accum + weigth_scalar;
          }
          accum_stack = xla::DynamicUpdateSlice(
              accum_stack, xla::Reshape(accum, {1, 1}), {idx_1, data_scalar});
          xla::Tuple(true_builder.get(), {accum, accum_stack});
          update = true_builder->Build().ValueOrDie();
        }

        xla::XlaComputation no_update;
        {
          std::unique_ptr<xla::XlaBuilder> false_builder =
              builder->CreateSubBuilder("no_update_rank2");
          auto param = Parameter(false_builder.get(), 0, condition_shape, "param");
          auto accum_stack  = xla::GetTupleElement(param, 3);
          auto accum = xla::GetTupleElement(param, 4);
          xla::Tuple(false_builder.get(), {accum, accum_stack});
          no_update = false_builder->Build().ValueOrDie();
        }
        auto limit = xla::ConstantR0<int64_t>(builder.get(), output_size);
        data_scalar = xla::ConvertElementType(data_scalar, xla::S64);


        auto pred = xla::Lt(data_scalar, limit);
        auto tuple = xla::Tuple(builder.get(), {data_scalar, idx_1, idx_2, accum_stack, accum, weights});
        auto cond = xla::Conditional(pred, tuple, update, tuple, no_update);
        accum = xla::GetTupleElement(cond, 0);
        accum_stack = xla::GetTupleElement(cond, 1);
      }
      counter = counter + xla::One(builder.get(), xla::S64);
      xla::Tuple(builder.get(), {counter, data_stack, accum_stack, weights});
      body = builder->Build().ConsumeValueOrDie();
    }

    // Create a While node with computations for the condition and the body.
    auto zero = xla::Zero(ctx->builder(), xla::S64);
    auto zero_out = xla::Zero(ctx->builder(), dtype);
    auto zero_broadcast = xla::Broadcast(zero_out, {output_shape.dimensions()});
    auto init = xla::Tuple(ctx->builder(), {zero, input, zero_broadcast, weights});
    auto result = xla::While(condition, body, init);
    auto output = xla::GetTupleElement(result,2);
    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("DenseBincount").CompileTimeConstantInput("size"), DenseBincountOp);

}  // namespace
}  // namespace tensorflow
