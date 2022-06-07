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

    ctx->SetOutput(0, input);
  }
};

REGISTER_XLA_OP(Name("DenseBincount").CompileTimeConstantInput("size"), DenseBincountOp);

}  // namespace
}  // namespace tensorflow
