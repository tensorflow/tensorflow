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
    bool has_weights;
    if (weights_size){
      has_weights = true;
      dtype = ctx->input_xla_type(2);
    }
    else {
      has_weights = false;
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
    input = xla::Reshape(input, {size, 1});
    auto idx = xla::Reshape(input, {size, 1});
    auto one = xla::One(ctx->builder(), input_xla_type);
    xla::XlaOp updates, output;
    if (has_weights) {
      updates = weights;
      auto zero = xla::Zero(ctx->builder(), dtype);
      output = xla::Broadcast(zero, {output_shape.dimensions()});
    }
    else  {
      auto zero = xla::Zero(ctx->builder(), input_xla_type);
      updates = xla::Broadcast(one, {output_shape.dimensions()});
      output = xla::Broadcast(zero, {output_shape.dimensions()});
    }
    
    xla::XlaComputation assn_computation = [&] {
      std::unique_ptr<xla::XlaBuilder> subb =
      ctx->builder()->CreateSubBuilder("scatter_bincount");
      xla::Shape param_shape = xla::ShapeUtil::MakeShape(dtype, {});
      auto p0 = xla::Parameter(subb.get(), 0, param_shape, "p0");
      auto p1 = xla::Parameter(subb.get(), 1, param_shape, "p1");
      if (binary_output_) {
        xla::One(subb.get(), xla::S32);
      }
      else {
        xla::Add(p0, p1);
      }
      return subb->BuildAndNoteError();
    }();
    xla::ScatterDimensionNumbers scatter_dnums;
    scatter_dnums.set_index_vector_dim(1);
    scatter_dnums.add_inserted_window_dims(0);
    scatter_dnums.add_scatter_dims_to_operand_dims(0);
    output = xla::Scatter(output, idx, updates, assn_computation, scatter_dnums, false, false);

    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("DenseBincount").CompileTimeConstantInput("size"), DenseBincountOp);

}  // namespace
}  // namespace tensorflow
