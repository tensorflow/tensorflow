/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace tensorflow {
namespace {

class DenseBincountOp : public XlaOpKernel {
 public:
  explicit DenseBincountOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    // It is optional for Bincount and required for DenseBincount
    (void)ctx->GetAttr("binary_output", &binary_output_);
  }

 private:
  bool binary_output_ = false;
  void Compile(XlaOpKernelContext* ctx) override {
    int64_t output_size;
    xla::XlaOp output_size_param = ctx->Input("size");
    StatusOr<xla::Shape> output_shape_or =
        ctx->builder()->GetShape(output_size_param);
    OP_REQUIRES_OK(ctx, output_shape_or.status());
    auto output_shape_param = output_shape_or.value();
    auto output_rank = output_shape_param.rank();
    OP_REQUIRES(ctx, output_rank == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        output_rank));
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("size", &output_size));
    OP_REQUIRES(ctx, output_size >= 0,
                errors::InvalidArgument("size (", output_size,
                                        ") must be non-negative"));
    xla::XlaOp idx, updates, output;
    xla::XlaOp input = ctx->Input(0);
    auto input_xla_type = ctx->input_xla_type(0);
    xla::PrimitiveType dtype = ctx->InputXlaType("weights");
    auto zero = xla::Zero(ctx->builder(), dtype);
    auto one = xla::One(ctx->builder(), dtype);
    StatusOr<xla::Shape> input_shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto input_shape = input_shape_or.value();
    auto size = input_shape.dimensions(0);

    if (!size) {
      output = xla::Broadcast(zero, {output_size});
      ctx->SetOutput(0, output);
      return;
    }
    auto rank = input_shape.rank();

    OP_REQUIRES(ctx, rank <= 2,
                errors::InvalidArgument(
                    "Shape must be at most rank 2 but is rank ", rank));

    xla::XlaOp weights = ctx->Input(2);
    StatusOr<xla::Shape> weights_shape_or = ctx->builder()->GetShape(weights);
    OP_REQUIRES_OK(ctx, weights_shape_or.status());

    auto weights_shape = weights_shape_or.value();
    OP_REQUIRES(ctx,
                xla::ShapeUtil::CompatibleIgnoringElementType(weights_shape,
                                                              input_shape) ||
                    (weights_shape.dimensions_size() > 0 &&
                     weights_shape.dimensions(0) == 0),
                errors::InvalidArgument(
                    "`weights` must be the same shape as `arr` or a length-0 "
                    "`Tensor`, in which case it acts as all weights equal to "
                    "1. Received ",
                    weights_shape.DebugString()));

    auto weights_size = weights_shape.dimensions(0);
    bool has_weights = false;
    if (weights_size) {
      has_weights = true;
    }
    xla::Shape output_shape = xla::ShapeUtil::MakeShape(dtype, {output_size});
    xla::ScatterDimensionNumbers scatter_dnums;
    scatter_dnums.set_index_vector_dim(1);
    scatter_dnums.add_inserted_window_dims(0);
    scatter_dnums.add_scatter_dims_to_operand_dims(0);

    if (rank == 2) {
      output_shape = xla::ShapeUtil::MakeShape(dtype, {size, output_size});
      scatter_dnums.add_inserted_window_dims(1);
      scatter_dnums.add_scatter_dims_to_operand_dims(1);
      auto i_shape =
          xla::ShapeUtil::MakeShape(input_xla_type, {input_shape.dimensions()});
      auto i = xla::Iota(ctx->builder(), i_shape, 0);
      i = xla::Reshape(
          i, {input_shape.dimensions(0) * input_shape.dimensions(1), 1});
      auto j = xla::Reshape(
          input, {input_shape.dimensions(0) * input_shape.dimensions(1), 1});
      std::vector<xla::XlaOp> iotas_to_concat;
      iotas_to_concat.push_back(i);
      iotas_to_concat.push_back(j);
      idx = xla::ConcatInDim(ctx->builder(), iotas_to_concat, 1);
      updates = xla::Broadcast(
          one, {input_shape.dimensions(0) * input_shape.dimensions(1)});
      output = xla::Broadcast(
          zero, {output_shape.dimensions(0), output_shape.dimensions(1)});
      if (has_weights && !binary_output_) {
        weights = xla::Reshape(
            weights, {input_shape.dimensions(0) * input_shape.dimensions(1)});
        updates = weights;
      }
    } else {
      input = xla::Reshape(input, {size, 1});
      idx = xla::Reshape(input, {size, 1});
      updates = xla::Broadcast(one, {size});
      output = xla::Broadcast(zero, {output_size});
      if (has_weights && !binary_output_) {
        updates = weights;
      }
    }

    xla::XlaComputation assn_computation = [&] {
      std::unique_ptr<xla::XlaBuilder> subb =
          ctx->builder()->CreateSubBuilder("scatter_bincount");
      xla::Shape param_shape = xla::ShapeUtil::MakeShape(dtype, {});
      auto p0 = xla::Parameter(subb.get(), 0, param_shape, "p0");
      auto p1 = xla::Parameter(subb.get(), 1, param_shape, "p1");
      if (!binary_output_) {
        xla::Add(p0, p1);
      }
      return subb->BuildAndNoteError();
    }();
    output = xla::Scatter(output, idx, updates, assn_computation, scatter_dnums,
                          false, false);
    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("DenseBincount").CompileTimeConstantInput("size"),
                DenseBincountOp);
REGISTER_XLA_OP(Name("Bincount").CompileTimeConstantInput("size"),
                DenseBincountOp);

}  // namespace
}  // namespace tensorflow
