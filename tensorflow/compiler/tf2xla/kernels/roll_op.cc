/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class RollOp : public XlaOpKernel {
 public:
  explicit RollOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    xla::XlaOp shift = ctx->Input(1);
    const TensorShape shift_shape = ctx->InputShape(1);
    const TensorShape axis_shape = ctx->InputShape(2);

    int64_t input_dims = input_shape.dims();
    OP_REQUIRES(ctx, input_dims >= 1,
                errors::InvalidArgument("input must be 1-D or higher"));
    OP_REQUIRES(ctx, shift_shape.dims() <= 1,
                errors::InvalidArgument(
                    "shift must be a scalar or a 1-D vector. Found: ",
                    shift_shape.DebugString()));
    OP_REQUIRES(ctx, axis_shape.dims() <= 1,
                errors::InvalidArgument(
                    "axis must be a scalar or a 1-D vector. Found: ",
                    shift_shape.DebugString()));
    OP_REQUIRES(
        ctx, shift_shape == axis_shape,
        errors::InvalidArgument("shift and axis must have the same size"));

    xla::Literal axis;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &axis));

    xla::XlaOp output = ctx->Input(0);
    xla::PrimitiveType shift_type = ctx->input_xla_type(1);
    int64_t num_axes = axis_shape.dims() == 0 ? 1 : axis_shape.dim_size(0);
    for (int64_t i = 0; i != num_axes; ++i) {
      int64_t cur_axis = axis_shape.dims() == 0 ? *axis.GetIntegralAsS64({})
                                                : *axis.GetIntegralAsS64({i});
      OP_REQUIRES(ctx, cur_axis >= -input_dims && cur_axis < input_dims,
                  errors::InvalidArgument(
                      absl::StrCat("axis ", cur_axis, " is out of range [-",
                                   input_dims, ", ", input_dims, ").")));
      if (cur_axis < 0) {
        cur_axis += input_dims;
      }

      xla::XlaOp offset =
          shift_shape.dims() == 0
              ? shift
              : xla::Reshape(xla::SliceInDim(shift, /*start_index=*/i,
                                             /*limit_index=*/i + 1,
                                             /*stride=*/1, /*dimno=*/0),
                             {});
      xla::XlaOp axis_size = xla::ConstantR0WithType(
          ctx->builder(), shift_type, input_shape.dim_size(cur_axis));
      // Adjust large offsets into [0, axis_size). This also makes negative
      // offsets positive.
      offset = ((offset % axis_size) + axis_size) % axis_size;

      // Stack two copies of the dimension, then slice from the calculated
      // offset.
      xla::XlaOp concat =
          xla::ConcatInDim(ctx->builder(), {output, output}, cur_axis);
      std::vector<xla::XlaOp> start_indices(
          input_shape.dims(), xla::Zero(ctx->builder(), shift_type));
      start_indices[cur_axis] = axis_size - offset;
      output =
          xla::DynamicSlice(concat, start_indices, input_shape.dim_sizes());
    }
    ctx->SetOutput(0, output);
  }

 private:
  RollOp(const RollOp&) = delete;
  void operator=(const RollOp&) = delete;
};

REGISTER_XLA_OP(Name("Roll").CompileTimeConstantInput("axis"), RollOp);

}  // namespace
}  // namespace tensorflow
