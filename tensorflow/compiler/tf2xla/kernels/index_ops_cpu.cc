/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Native XLA implementations of indexing ops.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {
namespace {

// The logic below uses a custom-call to implement argmax when possible. When
// custom-call is not allowed or input shapes are not supported, this kernel
// falls back to using XLA HLO native ArgMax.
//
// Also see b/29507024 for first-class XLA support for indexing ops.
class ArgMaxCustomCallOp : public XlaOpKernel {
 public:
  explicit ArgMaxCustomCallOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape dimension_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dimension_shape),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension_shape.DebugString()));

    // We require that the dimension argument is a constant, since it lets us
    // dispatch to a specialized custom-call function without any run-time
    // overhead, when compiling ahead-of-time.
    int64 dim;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &dim));

    const int input_dims = input_shape.dims();
    const int axis = dim < 0 ? dim + input_dims : dim;
    OP_REQUIRES(ctx, axis >= 0 && axis < input_dims,
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));

    const int64 axis_size = input_shape.dim_size(axis);
    OP_REQUIRES(ctx, axis_size > 0,
                errors::InvalidArgument(
                    "Reduction axis ", dim,
                    " is empty in shape: ", input_shape.DebugString()));

    const DataType dtype = output_type(0);
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &output_type));

    // Fall back to XLA ArgMax HLO when CustomCall is not allowed or when input
    // shape isn't supported.
    if (!ctx->compiler()->options().allow_cpu_custom_calls ||
        (input_dims != 1 && input_dims != 2)) {
      xla::XlaOp output = xla::ArgMax(ctx->Input(0), output_type, axis);
      ctx->SetOutput(0, output);
      return;
    }

    xla::XlaOp output;
    // The output shape is the input shape contracted along axis.
    TensorShape output_shape;
    for (int d = 0; d < input_shape.dims() - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
    }

    xla::XlaBuilder& b = *ctx->builder();

    // XLA passes <out> to the function, so it is not included here.
    std::vector<xla::XlaOp> args;
    args.push_back(ctx->Input(0));
    args.push_back(xla::ConstantLiteral(
        &b, xla::LiteralUtil::CreateR1<int64>(input_shape.dim_sizes())));
    if (input_shape.dims() > 1) {
      // Don't bother passing the output shape and dim for the 1d case, since
      // the shape is always a scalar and the dim is always 0.
      args.push_back(xla::ConstantLiteral(
          &b, xla::LiteralUtil::CreateR1<int64>(output_shape.dim_sizes())));
      args.push_back(
          xla::ConstantLiteral(&b, xla::LiteralUtil::CreateR0<int32>(axis)));
    }

    // The argmax function expects row-major layout.
    xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
        xla::S64, output_shape.dim_sizes());
    std::vector<xla::Shape> arg_shapes;
    for (const xla::XlaOp& arg : args) {
      auto shape_status = b.GetShape(arg);
      OP_REQUIRES_OK(ctx, shape_status.status());
      xla::Shape arg_shape = shape_status.ConsumeValueOrDie();
      *arg_shape.mutable_layout() =
          xla::LayoutUtil::MakeDescendingLayout(arg_shape.rank());
      arg_shapes.push_back(std::move(arg_shape));
    }

    // Tell XLA to call the custom code, defined in
    // index_ops_kernel_argmax_float_{1, 2}d.cc.
    if (input_dims == 1) {
      output = xla::CustomCallWithLayout(&b, "argmax_float_1d_xla_impl", args,
                                         xla_shape, arg_shapes);
    } else {
      output = xla::CustomCallWithLayout(&b, "argmax_float_2d_xla_impl", args,
                                         xla_shape, arg_shapes);
    }
    output = xla::ConvertElementType(output, output_type);
    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArgMaxCustomCallOp);
};

REGISTER_XLA_OP(Name("ArgMax")
                    .TypeConstraint("T", DT_FLOAT)
                    .Device(DEVICE_CPU_XLA_JIT)
                    .CompileTimeConstantInput("dimension"),
                ArgMaxCustomCallOp);

}  // namespace
}  // namespace tensorflow
