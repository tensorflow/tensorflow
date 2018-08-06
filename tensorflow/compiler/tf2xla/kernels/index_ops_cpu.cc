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
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {
namespace {

// The logic below uses a custom-call to implement argmax.
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
    xla::Literal literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &literal));
    const int32 dim = literal.Get<int32>({});
    OP_REQUIRES(ctx, dim >= 0, errors::InvalidArgument("dim must be >= 0"));
    OP_REQUIRES(
        ctx, dim < input_shape.dims(),
        errors::InvalidArgument("dim must be < input rank (",
                                input_shape.dims(), "), but got: ", dim));
    const int64 dim_size = input_shape.dim_size(dim);
    OP_REQUIRES(ctx, dim_size > 0,
                errors::InvalidArgument(
                    "Reduction axis ", dim,
                    " is empty in shape: ", input_shape.DebugString()));

    // The output shape is the input shape contracted along dim.
    TensorShape output_shape;
    for (int d = 0; d < input_shape.dims() - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < dim) ? d : d + 1));
    }

    // For now we use a custom-call, only for the 1d and 2d cases.
    OP_REQUIRES(ctx, XlaContext::Get(ctx).allow_cpu_custom_calls(),
                errors::InvalidArgument(
                    "ArgMax implementation requires a CustomCall on CPU"));
    xla::XlaBuilder& b = *ctx->builder();

    // XLA passes <out> to the function, so it is not included here.
    std::vector<xla::XlaOp> args;
    args.push_back(ctx->Input(0));
    args.push_back(xla::ConstantLiteral(
        &b, *xla::LiteralUtil::CreateR1<int64>(input_shape.dim_sizes())));
    if (input_shape.dims() > 1) {
      // Don't bother passing the output shape and dim for the 1d case, since
      // the shape is always a scalar and the dim is always 0.
      args.push_back(xla::ConstantLiteral(
          &b, *xla::LiteralUtil::CreateR1<int64>(output_shape.dim_sizes())));
      args.push_back(
          xla::ConstantLiteral(&b, *xla::LiteralUtil::CreateR0<int32>(dim)));
    }

    xla::Shape xla_shape =
        xla::ShapeUtil::MakeShape(xla::S64, output_shape.dim_sizes());

    // Tell XLA to call the custom code, defined in
    // index_ops_kernel_argmax_float_1d.cc.
    xla::XlaOp output;
    switch (input_shape.dims()) {
      case 1:
        output =
            xla::CustomCall(&b, "argmax_float_1d_xla_impl", args, xla_shape);
        break;
      case 2:
        output =
            xla::CustomCall(&b, "argmax_float_2d_xla_impl", args, xla_shape);
        break;
      default:
        OP_REQUIRES(ctx, false,
                    errors::Unimplemented(
                        "Argmax is only implemented for 1d and 2d tensors"
                        ", but got shape: ",
                        input_shape.DebugString()));
    }
    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArgMaxCustomCallOp);
};

REGISTER_XLA_OP(Name("ArgMax")
                    .TypeConstraint("T", DT_FLOAT)
                    .Device(DEVICE_CPU_XLA_JIT)
                    .CompileTimeConstInput("dimension"),
                ArgMaxCustomCallOp);

}  // namespace
}  // namespace tensorflow
