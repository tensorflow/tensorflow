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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class GatherOp : public XlaOpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape params_shape = ctx->InputShape(0);
    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(params_shape),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    DataType index_type = input_type(1);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // Check that we have enough index space.
    const int64 limit = index_type == DT_INT32
                            ? std::numeric_limits<int32>::max()
                            : std::numeric_limits<int64>::max();
    OP_REQUIRES(
        ctx, params_shape.dim_size(0) <= limit,
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(index_type), " indexing: ",
                                params_shape.dim_size(0), " > ", limit));

    // The result shape is indices.shape + params.shape[1:].
    TensorShape result_shape = indices_shape;
    for (int i = 1; i < params_shape.dims(); i++) {
      result_shape.AddDim(params_shape.dim_size(i));
    }

    XlaContext& tc = XlaContext::Get(ctx);
    OP_REQUIRES(
        ctx, tc.allow_cpu_custom_calls(),
        errors::InvalidArgument("Gather op requires CustomCall on CPU"));

    xla::ComputationBuilder& b = *ctx->builder();

    // Call gather_xla_float_kernel (from gather_op_kernel_float.cc).
    // XLA passes <out> to the function, so it is not included here.
    std::vector<xla::ComputationDataHandle> args;
    args.push_back(tc.GetOrCreateRuntimeContextParameter());
    args.push_back(b.ConstantLiteral(
        *xla::LiteralUtil::CreateR0<int64>(indices_shape.num_elements())));
    args.push_back(b.ConstantLiteral(
        *xla::LiteralUtil::CreateR0<int64>(params_shape.dim_size(0))));
    args.push_back(b.ConstantLiteral(*xla::LiteralUtil::CreateR0<int64>(
        params_shape.num_elements() / params_shape.dim_size(0))));
    args.push_back(ctx->Input(0));
    args.push_back(ctx->Input(1));

    xla::Shape xla_out_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(DT_FLOAT, result_shape, &xla_out_shape));

    // Call the custom code with args:
    xla::ComputationDataHandle output;
    if (index_type == DT_INT32) {
      output = b.CustomCall("gather_float_int32_xla_impl", args, xla_out_shape);
    } else {
      output = b.CustomCall("gather_float_int64_xla_impl", args, xla_out_shape);
    }

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GatherOp);
};

REGISTER_XLA_OP(Name("Gather")
                    .TypeConstraint("Tparams", DT_FLOAT)
                    .Device(DEVICE_CPU_XLA_JIT),
                GatherOp);

}  // namespace
}  // namespace tensorflow
