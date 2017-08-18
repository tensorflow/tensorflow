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

#include "tensorflow/compiler/tf2xla/kernels/gather_op.h"
#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {

xla::ComputationDataHandle XlaComputeGatherDynamicSlice(
    const xla::ComputationDataHandle& input, const TensorShape& input_shape,
    const xla::ComputationDataHandle& indices, const TensorShape& indices_shape,
    DataType dtype, xla::ComputationBuilder* builder) {
  const int num_indices = indices_shape.num_elements();

  // Flatten the indices into 1-D.
  auto indices_1d = builder->Reshape(indices, {num_indices});

  // Compute the slice for each of these indices separately.
  std::vector<xla::ComputationDataHandle> slices(num_indices);
  for (int i = 0; i < num_indices; ++i) {
    auto index = builder->Slice(indices_1d, {i}, {i + 1}, {1});

    auto start_indices =
        XlaHelpers::PadWithZeros(builder, index, input_shape.dims() - 1);

    auto slice_shape = input_shape.dim_sizes();
    slice_shape[0] = 1;

    slices[i] = builder->DynamicSlice(input, start_indices, slice_shape);
  }

  xla::ComputationDataHandle gather;
  if (slices.empty()) {
    auto shape = input_shape.dim_sizes();
    shape[0] = 0;
    gather = builder->Broadcast(XlaHelpers::Zero(builder, dtype), shape);
  } else {
    gather = builder->ConcatInDim(slices, 0);
  }

  // Compute the shape of the result tensor, which is:
  //    indices.shape + resource.shape[1:]
  TensorShape gather_shape = indices_shape;
  gather_shape.AppendShape(input_shape);
  gather_shape.RemoveDim(indices_shape.dims());

  // Reshape the concatenated slices into the shape expected of the result
  // tensor.
  return builder->Reshape(gather, gather_shape.dim_sizes());
}

namespace {

class GatherOpCustomCall : public XlaOpKernel {
 public:
  explicit GatherOpCustomCall(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape params_shape = ctx->InputShape(0);
    const auto params_dims = params_shape.dims();
    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(params_shape),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    DataType index_type = input_type(1);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // GatherV2 added an axis argument. We support both Gather and GatherV2 in
    // this kernel by defaulting axis to 0 if there are 2 inputs.
    int64 axis = 0;
    if (ctx->num_inputs() == 3) {
      const TensorShape axis_shape = ctx->InputShape(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_shape),
                  errors::InvalidArgument("axis must be scalar"));
      DataType axis_type = input_type(2);
      OP_REQUIRES(ctx, axis_type == DT_INT32 || axis_type == DT_INT64,
                  errors::InvalidArgument("axis must be int32 or int64"));

      xla::Literal literal;
      OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &literal));
      int64 axis_input = axis_type == DT_INT32 ? literal.Get<int32>({})
                                               : literal.Get<int64>({});
      axis = axis_input < 0 ? axis_input + params_dims : axis_input;
      OP_REQUIRES(ctx, 0 <= axis && axis < params_dims,
                  errors::InvalidArgument("Expected axis in the range [",
                                          -params_dims, ", ", params_dims,
                                          "), but got ", axis_input));
    }

    // Check that we have enough index space.
    const int64 limit = index_type == DT_INT32
                            ? std::numeric_limits<int32>::max()
                            : std::numeric_limits<int64>::max();
    OP_REQUIRES(ctx, params_shape.dim_size(axis) <= limit,
                errors::InvalidArgument(
                    "params.shape[", axis, "] too large for ",
                    DataTypeString(index_type),
                    " indexing: ", params_shape.dim_size(axis), " > ", limit));

    // The result shape is params.shape[0:axis] + indices.shape +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64 outer_size = 1;
    int64 inner_size = 1;
    for (int i = 0; i < axis; i++) {
      result_shape.AddDim(params_shape.dim_size(i));
      outer_size *= params_shape.dim_size(i);
    }
    result_shape.AppendShape(indices_shape);
    for (int i = axis + 1; i < params_dims; i++) {
      result_shape.AddDim(params_shape.dim_size(i));
      inner_size *= params_shape.dim_size(i);
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
        *xla::Literal::CreateR0<int64>(indices_shape.num_elements())));
    args.push_back(
        b.ConstantLiteral(*xla::Literal::CreateR0<int64>(outer_size)));
    args.push_back(b.ConstantLiteral(
        *xla::Literal::CreateR0<int64>(params_shape.dim_size(axis))));
    args.push_back(
        b.ConstantLiteral(*xla::Literal::CreateR0<int64>(inner_size)));
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
  TF_DISALLOW_COPY_AND_ASSIGN(GatherOpCustomCall);
};

REGISTER_XLA_OP(Name("Gather")
                    .TypeConstraint("Tparams", DT_FLOAT)
                    .Device(DEVICE_CPU_XLA_JIT),
                GatherOpCustomCall);
REGISTER_XLA_OP(Name("GatherV2")
                    .TypeConstraint("Tparams", DT_FLOAT)
                    .Device(DEVICE_CPU_XLA_JIT),
                GatherOpCustomCall);

}  // namespace

GatherOpDynamicSlice::GatherOpDynamicSlice(OpKernelConstruction* context)
    : XlaOpKernel(context) {}

void GatherOpDynamicSlice::Compile(XlaOpKernelContext* context) {
  xla::ComputationBuilder* builder = context->builder();
  auto input = context->Input(0);
  auto input_shape = context->InputShape(0);
  auto indices = context->Input(1);
  auto indices_shape = context->InputShape(1);
  xla::ComputationDataHandle gather = XlaComputeGatherDynamicSlice(
      input, input_shape, indices, indices_shape, DT_FLOAT, builder);
  context->SetOutput(0, gather);
}

REGISTER_XLA_OP(Name("Gather")
                    .TypeConstraint("Tparams", DT_FLOAT)
                    .Device(DEVICE_GPU_XLA_JIT),
                GatherOpDynamicSlice);

}  // namespace tensorflow
