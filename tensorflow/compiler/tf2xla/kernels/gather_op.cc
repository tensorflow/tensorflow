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
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

xla::ComputationDataHandle XlaComputeGatherDynamicSlice(
    XlaOpKernelContext* context, const xla::ComputationDataHandle& input,
    const TensorShape& input_shape, const xla::ComputationDataHandle& indices,
    const TensorShape& indices_shape, DataType dtype,
    xla::ComputationBuilder* builder) {
  // Although the indices Tensor is flattened into rank 1 during the lookup,
  // and each scalar entry is used as an index into the first dimension of the
  // input, the output is returned with shape indices.shape + input.shape[1:]
  const int num_indices = indices_shape.num_elements();
  TensorShape input_shape_1(input_shape);
  input_shape_1.RemoveDim(0);

  // Each slice of the input tensor is [1, <input shape_1>]
  TensorShape slice_shape(input_shape);
  slice_shape.set_dim(0, 1);

  // TODO(b/37575001) The tensor in which we construct the output during
  // the loop must have rank >= 3 as a workaround for lowering issues.
  int64 extra_dims = 0;
  if (input_shape.dims() < 3) extra_dims = 3 - input_shape.dims();

  TensorShape loop_out_shape;
  for (int64 k = 0; k < extra_dims; ++k) loop_out_shape.AddDim(1);
  loop_out_shape.AddDim(num_indices);
  loop_out_shape.AppendShape(input_shape_1);

  // Slices are reshaped into the rank >= 3 shape of the loop carried output.
  TensorShape loop_out_slice_shape;
  for (int64 k = 0; k < extra_dims; ++k) loop_out_slice_shape.AddDim(1);
  loop_out_slice_shape.AddDim(1);
  loop_out_slice_shape.AppendShape(input_shape_1);

  // Finally, the loop-carried rank >= 3 output is reshaped to the op's
  // specified result shape.
  TensorShape out_shape(indices_shape);
  out_shape.AppendShape(input_shape_1);

  // Degenerate case: empty indices.
  if (num_indices == 0) {
    return builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                              out_shape.dim_sizes());
  }

  // Specify the shape of the loop-carried Tensor tuple.
  xla::PrimitiveType ptype;
  TF_CHECK_OK(DataTypeToPrimitiveType(dtype, &ptype));
  std::vector<xla::Shape> tuple_shapes(
      {// The iteration counter i is a scalar, incremented each iteration.
       xla::ShapeUtil::MakeShape(xla::S32, {}),
       // The input array has shape input_shape. Loop invariant.
       xla::ShapeUtil::MakeShape(ptype, input_shape.dim_sizes()),
       // The gather indices are reshaped to rank 1. Loop invariant.
       xla::ShapeUtil::MakeShape(xla::S32, {num_indices}),
       // The output array is rank >= 3, and is updated on each loop iteration.
       xla::ShapeUtil::MakeShape(ptype, loop_out_shape.dim_sizes())});
  xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(tuple_shapes);

  // Construct the initial values of the loop-carried Tensors.
  auto init_i = builder->ConstantR0<int32>(0);
  auto init_out =
      builder->Broadcast(builder->ConstantLiteral(xla::Literal::Zero(ptype)),
                         loop_out_shape.dim_sizes());
  // Flatten the indices into 1-D for ease of iteration.
  auto indices_1d = builder->Reshape(indices, {num_indices});
  auto init = builder->Tuple({init_i, input, indices_1d, init_out});

  // Construct the while loop condition (i < num_indices)
  xla::ComputationBuilder condb(context->builder()->client(),
                                "GatherWhileCond");
  condb.Lt(condb.GetTupleElement(
               condb.Parameter(0, tuple_shape, "GatherWhileTuple"), 0),
           condb.ConstantR0<int32>(num_indices));
  auto cond_status = condb.Build();
  auto cond = cond_status.ConsumeValueOrDie();

  // Construct the while loop body's function. The implementation of gather is:
  // for i in range(num_indices):
  //   index = dynamic-slice(indices, i)
  //   xi = dynamic-slice(input, index)
  //   output = dynamic-update-slice(output, xi, i)
  xla::ComputationBuilder bodyb(context->builder()->client(),
                                "GatherWhileBody");
  {
    // The four loop carried values.
    auto loop_tuple = bodyb.Parameter(0, tuple_shape, "GatherWhileTuple");
    auto i = bodyb.GetTupleElement(loop_tuple, 0);
    auto input = bodyb.GetTupleElement(loop_tuple, 1);
    auto indices = bodyb.GetTupleElement(loop_tuple, 2);
    auto output = bodyb.GetTupleElement(loop_tuple, 3);

    // Slice from the input array.
    auto index = bodyb.DynamicSlice(indices, bodyb.Reshape(i, {1}), {1});
    auto start_indices =
        XlaHelpers::PadWithZeros(&bodyb, index, input_shape.dims() - 1);
    auto slice_i = bodyb.Reshape(
        bodyb.DynamicSlice(input, start_indices, slice_shape.dim_sizes()),
        loop_out_slice_shape.dim_sizes());

    // Construct the index into the R3+ output Tensor 0, ..., <index>, 0, ...
    std::vector<xla::ComputationDataHandle> out_index_vals(
        loop_out_shape.dims(), bodyb.ConstantR1<int32>({0}));
    out_index_vals[extra_dims] = bodyb.Reshape(i, {1});
    auto out_index = bodyb.ConcatInDim(out_index_vals, 0);

    // Update the output Tensor
    auto updated_output = bodyb.DynamicUpdateSlice(output, slice_i, out_index);

    bodyb.Tuple({bodyb.Add(i, bodyb.ConstantR0<int32>(1)), input, indices,
                 updated_output});
  }
  auto body_status = bodyb.Build();
  auto body = body_status.ConsumeValueOrDie();

  // Construct the While loop, extract and reshape the output.
  auto gather_while = builder->While(cond, body, init);
  auto gather_output = builder->GetTupleElement(gather_while, 3);
  return builder->Reshape(gather_output, out_shape.dim_sizes());
}

namespace {

class GatherOpCustomCall : public XlaOpKernel {
 public:
  explicit GatherOpCustomCall(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape params_shape = context->InputShape(0);
    const auto params_dims = params_shape.dims();
    const TensorShape indices_shape = context->InputShape(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVectorOrHigher(params_shape),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    DataType index_type = input_type(1);
    OP_REQUIRES(context, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // GatherV2 added an axis argument. We support both Gather and GatherV2 in
    // this kernel by defaulting axis to 0 if there are 2 inputs.
    int64 axis = 0;
    if (context->num_inputs() == 3) {
      const TensorShape axis_shape = context->InputShape(2);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(axis_shape),
                  errors::InvalidArgument("axis must be scalar"));
      DataType axis_type = input_type(2);
      OP_REQUIRES(context, axis_type == DT_INT32 || axis_type == DT_INT64,
                  errors::InvalidArgument("axis must be int32 or int64"));

      xla::Literal literal;
      OP_REQUIRES_OK(context, context->ConstantInput(2, &literal));
      int64 axis_input = axis_type == DT_INT32 ? literal.Get<int32>({})
                                               : literal.Get<int64>({});
      axis = axis_input < 0 ? axis_input + params_dims : axis_input;
      OP_REQUIRES(context, 0 <= axis && axis < params_dims,
                  errors::InvalidArgument("Expected axis in the range [",
                                          -params_dims, ", ", params_dims,
                                          "), but got ", axis_input));
    }

    // Check that we have enough index space.
    const int64 limit = index_type == DT_INT32
                            ? std::numeric_limits<int32>::max()
                            : std::numeric_limits<int64>::max();
    OP_REQUIRES(context, params_shape.dim_size(axis) <= limit,
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

    XlaContext& tc = XlaContext::Get(context);
    OP_REQUIRES(
        context, tc.allow_cpu_custom_calls(),
        errors::InvalidArgument("Gather op requires CustomCall on CPU"));

    xla::ComputationBuilder& b = *context->builder();

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
    args.push_back(context->Input(0));
    args.push_back(context->Input(1));

    xla::Shape xla_out_shape;
    OP_REQUIRES_OK(
        context, TensorShapeToXLAShape(DT_FLOAT, result_shape, &xla_out_shape));

    // Call the custom code with args:
    xla::ComputationDataHandle output;
    if (index_type == DT_INT32) {
      output = b.CustomCall("gather_float_int32_xla_impl", args, xla_out_shape);
    } else {
      output = b.CustomCall("gather_float_int64_xla_impl", args, xla_out_shape);
    }

    context->SetOutput(0, output);
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
      context, input, input_shape, indices, indices_shape, DT_FLOAT, builder);
  context->SetOutput(0, gather);
}

REGISTER_XLA_OP(Name("Gather")
                    .TypeConstraint("Tparams", DT_FLOAT)
                    .Device(DEVICE_GPU_XLA_JIT),
                GatherOpDynamicSlice);

}  // namespace tensorflow
