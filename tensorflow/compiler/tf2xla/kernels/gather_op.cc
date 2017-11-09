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
    const TensorShape& indices_shape, int64 axis, DataType dtype,
    DataType index_type, xla::ComputationBuilder* builder) {
  // Although the indices Tensor is flattened into rank 1 during the lookup,
  // and each scalar entry is used as an index into the first dimension of the
  // input, the output is returned with shape:
  // input.shape[:axis] + indices.shape + input.shape[axis+1:]
  const int num_indices = indices_shape.num_elements();
  TensorShape input_shape_pre_axis(input_shape);
  input_shape_pre_axis.RemoveDimRange(axis, input_shape.dims());
  TensorShape input_shape_post_axis(input_shape);
  input_shape_post_axis.RemoveDimRange(0, axis + 1);

  // Each slice of the input tensor has shape:
  // [<input_shape_pre_axis>, 1, <input shape_post_axis>]
  TensorShape slice_shape(input_shape);
  slice_shape.set_dim(axis, 1);

  // TODO(b/37575001) The tensor in which we construct the output during
  // the loop must have rank >= 3 as a workaround for lowering issues.
  int64 extra_dims = 0;
  if (input_shape.dims() < 3) extra_dims = 3 - input_shape.dims();

  TensorShape loop_out_shape;
  for (int64 k = 0; k < extra_dims; ++k) loop_out_shape.AddDim(1);
  loop_out_shape.AppendShape(input_shape_pre_axis);
  loop_out_shape.AddDim(num_indices);
  loop_out_shape.AppendShape(input_shape_post_axis);

  // Slices are reshaped into the rank >= 3 shape of the loop carried output.
  TensorShape loop_out_slice_shape;
  for (int64 k = 0; k < extra_dims; ++k) loop_out_slice_shape.AddDim(1);
  loop_out_slice_shape.AppendShape(input_shape_pre_axis);
  loop_out_slice_shape.AddDim(1);
  loop_out_slice_shape.AppendShape(input_shape_post_axis);

  // Finally, the loop-carried rank >= 3 output is reshaped to the op's
  // specified result shape.
  TensorShape out_shape;
  out_shape.AppendShape(input_shape_pre_axis);
  out_shape.AppendShape(indices_shape);
  out_shape.AppendShape(input_shape_post_axis);

  // Degenerate case: empty indices.
  if (num_indices == 0) {
    return builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                              out_shape.dim_sizes());
  }

  // Specify the shape of the loop-carried Tensor tuple.
  xla::PrimitiveType ptype;
  TF_CHECK_OK(DataTypeToPrimitiveType(dtype, &ptype));
  xla::PrimitiveType idxtype;
  TF_CHECK_OK(DataTypeToPrimitiveType(index_type, &idxtype));
  std::vector<xla::Shape> tuple_shapes(
      {// The iteration counter i is a scalar, incremented each iteration.
       xla::ShapeUtil::MakeShape(idxtype, {}),
       // The input array has shape input_shape. Loop invariant.
       xla::ShapeUtil::MakeShape(ptype, input_shape.dim_sizes()),
       // The gather indices are reshaped to rank 1. Loop invariant.
       xla::ShapeUtil::MakeShape(idxtype, {num_indices}),
       // The output array is rank >= 3, and is updated on each loop iteration.
       xla::ShapeUtil::MakeShape(ptype, loop_out_shape.dim_sizes())});
  xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(tuple_shapes);

  // Construct the initial values of the loop-carried Tensors.
  auto init_i = XlaHelpers::Zero(builder, index_type);
  auto init_out = builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                                     loop_out_shape.dim_sizes());
  // Flatten the indices into 1-D for ease of iteration.
  auto indices_1d = builder->Reshape(indices, {num_indices});
  auto init = builder->Tuple({init_i, input, indices_1d, init_out});

  // Construct the while loop condition (i < num_indices)
  xla::ComputationBuilder condb(context->builder()->client(),
                                "GatherWhileCond");
  condb.Lt(condb.GetTupleElement(
               condb.Parameter(0, tuple_shape, "GatherWhileTuple"), 0),
           XlaHelpers::IntegerLiteral(&condb, index_type, num_indices));
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
    auto start_indices = bodyb.Pad(
        bodyb.Reshape(index, {1}), XlaHelpers::Zero(&bodyb, index_type),
        xla::MakeEdgePaddingConfig(
            {{input_shape_pre_axis.dims(), input_shape_post_axis.dims()}}));
    auto slice_i = bodyb.Reshape(
        bodyb.DynamicSlice(input, start_indices, slice_shape.dim_sizes()),
        loop_out_slice_shape.dim_sizes());

    // Construct the index into the R3+ output Tensor 0, ..., <index>, 0, ...
    std::vector<xla::ComputationDataHandle> out_index_vals(
        loop_out_shape.dims(),
        bodyb.Reshape(XlaHelpers::Zero(&bodyb, index_type), {1}));
    out_index_vals[input_shape_pre_axis.dims() + extra_dims] =
        bodyb.Reshape(i, {1});
    auto out_index = bodyb.ConcatInDim(out_index_vals, 0);

    // Update the output Tensor
    auto updated_output = bodyb.DynamicUpdateSlice(output, slice_i, out_index);

    bodyb.Tuple({bodyb.Add(i, XlaHelpers::One(&bodyb, index_type)), input,
                 indices, updated_output});
  }
  auto body_status = bodyb.Build();
  auto body = body_status.ConsumeValueOrDie();

  // Construct the While loop, extract and reshape the output.
  auto gather_while = builder->While(cond, body, init);
  auto gather_output = builder->GetTupleElement(gather_while, 3);
  return builder->Reshape(gather_output, out_shape.dim_sizes());
}

GatherOpDynamicSlice::GatherOpDynamicSlice(OpKernelConstruction* context)
    : XlaOpKernel(context) {}

void GatherOpDynamicSlice::Compile(XlaOpKernelContext* context) {
  xla::ComputationBuilder* builder = context->builder();
  auto input = context->Input(0);
  auto input_shape = context->InputShape(0);
  auto indices = context->Input(1);
  auto indices_shape = context->InputShape(1);
  int64 axis = 0;
  if (context->num_inputs() == 3) {
    const TensorShape axis_shape = context->InputShape(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(axis_shape),
                errors::InvalidArgument("axis must be scalar"));
    DataType axis_type = input_type(2);
    OP_REQUIRES(context, axis_type == DT_INT32 || axis_type == DT_INT64,
                errors::InvalidArgument("axis must be int32 or int64"));

    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(2, &axis));
    const auto params_dims = input_shape.dims();
    if (axis < 0) {
      axis += params_dims;
    }
    OP_REQUIRES(
        context, 0 <= axis && axis < params_dims,
        errors::InvalidArgument("Expected axis in the range [", -params_dims,
                                ", ", params_dims, "), but got ", axis));
  }

  DataType index_type = input_type(1);
  OP_REQUIRES(context, index_type == DT_INT32 || index_type == DT_INT64,
              errors::InvalidArgument("indices must be int32 or int64"));

  xla::ComputationDataHandle gather = XlaComputeGatherDynamicSlice(
      context, input, input_shape, indices, indices_shape, axis, input_type(0),
      index_type, builder);
  context->SetOutput(0, gather);
}

REGISTER_XLA_OP(Name("Gather"), GatherOpDynamicSlice);
REGISTER_XLA_OP(Name("GatherV2"), GatherOpDynamicSlice);

}  // namespace tensorflow
