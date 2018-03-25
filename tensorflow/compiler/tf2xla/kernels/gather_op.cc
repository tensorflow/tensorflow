/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/while_loop.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

Status XlaGather(const xla::ComputationDataHandle& input,
                 const TensorShape& input_shape,
                 const xla::ComputationDataHandle& indices,
                 TensorShape indices_shape, int64 axis, bool indices_are_nd,
                 DataType dtype, DataType index_type,
                 xla::ComputationBuilder* builder,
                 xla::ComputationDataHandle* gather_output) {
  // If the indices are N-dimensional, then the minor dimension of indices
  // should be of size N and correspond to the N indices.
  int64 num_index_dims = 1;
  if (indices_are_nd) {
    CHECK_GE(indices_shape.dims(), 1);
    num_index_dims = indices_shape.dim_size(indices_shape.dims() - 1);
    indices_shape.RemoveLastDims(1);
  }

  // Although the indices Tensor is flattened into rank 1 during the lookup,
  // and each scalar entry is used as an index into the first dimension of the
  // input, the output is returned with shape:
  // input.shape[:axis] + indices.shape + input.shape[axis+1:]

  const int64 num_indices = indices_shape.num_elements();
  TensorShape input_shape_pre_axis(input_shape);
  input_shape_pre_axis.RemoveDimRange(axis, input_shape.dims());
  TensorShape input_shape_post_axis(input_shape);
  input_shape_post_axis.RemoveDimRange(0, axis + num_index_dims);
  // Each slice of the input tensor has shape:
  // [<input_shape_pre_axis>, 1, ..., 1, <input shape_post_axis>]
  TensorShape slice_shape(input_shape);
  for (int64 i = 0; i < num_index_dims; ++i) {
    slice_shape.set_dim(axis + i, 1);
  }

  TensorShape loop_out_shape;
  loop_out_shape.AppendShape(input_shape_pre_axis);
  loop_out_shape.AddDim(num_indices);
  loop_out_shape.AppendShape(input_shape_post_axis);
  TensorShape loop_out_slice_shape;
  loop_out_slice_shape.AppendShape(input_shape_pre_axis);
  loop_out_slice_shape.AddDim(1);
  loop_out_slice_shape.AppendShape(input_shape_post_axis);

  TensorShape out_shape;
  out_shape.AppendShape(input_shape_pre_axis);
  out_shape.AppendShape(indices_shape);
  out_shape.AppendShape(input_shape_post_axis);

  // Degenerate case: empty indices.
  if (num_indices == 0) {
    *gather_output = builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                                        out_shape.dim_sizes());
    return Status::OK();
  }

  for (int64 i = 0; i < num_index_dims; ++i) {
    if (input_shape.dim_size(axis + i) == 0) {
      return errors::InvalidArgument("Gather dimension ", axis + i,
                                     " is of size zero in tensor with shape ",
                                     input_shape.DebugString());
    }
  }

  // Flatten the major dimensions of indices into a single dimension for ease of
  // iteration. If there is an axis dimension, we must leave it alone.
  std::vector<int64> flat_indices_shape = {num_indices};
  if (indices_are_nd) {
    flat_indices_shape.push_back(num_index_dims);
  }

  // Specify the shape of the loop-carried Tensor tuple.

  // Construct the initial values of the loop-carried Tensors.
  auto flat_indices = builder->Reshape(indices, flat_indices_shape);
  auto init_out = builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                                     loop_out_shape.dim_sizes());
  auto init = {input, flat_indices, init_out};

  // Construct the while loop body's function. The implementation of gather is:
  // for i in range(num_indices):
  //   index = dynamic-slice(indices, i)
  //   xi = dynamic-slice(input, index)
  //   output = dynamic-update-slice(output, xi, i)
  auto body_fn = [&](xla::ComputationDataHandle i,
                     gtl::ArraySlice<xla::ComputationDataHandle> loop_vars,
                     xla::ComputationBuilder* bodyb) {
    auto input = loop_vars[0];
    auto indices = loop_vars[1];
    auto output = loop_vars[2];

    auto zero_index = XlaHelpers::Zero(bodyb, index_type);

    // Slice the i-th index from the indices array.
    xla::ComputationDataHandle index;
    auto indices_offset = bodyb->Reshape(i, {1});
    if (indices_are_nd) {
      // Slice out the entire nd index, if applicable.
      indices_offset = bodyb->Pad(indices_offset, zero_index,
                                  xla::MakeEdgePaddingConfig({{0, 1}}));
      index = bodyb->DynamicSlice(indices, indices_offset, {1, num_index_dims});
      index = bodyb->Collapse(index, {0, 1});
    } else {
      index = bodyb->DynamicSlice(indices, indices_offset, {1});
    }

    // Slice the corresponding data from the input array.
    auto start_indices = bodyb->Pad(
        index, zero_index,
        xla::MakeEdgePaddingConfig(
            {{input_shape_pre_axis.dims(), input_shape_post_axis.dims()}}));
    auto slice_i = bodyb->Reshape(
        bodyb->DynamicSlice(input, start_indices, slice_shape.dim_sizes()),
        loop_out_slice_shape.dim_sizes());

    // Construct the index into the output Tensor 0, ..., <index>, 0, ...
    std::vector<xla::ComputationDataHandle> out_index_vals(
        loop_out_shape.dims(), bodyb->Reshape(zero_index, {1}));
    out_index_vals[input_shape_pre_axis.dims()] = bodyb->Reshape(i, {1});
    auto out_index = bodyb->ConcatInDim(out_index_vals, 0);

    // Update the output Tensor
    auto updated_output = bodyb->DynamicUpdateSlice(output, slice_i, out_index);

    return std::vector<xla::ComputationDataHandle>{input, indices,
                                                   updated_output};
  };

  // Construct the While loop, extract and reshape the output.
  xla::PrimitiveType ptype;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(index_type, &ptype));
  TF_ASSIGN_OR_RETURN(auto outputs, XlaForEachIndex(num_indices, ptype, body_fn,
                                                    init, "gather", builder));
  *gather_output = builder->Reshape(outputs[2], out_shape.dim_sizes());
  return Status::OK();
}

class GatherOp : public XlaOpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
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

    xla::ComputationDataHandle gather;
    OP_REQUIRES_OK(
        context, XlaGather(input, input_shape, indices, indices_shape, axis,
                           /*indices_are_nd=*/false, input_type(0), index_type,
                           builder, &gather));
    context->SetOutput(0, gather);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GatherOp);
};

REGISTER_XLA_OP(Name("Gather"), GatherOp);
REGISTER_XLA_OP(Name("GatherV2").CompileTimeConstInput("axis"), GatherOp);

class GatherNdOp : public XlaOpKernel {
 public:
  explicit GatherNdOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    DataType params_type = context->input_type(0);
    DataType indices_type = context->input_type(1);

    TensorShape params_shape = context->InputShape(0);
    TensorShape indices_shape = context->InputShape(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(params_shape),
                errors::InvalidArgument("params must be at least a vector"));
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(indices_shape),
                errors::InvalidArgument("indices must be at least a vector"));
    const int64 num_index_dims =
        indices_shape.dim_size(indices_shape.dims() - 1);
    OP_REQUIRES(
        context, num_index_dims <= params_shape.dims(),
        errors::InvalidArgument(
            "index innermost dimension length must be <= params rank; saw: ",
            indices_shape.dim_size(indices_shape.dims() - 1), " vs. ",
            params_shape.dims()));

    xla::ComputationBuilder* builder = context->builder();
    auto params = context->Input(0);
    auto indices = context->Input(1);
    xla::ComputationDataHandle gather;
    OP_REQUIRES_OK(context, XlaGather(params, params_shape, indices,
                                      indices_shape, /*axis=*/0,
                                      /*indices_are_nd=*/true, params_type,
                                      indices_type, builder, &gather));
    context->SetOutput(0, gather);
  }
};

REGISTER_XLA_OP(Name("GatherNd"), GatherNdOp);

}  // namespace tensorflow
