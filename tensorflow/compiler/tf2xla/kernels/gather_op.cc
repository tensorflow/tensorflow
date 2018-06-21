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

Status XlaGather(const xla::XlaOp& input, const TensorShape& input_shape,
                 const xla::XlaOp& indices, const TensorShape& indices_shape,
                 int64 axis, bool indices_are_nd, DataType dtype,
                 DataType index_type, xla::XlaBuilder* builder,
                 xla::XlaOp* gather_output) {
  // There is no deep reason why we need this precondition, but this is the only
  // combination that is used and tested today.
  CHECK(!indices_are_nd || axis == 0);

  // num_index_dims is the number of components in each index in the indices
  // tensor.
  //
  // num_indices is the total number of (n dimensional or scalar) indices in the
  // indices tensor.
  //
  // If the indices are N-dimensional, then the minor dimension of indices
  // should be of size N and correspond to the N indices.
  int64 num_index_dims;
  int64 num_indices = 1;
  if (indices_are_nd) {
    CHECK_GE(indices_shape.dims(), 1);
    num_index_dims = indices_shape.dim_size(indices_shape.dims() - 1);
    for (int64 i = 0, e = indices_shape.dims() - 1; i < e; i++) {
      num_indices *= indices_shape.dim_size(i);
    }
  } else {
    num_index_dims = 1;
    for (int64 i = 0, e = indices_shape.dims(); i < e; i++) {
      num_indices *= indices_shape.dim_size(i);
    }
  }

  // Degenerate case: empty indices.
  if (num_indices == 0) {
    TensorShape input_shape_pre_axis{input_shape};
    input_shape_pre_axis.RemoveDimRange(axis, input_shape.dims());
    TensorShape input_shape_post_axis{input_shape};
    input_shape_post_axis.RemoveDimRange(0, axis + num_index_dims);

    TensorShape indices_shape_no_index_vectors{indices_shape};
    if (indices_are_nd) {
      indices_shape_no_index_vectors.RemoveLastDims(1);
    }

    TensorShape out_shape;
    out_shape.AppendShape(input_shape_pre_axis);
    out_shape.AppendShape(indices_shape_no_index_vectors);
    out_shape.AppendShape(input_shape_post_axis);

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

  // Example of a 1-D gather with axis=1, pulling two [3,1] tensors out of a
  // tensor of shape [3,3].
  //
  //  operand = s32[3,3] parameter(0)
  //  indices = s32[2] parameter(1)
  //  gather = s32[3,2] gather(operand, indices),
  //       output_window_dims={0},
  //       elided_window_dims={1},
  //       gather_dims_to_operand_dims={1},
  //       index_vector_dim=1,
  //       window_bounds={3, 1}
  //
  //
  // Example of an N-D gather pulling out slices of shape [1,1,2] out of a
  // tensor of shape [3,3,2].
  //
  //  operand = s32[3,3,2] parameter(0)
  //  indices = s32[2,2] parameter(1)
  //  gather = s32[2,2] gather(operand, indices),
  //       output_window_dims={1},
  //       elided_window_dims={0,1},
  //       gather_dims_to_operand_dims={0,1},
  //       index_vector_dim=0,
  //       window_bounds={1,1,2}

  xla::GatherDimensionNumbers dim_numbers;
  std::vector<int64> window_bounds;
  window_bounds.reserve(input_shape.dims());
  for (int64 i = 0; i < input_shape.dims(); i++) {
    int64 window_bound;
    if (axis <= i && i < (axis + num_index_dims)) {
      dim_numbers.add_elided_window_dims(i);
      window_bound = 1;
    } else {
      window_bound = input_shape.dim_size(i);
    }

    window_bounds.push_back(window_bound);

    if (i < axis) {
      dim_numbers.add_output_window_dims(i);
    } else if (i >= (axis + num_index_dims)) {
      int64 indices_rank =
          indices_are_nd ? (indices_shape.dims() - 1) : indices_shape.dims();
      dim_numbers.add_output_window_dims(i + indices_rank - num_index_dims);
    }
  }

  dim_numbers.set_index_vector_dim(indices_are_nd ? (indices_shape.dims() - 1)
                                                  : indices_shape.dims());
  for (int64 i = axis; i < axis + num_index_dims; i++) {
    dim_numbers.add_gather_dims_to_operand_dims(i);
  }

  *gather_output = builder->Gather(input, indices, dim_numbers, window_bounds);
  return Status::OK();
}

class GatherOp : public XlaOpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaBuilder* builder = context->builder();
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

    xla::XlaOp gather;
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

    xla::XlaBuilder* builder = context->builder();
    auto params = context->Input(0);
    auto indices = context->Input(1);
    xla::XlaOp gather;
    OP_REQUIRES_OK(context, XlaGather(params, params_shape, indices,
                                      indices_shape, /*axis=*/0,
                                      /*indices_are_nd=*/true, params_type,
                                      indices_type, builder, &gather));
    context->SetOutput(0, gather);
  }
};

REGISTER_XLA_OP(Name("GatherNd"), GatherNdOp);

}  // namespace tensorflow
