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

#include <algorithm>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status XlaGather(const xla::XlaOp& input, const TensorShape& input_shape,
                 const xla::XlaOp& indices, const TensorShape& indices_shape,
                 int64_t axis, bool indices_are_nd, DataType dtype,
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
  int64_t num_index_dims;
  int64_t num_indices = 1;
  if (indices_are_nd) {
    CHECK_GE(indices_shape.dims(), 1);
    num_index_dims = indices_shape.dim_size(indices_shape.dims() - 1);
    for (int64_t i = 0, e = indices_shape.dims() - 1; i < e; i++) {
      num_indices *= indices_shape.dim_size(i);
    }
  } else {
    num_index_dims = 1;
    for (int64_t i = 0, e = indices_shape.dims(); i < e; i++) {
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

    *gather_output =
        xla::Broadcast(XlaHelpers::Zero(builder, dtype), out_shape.dim_sizes());
    return OkStatus();
  }

  for (int64_t i = 0; i < num_index_dims; ++i) {
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
  //       offset_dims={0},
  //       collapsed_slice_dims={1},
  //       start_index_map={1},
  //       index_vector_dim=1,
  //       slice_sizes={3, 1}
  //
  //
  // Example of an N-D gather pulling out slices of shape [1,1,2] out of a
  // tensor of shape [3,3,2].
  //
  //  operand = s32[3,3,2] parameter(0)
  //  indices = s32[2,2] parameter(1)
  //  gather = s32[2,2] gather(operand, indices),
  //       offset_dims={1},
  //       collapsed_slice_dims={0,1},
  //       start_index_map={0,1},
  //       index_vector_dim=0,
  //       slice_sizes={1,1,2}

  xla::GatherDimensionNumbers dim_numbers;
  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(input_shape.dims());
  for (int64_t i = 0; i < input_shape.dims(); i++) {
    int64_t window_bound;
    if (axis <= i && i < (axis + num_index_dims)) {
      dim_numbers.add_collapsed_slice_dims(i);
      window_bound = 1;
    } else {
      window_bound = input_shape.dim_size(i);
    }

    slice_sizes.push_back(window_bound);

    if (i < axis) {
      dim_numbers.add_offset_dims(i);
    } else if (i >= (axis + num_index_dims)) {
      int64_t indices_rank =
          indices_are_nd ? (indices_shape.dims() - 1) : indices_shape.dims();
      dim_numbers.add_offset_dims(i + indices_rank - num_index_dims);
    }
  }

  dim_numbers.set_index_vector_dim(indices_are_nd ? (indices_shape.dims() - 1)
                                                  : indices_shape.dims());
  for (int64_t i = axis; i < axis + num_index_dims; i++) {
    dim_numbers.add_start_index_map(i);
  }

  *gather_output = xla::Gather(input, indices, dim_numbers, slice_sizes);
  return OkStatus();
}

Status XlaGatherWithBatchDimsOpImpl(XlaOpKernelContext* context,
                                    const xla::XlaOp input,
                                    const TensorShape& input_shape,
                                    int batch_dims, xla::XlaOp* gather_output) {
  auto indices = context->Input(1);
  auto indices_shape = context->InputShape(1);

  absl::optional<int64_t> axis;
  if (context->num_inputs() == 3) {
    const TensorShape axis_shape = context->InputShape(2);
    if (!TensorShapeUtils::IsScalar(axis_shape)) {
      return errors::InvalidArgument("axis must be scalar");
    }
    DataType axis_type = context->input_type(2);
    if (axis_type != DT_INT32 && axis_type != DT_INT64) {
      return errors::InvalidArgument("axis must be int32 or int64");
    }

    int64_t axis_input;
    TF_RETURN_IF_ERROR(context->ConstantInputAsIntScalar(2, &axis_input));

    const auto params_dims = input_shape.dims();
    if (-params_dims > axis_input || axis_input >= params_dims) {
      // Check that params has rank of at least axis + 1.
      const auto min_params_rank =
          axis_input < 0 ? -axis_input : axis_input + 1;
      return errors::InvalidArgument("Shape must be at least rank ",
                                     min_params_rank, " but is rank ",
                                     params_dims);
    }
    if (axis_input < 0) {
      axis_input += params_dims;
    }
    axis = axis_input;
  }

  if (batch_dims != 0) {
    if (batch_dims < 0) {
      batch_dims = indices_shape.dims() + batch_dims;
    }

    axis = axis.value_or(batch_dims);

    if (batch_dims < -indices_shape.dims() ||
        batch_dims > indices_shape.dims()) {
      return errors::InvalidArgument(
          "Expected batch_dims in the range [", -indices_shape.dims(), ", ",
          indices_shape.dims(), "], but got ", batch_dims);
    }

    if (batch_dims >= input_shape.dims()) {
      return errors::InvalidArgument("batch_dims (", batch_dims,
                                     ") must be less than rank(input) (",
                                     input_shape.dims(), ").");
    }

    if (*axis < batch_dims) {
      return errors::InvalidArgument("batch_dims (", batch_dims,
                                     ") must be less than or equal to ",
                                     "axis (", *axis, ").");
    }
  }

  axis = axis.value_or(0);
  DataType index_type = context->input_type(1);
  if (index_type != DT_INT32 && index_type != DT_INT64) {
    return errors::InvalidArgument("indices must be int32 or int64");
  }

  xla::XlaOp gather;
  if (batch_dims > 0) {
    *gather_output = xla::TorchIndexSelect(input, indices, *axis, batch_dims);
  } else {
    // XlaGather() manages degenerate cases, like empty-indices, which are
    // error conditions and caught above if batch_dims is not 0.
    TF_RETURN_IF_ERROR(
        XlaGather(input, input_shape, indices, indices_shape, *axis,
                  /*indices_are_nd=*/false, context->expected_output_dtype(0),
                  index_type, context->builder(), gather_output));
  }
  return OkStatus();
}
class GatherOp : public XlaOpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    // Set batch_dims_ to 0 if the attribute does not exist.
    if (context->HasAttr("batch_dims")) {
      OP_REQUIRES_OK(context, context->GetAttr("batch_dims", &batch_dims_));
    } else {
      batch_dims_ = 0;
    }
  }

  void Compile(XlaOpKernelContext* context) override {
    auto input = context->Input(0);
    auto input_shape = context->InputShape(0);

    xla::XlaOp gather;
    OP_REQUIRES_OK(context,
                   XlaGatherWithBatchDimsOpImpl(context, input, input_shape,
                                                batch_dims_, &gather));
    context->SetOutput(0, gather);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GatherOp);

  // The number of batch dimensions, as passed in the batch_dims attribute.
  // It must be less than or equal to rank(indices).
  int32 batch_dims_ = 0;
};

REGISTER_XLA_OP(Name("Gather"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("GatherV2").CompileTimeConstantInput("axis"), GatherOp);

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
    const int64_t num_index_dims =
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
