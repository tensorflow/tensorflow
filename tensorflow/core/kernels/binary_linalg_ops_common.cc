/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/binary_linalg_ops_common.h"

namespace tensorflow {

void BinaryLinearAlgebraOpBase::Compute(OpKernelContext* context) {
  const Tensor& in_lhs = context->input(0);
  const Tensor& in_rhs = context->input(1);

  const int input_rank = in_lhs.dims();
  if (SupportsBatchOperation()) {
    OP_REQUIRES(context, input_rank >= 2,
                errors::InvalidArgument("Input tensor must have rank >= 2"));
  } else {
    OP_REQUIRES(context, input_rank == 2,
                errors::InvalidArgument("Input tensor must have rank == 2"));
  }
  // TODO(rmlarsen): Add support for broadcasting.
  OP_REQUIRES(
      context, input_rank == in_rhs.dims(),
      errors::InvalidArgument(
          "Tensors must have the same rank: rank(lhs) (%d) != rank(rhs) (%d)",
          input_rank, in_rhs.dims()));
  for (int dim = 0; dim < (in_rhs.dims() - 2); ++dim) {
    OP_REQUIRES(context, in_rhs.dim_size(dim) == in_lhs.dim_size(dim),
                errors::InvalidArgument(
                    "Dimension mismatch: %d != %d for dimension %d",
                    in_lhs.dim_size(dim), in_rhs.dim_size(dim), dim));
  }

  // If the tensor rank is greater than 2, we consider the inner-most
  // dimensions as matrices, and loop over all the other outer
  // dimensions to compute the results.
  const int row_dimension = input_rank - 2;
  const int col_dimension = input_rank - 1;
  const int64 lhs_num_rows = in_lhs.dim_size(row_dimension);
  const int64 lhs_num_cols = in_lhs.dim_size(col_dimension);
  const int64 rhs_num_rows = in_rhs.dim_size(row_dimension);
  const int64 rhs_num_cols = in_rhs.dim_size(col_dimension);
  const TensorShape in_lhs_matrix_shape =
      TensorShape({lhs_num_rows, lhs_num_cols});
  const TensorShape in_rhs_matrix_shape =
      TensorShape({rhs_num_rows, rhs_num_cols});
  const TensorShape output_matrix_shape =
      GetOutputMatrixShape(in_lhs_matrix_shape, in_rhs_matrix_shape);
  OP_REQUIRES(context, output_matrix_shape.dims() <= 2,
              errors::InvalidArgument("Output rank must be 1 or 2."));

  int num_matrices = 1;
  // The output has the shape of all the outer dimensions of the input
  // except for the last two, plus the output_matrix_shape (if the output
  // is not scalar). This still assumes that each input matrix is
  // 2-dimensional, in accordance with the TODO above.
  TensorShape output_shape;
  if (input_rank == 2) {
    output_shape = output_matrix_shape;
  } else {
    // Add the common outer dimensions.
    for (int dim = 0; dim < input_rank - 2; ++dim) {
      num_matrices *= in_lhs.dim_size(dim);
      output_shape.AddDim(in_lhs.dim_size(dim));
    }
    // Add the inner dimensions that depend on the operation implemented by the
    // derived class.
    for (int dim = 0; dim < output_matrix_shape.dims(); ++dim) {
      output_shape.AddDim(output_matrix_shape.dim_size(dim));
    }
  }

  Tensor* out = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out));

  auto shard = [this, &in_lhs, &in_lhs_matrix_shape, &in_rhs,
                &in_rhs_matrix_shape, &output_matrix_shape, context,
                out](int64 begin, int64 end) {
    for (int64 i = begin; i < end; ++i) {
      ComputeMatrix(context, i, in_lhs, in_lhs_matrix_shape, in_rhs,
                    in_rhs_matrix_shape, out, output_matrix_shape);
    }
  };

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  Shard(worker_threads.num_threads, worker_threads.workers, num_matrices,
        GetCostPerUnit(in_lhs_matrix_shape, in_rhs_matrix_shape), shard);
}

template <typename Scalar, bool SupportsBatchOperationT>
void BinaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::ComputeMatrix(
    OpKernelContext* context, int64 matrix_index, const Tensor& in_lhs,
    const TensorShape& in_lhs_matrix_shape, const Tensor& in_rhs,
    const TensorShape& in_rhs_matrix_shape, Tensor* out,
    const TensorShape& output_matrix_shape) {
  // TODO(kalakris): Handle alignment if possible. Eigen::Map is
  // unaligned by default.
  ConstMatrixMap in_lhs_map(
      in_lhs.flat<Scalar>().data() +
          matrix_index * in_lhs_matrix_shape.num_elements(),
      in_lhs_matrix_shape.dim_size(0), in_lhs_matrix_shape.dim_size(1));
  ConstMatrixMap in_rhs_map(
      in_rhs.flat<Scalar>().data() +
          matrix_index * in_rhs_matrix_shape.num_elements(),
      in_rhs_matrix_shape.dim_size(0), in_rhs_matrix_shape.dim_size(1));

  // The output matrix shape may not be a matrix.
  int num_output_rows =
      output_matrix_shape.dims() >= 1 ? output_matrix_shape.dim_size(0) : 1;
  int num_output_cols =
      output_matrix_shape.dims() == 2 ? output_matrix_shape.dim_size(1) : 1;
  MatrixMap output(out->flat<Scalar>().data() +
                       matrix_index * output_matrix_shape.num_elements(),
                   num_output_rows, num_output_cols);
  ComputeMatrix(context, in_lhs_map, in_rhs_map, &output);
}

// Explicitly instantiate BinaryLinearAlgebraOp for the scalar types we expect
// to use.
template class BinaryLinearAlgebraOp<float, false>;
template class BinaryLinearAlgebraOp<float, true>;
template class BinaryLinearAlgebraOp<double, false>;
template class BinaryLinearAlgebraOp<double, true>;

}  // namespace tensorflow
