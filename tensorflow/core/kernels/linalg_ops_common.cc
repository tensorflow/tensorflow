#include "tensorflow/core/kernels/linalg_ops_common.h"

namespace tensorflow {

void LinearAlgebraOpBase::Compute(OpKernelContext* context) {
  const Tensor& in = context->input(0);

  const int input_rank = GetInputMatrixRank();
  OP_REQUIRES(
      context, input_rank == 2,
      errors::InvalidArgument("Only matrix inputs are supported so far."));
  if (SupportsBatchOperation()) {
    OP_REQUIRES(context, in.dims() > input_rank,
                errors::InvalidArgument("Input tensor must have rank >= %d",
                                        input_rank + 1));
  } else {
    OP_REQUIRES(context, in.dims() == input_rank,
                errors::InvalidArgument("Input tensor must have rank == %d",
                                        input_rank));
  }

  // If the tensor rank is greater than input_rank, we consider the inner-most
  // dimensions as matrices, and loop over all the other outer
  // dimensions to compute the results.
  // TODO(kalakris): Only matrix inputs are currently supported.
  const int row_dimension = in.dims() - 2;
  const int col_dimension = in.dims() - 1;
  const int64 num_rows = in.dim_size(row_dimension);
  const int64 num_cols = in.dim_size(col_dimension);
  const TensorShape input_matrix_shape = TensorShape({num_rows, num_cols});
  const TensorShape output_matrix_shape =
      GetOutputMatrixShape(input_matrix_shape);
  OP_REQUIRES(context, output_matrix_shape.dims() <= 2,
              errors::InvalidArgument("Output rank must be 1 or 2."));

  int num_matrices = 1;
  // The output has the shape of all the outer dimensions of the input
  // except for the last two, plus the output_matrix_shape (if the output
  // is not scalar). This still assumes that each input matrix is
  // 2-dimensional, in accordance with the TODO above.
  TensorShape output_shape;
  if (in.dims() == 2) {
    output_shape = output_matrix_shape;
  } else {
    for (int dim = 0; dim <= in.dims() - 3; ++dim) {
      num_matrices *= in.dim_size(dim);
      output_shape.AddDim(in.dim_size(dim));
    }
    for (int dim = 0; dim < output_matrix_shape.dims(); ++dim) {
      output_shape.AddDim(output_matrix_shape.dim_size(dim));
    }
  }

  Tensor* out = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out));

  auto shard = [this, &in, &input_matrix_shape, &output_matrix_shape, context,
                out](int64 begin, int64 end) {
    for (int64 i = begin; i < end; ++i) {
      ComputeMatrix(context, i, in, input_matrix_shape, out,
                    output_matrix_shape);
    }
  };

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  Shard(worker_threads.num_threads, worker_threads.workers, num_matrices,
        GetCostPerUnit(input_matrix_shape), shard);
}

template <typename Scalar, bool SupportsBatchOperationT>
void LinearAlgebraOp<Scalar, SupportsBatchOperationT>::ComputeMatrix(
    OpKernelContext* context, int64 matrix_index, const Tensor& in,
    const TensorShape& input_matrix_shape, Tensor* out,
    const TensorShape& output_matrix_shape) {
  // TODO(kalakris): Handle alignment if possible. Eigen::Map is
  // unaligned by default.
  ConstMatrixMap input(in.flat<Scalar>().data() +
                           matrix_index * input_matrix_shape.num_elements(),
                       input_matrix_shape.dim_size(0),
                       input_matrix_shape.dim_size(1));

  // The output matrix shape may not be a matrix.
  int num_output_rows =
      output_matrix_shape.dims() >= 1 ? output_matrix_shape.dim_size(0) : 1;
  int num_output_cols =
      output_matrix_shape.dims() == 2 ? output_matrix_shape.dim_size(1) : 1;
  MatrixMap output(out->flat<Scalar>().data() +
                       matrix_index * output_matrix_shape.num_elements(),
                   num_output_rows, num_output_cols);
  ComputeMatrix(context, input, &output);
}

// Explicitly instantiate LinearAlgebraOp for the scalar types we expect to use.
template class LinearAlgebraOp<float, false>;
template class LinearAlgebraOp<float, true>;
template class LinearAlgebraOp<double, false>;
template class LinearAlgebraOp<double, true>;

}  // namespace tensorflow
