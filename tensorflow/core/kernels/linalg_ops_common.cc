/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/linalg_ops_common.h"

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// static
template <typename Scalar, bool SupportsBatchOperation>
void LinearAlgebraOp<Scalar, SupportsBatchOperation>::ValidateSingleMatrix(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 1,
              errors::InvalidArgument("Expected a single input matrix, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[0]),
              errors::InvalidArgument("Input must be a matrix."));
}

// static
template <typename Scalar, bool SupportsBatchOperation>
void LinearAlgebraOp<Scalar, SupportsBatchOperation>::
    ValidateSingleSquareMatrix(OpKernelContext* context,
                               const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 1,
              errors::InvalidArgument("Expected a single input matrix, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(context, TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
              errors::InvalidArgument("Input matrix must be square."));
}

// static
template <typename Scalar, bool SupportsBatchOperation>
void LinearAlgebraOp<Scalar, SupportsBatchOperation>::ValidateSolver(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 2,
              errors::InvalidArgument("Expected two input matrices, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[0]),
              errors::InvalidArgument("First input (lhs) must be a matrix."));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[1]),
              errors::InvalidArgument("Second input (rhs) must be a matrix."));
  OP_REQUIRES(
      context,
      input_matrix_shapes[0].dim_size(0) == input_matrix_shapes[1].dim_size(0),
      errors::InvalidArgument("Input matrix and rhs are incompatible."));
}

// static
template <typename Scalar, bool SupportsBatchOperation>
void LinearAlgebraOp<Scalar, SupportsBatchOperation>::ValidateSquareSolver(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 2,
              errors::InvalidArgument("Expected two input matrices, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(
      context, TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
      errors::InvalidArgument("First input (lhs) must be a square matrix."));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[1]),
              errors::InvalidArgument("Second input (rhs) must be a matrix."));
  OP_REQUIRES(
      context,
      input_matrix_shapes[0].dim_size(0) == input_matrix_shapes[1].dim_size(0),
      errors::InvalidArgument("Input matrix and rhs are incompatible."));
}

template <typename Scalar, bool SupportsBatchOperation>
void LinearAlgebraOp<Scalar, SupportsBatchOperation>::Compute(
    OpKernelContext* context) {
  TensorInputs inputs;
  TensorShapes input_matrix_shapes;
  TensorShape batch_shape;
  int input_rank = -1;
  int num_batch_matrices = 1;
  for (int i = 0; i < NumMatrixInputs(context); ++i) {
    const Tensor& in = context->input(i);
    if (i == 0) {
      // If the tensor rank is greater than 2, we consider the inner-most
      // dimensions as matrices, and loop over all the other outer ("batch")
      // dimensions to compute the results.
      input_rank = in.dims();
      for (int dim = 0; dim < input_rank - 2; ++dim) {
        num_batch_matrices *= in.dim_size(dim);
        batch_shape.AddDim(in.dim_size(dim));
      }
      if (SupportsBatchOperation) {
        OP_REQUIRES(
            context, input_rank >= 2,
            errors::InvalidArgument("Input tensor ", i,
                                    " must have rank >= 2, got", input_rank));
      } else {
        OP_REQUIRES(
            context, input_rank == 2,
            errors::InvalidArgument("Input tensor ", i,
                                    " must have rank == 2, got", input_rank));
      }
    } else {
      // Make sure that all inputs have the same rank and outer dimensions.
      OP_REQUIRES(context, input_rank == in.dims(),
                  errors::InvalidArgument(
                      "All input tensors must have the same rank."));
      for (int dim = 0; dim < input_rank - 2; ++dim) {
        OP_REQUIRES(
            context, in.dim_size(dim) == batch_shape.dim_size(dim),
            errors::InvalidArgument(
                "All input tensors must have the same outer dimensions."));
      }
    }

    const int row_dimension = input_rank - 2;
    const int col_dimension = input_rank - 1;
    const int64 num_rows = in.dim_size(row_dimension);
    const int64 num_cols = in.dim_size(col_dimension);
    input_matrix_shapes.push_back(TensorShape({num_rows, num_cols}));
    inputs.push_back(in);
  }
  // Have the derived class validate that the inputs are as expected.
  ValidateInputMatrixShapes(context, input_matrix_shapes);

  // Get shape for each of the matrix outputs.
  const TensorShapes output_matrix_shapes =
      GetOutputMatrixShapes(input_matrix_shapes);
  // Make sure the number of outputs is what the derived class expects.
  OP_REQUIRES(
      context, output_matrix_shapes.size() == context->num_outputs(),
      errors::Internal(
          "Derived class expected (%d) output matrices for op, got (%d).",
          output_matrix_shapes.size(), context->num_outputs()));

  // Allocate outputs.
  TensorShapes output_shapes;
  TensorOutputs outputs;
  for (int i = 0; i < context->num_outputs(); ++i) {
    OP_REQUIRES(context, output_matrix_shapes[i].dims() <= 2,
                errors::InvalidArgument(
                    "Rank of matrix output no. %d must be 0, 1 or 2, got %d.",
                    i, output_matrix_shapes[i].dims()));

    // The final output has the shape of the outer batch dimensions concatenated
    // with the output_matrix_shape (if the output is not scalar).
    TensorShape output_shape;
    if (input_rank == 2) {
      output_shape = output_matrix_shapes[i];
    } else {
      output_shape = batch_shape;
      // Add the inner dimensions that depend on the operation implemented by
      // the derived class.
      for (int dim = 0; dim < output_matrix_shapes[i].dims(); ++dim) {
        output_shape.AddDim(output_matrix_shapes[i].dim_size(dim));
      }
    }
    output_shapes.push_back(output_shape);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(i, output_shape, &out));
    outputs.push_back(out);
  }

  auto shard = [this, &inputs, &input_matrix_shapes, &outputs,
                &output_matrix_shapes, context](int64 begin, int64 end) {
    for (int64 i = begin; i < end; ++i) {
      ComputeTensorSlice(context, i, inputs, input_matrix_shapes, outputs,
                         output_matrix_shapes);
    }
  };
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  Shard(worker_threads.num_threads, worker_threads.workers, num_batch_matrices,
        GetCostPerUnit(input_matrix_shapes), shard);
}

template <typename Scalar, bool SupportsBatchOperationT>
void LinearAlgebraOp<Scalar, SupportsBatchOperationT>::ComputeTensorSlice(
    OpKernelContext* context, int64 matrix_index, const TensorInputs& inputs,
    const TensorShapes& input_matrix_shapes, const TensorOutputs& outputs,
    const TensorShapes& output_matrix_shapes) {
  ConstMatrixMaps matrix_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    // TODO(kalakris): Handle alignment if possible. Eigen::Map is
    // unaligned by default.
    matrix_inputs.push_back(
        ConstMatrixMap(inputs[i].flat<Scalar>().data() +
                           matrix_index * input_matrix_shapes[i].num_elements(),
                       input_matrix_shapes[i].dim_size(0),
                       input_matrix_shapes[i].dim_size(1)));
  }

  MatrixMaps matrix_outputs;
  for (int i = 0; i < outputs.size(); ++i) {
    // The output matrix shape may not be a matrix.
    int num_output_rows = output_matrix_shapes[i].dims() >= 1
                              ? output_matrix_shapes[i].dim_size(0)
                              : 1;
    int num_output_cols = output_matrix_shapes[i].dims() == 2
                              ? output_matrix_shapes[i].dim_size(1)
                              : 1;
    matrix_outputs.push_back(
        MatrixMap(outputs[i]->flat<Scalar>().data() +
                      matrix_index * output_matrix_shapes[i].num_elements(),
                  num_output_rows, num_output_cols));
  }
  ComputeMatrix(context, matrix_inputs, &matrix_outputs);
}

// Explicitly instantiate LinearAlgebraOp for the scalar types we expect to
// use.
template class LinearAlgebraOp<float, false>;
template class LinearAlgebraOp<float, true>;
template class LinearAlgebraOp<double, false>;
template class LinearAlgebraOp<double, true>;

}  // namespace tensorflow
