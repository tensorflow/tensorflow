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

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/LU"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Scalar, typename Tidx>
class LuOp : public OpKernel {
 public:
  explicit LuOp(OpKernelConstruction* context) : OpKernel(context) {}

 protected:
  using TensorShapes = absl::InlinedVector<TensorShape, 4UL>;
  using TensorOutputs = absl::InlinedVector<Tensor*, 4UL>;

  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  using Indices =
      Eigen::Matrix<Tidx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IndicesMap = Eigen::Map<Indices>;
  using ConstIndicesMap = Eigen::Map<const Indices>;

 public:
  // Returns the cost per matrix operation. This is used to determine the
  // number of threads to use for parallelizing factorization in batch mode.
  // Cost per unit is assumed to be roughly 1ns, based on comments
  // in core/util/work_sharder.cc.
  // LU decomposition for a square matrix takes roughly (2/3) * (num_rows)^3.
  // TODO(anudhyan): Refine this estimate after taking constant factors into
  // account.
  int64_t GetCostPerUnit(const TensorShape& input_matrix_shape) const {
    double num_rows = static_cast<double>(input_matrix_shape.dim_size(0));
    double cost = (2 / 3.0) * MathUtil::IPow(num_rows, 3);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, context->num_inputs() == 1,
                errors::InvalidArgument("Expecting exactly one input, got ",
                                        context->num_inputs()));

    const Tensor& input = context->input(0);
    int input_rank = input.dims();
    OP_REQUIRES(context, input_rank >= 2,
                errors::InvalidArgument(
                    "Input tensor must have rank >= 2, got ", input_rank));

    // If the tensor rank is greater than 2, we consider the inner-most
    // dimensions as matrices, and loop over all the other outer ("batch")
    // dimensions to compute the results.
    TensorShape input_matrix_shape;
    TensorShape batch_shape;
    for (int dim = 0; dim < input_rank - 2; ++dim) {
      OP_REQUIRES_OK(context,
                     batch_shape.AddDimWithStatus(input.dim_size(dim)));
    }
    const int64_t num_rows = input.dim_size(input_rank - 2);
    const int64_t num_cols = input.dim_size(input_rank - 1);

    input_matrix_shape.AppendShape({num_rows, num_cols});
    OP_REQUIRES(context, TensorShapeUtils::IsSquareMatrix(input_matrix_shape),
                errors::InvalidArgument("Input matrix must be square."));

    // packed_triangular_factors is a matrix with the same shape as the input;
    // permutation is a vector.
    TensorShape permutation_shape = batch_shape;
    OP_REQUIRES_OK(context, permutation_shape.AddDimWithStatus(num_rows));

    TensorShapes output_matrix_shapes({input.shape(), permutation_shape});

    TensorOutputs outputs;
    Tensor* output_packed_triangular_factors = nullptr;
    OP_REQUIRES_OK(
        context, context->forward_input_or_allocate_output(
                     {0}, 0, input.shape(), &output_packed_triangular_factors));
    outputs.emplace_back(output_packed_triangular_factors);

    Tensor* output_permutation = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, permutation_shape,
                                                     &output_permutation));
    outputs.emplace_back(output_permutation);

    if (num_rows == 0) {
      return;
    }

    // Process the individual matrix problems in parallel using a threadpool.
    auto shard = [this, &input, &num_rows, &num_cols, &outputs,
                  &output_matrix_shapes, context](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        ComputeTensorSlice(context, i, input, num_rows, num_cols, outputs,
                           output_matrix_shapes);
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          batch_shape.num_elements(), GetCostPerUnit(input_matrix_shape),
          shard);
  }

  void ComputeTensorSlice(OpKernelContext* context, int64_t matrix_index,
                          const Tensor& input, int64_t num_rows,
                          int64_t num_cols, const TensorOutputs& outputs,
                          const TensorShapes& output_matrix_shapes) {
    // TODO(kalakris): Handle alignment if possible. Eigen::Map is
    // unaligned by default.
    ConstMatrixMap input_matrix(
        input.flat<Scalar>().data() + matrix_index * num_rows * num_cols,
        num_rows, num_cols);

    // packed_triangular_factors has shape [num_rows, num_cols]
    MatrixMap packed_triangular_factors(
        outputs[0]->flat<Scalar>().data() + matrix_index * num_rows * num_cols,
        num_rows, num_rows);

    // permutation has shape [num_rows, 1]
    IndicesMap permutation_indices(
        outputs[1]->flat<Tidx>().data() + matrix_index * num_rows, num_rows, 1);

    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        lu_decomposition(input_matrix);

    // Output the packed triangular factors in a dense form.
    // The lower triangular factor L corresponds to the strictly lower
    // triangular part of packed_triangular_factors with an implicit unit
    // diagonal. The upper triangular factor U is the upper triangular part of
    // packed_triangular_factors. The triangular factors satisfy the equation
    //     P * input_matrix = L * U
    // where P is the permutation matrix corresponding to the indices in
    // permutation_indices.
    packed_triangular_factors = lu_decomposition.matrixLU();
    // Output the permutation matrix used for pivoting.
    Eigen::PermutationMatrix<-1, -1, Tidx> permutation =
        lu_decomposition.permutationP().transpose();
    permutation_indices = permutation.indices();

    // PartialPivLU cannot give strong guarantees on invertibility,
    // but we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes such providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    const RealScalar min_abs_pivot =
        packed_triangular_factors.diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > RealScalar(0),
                errors::InvalidArgument("Input is not invertible."));
  }
};

#define REGISTER_LU(type, idx_type)                                         \
  REGISTER_KERNEL_BUILDER(Name("Lu")                                        \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<idx_type>("output_idx_type"), \
                          LuOp<type, idx_type>);

REGISTER_LU(float, int32);
REGISTER_LU(double, int32);
REGISTER_LU(complex64, int32);
REGISTER_LU(complex128, int32);

REGISTER_LU(float, int64_t);
REGISTER_LU(double, int64_t);
REGISTER_LU(complex64, int64_t);
REGISTER_LU(complex128, int64_t);

}  // namespace tensorflow
