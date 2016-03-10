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

// See docs in ../ops/linalg_ops.cc.
// TODO(konstantinos): Enable complex inputs. This will require additional tests
//                     and OP_REQUIRES.

#include <cmath>

#include "third_party/eigen3/Eigen/Cholesky"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperationT>
class CholeskyOp
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit CholeskyOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    return input_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint64max;
    } else {
      return rows * rows * rows;
    }
  }

  using
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap;
  using typename UnaryLinearAlgebraOp<Scalar,
                                      SupportsBatchOperationT>::ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& input,
                     MatrixMap* output) override {
    OP_REQUIRES(context, input.rows() == input.cols(),
                errors::InvalidArgument("Input matrix must be square."));
    if (input.rows() == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }
    // Perform the actual LL^T Cholesky decomposition. This will only use
    // the lower triangular part of data_in by default. The upper triangular
    // part of the matrix will not be read.
    Eigen::LLT<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        llt_decomposition(input);

    // Output the lower triangular in a dense form.
    *output = llt_decomposition.matrixL();

    OP_REQUIRES(context, llt_decomposition.info() == Eigen::Success,
                errors::InvalidArgument("LLT decomposition was not successful. "
                                        "The input might not be valid."));
  }
};

REGISTER_LINALG_OP("Cholesky", (CholeskyOp<float, false>), float);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<double, false>), double);
REGISTER_LINALG_OP("BatchCholesky", (CholeskyOp<float, true>), float);
REGISTER_LINALG_OP("BatchCholesky", (CholeskyOp<double, true>), double);
}  // namespace tensorflow
