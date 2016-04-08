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
#include <cmath>

#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperationT>
class DeterminantOp
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit DeterminantOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) {}
  ~DeterminantOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    return TensorShape({});
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
    Scalar determinant;
    if (input.rows() == 0) {
      // An empty matrix' determinant is defined to be 1.  See
      // wikipedia.
      determinant = 1;
    } else {
      determinant = input.determinant();
    }
    OP_REQUIRES(context, std::isfinite(determinant),
                errors::Internal("The determinant is not finite."));
    (*output)(0, 0) = determinant;
  }
};

REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<float, false>), float);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<double, false>), double);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<float, true>),
                   float);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<double, true>),
                   double);

}  // namespace tensorflow
