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

// See docs in ../ops/linalg_ops.cc.

#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class MatrixSolveLsOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit MatrixSolveLsOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fast", &fast_));
  }

  using TensorShapes = typename Base::TensorShapes;
  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  // Tell the base class to ignore the regularization parameter
  // in context->input(2).
  int NumMatrixInputs(const OpKernelContext* context) const final { return 2; }

  virtual void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    Base::ValidateSolver(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({TensorShape({input_matrix_shapes[0].dim_size(1),
                                      input_matrix_shapes[1].dim_size(1)})});
  }

  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double num_rhss = static_cast<double>(input_matrix_shapes[1].dim_size(1));
    double cost = std::max(m, n) * std::min(m, n) * (std::min(m, n) + num_rhss);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64>(cost);
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& matrix = inputs[0];
    const ConstMatrixMap& rhs = inputs[1];
    const auto& l2_regularizer_in = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(l2_regularizer_in.shape()),
        errors::InvalidArgument("l2_regularizer must be scalar, got shape ",
                                l2_regularizer_in.shape().DebugString()));
    const double l2_regularizer = l2_regularizer_in.scalar<double>()();
    OP_REQUIRES(context, l2_regularizer >= 0,
                errors::InvalidArgument("l2_regularizer must be >= 0."));

    const int64 rows = matrix.rows();
    const int64 cols = matrix.cols();
    if (rows == 0 || cols == 0) {
      // The result is the empty matrix.
      return;
    }
    if (fast_) {
      // The fast branch assumes that matrix is not rank deficient and
      // not too ill-conditioned. Specifically, the reciprocal condition number
      // should be greater than the square root of the machine precision, i.e.
      //   1 / cond(matrix) > sqrt(std::numeric_limits<Scalar>::epsilon()).
      // This branch solves over- or underdetermined least-squares problems
      // via the normal equations and Cholesky decomposition.
      if (matrix.rows() >= matrix.cols()) {
        // Overdetermined case (rows >= cols): Solves the ordinary (possibly
        // regularized) least-squares problem
        //   min || A * X - RHS ||_F^2 + l2_regularizer ||X||_F^2
        // by solving the normal equations
        //    (A^T * A + l2_regularizer * I) X = A^T RHS
        // using Cholesky decomposition.
        Matrix gramian(cols, cols);
        gramian.template triangularView<Eigen::Lower>() =
            matrix.transpose() * matrix;
        if (l2_regularizer > 0) {
          gramian +=
              (Scalar(l2_regularizer) * Matrix::Ones(cols, 1)).asDiagonal();
        }
        const Eigen::LLT<Matrix, Eigen::Lower> llt(gramian);
        OP_REQUIRES(
            context, llt.info() == Eigen::Success,
            errors::InvalidArgument("Input matrix was rank deficient or "
                                    "ill-conditioned. Try setting fast=False "
                                    "or provide a larger l2_regularizer > 0."));
        outputs->at(0) = llt.solve(matrix.transpose() * rhs);
      } else {
        // Underdetermined case (rows < cols): Solves the minimum-norm problem
        //   min ||X||_F^2 s.t. A*X = RHS
        // by solving the normal equations of the second kind
        //   (A * A^T + l2_regularizer * I) Z = RHS,  X = A^T * Z
        // using Cholesky decomposition.
        Matrix gramian(rows, rows);
        gramian.template triangularView<Eigen::Lower>() =
            matrix * matrix.transpose();
        if (l2_regularizer > 0) {
          gramian +=
              (Scalar(l2_regularizer) * Matrix::Ones(rows, 1)).asDiagonal();
        }
        const Eigen::LLT<Matrix, Eigen::Lower> llt(gramian);
        OP_REQUIRES(
            context, llt.info() == Eigen::Success,
            errors::InvalidArgument("Input matrix was rank deficient or "
                                    "ill-conditioned. Try setting fast=False "
                                    "or provide an l2_regularizer > 0."));
        outputs->at(0) = matrix.transpose() * llt.solve(rhs);
      }
    } else {
      // Use complete orthogonal decomposition which is backwards stable and
      // will compute the minimum-norm solution for rank-deficient matrices.
      // This is 6-7 times slower than the fast path.
      //
      // TODO(rmlarsen): The implementation of
      //   Eigen::CompleteOrthogonalDecomposition is not blocked, so for
      //   matrices that do not fit in cache, it is significantly slower than
      //   the equivalent blocked LAPACK routine xGELSY (e.g. Eigen is ~3x
      //   slower for 4k x 4k matrices).
      //   See http://www.netlib.org/lapack/lawnspdf/lawn114.pdf
      outputs->at(0) = matrix.completeOrthogonalDecomposition().solve(rhs);
    }
  }

 private:
  bool fast_;
};

REGISTER_LINALG_OP("MatrixSolveLs", (MatrixSolveLsOp<float>), float);
REGISTER_LINALG_OP("MatrixSolveLs", (MatrixSolveLsOp<double>), double);
REGISTER_LINALG_OP("BatchMatrixSolveLs", (MatrixSolveLsOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixSolveLs", (MatrixSolveLsOp<double>), double);

}  // namespace tensorflow
