/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

static const char kNotInvertibleMsg[] = "The matrix is not invertible.";

static const char kNotInvertibleScalarMsg[] =
    "The matrix is not invertible: it is a scalar with value zero.";

template <class Scalar>
class TridiagonalSolveOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit TridiagonalSolveOp(OpKernelConstruction* context) : Base(context) {}

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    auto num_inputs = input_matrix_shapes.size();
    OP_REQUIRES(context, num_inputs == 2,
                errors::InvalidArgument("Expected two input matrices, got ",
                                        num_inputs, "."));

    auto num_diags = input_matrix_shapes[0].dim_size(0);
    OP_REQUIRES(
        context, num_diags == 3,
        errors::InvalidArgument("Expected diagonals to be provided as a "
                                "matrix with 3 rows, got ",
                                num_diags, " rows."));

    auto num_eqs_left = input_matrix_shapes[0].dim_size(1);
    auto num_eqs_right = input_matrix_shapes[1].dim_size(0);
    OP_REQUIRES(
        context, num_eqs_left == num_eqs_right,
        errors::InvalidArgument("Expected the same number of left-hand sides "
                                "and right-hand sides, got ",
                                num_eqs_left, " and ", num_eqs_right, "."));
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({input_matrix_shapes[1]});
  }

  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    const int num_eqs = static_cast<int>(input_matrix_shapes[0].dim_size(1));
    const int num_rhss = static_cast<int>(input_matrix_shapes[1].dim_size(0));

    const double add_cost = Eigen::TensorOpCost::AddCost<Scalar>();
    const double mult_cost = Eigen::TensorOpCost::MulCost<Scalar>();
    const double div_cost = Eigen::TensorOpCost::DivCost<Scalar>();

    // Assuming cases with and without row interchange are equiprobable.
    const double cost =
        num_eqs * (div_cost * (num_rhss + 1) +
                   (add_cost + mult_cost) * (2.5 * num_rhss + 1.5));
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64>(cost);
  }

  bool EnableInputForwarding() const final { return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const auto diagonals = inputs[0];

    // Subdiagonal elements, first is ignored.
    const auto& superdiag = diagonals.row(0);
    // Diagonal elements.
    const auto& diag = diagonals.row(1);
    // Superdiagonal elements, n-th is ignored.
    const auto& subdiag = diagonals.row(2);
    // Right-hand sides.
    const auto& rhs = inputs[1];

    const int n = diag.size();
    MatrixMap& x = outputs->at(0);
    const Scalar zero(0);

    if (n == 0) {
      return;
    }
    if (n == 1) {
      OP_REQUIRES(context, diag(0) != zero,
                  errors::InvalidArgument(kNotInvertibleScalarMsg));
      x.row(0) = rhs.row(0) / diag(0);
      return;
    }

    // The three columns in u are the diagonal, superdiagonal, and second
    // superdiagonal, respectively, of the U matrix in the LU decomposition of
    // the input matrix (subject to row exchanges due to pivoting). For pivoted
    // tridiagonal matrix, the U matrix has at most two non-zero superdiagonals.
    Eigen::Array<Scalar, Eigen::Dynamic, 3> u(n, 3);

    // The code below roughly follows LAPACK's dgtsv routine, with main
    // difference being not overwriting the input.
    u(0, 0) = diag(0);
    u(0, 1) = superdiag(0);
    x.row(0) = rhs.row(0);
    for (int i = 0; i < n - 1; ++i) {
      if (std::abs(u(i)) >= std::abs(subdiag(i + 1))) {
        // No row interchange.
        OP_REQUIRES(context, u(i) != zero,
                    errors::InvalidArgument(kNotInvertibleMsg));
        const Scalar factor = subdiag(i + 1) / u(i, 0);
        u(i + 1, 0) = diag(i + 1) - factor * u(i, 1);
        x.row(i + 1) = rhs.row(i + 1) - factor * x.row(i);
        if (i != n - 2) {
          u(i + 1, 1) = superdiag(i + 1);
          u(i, 2) = 0;
        }
      } else {
        // Interchange rows i and i + 1.
        const Scalar factor = u(i, 0) / subdiag(i + 1);
        u(i, 0) = subdiag(i + 1);
        u(i + 1, 0) = u(i, 1) - factor * diag(i + 1);
        u(i, 1) = diag(i + 1);
        x.row(i + 1) = x.row(i) - factor * rhs.row(i + 1);
        x.row(i) = rhs.row(i + 1);
        if (i != n - 2) {
          u(i, 2) = superdiag(i + 1);
          u(i + 1, 1) = -factor * superdiag(i + 1);
        }
      }
    }
    OP_REQUIRES(context, u(n - 1, 0) != zero,
                errors::InvalidArgument(kNotInvertibleMsg));
    x.row(n - 1) /= u(n - 1, 0);
    x.row(n - 2) = (x.row(n - 2) - u(n - 2, 1) * x.row(n - 1)) / u(n - 2, 0);
    for (int i = n - 3; i >= 0; --i) {
      x.row(i) = (x.row(i) - u(i, 1) * x.row(i + 1) - u(i, 2) * x.row(i + 2)) /
                 u(i, 0);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TridiagonalSolveOp);
};

REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<float>), float);
REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<double>),
                       double);
REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<complex64>),
                       complex64);
REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<complex128>),
                       complex128);
}  // namespace tensorflow
