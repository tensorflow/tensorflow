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
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

static const char kNotInvertibleMsg[] = "The matrix is not invertible.";

static const char kNotInvertibleScalarMsg[] =
    "The matrix is not invertible: it is a scalar with value zero.";

static const char kThomasFailedMsg[] =
    "The matrix is either not invertible, or requires pivoting. "
    "Try setting partial_pivoting = True.";

template <class Scalar>
class TridiagonalSolveOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);
  using MatrixMapRow =
      decltype(std::declval<const ConstMatrixMaps>()[0].row(0));

  explicit TridiagonalSolveOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("partial_pivoting", &pivoting_));
    perturb_singular_ = false;
    if (context->HasAttr("perturb_singular")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("perturb_singular", &perturb_singular_));
    }
    OP_REQUIRES(context, pivoting_ || !perturb_singular_,
                errors::InvalidArgument("Setting perturb_singular requires "
                                        "also setting partial_pivoting."));
  }

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

    double cost;
    if (pivoting_) {
      // Assuming cases with and without row interchange are equiprobable.
      cost = num_eqs * (div_cost * (num_rhss + 1) +
                        (add_cost + mult_cost) * (2.5 * num_rhss + 1.5));
    } else {
      cost = num_eqs * (div_cost * (num_rhss + 1) +
                        (add_cost + mult_cost) * (2 * num_rhss + 1));
    }
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64>(cost);
  }

  bool EnableInputForwarding() const final { return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const auto diagonals = inputs[0];

    // Superdiagonal elements, first is ignored.
    const auto& superdiag = diagonals.row(0);
    // Diagonal elements.
    const auto& diag = diagonals.row(1);
    // Subdiagonal elements, n-th is ignored.
    const auto& subdiag = diagonals.row(2);
    // Right-hand sides.
    const auto& rhs = inputs[1];

    const int n = diag.size();
    MatrixMap& x = outputs->at(0);
    constexpr Scalar zero(0);

    if (n == 0) {
      return;
    }
    if (pivoting_ && perturb_singular_) {
      SolveWithGaussianEliminationWithPivotingAndPerturbSingular(
          context, superdiag, diag, subdiag, rhs, x);
      return;
    }

    if (n == 1) {
      if (diag(0) == zero) {
        LOG(WARNING) << kNotInvertibleScalarMsg;
        x.fill(std::numeric_limits<Scalar>::quiet_NaN());
      } else {
        x.row(0) = rhs.row(0) / diag(0);
      }
      return;
    }

    if (pivoting_) {
      SolveWithGaussianEliminationWithPivoting(context, superdiag, diag,
                                               subdiag, rhs, x);
    } else {
      SolveWithThomasAlgorithm(context, superdiag, diag, subdiag, rhs, x);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TridiagonalSolveOp);

  // Adjust pivot such that neither 'rhs[i,:] / pivot' nor '1 / pivot' cause
  // overflow, where i numerates the multiple right-hand-sides. During the
  // back-substitution phase in
  // SolveWithGaussianEliminationWithPivotingAndPerturbSingular, we compute
  // the i'th row of the solution as rhs[i,:] * (1 / pivot). This logic is
  // extracted from the LAPACK routine xLAGTS.
  void MaybePerturbPivot(RealScalar perturb, Scalar& pivot,
                         Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& rhs_row) {
    constexpr RealScalar one(1);
    // The following logic is extracted from xLAMCH in LAPACK.
    constexpr RealScalar tiny = std::numeric_limits<RealScalar>::min();
    constexpr RealScalar small = one / std::numeric_limits<RealScalar>::max();
    constexpr RealScalar safemin =
        (small < tiny
             ? tiny
             : (one + std::numeric_limits<RealScalar>::epsilon()) * safemin);
    constexpr RealScalar bignum = one / safemin;

    RealScalar abs_pivot = std::abs(pivot);
    if (abs_pivot >= one) {
      return;
    }
    // Safeguard against infinite loop if 'perturb' is zero.
    // 'perturb' should never have magnitude smaller than safemin.
    perturb = std::max(std::abs(perturb), safemin);
    // Make sure perturb and pivot have the same sign.
    perturb = std::copysign(perturb, std::real(pivot));

    bool stop = false;
    const RealScalar max_factor = rhs_row.array().abs().maxCoeff();
    while (abs_pivot < one && !stop) {
      if (abs_pivot < safemin) {
        if (abs_pivot == 0 || max_factor * safemin > abs_pivot) {
          pivot += perturb;
          perturb *= 2;
        } else {
          pivot *= bignum;
          rhs_row *= bignum;
          stop = true;
        }
      } else if (max_factor > abs_pivot * bignum) {
        pivot += perturb;
        perturb *= 2;
      } else {
        stop = true;
      }
      abs_pivot = std::abs(pivot);
    }
  }

  // This function roughly follows LAPACK's xLAGTF + xLAGTS routines.
  //
  // It computes the solution to the a linear system with multiple
  // right-hand sides
  //     T * X = RHS
  // where T is a tridiagonal matrix using a row-pivoted LU decomposition.

  // This routine differs from SolveWithGaussianEliminationWithPivoting by
  // allowing the tridiagonal matrix to be numerically singular.
  // If tiny diagonal elements of U are encountered, signaling that T is
  // numerically singular, the diagonal elements are perturbed by
  // an amount proportional to eps*max_abs_u to avoid overflow, where
  // max_abs_u is max_{i,j} | U(i,j) |. This is useful when using this
  // routine for computing eigenvectors of a matrix T' via inverse
  // iteration by solving the singular system
  //   (T' - lambda*I) X = RHS,
  // where lambda is an eigenvalue of T'.
  //
  // By fusing the factorization and solution, we avoid storing L
  // and pivoting information, and the forward solve is done on-the-fly
  // during factorization, instead of requiring a separate loop.
  void SolveWithGaussianEliminationWithPivotingAndPerturbSingular(
      OpKernelContext* context, const MatrixMapRow& superdiag,
      const MatrixMapRow& diag, const MatrixMapRow& subdiag,
      const ConstMatrixMap& rhs, MatrixMap& x) {
    constexpr Scalar zero(0);
    constexpr RealScalar realzero(0);
    constexpr Scalar one(1);
    constexpr RealScalar eps = std::numeric_limits<RealScalar>::epsilon();

    const int n = diag.size();
    if (n == 0) return;
    if (n == 1) {
      Scalar denom = diag(0);
      RealScalar tol = eps * std::abs(denom);
      Eigen::Matrix<Scalar, 1, Eigen::Dynamic> row = rhs.row(0);
      MaybePerturbPivot(tol, denom, row);
      x = row * (one / denom);
      return;
    }

    // The three columns in u are the diagonal, superdiagonal, and second
    // superdiagonal, respectively, of the U matrix in the LU decomposition
    // of the input matrix (subject to row exchanges due to pivoting). For
    // a pivoted tridiagonal matrix, the U matrix has at most two non-zero
    // superdiagonals.
    Eigen::Array<Scalar, Eigen::Dynamic, 3> u(n, 3);

    // We accumulate max( abs( U(i,j) ) ) in max_abs_u for use in perturbing
    // near-zero pivots during the solution phase.
    u(0, 0) = diag(0);
    u(0, 1) = superdiag(0);
    RealScalar max_abs_u = std::max(std::abs(u(0, 0)), std::abs(u(0, 1)));
    RealScalar scale1 = std::abs(u(0, 0)) + std::abs(u(0, 1));
    x.row(0) = rhs.row(0);
    for (int k = 0; k < n - 1; ++k) {
      // The non-zeros in the (k+1)-st row are
      //    [ ... subdiag(k+1) (diag(k+1)-shift) superdiag(k+1) ... ]
      u(k + 1, 0) = diag(k + 1);
      RealScalar scale2 = std::abs(subdiag(k + 1)) + std::abs(u(k + 1, 0));
      if (k < n - 2) scale2 += std::abs(superdiag(k + 1));
      if (subdiag(k + 1) == zero) {
        // The sub-diagonal in the k+1 row is already zero. Move to the next
        // row.
        scale1 = scale2;
        u(k + 1, 1) = superdiag(k + 1);
        u(k, 2) = zero;
        x.row(k + 1) = rhs.row(k + 1);
      } else {
        const RealScalar piv1 =
            u(k, 0) == zero ? realzero : std::abs(u(k, 0)) / scale1;
        const RealScalar piv2 = std::abs(subdiag(k + 1)) / scale2;
        if (piv2 <= piv1) {
          // No row pivoting needed.
          scale1 = scale2;
          Scalar factor = subdiag(k + 1) / u(k, 0);
          u(k + 1, 0) = diag(k + 1) - factor * u(k, 1);
          u(k + 1, 1) = superdiag(k + 1);
          u(k, 2) = zero;
          x.row(k + 1) = rhs.row(k + 1) - factor * x.row(k);
        } else {
          // Swap rows k and k+1.
          Scalar factor = u(k, 0) / subdiag(k + 1);
          u(k, 0) = subdiag(k + 1);
          u(k + 1, 0) = u(k, 1) - factor * diag(k + 1);
          u(k, 1) = diag(k + 1);
          if (k < n - 2) {
            u(k, 2) = superdiag(k + 1);
            u(k + 1, 1) = -factor * superdiag(k + 1);
          }
          x.row(k + 1) = x.row(k) - factor * rhs.row(k + 1);
          x.row(k) = rhs.row(k + 1);
        }
      }
      if (k < n - 2) {
        for (int i = 0; i < 3; ++i) {
          max_abs_u = std::max(max_abs_u, std::abs(u(k, i)));
        }
      }
    }
    max_abs_u = std::max(max_abs_u, std::abs(u(n - 1, 0)));

    // We have already solved L z = P rhs above. Now we solve U x = z,
    // possibly perturbing small pivots to avoid overflow. The variable tol
    // contains eps * max( abs( u(:,:) ) ). If tiny pivots are encoutered,
    // they are perturbed by a small amount on the scale of tol to avoid
    // overflow or scaled up to avoid underflow.
    RealScalar tol = eps * max_abs_u;
    Scalar denom = u(n - 1, 0);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> row = x.row(n - 1);
    MaybePerturbPivot(tol, denom, row);
    x.row(n - 1) = row * (one / denom);
    if (n > 1) {
      denom = u(n - 2, 0);
      row = x.row(n - 2) - u(n - 2, 1) * x.row(n - 1);
      MaybePerturbPivot(std::copysign(tol, std::real(denom)), denom, row);
      x.row(n - 2) = row * (one / denom);

      for (int k = n - 3; k >= 0; --k) {
        row = x.row(k) - u(k, 1) * x.row(k + 1) - u(k, 2) * x.row(k + 2);
        denom = u(k, 0);
        MaybePerturbPivot(std::copysign(tol, std::real(denom)), denom, row);
        x.row(k) = row * (one / denom);
      }
    }
  }

  void SolveWithGaussianEliminationWithPivoting(OpKernelContext* context,
                                                const MatrixMapRow& superdiag,
                                                const MatrixMapRow& diag,
                                                const MatrixMapRow& subdiag,
                                                const ConstMatrixMap& rhs,
                                                MatrixMap& x) {
    const int n = diag.size();
    const Scalar zero(0);

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
        if (u(i) == zero) {
          LOG(WARNING) << kNotInvertibleMsg;
          x.fill(std::numeric_limits<Scalar>::quiet_NaN());
          return;
        }
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
    if (u(n - 1, 0) == zero) {
      LOG(WARNING) << kNotInvertibleMsg;
      x.fill(std::numeric_limits<Scalar>::quiet_NaN());
      return;
    }
    x.row(n - 1) /= u(n - 1, 0);
    x.row(n - 2) = (x.row(n - 2) - u(n - 2, 1) * x.row(n - 1)) / u(n - 2, 0);
    for (int i = n - 3; i >= 0; --i) {
      x.row(i) = (x.row(i) - u(i, 1) * x.row(i + 1) - u(i, 2) * x.row(i + 2)) /
                 u(i, 0);
    }
  }

  void SolveWithThomasAlgorithm(OpKernelContext* context,
                                const MatrixMapRow& superdiag,
                                const MatrixMapRow& diag,
                                const MatrixMapRow& subdiag,
                                const ConstMatrixMap& rhs, MatrixMap& x) {
    const int n = diag.size();
    const Scalar zero(0);

    // The superdiagonal of the U matrix in the LU decomposition of the input
    // matrix (in Thomas algorithm, the U matrix has ones on the diagonal and
    // one superdiagonal).
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u(n);

    if (diag(0) == zero) {
      LOG(WARNING) << kThomasFailedMsg;
      x.fill(std::numeric_limits<Scalar>::quiet_NaN());
      return;
    }

    u(0) = superdiag(0) / diag(0);
    x.row(0) = rhs.row(0) / diag(0);
    for (int i = 1; i < n; ++i) {
      auto denom = diag(i) - subdiag(i) * u(i - 1);
      if (denom == zero) {
        LOG(WARNING) << kThomasFailedMsg;
        x.fill(std::numeric_limits<Scalar>::quiet_NaN());
        return;
      }
      u(i) = superdiag(i) / denom;
      x.row(i) = (rhs.row(i) - subdiag(i) * x.row(i - 1)) / denom;
    }
    for (int i = n - 2; i >= 0; --i) {
      x.row(i) -= u(i) * x.row(i + 1);
    }
  }

  bool pivoting_;
  bool perturb_singular_;
};

REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<float>), float);
REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<double>),
                       double);
REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<complex64>),
                       complex64);
REGISTER_LINALG_OP_CPU("TridiagonalSolve", (TridiagonalSolveOp<complex128>),
                       complex128);
}  // namespace tensorflow
