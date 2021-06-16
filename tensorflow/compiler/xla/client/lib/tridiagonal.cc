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

#include "tensorflow/compiler/xla/client/lib/tridiagonal.h"

#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace tridiagonal {

namespace {

Status CheckSecondToLastDimension(const Shape& op_shape, int64 rank,
                                  int64 expected, const std::string& op_name) {
  const auto actual_num_dims = ShapeUtil::GetDimension(op_shape, rank - 2);

  if (actual_num_dims != expected) {
    return InvalidArgument(
        "Second to last dimension of %s should be %d but is %d.", op_name,
        expected, actual_num_dims);
  }

  return Status::OK();
}

StatusOr<int64> CheckSystemAndReturnNumEquations(XlaOp lower_diagonal,
                                                 XlaOp main_diagonal,
                                                 XlaOp upper_diagonal,
                                                 XlaOp rhs) {
  XlaBuilder* builder = lower_diagonal.builder();

  TF_ASSIGN_OR_RETURN(Shape lower_diagonal_shape,
                      builder->GetShape(lower_diagonal));
  TF_ASSIGN_OR_RETURN(Shape main_diagonal_shape,
                      builder->GetShape(main_diagonal));
  TF_ASSIGN_OR_RETURN(Shape upper_diagonal_shape,
                      builder->GetShape(upper_diagonal));
  TF_ASSIGN_OR_RETURN(Shape rhs_shape, builder->GetShape(rhs));

  const auto lower_diagonal_rank = lower_diagonal_shape.rank();
  const auto main_diagonal_rank = main_diagonal_shape.rank();
  const auto upper_diagonal_rank = upper_diagonal_shape.rank();
  const auto rhs_rank = rhs_shape.rank();
  if (!((lower_diagonal_rank == main_diagonal_rank) &&
        (lower_diagonal_rank == upper_diagonal_rank) &&
        (lower_diagonal_rank == rhs_rank))) {
    return InvalidArgument(
        "All inputs should have the same rank but got rank "
        "%d for lower diagonal, %d for diagonal, %d for upper diagonal, "
        "%d for rhs",
        lower_diagonal_rank, main_diagonal_rank, upper_diagonal_rank, rhs_rank);
  }
  const auto rank = lower_diagonal_rank;
  if (rank < 2) {
    return InvalidArgument("Arguments must have rank >=2; got rank %d.", rank);
  }

  const auto lower_diagonal_num_eqs =
      ShapeUtil::GetDimension(lower_diagonal_shape, rank - 1);
  const auto main_diagonal_num_eqs =
      ShapeUtil::GetDimension(main_diagonal_shape, rank - 1);
  const auto upper_diagonal_num_eqs =
      ShapeUtil::GetDimension(upper_diagonal_shape, rank - 1);
  const auto rhs_num_eqs = ShapeUtil::GetDimension(rhs_shape, rank - 1);
  if (!((lower_diagonal_num_eqs == main_diagonal_num_eqs) &&
        (lower_diagonal_num_eqs == upper_diagonal_num_eqs) &&
        (lower_diagonal_num_eqs == rhs_num_eqs))) {
    return InvalidArgument(
        "All inputs should have the same innermost dimension but got "
        "%d for lower diagonal, %d for diagonal, %d for upper diagonal, "
        "%d for rhs",
        lower_diagonal_num_eqs, main_diagonal_num_eqs, upper_diagonal_num_eqs,
        rhs_num_eqs);
  }
  const auto num_equations = lower_diagonal_num_eqs;

  TF_RETURN_IF_ERROR(CheckSecondToLastDimension(lower_diagonal_shape, rank, 1,
                                                "lower diagonal"));
  TF_RETURN_IF_ERROR(
      CheckSecondToLastDimension(main_diagonal_shape, rank, 1, "diagonal"));
  TF_RETURN_IF_ERROR(CheckSecondToLastDimension(upper_diagonal_shape, rank, 1,
                                                "upper diagonal"));

  return num_equations;
}

XlaOp Coefficient(XlaOp operand, int32 i) {
  return DynamicSliceInMinorDims(operand,
                                 /*starts=*/{ConstantR0(operand.builder(), i)},
                                 /*sizes=*/{1});
}

XlaOp Coefficient(XlaOp operand, XlaOp i) {
  return DynamicSliceInMinorDims(operand,
                                 /*starts=*/{i}, /*sizes=*/{1});
}

XlaOp UpdateEq(XlaOp updated, int32 i, XlaOp update) {
  return DynamicUpdateSliceInMinorDims(
      updated, update, /*starts=*/{ConstantR0(updated.builder(), i)});
}

XlaOp UpdateEq(XlaOp updated, XlaOp i, XlaOp update) {
  return DynamicUpdateSliceInMinorDims(updated, update, /*starts=*/{i});
}

template <SolverAlgorithm algo>
StatusOr<XlaOp> TridiagonalSolverImpl(XlaOp lower_diagonal, XlaOp main_diagonal,
                                      XlaOp upper_diagonal, XlaOp rhs);

// Applies Thomas algorithm to solve a linear system where the linear operand
// is a tri-diagonal matrix.
// See https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm for a simple
// reference on the Thomas algorithm.
// It is expected that the three diagonals are represented as tensors of shape
// [..., 1, num_equations] where num_equations is the number of dimensions of
// the unknowns considered in the linear systems.
// The first innermost dimension of `lower_diagonal` (`lower_diagonal[..., :,
// 0]`) will be ignored. The last innermost dimension of `upper_diagonal`
// (`upper_diagonal[..., :, num_equations - 1]`) will be ignored. The shape of
// the right-hand-side `rhs` should be [..., num_rhs, num_equations]. The
// solution will have the shape [..., num_rhs, num_equations].
template <>
StatusOr<XlaOp> TridiagonalSolverImpl<kThomas>(XlaOp lower_diagonal,
                                               XlaOp main_diagonal,
                                               XlaOp upper_diagonal,
                                               XlaOp rhs) {
  XlaBuilder* builder = lower_diagonal.builder();
  TF_ASSIGN_OR_RETURN(int64 num_eqs,
                      CheckSystemAndReturnNumEquations(
                          lower_diagonal, main_diagonal, upper_diagonal, rhs));

  // x = rhs
  // x[:,0] = x[:,0] / main_diagoal[:, 0]
  auto x =
      UpdateEq(rhs, 0, Coefficient(rhs, 0) / Coefficient(main_diagonal, 0));

  // main_diagonal[:,0] = upper_diagonal[:,0] / main_diagoal[:, 0]
  main_diagonal =
      UpdateEq(main_diagonal, 0,
               Coefficient(upper_diagonal, 0) / Coefficient(main_diagonal, 0));

  auto forward_transformation_fn =
      [](XlaOp i_minus_one, absl::Span<const XlaOp> values,
         XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    auto lower_diagonal = values[0];
    auto main_diagonal = values[1];
    auto upper_diagonal = values[2];
    auto x = values[3];

    auto one = ScalarLike(i_minus_one, 1);
    auto i = i_minus_one + one;
    // denom = main_diagonal[:, i] -
    //          lower_diagonal[:, i] * main_diagonal[:, i - 1]
    auto denom = Coefficient(main_diagonal, i) -
                 Coefficient(lower_diagonal, i) *
                     Coefficient(main_diagonal, i_minus_one);

    // x[:, i] -= x[:, i - 1] / denom;
    x = UpdateEq(x, i,
                 (Coefficient(x, i) - Coefficient(lower_diagonal, i) *
                                          Coefficient(x, i_minus_one)) /
                     denom);

    // main_diagonal[:, i] = upper_diagonal[:, i] / denom;
    main_diagonal =
        UpdateEq(main_diagonal, i, Coefficient(upper_diagonal, i) / denom);

    return std::vector<XlaOp>{lower_diagonal, main_diagonal, upper_diagonal, x};
  };
  TF_ASSIGN_OR_RETURN(
      auto values_after_fwd_transformation,
      ForEachIndex(num_eqs - 1, S32, forward_transformation_fn,
                   {lower_diagonal, main_diagonal, upper_diagonal, x},
                   "forward_transformation", builder));

  main_diagonal = values_after_fwd_transformation[1];
  x = values_after_fwd_transformation[3];

  // Backward reduction.
  auto bwd_reduction_fn =
      [num_eqs](XlaOp j, absl::Span<const XlaOp> values,
                XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    auto x = values[0];
    auto main_diagonal = values[1];
    auto n = ScalarLike(j, num_eqs - 2);
    auto one = ScalarLike(j, 1);
    auto i = n - j;
    // for (int i = num_eqs - 2; i >= 0; i--)
    //   x[:, i] -=  main_diagonal[:, i] * x[:, i + 1])
    x = UpdateEq(x, i,
                 (Coefficient(x, i) -
                  Coefficient(main_diagonal, i) * Coefficient(x, i + one)));
    return std::vector<XlaOp>{x, main_diagonal};
  };

  TF_ASSIGN_OR_RETURN(
      auto values_after_bwd_reduction,
      ForEachIndex(num_eqs - 1, S32, bwd_reduction_fn, {x, main_diagonal},
                   "backward_reduction", builder));
  return values_after_bwd_reduction[0];
}

// Applies partially pivoted Gaussian elimination to solve a tridiagonal linear
// system. It is expected that the three diagonals are represented as tensors of
// shape [..., 1, num_equations] where num_equations is the number of dimensions
// of the unknowns considered in the linear systems. The first innermost
// dimension of `lower_diagonal` (`lower_diagonal[..., :, 0]`) will be ignored.
// The last innermost dimension of `upper_diagonal`
// (`upper_diagonal[..., :, num_equations - 1]`) will be ignored. The shape of
// the right-hand-side `rhs` should be [..., num_rhs, num_equations]. The
// solution will have the shape [..., num_rhs, num_equations].
template <>
StatusOr<XlaOp> TridiagonalSolverImpl<kPartialPivoting>(XlaOp lower_diagonal,
                                                        XlaOp main_diagonal,
                                                        XlaOp upper_diagonal,
                                                        XlaOp rhs) {
  XlaBuilder* builder = lower_diagonal.builder();
  TF_ASSIGN_OR_RETURN(
      int64 n, CheckSystemAndReturnNumEquations(lower_diagonal, main_diagonal,
                                                upper_diagonal, rhs));
  if (n == 0) return rhs;
  // Create the rhs_pivot_mask_shape [..., num_rhs, 1] to which we need to
  // broadcast the pivot mask of shape [..., 1, 1] created in each step during
  // the forward transformation.
  TF_ASSIGN_OR_RETURN(Shape rhs_shape, builder->GetShape(rhs));
  const int64 rhs_rank = rhs_shape.rank();
  std::vector<int64> rhs_pivot_mask_shape(rhs_shape.dimensions().begin(),
                                          rhs_shape.dimensions().end());
  rhs_pivot_mask_shape[rhs_rank - 1] = 1;
  std::vector<int64> dims_to_broadcast(rhs_rank);
  std::iota(dims_to_broadcast.begin(), dims_to_broadcast.end(),
            static_cast<int64>(0));

  auto forward_transformation_fn =
      [&rhs_pivot_mask_shape, &dims_to_broadcast](
          XlaOp i, absl::Span<const XlaOp> values, XlaBuilder* builder,
          bool is_penultimate_row = false) -> StatusOr<std::vector<XlaOp>> {
    auto lower_diagonal = values[0];
    auto main_diagonal = values[1];
    auto upper_diagonal = values[2];
    auto rhs = values[3];

    auto one = ScalarLike(i, 1);

    // Values referenced in the i'th step:
    // rhs[:, i]
    auto rhs_i = Coefficient(rhs, i);
    // rhs[:, i+1]
    auto rhs_ip1 = Coefficient(rhs, i + one);
    // diagonal[:, i]
    auto diag_i = Coefficient(main_diagonal, i);
    // diagonal[:, i+1]
    auto diag_ip1 = Coefficient(main_diagonal, i + one);
    // upper_diagonal[:, i]
    auto upper_diag_i = Coefficient(upper_diagonal, i);
    // upper_diagonal[:, i + 1]
    auto upper_diag_ip1 = Coefficient(upper_diagonal, i + one);
    // lower_diagonal[:, i + 1]
    auto lower_diag_ip1 = Coefficient(lower_diagonal, i + one);
    // Pivoting mask. We implicitly interchange rows i and i + 1
    // for values of j where |lower_diag[j, i + 1]| > |diag[j, i]|.
    auto pivot_mask = Gt(Abs(lower_diag_ip1), Abs(diag_i));
    auto pivot_mask_rhs =
        BroadcastInDim(pivot_mask, rhs_pivot_mask_shape, dims_to_broadcast);

    // Notice: The second upper diagonal of the matrix U in the
    // partially pivoted LU decomposition overwrites lower_diagonal.

    // Updated values for the non-pivoted case:
    auto factor_no_pivot = lower_diag_ip1 / diag_i;
    auto diag_i_no_pivot = diag_i;
    auto diag_ip1_no_pivot = diag_ip1 - factor_no_pivot * upper_diag_i;
    auto upper_diag_i_no_pivot = upper_diag_i;
    auto upper_diag_ip1_no_pivot = upper_diag_ip1;
    auto lower_diag_i_no_pivot = ZerosLike(upper_diag_ip1);
    auto rhs_i_no_pivot = rhs_i;
    auto rhs_ip1_no_pivot = rhs_ip1 - factor_no_pivot * rhs_i;

    // Updated values for the pivoted case:
    auto factor_pivot = diag_i / lower_diag_ip1;
    auto diag_i_pivot = lower_diag_ip1;
    auto diag_ip1_pivot = upper_diag_i - factor_pivot * diag_ip1;
    auto upper_diag_i_pivot = diag_ip1;
    auto upper_diag_ip1_pivot = -factor_pivot * upper_diag_ip1;
    auto lower_diag_i_pivot = upper_diag_ip1;
    auto rhs_i_pivot = rhs_ip1;
    auto rhs_ip1_pivot = rhs_i - factor_pivot * rhs_ip1;

    // Write the updated values. For each batch dimension (outer batch x rhs),
    // select either the pivoted or non-pivoted values, depending on the
    // corresponding value in pivot_mask.
    main_diagonal = UpdateEq(main_diagonal, i,
                             Select(pivot_mask, diag_i_pivot, diag_i_no_pivot));
    main_diagonal =
        UpdateEq(main_diagonal, i + one,
                 Select(pivot_mask, diag_ip1_pivot, diag_ip1_no_pivot));
    upper_diagonal =
        UpdateEq(upper_diagonal, i,
                 Select(pivot_mask, upper_diag_i_pivot, upper_diag_i_no_pivot));
    rhs = UpdateEq(rhs, i, Select(pivot_mask_rhs, rhs_i_pivot, rhs_i_no_pivot));
    rhs = UpdateEq(rhs, i + one,
                   Select(pivot_mask_rhs, rhs_ip1_pivot, rhs_ip1_no_pivot));

    if (!is_penultimate_row) {
      upper_diagonal = UpdateEq(
          upper_diagonal, i + one,
          Select(pivot_mask, upper_diag_ip1_pivot, upper_diag_ip1_no_pivot));
      lower_diagonal = UpdateEq(
          lower_diagonal, i,
          Select(pivot_mask, lower_diag_i_pivot, lower_diag_i_no_pivot));
    }

    return std::vector<XlaOp>{lower_diagonal, main_diagonal, upper_diagonal,
                              rhs};
  };
  TF_ASSIGN_OR_RETURN(
      auto values_after_fwd_transformation,
      ForEachIndex(
          n - 2, S32,
          std::bind(forward_transformation_fn, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, false),
          {lower_diagonal, main_diagonal, upper_diagonal, rhs},
          "forward_transformation", builder));

  // Transform the final rows.
  lower_diagonal = values_after_fwd_transformation[0];
  main_diagonal = values_after_fwd_transformation[1];
  upper_diagonal = values_after_fwd_transformation[2];
  rhs = values_after_fwd_transformation[3];
  auto n_minus_2 = ConstantR0<int32>(builder, n - 2);
  TF_ASSIGN_OR_RETURN(
      values_after_fwd_transformation,
      forward_transformation_fn(
          n_minus_2, {lower_diagonal, main_diagonal, upper_diagonal, rhs},
          builder, true));

  // Get the diagonals of the U matrix constructed during the partially pivoted
  // LU decomposition.
  auto second_upper_diagonal = values_after_fwd_transformation[0];
  main_diagonal = values_after_fwd_transformation[1];
  upper_diagonal = values_after_fwd_transformation[2];
  auto x = values_after_fwd_transformation[3];

  // Backward reduction.
  x = UpdateEq(x, n - 1,
               Coefficient(x, n - 1) / Coefficient(main_diagonal, n - 1));
  if (n < 2) return x;
  x = UpdateEq(x, n - 2,
               (Coefficient(x, n - 2) -
                Coefficient(upper_diagonal, n - 2) * Coefficient(x, n - 1)) /
                   Coefficient(main_diagonal, n - 2));
  auto bwd_reduction_fn =
      [n](XlaOp j, absl::Span<const XlaOp> values,
          XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    auto main_diagonal = values[0];
    auto upper_diagonal = values[1];
    auto second_upper_diagonal = values[2];
    auto x = values[3];

    auto one = ScalarLike(j, 1);
    auto two = ScalarLike(j, 2);
    auto i = ScalarLike(j, n - 3) - j;
    // for (int i = num_eqs - 3; i >= 0; i--)
    //   x[:, i] =  ( x[:, i] - upper_diagonal[:, i] * x[:, i + 1]) -
    //                second_upper_diagnal[:,i] * x[:, i + 2] ) /
    //                main_diagoal[:,i]
    x = UpdateEq(
        x, i,
        (Coefficient(x, i) -
         Coefficient(upper_diagonal, i) * Coefficient(x, i + one) -
         Coefficient(second_upper_diagonal, i) * Coefficient(x, i + two)) /
            Coefficient(main_diagonal, i));

    return std::vector<XlaOp>{main_diagonal, upper_diagonal,
                              second_upper_diagonal, x};
  };

  TF_ASSIGN_OR_RETURN(
      auto values_after_bwd_reduction,
      ForEachIndex(n - 2, S32, bwd_reduction_fn,
                   {main_diagonal, upper_diagonal, second_upper_diagonal, x},
                   "backward_reduction", builder));
  return values_after_bwd_reduction[3];
}

}  // namespace

StatusOr<XlaOp> TridiagonalSolver(SolverAlgorithm algo, XlaOp lower_diagonal,
                                  XlaOp main_diagonal, XlaOp upper_diagonal,
                                  XlaOp rhs) {
  switch (algo) {
    case kThomas:
      return TridiagonalSolverImpl<kThomas>(lower_diagonal, main_diagonal,
                                            upper_diagonal, rhs);
    case kPartialPivoting:
      return TridiagonalSolverImpl<kPartialPivoting>(
          lower_diagonal, main_diagonal, upper_diagonal, rhs);
    default:
      return Unimplemented("Got unknown algorithm: %d", algo);
  }
}

// Solves a linear system where the linear operand is a tri-diagonal matrix.
// It is expected that the tree diagonals are stacked into a tensors of shape
// [..., 3, num_equations] where num_equations is the number of spatial
// dimensions considered in the system.
// diagonals[..., 0, :] represents the upper diagonal whose last inner
// dimension will be ignored.
// diagonals[..., 1, :] represents the main diagonal.
// diagonals[..., 2, :] represents the lower diagonal whose first inner
// dimension will be ignored.
// The right-hand-side d is expected to have dimension
// [..., num_rhs, num_equations].
// The solution will have size [..., num_rhs, num_equations].
StatusOr<XlaOp> TridiagonalSolver(SolverAlgorithm algo, XlaOp diagonals,
                                  XlaOp rhs) {
  XlaBuilder* builder = diagonals.builder();
  TF_ASSIGN_OR_RETURN(Shape diagonals_shape, builder->GetShape(diagonals));
  const int64 rank = diagonals_shape.rank();

  auto upper_diagonal =
      SliceInDim(diagonals, /*start_index=*/0, /*limit_index=*/1,
                 /*stride=*/1, /*dimno=*/rank - 2);
  auto main_diagonal =
      SliceInDim(diagonals, /*start_index=*/1, /*limit_index=*/2,
                 /*stride=*/1, /*dimno=*/rank - 2);
  auto lower_diagonal =
      SliceInDim(diagonals, /*start_index=*/2, /*limit_index=*/3,
                 /*stride=*/1, /*dimno=*/rank - 2);

  // TODO(belletti): Get rid of the transposes here.
  std::vector<int64> transpose_order(rank);
  std::iota(transpose_order.begin(), transpose_order.end(), 0);
  transpose_order[rank - 2] = rank - 1;
  transpose_order[rank - 1] = rank - 2;
  // Swap the last two dimensions.
  rhs = Transpose(rhs, transpose_order);

  switch (algo) {
    case kThomas: {
      TF_ASSIGN_OR_RETURN(
          XlaOp x, TridiagonalSolverImpl<kThomas>(lower_diagonal, main_diagonal,
                                                  upper_diagonal, rhs));
      return Transpose(x, transpose_order);
    }
    case kPartialPivoting: {
      TF_ASSIGN_OR_RETURN(
          XlaOp x, TridiagonalSolverImpl<kPartialPivoting>(
                       lower_diagonal, main_diagonal, upper_diagonal, rhs));
      return Transpose(x, transpose_order);
    }
    default:
      return Unimplemented("Got unknown algorithm: %d", algo);
  }
}

}  // namespace tridiagonal
}  // namespace xla
