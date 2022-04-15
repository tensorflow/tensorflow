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

Status CheckSecondToLastDimension(const Shape& op_shape, int64_t rank,
                                  int64_t expected,
                                  const std::string& op_name) {
  const auto actual_num_dims = ShapeUtil::GetDimension(op_shape, rank - 2);

  if (actual_num_dims != expected) {
    return InvalidArgument(
        "Second to last dimension of %s should be %d but is %d.", op_name,
        expected, actual_num_dims);
  }

  return Status::OK();
}

StatusOr<int64_t> CheckSystemAndReturnNumEquations(XlaOp lower_diagonal,
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

XlaOp Coefficient(XlaOp operand, int32_t i) {
  return DynamicSliceInMinorDims(operand,
                                 /*starts=*/{ConstantR0(operand.builder(), i)},
                                 /*sizes=*/{1});
}

XlaOp Coefficient(XlaOp operand, XlaOp i) {
  return DynamicSliceInMinorDims(operand,
                                 /*starts=*/{i}, /*sizes=*/{1});
}

XlaOp UpdateEq(XlaOp updated, int32_t i, XlaOp update) {
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

  TF_ASSIGN_OR_RETURN(int64_t num_eqs,
                      CheckSystemAndReturnNumEquations(
                          lower_diagonal, main_diagonal, upper_diagonal, rhs));

  XlaOp main_diag_after_elimination = ZerosLike(main_diagonal);
  XlaOp rhs_after_elimination = ZerosLike(rhs);
  XlaOp upper_diagonal_coeffs = ZerosLike(upper_diagonal);
  XlaOp x_coeffs = ZerosLike(rhs);

  // main_diag_after_elimination[:, 0] = main_diagonal[:, 0];
  main_diag_after_elimination =
      UpdateEq(main_diag_after_elimination, 0, Coefficient(main_diagonal, 0));

  // rhs_after_elimination[:, 0] = rhs[:, 0];
  rhs_after_elimination =
      UpdateEq(rhs_after_elimination, 0, Coefficient(rhs, 0));

  auto preparation_body_fn =
      [](XlaOp i, absl::Span<const XlaOp> values,
         XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    auto upper_diagonal_coeffs = values[0];
    auto upper_diagonal = values[1];
    // upper_diagonal_coeffs[:, i] = upper_diagonal[:, i];
    upper_diagonal_coeffs =
        UpdateEq(upper_diagonal_coeffs, i, Coefficient(upper_diagonal, i));
    return std::vector<XlaOp>{upper_diagonal_coeffs, upper_diagonal};
  };
  TF_ASSIGN_OR_RETURN(auto values_after_preparation,
                      ForEachIndex(num_eqs - 1, S32, preparation_body_fn,
                                   {upper_diagonal_coeffs, upper_diagonal},
                                   "preparation", builder));
  upper_diagonal_coeffs = values_after_preparation[0];

  // Forward transformation.
  auto forward_transformation_fn =
      [](XlaOp i_minus_one, absl::Span<const XlaOp> values,
         XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    auto lower_diagonal = values[0];
    auto main_diagonal = values[1];
    auto rhs = values[2];
    auto main_diag_after_elimination = values[3];
    auto upper_diagonal_coeffs = values[4];
    auto rhs_after_elimination = values[5];

    auto one = ScalarLike(i_minus_one, 1);
    auto i = i_minus_one + one;
    auto lower_diagonal_i = Coefficient(lower_diagonal, i);
    auto main_diagonal_i = Coefficient(main_diagonal, i);
    auto rhs_i = Coefficient(rhs, i);

    auto w_i =
        lower_diagonal_i / Coefficient(main_diag_after_elimination, i - one);

    // main_diag_after_elimination[:, i] =
    //     main_diagonal_i - w_i * upper_diagonal_coeffs[:, i - 1];
    main_diag_after_elimination = UpdateEq(
        main_diag_after_elimination, i,
        main_diagonal_i - w_i * Coefficient(upper_diagonal_coeffs, i - one));
    // rhs_after_elimination[:, i] =
    //     rhs_i - w_i * rhs_after_elimination[:, i - 1];
    rhs_after_elimination =
        UpdateEq(rhs_after_elimination, i,
                 rhs_i - w_i * Coefficient(rhs_after_elimination, i - one));

    return std::vector<XlaOp>{lower_diagonal,
                              main_diagonal,
                              rhs,
                              main_diag_after_elimination,
                              upper_diagonal_coeffs,
                              rhs_after_elimination};
  };
  TF_ASSIGN_OR_RETURN(
      auto values_after_fwd_transformation,
      ForEachIndex(
          num_eqs - 1, S32, forward_transformation_fn,
          {lower_diagonal, main_diagonal, rhs, main_diag_after_elimination,
           upper_diagonal_coeffs, rhs_after_elimination},
          "forward_transformation", builder));
  lower_diagonal = values_after_fwd_transformation[0];
  main_diagonal = values_after_fwd_transformation[1];
  rhs = values_after_fwd_transformation[2];
  main_diag_after_elimination = values_after_fwd_transformation[3];
  upper_diagonal_coeffs = values_after_fwd_transformation[4];
  rhs_after_elimination = values_after_fwd_transformation[5];

  // Backward reduction.
  // x_coeffs[:, num_eqs - 1] = rhs_after_elimination[:, num_eqs - 1] /
  //                              main_diag_after_elimination[:, num_eqs - 1];
  x_coeffs =
      UpdateEq(x_coeffs, num_eqs - 1,
               Coefficient(rhs_after_elimination, num_eqs - 1) /
                   Coefficient(main_diag_after_elimination, num_eqs - 1));
  auto bwd_reduction_fn =
      [num_eqs](XlaOp j, absl::Span<const XlaOp> values,
                XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    auto x_coeffs = values[0];
    auto rhs_after_elimination = values[1];
    auto upper_diagonal_coeffs = values[2];
    auto main_diag_after_elimination = values[3];
    auto n = ScalarLike(j, num_eqs - 2);
    auto one = ScalarLike(j, 1);
    auto i = n - j;
    // for (int i = num_eqs - 2; i >= 0; i--)
    //   x_coeffs[:, i] = (rhs_after_elimination[:, i] -
    //     upper_diagonal_coeffs[:, i] * x_coeffs[:, i + 1]) /
    //       main_diag_after_elimination[:, i];
    x_coeffs = UpdateEq(x_coeffs, i,
                        (Coefficient(rhs_after_elimination, i) -
                         Coefficient(upper_diagonal_coeffs, i) *
                             Coefficient(x_coeffs, i + one)) /
                            Coefficient(main_diag_after_elimination, i));
    return std::vector<XlaOp>{x_coeffs, rhs_after_elimination,
                              upper_diagonal_coeffs,
                              main_diag_after_elimination};
  };

  TF_ASSIGN_OR_RETURN(
      auto values_after_bwd_reduction,
      ForEachIndex(num_eqs - 1, S32, bwd_reduction_fn,
                   {x_coeffs, rhs_after_elimination, upper_diagonal_coeffs,
                    main_diag_after_elimination},
                   "backward_reduction", builder));
  x_coeffs = values_after_bwd_reduction[0];

  return x_coeffs;
}

}  // namespace

StatusOr<XlaOp> TridiagonalSolver(SolverAlgorithm algo, XlaOp lower_diagonal,
                                  XlaOp main_diagonal, XlaOp upper_diagonal,
                                  XlaOp rhs) {
  switch (algo) {
    case kThomas:
      return TridiagonalSolverImpl<kThomas>(lower_diagonal, main_diagonal,
                                            upper_diagonal, rhs);
    default:
      return Unimplemented(
          "Only algorithm kThomas (%d) is implemented, got: %d",
          static_cast<int>(kThomas), algo);
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
  const int64_t rank = diagonals_shape.rank();

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
  std::vector<int64_t> transpose_order(rank);
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
    default:
      return Unimplemented(
          "Only algorithm kThomas (%d) is implemented, got: %d",
          static_cast<int>(kThomas), algo);
  }
}

}  // namespace tridiagonal
}  // namespace xla
