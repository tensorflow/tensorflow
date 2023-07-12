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

#include "tensorflow/compiler/xla/service/eigh_expander.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"

// Parallel two-sided Jacobi symmetric eigendecomposition.
//
// The implementation follows the approach described in:
// Brent, Richard P., and Franklin T. Luk. "The solution of singular-value and
// symmetric eigenvalue problems on multiprocessor arrays." SIAM Journal on
// Scientific and Statistical Computing 6.1 (1985): 69-84.
//
// Where the Brent/Luk paper uses "processors", we use "vector elements".
namespace xla {

namespace {

// A 2x2 symmetric Eigendecomposition of a matrix A.
// If
// G = [[ c, s],
//      [-s, c]]
// matmul(G_T, G) = I
// and
// G @ [[rt1, 0  ],  @ G.T = A
//      [  0, rt2]]
struct Eigh2x2 {
  // Eigenvalues
  XlaOp rt1;
  XlaOp rt2;
  // First row of Eigenvector matrix.
  XlaOp c;  // cosine.
  XlaOp s;  // sine.
};

// Given an n-by-n symmetric A and integers p and q that satisfy 0 <= p < q < n,
// a Jacobi rotation computes a rotation matrix G = [[c, s], [-s, c]], such that
//   G_T * A[[p, q], [p, q]] * G
// is diagonalized.
//
// In this parallel Jacobi algorithm, we simultaneously compute Jacobi rotations
// for all of the matrix diagonal elements at the same time. The matrix diagonal
// elements correspond to different rows and columns of the original matrix and
// their rotations do not interfere and hence can be computed in parallel.
//
// def sym_schur2x2(w_tl, w_tr, w_br):
//   a = np.diag(w_br)
//   b = np.diag(w_tr)
//   c = np.diag(w_tl)
//   tau = (a - c) / (2 * b)
//   t = np.where(tau >= 0, 1.0 / (tau + np.sqrt(1 + tau ** 2)),
//                -1.0 / (-tau + np.sqrt(1 + tau ** 2)))
//   fudge_factor = 0.1
//   b_is_tiny = np.abs(b) <= (fudge_factor*eps*
//     np.min(np.abs(a), np.abs(c)))
//   t = np.where(b_is_tiny, 0., t)
//   c = 1.0 / np.sqrt(1.0 + t ** 2)
//   s = t * c
//   rt1 = w_tl - t * w_tr
//   rt2 = w_br + t * w_tr
//   return rt1, rt2, c, s
StatusOr<Eigh2x2> HermitianEigenDecomposition2x2(XlaOp w_tl, XlaOp w_tr,
                                                 XlaOp w_br) {
  TF_ASSIGN_OR_RETURN(Shape w_tl_shape, w_tl.builder()->GetShape(w_tl));
  bool is_complex = primitive_util::IsComplexType(w_tl_shape.element_type());

  w_tl = GetMatrixDiagonal(Real(w_tl));
  w_tr = GetMatrixDiagonal(w_tr);
  w_br = GetMatrixDiagonal(Real(w_br));
  auto zero = ScalarLike(w_tl, 0.0);
  auto one = ScalarLike(w_tl, 1.0);
  auto two = ScalarLike(w_tl, 2.0);

  XlaOp w;
  if (is_complex) {
    auto abs_tr = Abs(w_tr);
    w = Select(Eq(abs_tr, ZerosLike(abs_tr)), FullLike(w_tr, 1),
               Conj(w_tr) / Complex(abs_tr, ZerosLike(abs_tr)));
    w_tr = abs_tr;
  }

  auto tau = (w_br - w_tl) / (two * w_tr);
  auto t = Sqrt(one + Square(tau));
  t = Reciprocal(tau + Select(Ge(tau, zero), t, Neg(t)));

  constexpr float kFudgeFactor = 0.1f;
  auto tiny =
      ScalarLike(w_tr, kFudgeFactor * std::numeric_limits<float>::epsilon());
  auto off_diag_is_tiny = Le(Abs(w_tr), Mul(tiny, Min(Abs(w_tl), Abs(w_br))));
  t = Select(off_diag_is_tiny, ZerosLike(t), t);
  auto c = Rsqrt(one + Square(t));
  auto s = t * c;

  auto rt1 = w_tl - t * w_tr;
  auto rt2 = w_br + t * w_tr;

  if (is_complex) {
    rt1 = Complex(rt1, ZerosLike(rt1));
    rt2 = Complex(rt2, ZerosLike(rt2));
    c = Complex(c, ZerosLike(c));
    s = Complex(s, ZerosLike(s)) * w;
  }

  return Eigh2x2{rt1, rt2, c, s};
}

// tl, tr, bl, br = (
//   tl * c[:, None] - bl * s[:, None],
//   tr * c[:, None] - br * s[:, None],
//   tl * s[:, None] + bl * c[:, None],
//   tr * s[:, None] + br * c[:, None],
// )
void ApplyJacobiRotationOverRows(Eigh2x2 rotation, XlaOp& tl, XlaOp& tr,
                                 XlaOp& bl, XlaOp& br) {
  Shape shape = tl.builder()->GetShape(tl).value();
  std::vector<int64_t> broadcast_dims(shape.dimensions().size() - 1);
  absl::c_iota(broadcast_dims, 0);
  auto c = BroadcastInDim(rotation.c, shape.dimensions(), broadcast_dims);
  auto s = BroadcastInDim(rotation.s, shape.dimensions(), broadcast_dims);

  auto s_conj = MaybeConjugate(s, true);
  std::tie(tl, tr, bl, br) =
      std::make_tuple(tl * c - bl * s_conj, tr * c - br * s_conj,
                      tl * s + bl * c, tr * s + br * c);
}

// tl, tr, bl, br = (
//   tl * c[None, :] - tr * s[None, :],
//   tl * s[None, :] + tr * c[None, :],
//   bl * c[None, :] - br * s[None, :],
//   bl * s[None, :] + br * c[None, :],
// )
void ApplyJacobiRotationOverCols(Eigh2x2 rotation, XlaOp& tl, XlaOp& tr,
                                 XlaOp& bl, XlaOp& br) {
  Shape shape = tl.builder()->GetShape(tl).value();
  std::vector<int64_t> broadcast_dims(shape.dimensions().size() - 1);
  absl::c_iota(broadcast_dims, 0);
  broadcast_dims.back() = shape.dimensions().size() - 1;
  auto c = BroadcastInDim(rotation.c, shape.dimensions(), broadcast_dims);
  auto s = BroadcastInDim(rotation.s, shape.dimensions(), broadcast_dims);

  auto s_conj = MaybeConjugate(s, true);
  std::tie(tl, tr, bl, br) =
      std::make_tuple(tl * c - tr * s, tl * s_conj + tr * c, bl * c - br * s,
                      bl * s_conj + br * c);
}

// def permute_rows_in_col(top, bottom):
//   top_out = np.zeros_like(l)
//   top_out[0] = top[0]
//   top_out[1] = bottom[0]
//   top_out[2:] = top[1:-1]
//   bottom_out = np.zeros_like(r)
//   bottom_out[:-1] = bottom[1:]
//   bottom_out[-1] = top[-1]
//   return top_out, bottom_out
void PermuteRowsInColumn(XlaOp& top, XlaOp& bottom) {
  XlaBuilder* builder = top.builder();
  Shape shape = builder->GetShape(top).value();
  int64_t k = ShapeUtil::GetDimension(shape, -1);
  if (k <= 1) {
    return;
  }
  int ndim = shape.dimensions_size();
  std::tie(top, bottom) =
      std::make_tuple(ConcatInDim(builder,
                                  {SliceInMinorDims(top, {0, 0}, {1, k}),
                                   SliceInMinorDims(bottom, {0, 0}, {1, k}),
                                   SliceInMinorDims(top, {1, 0}, {k - 1, k})},
                                  ndim - 2),
                      ConcatInDim(builder,
                                  {SliceInMinorDims(bottom, {1, 0}, {k, k}),
                                   SliceInMinorDims(top, {k - 1, 0}, {k, k})},
                                  ndim - 2));
}

void PermuteColumnsInRow(XlaOp& left, XlaOp& right) {
  XlaBuilder* builder = left.builder();
  Shape shape = builder->GetShape(left).value();
  int64_t k = ShapeUtil::GetDimension(shape, -1);
  if (k <= 1) {
    return;
  }
  int ndim = shape.dimensions_size();
  std::tie(left, right) =
      std::make_tuple(ConcatInDim(builder,
                                  {SliceInMinorDims(left, {0}, {1}),
                                   SliceInMinorDims(right, {0}, {1}),
                                   SliceInMinorDims(left, {1}, {k - 1})},
                                  ndim - 1),
                      ConcatInDim(builder,
                                  {SliceInMinorDims(right, {1}, {k}),
                                   SliceInMinorDims(left, {k - 1}, {k})},
                                  ndim - 1));
}

// Performs one round of parallel Jacobi rotations; n-1 rounds make a sweep.
// After each rotation, we permute the rows and columns of the quadrants of the
// matrix. The effect of the permutations is that all pairs of rows end up
// on the diagonal of the quadrants after n-1 rounds. The permutations are an
// implicit way of computing a tournament for n players such that each player
// plays every other player exactly once in n - 1 rounds. See the Brent/Luk
// paper for more details.
Status ApplyRotations(int64_t n, XlaOp& w_tl, XlaOp& w_tr, XlaOp& w_bl,
                      XlaOp& w_br, XlaOp& v_tl, XlaOp& v_tr, XlaOp& v_bl,
                      XlaOp& v_br) {
  TF_ASSIGN_OR_RETURN(Eigh2x2 rotation,
                      HermitianEigenDecomposition2x2(w_tl, w_tr, w_br));

  ApplyJacobiRotationOverRows(rotation, w_tl, w_tr, w_bl, w_br);
  ApplyJacobiRotationOverCols(rotation, w_tl, w_tr, w_bl, w_br);
  w_tl = SetMatrixDiagonal(w_tl, rotation.rt1);
  w_tr = SetMatrixDiagonal(w_tr, ZerosLike(rotation.rt1));
  w_bl = SetMatrixDiagonal(w_bl, ZerosLike(rotation.rt1));
  w_br = SetMatrixDiagonal(w_br, rotation.rt2);

  PermuteColumnsInRow(w_tl, w_tr);
  PermuteColumnsInRow(w_bl, w_br);
  PermuteRowsInColumn(w_tl, w_bl);
  PermuteRowsInColumn(w_tr, w_br);

  // Apply the rotations to the eigenvector matrix.
  // TODO(phawkins): we could omit this if we aren't interested in computing the
  // eigenvectors.
  ApplyJacobiRotationOverRows(rotation, v_tl, v_tr, v_bl, v_br);
  PermuteRowsInColumn(v_tl, v_bl);
  PermuteRowsInColumn(v_tr, v_br);
  return OkStatus();
}

struct FrobeniusNorms {
  XlaOp off_diagonal_sq_norm;
  XlaOp frobenius_sq_norm;
};

StatusOr<FrobeniusNorms> ComputeFrobeniusNorms(XlaOp w_tl, XlaOp w_tr,
                                               XlaOp w_bl, XlaOp w_br) {
  XlaBuilder* builder = w_tl.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(w_tl));
  const int64_t num_dims = shape.rank();
  auto square_norm = [](XlaOp x) -> XlaOp {
    return Real(x * MaybeConjugate(x, true));
  };
  auto off_diag = [](XlaOp x) {
    return Select(GetDiagonalMask(x), ZerosLike(x), x);
  };
  PrimitiveType norm_type =
      primitive_util::IsComplexType(shape.element_type())
          ? primitive_util::ComplexComponentType(shape.element_type())
          : shape.element_type();
  auto zero = ScalarLike(Real(w_tl), 0.0);
  FrobeniusNorms norms;
  norms.frobenius_sq_norm =
      Reduce(square_norm(w_tl) + square_norm(w_tr) + square_norm(w_bl) +
                 square_norm(w_br),
             zero, CreateScalarAddComputation(norm_type, builder),
             {num_dims - 2, num_dims - 1});
  norms.off_diagonal_sq_norm =
      Reduce(square_norm(off_diag(w_tl)) + square_norm(w_tr) +
                 square_norm(w_bl) + square_norm(off_diag(w_br)),
             zero, CreateScalarAddComputation(norm_type, builder),
             {num_dims - 2, num_dims - 1});

  return norms;
}

StatusOr<std::vector<XlaOp>> Sweeps(absl::Span<const XlaOp> initial_values,
                                    int64_t n, int max_iters,
                                    PrimitiveType index_type,
                                    XlaBuilder* builder) {
  auto while_cond_fn = [&](absl::Span<const XlaOp> values,
                           XlaBuilder* cond_builder) -> StatusOr<XlaOp> {
    auto iter_cond = Lt(values[0], ScalarLike(values[0], max_iters));

    XlaOp w_tl, w_tr, w_bl, w_br;
    std::tie(w_tl, w_tr, w_bl, w_br) =
        std::make_tuple(values[2], values[3], values[4], values[5]);
    TF_ASSIGN_OR_RETURN(auto norms,
                        ComputeFrobeniusNorms(w_tl, w_tr, w_bl, w_br));
    auto tol = norms.frobenius_sq_norm * Square(values[1]);
    auto tol_cond = ReduceAll(Lt(tol, norms.off_diagonal_sq_norm),
                              xla::ConstantR0<bool>(cond_builder, false),
                              CreateScalarOrComputation(PRED, cond_builder));

    return And(iter_cond, tol_cond);
  };

  auto while_body_fn =
      [&](absl::Span<const XlaOp> values,
          XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
    std::vector<XlaOp> sweep_values(values.begin() + 1, values.end());
    TF_ASSIGN_OR_RETURN(
        sweep_values,
        ForEachIndex(
            n - 1, S32,
            [&](XlaOp iter, absl::Span<const XlaOp> values,
                XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
              XlaOp tol, w_tl, w_tr, w_bl, w_br, v_tl, v_tr, v_bl, v_br;
              std::tie(tol, w_tl, w_tr, w_bl, w_br, v_tl, v_tr, v_bl, v_br) =
                  std::make_tuple(values[0], values[1], values[2], values[3],
                                  values[4], values[5], values[6], values[7],
                                  values[8]);
              TF_RETURN_IF_ERROR(ApplyRotations(n, w_tl, w_tr, w_bl, w_br, v_tl,
                                                v_tr, v_bl, v_br));
              return std::vector<XlaOp>{tol,  w_tl, w_tr, w_bl, w_br,
                                        v_tl, v_tr, v_bl, v_br};
            },
            sweep_values, "ApplyRotations", body_builder));
    std::vector<XlaOp> output(values.size());
    output[0] = values[0] + ScalarLike(values[0], 1);
    std::copy(sweep_values.begin(), sweep_values.end(), output.begin() + 1);
    return output;
  };
  return WhileLoopHelper(while_cond_fn, while_body_fn, initial_values,
                         "EighJacobiSweeps", builder);
}

}  // namespace

Status EighExpander::SortByEigenvalues(XlaOp& v, XlaOp& w) {
  XlaBuilder* builder = v.builder();
  TF_ASSIGN_OR_RETURN(Shape v_shape, builder->GetShape(v));
  TF_ASSIGN_OR_RETURN(Shape w_shape, builder->GetShape(w));
  const int64_t num_dims = v_shape.rank();
  auto dimensions = v_shape.dimensions();

  std::vector<int64_t> broadcast_dims(num_dims - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims[num_dims - 2] = num_dims - 1;
  w = BroadcastInDim(w, dimensions, broadcast_dims);

  XlaOp sort_result =
      Sort({w, v},
           CreateScalarLtComputation(
               {w_shape.element_type(), v_shape.element_type()}, builder),
           num_dims - 1);
  w = GetMatrixDiagonal(GetTupleElement(sort_result, 0));
  v = GetTupleElement(sort_result, 1);
  return OkStatus();
}

// This is the cyclic Jacobi iteration.
//
// def jacobi(A):
//   n, _ = A.shape
//   tl = A[:n // 2, :n // 2]
//   bl = A[n // 2:, :n // 2]
//   tr = A[:n // 2, n // 2:]
//   br = A[n // 2:, n // 2:]
//   v_tl = np.eye(n // 2, dtype=A.dtype)
//   v_tr = np.zeros((n // 2, n // 2), A.dtype)
//   v_bl = np.zeros((n // 2, n // 2), A.dtype)
//   v_br = np.eye(n // 2, dtype=A.dtype)
//   frobenius_norm = np.sqrt(np.sum(np.square(tl) + np.square(tr) +
//                            np.square(bl) + np.square(br)))
//   diag_norm = np.sqrt(np.sum(np.square(np.diag(tl)) +
//                              np.square(np.diag(br))))
//    off_diag_norm = np.sqrt(frobenius_norm - diag_norm) * np.sqrt(
//            frobenius_norm + diag_norm)
//   while off_diag_norm > 1e-6 * frobenius_norm:
//     for i in range(n - 1):
//       c, s = sym_schur2x2(tl, tr, br)
//        tl, tr, bl, br = (
//          tl * c[:, None] - bl * s[:, None],
//          tr * c[:, None] - br * s[:, None],
//          tl * s[:, None] + bl * c[:, None],
//          tr * s[:, None] + br * c[:, None],
//        )
//        tl, tr, bl, br = (
//          tl * c[None, :] - tr * s[None, :],
//          tl * s[None, :] + tr * c[None, :],
//          bl * c[None, :] - br * s[None, :],
//          bl * s[None, :] + br * c[None, :],
//        )
//        tl, bl = permute_rows_in_col(tl, bl)
//        tr, br = permute_rows_in_col(tr, br)
//        tl, tr = permute_cols_in_row(tl, tr)
//        bl, br = permute_cols_in_row(bl, br)
//        v_tl, v_tr, v_bl, v_br = (
//          v_tl * c[:, None] - v_bl * s[:, None],
//          v_tr * c[:, None] - v_br * s[:, None],
//          v_tl * s[:, None] + v_bl * c[:, None],
//          v_tr * s[:, None] + v_br * c[:, None],
//        )
//        v_tl, v_bl = permute_rovs_in_col(v_tl, v_bl)
//        v_tr, v_br = permute_rovs_in_col(v_tr, v_br)
//
//     frobenius_norm = np.sqrt(np.sum(np.square(tl) + np.square(tr) +
//                              np.square(bl) + np.square(br)))
//     diag_norm = np.sqrt(np.sum(np.square(np.diag(tl)) +
//                         np.square(np.diag(br))))
//     off_diag_norm = np.sqrt(frobenius_norm - diag_norm) * np.sqrt(
//             frobenius_norm + diag_norm)
//   return A, V
XlaOp EighExpander::BuildEigh(XlaOp a, bool lower, int64_t max_iter, float tol,
                              bool sort_eigenvalues) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int64_t num_dims = a_shape.rank();
    if (num_dims < 2) {
      return InvalidArgument(
          "Arguments to Eigen decomposition must have rank >= 2: got shape %s.",
          a_shape.ToString());
    }
    PrimitiveType type = a_shape.element_type();
    if (!primitive_util::IsFloatingPointType(type) &&
        !primitive_util::IsComplexType(type)) {
      return InvalidArgument(
          "Type of the input matrix must be floating point "
          "or complex: got %s.",
          a_shape.ToString());
    }

    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);

    if (m != n) {
      return InvalidArgument(
          "Arguments to symmetric eigendecomposition must be square matrices: "
          "got shape (%d, %d).",
          m, n);
    }

    const int64_t num_batch_dims = num_dims - 2;
    std::vector<int64_t> batch_dims(num_batch_dims);
    for (int i = 0; i < num_batch_dims; ++i) {
      batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
    }

    if (m <= 1) {
      return Tuple(builder, {FullLike(a, 1), GetMatrixDiagonal(Real(a))});
    }

    a = Symmetrize(a, lower);

    const int64_t k = CeilOfRatio(n, int64_t{2});
    // tl = A[:n // 2, :n // 2]
    // bl = A[n // 2:, :n // 2]
    // tr = A[:n // 2, n // 2:]
    // br = A[n // 2:, n // 2:]
    auto tl = SliceInMinorDims(a, {0, 0}, {k, k});
    auto bl = SliceInMinorDims(a, {k, 0}, {n, k});
    auto tr = SliceInMinorDims(a, {0, k}, {k, n});
    auto br = SliceInMinorDims(a, {k, k}, {n, n});
    if (n % 2) {
      auto zero = Zero(builder, type);
      tr = PadInDim(tr, zero, num_dims - 1, /*pad_lo=*/0, /*pad_hi=*/1);
      bl = PadInDim(bl, zero, num_dims - 2, /*pad_lo=*/0, /*pad_hi=*/1);
      PaddingConfig config = MakeNoPaddingConfig(num_dims);
      config.mutable_dimensions(num_dims - 2)->set_edge_padding_high(1);
      config.mutable_dimensions(num_dims - 1)->set_edge_padding_high(1);
      br = Pad(br, zero, config);
    }
    // v_tl = np.eye(n // 2, dtype=A.dtype)
    // v_tr = np.zeros((n // 2, n // 2), A.dtype)
    // v_bl = np.zeros((n // 2, n // 2), A.dtype)
    // v_br = np.eye(n // 2, dtype=A.dtype)
    auto v_tl = Broadcast(IdentityMatrix(builder, type, k, k), batch_dims);
    auto v_br = v_tl;
    auto v_tr = ZerosLike(v_tl);
    auto v_bl = v_tr;

    TF_ASSIGN_OR_RETURN(auto output, Sweeps(
                                         {
                                             Zero(builder, S32),
                                             ScalarLike(Real(a), tol),
                                             tl,
                                             tr,
                                             bl,
                                             br,
                                             v_tl,
                                             v_tr,
                                             v_bl,
                                             v_br,
                                         },
                                         k * 2, max_iter, S32, builder));

    std::tie(tl, tr, bl, br) =
        std::make_tuple(output[2], output[3], output[4], output[5]);
    std::tie(v_tl, v_tr, v_bl, v_br) =
        std::make_tuple(output[6], output[7], output[8], output[9]);

    auto w = ConcatInDim(
        builder, {GetMatrixDiagonal(Real(tl)), GetMatrixDiagonal(Real(br))},
        num_dims - 2);
    auto v = ConcatInDim(builder,
                         {ConcatInDim(builder, {v_tl, v_tr}, num_dims - 1),
                          ConcatInDim(builder, {v_bl, v_br}, num_dims - 1)},
                         num_dims - 2);
    if (n % 2) {
      w = SliceInMinorDims(w, {0}, {n});
      v = SliceInMinorDims(v, {0, 0}, {n, n});
    }
    v = MaybeConjugate(TransposeInMinorDims(v), true);

    if (sort_eigenvalues) {
      TF_RETURN_IF_ERROR(SortByEigenvalues(v, w));
    }
    return Tuple(builder, {v, w});
  });
}

static const char* kEighCustomCallName = "Eigh";

bool EighExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCustomCall &&
         instruction->custom_call_target() == kEighCustomCallName;
}

StatusOr<HloInstruction*> EighExpander::ExpandInstruction(
    HloInstruction* instruction) {
  const std::string name =
      absl::StrFormat("xla.%s_%s", instruction->custom_call_target(),
                      instruction->operand(0)->shape().ToString());

  HloModule* module = instruction->GetModule();

  HloComputation*& computation =
      computation_cache_.emplace(name, nullptr).first->second;
  if (!computation) {
    // Builds a new expansion.
    //
    // TODO(b/62327888): We do something unusual here: we build the computation
    // using the XlaBuilder API, which is nominally an XLA client API. We do
    // this because the external APIs for building complicated computations
    // (XlaBuilder) are much more ergonomic than the internal ones. As it turns
    // out, XlaBuilder isn't really a client API—what it does is build a
    // HloModuleProto protocol buffer, that we can then deserialize and clone
    // into our HloModule. Ideally we would avoid the protocol buffer step;
    // that is left as an exercise for future work.
    XlaBuilder builder(name);
    TF_RET_CHECK(instruction->operand_count() == 1);
    XlaOp a = Parameter(&builder, 0, instruction->operand(0)->shape(), "a");

    std::vector<std::string> config_strs =
        absl::StrSplit(instruction->raw_backend_config_string(), ',');
    int lower;
    int64_t max_iter;
    int sort_eigenvalues;
    float tol;
    if (config_strs.size() != 4 || !absl::SimpleAtoi(config_strs[0], &lower) ||
        !absl::SimpleAtoi(config_strs[1], &sort_eigenvalues) ||
        !absl::SimpleAtoi(config_strs[2], &max_iter) ||
        !absl::SimpleAtof(config_strs[3], &tol)) {
      return Internal("Unable to parse arguments to Eigh custom call, got: %s",
                      instruction->raw_backend_config_string());
    }
    XlaOp result = BuildEigh(a, lower, max_iter, tol, sort_eigenvalues);
    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build(result));

    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        xla_computation.GetProgramShape());
    HloModuleConfig config(program_shape);
    TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                             xla_computation.proto(), config));
    HloCloneContext context(module);
    computation =
        module->DeepCloneComputation(new_module->entry_computation(), &context);
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

}  // namespace xla
