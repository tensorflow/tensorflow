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

#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"

#include <memory>
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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

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

// sqrt(x**2 + y**2), calculated avoiding overflow.
XlaOp Hypot(XlaOp x, XlaOp y) {
  x = Abs(x);
  y = Abs(y);
  auto xy_min = Min(x, y);
  auto xy_max = Max(x, y);
  auto out = xy_max * Sqrt(ScalarLike(x, 1) + Square(xy_min / xy_max));
  return Select(Eq(xy_min, xy_max), xy_min * ScalarLike(xy_min, std::sqrt(2.)),
                out);
}

// Given an n-by-n symmetric A and integers p and q that satisfy 0 <= p < q < n,
// a Jacobi rotation computes a rotation matrix G = [[c, s], [-s, c]], such that
//   G_T * A[[p, q], [p, q]] * G
// is diagonalized. We do this by computing a 2x2 eigendecomposition.
//
// In this parallel Jacobi algorithm, we simultaneously compute Jacobi rotations
// for all of the matrix diagonal elements at the same time. The matrix diagonal
// elements correspond to different rows and columns of the original matrix and
// their rotations do not interfere and hence can be computed in parallel.
//
// The algorithm is based on slaev2 from LAPACK, modified to allow for
// vectorization.
// In addition, slaev2 always returns the largest eigenvalue as rt1, which has
// the effect of swapping eigenvalues around in the Jacob algorithm. This does
// not converge when used in a parallel Jacobi algorithm, so we modify the
// algorithm to maintain the following symmetry property:
// slaev2(a, b, c) has the opposite Eigenvalue order from slaev2(c, b, a)

// def symmetric_eigendecomposition_2x2(a, b, c):
//   # Input matrix [[a, b], [b, c]].
//   ac_sum = a + c
//   ac_diff = a - c
//   two_b = 2*b
//
//   rt = hypot(ac_diff, two_b)
//
//   which_max_abs = np.abs(a) > np.abs(c)
//   ac_max = np.where(which_max_abs, a, c)
//   ac_min = np.where(which_max_abs, c, a)
//   rt1 = np.float32(0.5)*(ac_sum + np.where(ac_sum < 0, -rt, rt))
//   rt2 = np.where(ac_sum != 0, (ac_max / rt1)*ac_min - (b/rt1)*b,
//                  -np.float32(0.5)*rt)
//
//
//   # Modification: don't sort the Eigenvalues.
//   rt1, rt2 = (np.where(which_max_abs, rt1, rt2),
//               np.where(which_max_abs, rt2, rt1))
//
//   # Compute eigenvectors
//   cs = ac_diff + np.where(ac_diff >= 0, rt, -rt)
//
//   ct = -two_b / cs
//   tn = -cs / two_b
//
//   cosine = np.where(two_b != 0, np.float32(1) / np.sqrt(1 + tn*tn),
//                  np.float32(1))
//   sine = np.where(two_b != 0, tn * cosine, np.float32(0))
//
//   tmp = 1 / np.sqrt(1 + ct*ct)
//   cosine = np.where(np.abs(cs) > np.abs(two_b), ct*tmp, cosine)
//   sine = np.where(np.abs(cs) > np.abs(two_b), tmp, sine)
//   same_sign = (ac_sum >= 0) == (ac_diff >= 0)
//   # Modification: use Eigenvalues corresponding to the Eigenvectors above.
//   same_sign = (same_sign == which_max_abs)
//   cosine, sine = (np.where(same_sign, -sine, cosine),
//                   np.where(same_sign, cosine, sine))
//   return rt1, rt2, cosine, sine
Eigh2x2 SymmetricEigenDecomposition2x2(XlaOp w_tl, XlaOp w_tr, XlaOp w_br) {
  auto a = GetMatrixDiagonal(w_tl);
  auto b = GetMatrixDiagonal(w_tr);
  auto c = GetMatrixDiagonal(w_br);
  auto zero = ScalarLike(w_tl, 0.0);
  auto half = ScalarLike(w_tl, 0.5);
  auto neg_half = ScalarLike(w_tl, -0.5);
  auto one = ScalarLike(w_tl, 1.0);
  auto two = ScalarLike(w_tl, 2.0);

  auto ac_sum = a + c;
  auto ac_diff = a - c;
  auto two_b = two * b;
  auto rt = Hypot(ac_diff, two_b);

  // Compute eigenvalues
  auto which_max_abs = Gt(Abs(a), Abs(c));
  auto ac_max = Select(which_max_abs, a, c);
  auto ac_min = Select(which_max_abs, c, a);
  auto rt1 = half * (ac_sum + Select(Lt(ac_sum, zero), -rt, rt));
  auto rt2 = Select(Ne(ac_sum, zero), (ac_max / rt1) * ac_min - (b / rt1) * b,
                    neg_half * rt);
  std::tie(rt1, rt2) = std::make_tuple(Select(which_max_abs, rt1, rt2),
                                       Select(which_max_abs, rt2, rt1));

  // Compute eigenvectors
  auto cs = ac_diff + Select(Ge(ac_diff, zero), rt, -rt);
  auto ct = -two_b / cs;
  auto tn = -cs / two_b;

  auto cosine = Select(Ne(two_b, zero), Rsqrt(one + Square(tn)), one);
  auto sine = Select(Ne(two_b, zero), tn * cosine, zero);

  auto tmp = Rsqrt(one + Square(ct));
  auto abs_cs_larger = Gt(Abs(cs), Abs(two_b));
  cosine = Select(abs_cs_larger, ct * tmp, cosine);
  sine = Select(abs_cs_larger, tmp, sine);
  auto same_sign = Eq(Ge(ac_sum, zero), Ge(ac_diff, zero));
  same_sign = Eq(same_sign, which_max_abs);
  std::tie(cosine, sine) = std::make_tuple(Select(same_sign, -sine, cosine),
                                           Select(same_sign, cosine, sine));

  // Negate 'sine' because we are returning the first row of the rotation matrix
  // not the first eigenvector.
  return {rt1, rt2, cosine, -sine};
}

// tl, tr, bl, br = (
//   tl * c[:, None] - bl * s[:, None],
//   tr * c[:, None] - br * s[:, None],
//   tl * s[:, None] + bl * c[:, None],
//   tr * s[:, None] + br * c[:, None],
// )
void ApplyJacobiRotationOverRows(Eigh2x2 rotation, XlaOp& tl, XlaOp& tr,
                                 XlaOp& bl, XlaOp& br) {
  Shape shape = tl.builder()->GetShape(tl).ValueOrDie();
  std::vector<int64> broadcast_dims(shape.dimensions().size() - 1);
  absl::c_iota(broadcast_dims, 0);
  auto c = BroadcastInDim(rotation.c, shape.dimensions(), broadcast_dims);
  auto s = BroadcastInDim(rotation.s, shape.dimensions(), broadcast_dims);

  std::tie(tl, tr, bl, br) = std::make_tuple(tl * c - bl * s, tr * c - br * s,
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
  Shape shape = tl.builder()->GetShape(tl).ValueOrDie();
  std::vector<int64> broadcast_dims(shape.dimensions().size() - 1);
  absl::c_iota(broadcast_dims, 0);
  broadcast_dims.back() = shape.dimensions().size() - 1;
  auto c = BroadcastInDim(rotation.c, shape.dimensions(), broadcast_dims);
  auto s = BroadcastInDim(rotation.s, shape.dimensions(), broadcast_dims);

  std::tie(tl, tr, bl, br) = std::make_tuple(tl * c - tr * s, tl * s + tr * c,
                                             bl * c - br * s, bl * s + br * c);
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
  Shape shape = builder->GetShape(top).ValueOrDie();
  int64 k = ShapeUtil::GetDimension(shape, -1);
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
  Shape shape = builder->GetShape(left).ValueOrDie();
  int64 k = ShapeUtil::GetDimension(shape, -1);
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
void ApplyRotations(int64 n, XlaOp& w_tl, XlaOp& w_tr, XlaOp& w_bl, XlaOp& w_br,
                    XlaOp& v_tl, XlaOp& v_tr, XlaOp& v_bl, XlaOp& v_br) {
  Eigh2x2 rotation = SymmetricEigenDecomposition2x2(w_tl, w_tr, w_br);

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
}

struct FrobeniusNorms {
  XlaOp off_diagonal_norm;
  XlaOp total_norm;
};

StatusOr<FrobeniusNorms> ComputeFrobeniusNorms(XlaOp w_tl, XlaOp w_tr,
                                               XlaOp w_bl, XlaOp w_br) {
  XlaBuilder* builder = w_tl.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(w_tl));
  const int64 num_dims = shape.rank();
  auto frobenius_norm =
      Sqrt(Reduce(Square(w_tl) + Square(w_tr) + Square(w_bl) + Square(w_br),
                  ScalarLike(w_tl, 0.0),
                  CreateScalarAddComputation(shape.element_type(), builder),
                  {num_dims - 2, num_dims - 1}));
  auto diag_square =
      Reduce(Square(GetMatrixDiagonal(w_tl)) + Square(GetMatrixDiagonal(w_br)),
             ScalarLike(w_tl, 0.0),
             CreateScalarAddComputation(shape.element_type(), builder),
             {num_dims - 2});

  FrobeniusNorms frobenius_norms;

  frobenius_norms.off_diagonal_norm =
      Sqrt(Max(Square(frobenius_norm) - diag_square, ScalarLike(w_tl, 0.0)));
  frobenius_norms.total_norm = frobenius_norm;

  return frobenius_norms;
}

StatusOr<std::vector<XlaOp>> Sweeps(absl::Span<const XlaOp> initial_values,
                                    int64 n, int max_iters,
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
    auto tol = norms.total_norm * values[1];
    auto tol_cond = ReduceAll(Lt(tol, norms.off_diagonal_norm),
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
              ApplyRotations(n, w_tl, w_tr, w_bl, w_br, v_tl, v_tr, v_bl, v_br);
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

StatusOr<SelfAdjointEigResult> SortByEigenvalues(SelfAdjointEigResult result) {
  XlaBuilder* builder = result.v.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(result.v));
  const int64 num_dims = shape.rank();
  auto dimensions = shape.dimensions();

  std::vector<int64> broadcast_dims(num_dims - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims[num_dims - 2] = num_dims - 1;
  result.w = BroadcastInDim(result.w, dimensions, broadcast_dims);

  XlaOp sort_result =
      Sort({result.w, result.v},
           CreateScalarLtComputation(
               {shape.element_type(), shape.element_type()}, builder),
           num_dims - 1);
  result.w = GetMatrixDiagonal(GetTupleElement(sort_result, 0));
  result.v = GetTupleElement(sort_result, 1);
  return result;
}

}  // namespace

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
SelfAdjointEigResult SelfAdjointEig(XlaOp a, bool lower, int64 max_iter,
                                    float epsilon) {
  XlaBuilder* builder = a.builder();
  auto return_error = [&](const Status& status) {
    SelfAdjointEigResult result;
    result.v = builder->ReportError(status);
    result.w = builder->ReportError(status);
    return result;
  };
  auto shape_with_status = builder->GetShape(a);
  if (!shape_with_status.ok()) {
    return return_error(shape_with_status.status());
  }
  Shape a_shape = shape_with_status.ValueOrDie();
  const int64 num_dims = a_shape.rank();
  if (num_dims < 2) {
    return return_error(InvalidArgument(
        "Arguments to Eigen decomposition must have rank >= 2: got shape %s.",
        a_shape.ToString()));
  }
  PrimitiveType type = a_shape.element_type();
  if (!primitive_util::IsFloatingPointType(type)) {
    return return_error(InvalidArgument(
        "Type of the input matrix must be float: got %s.", a_shape.ToString()));
  }

  const int64 m = ShapeUtil::GetDimension(a_shape, -2);
  const int64 n = ShapeUtil::GetDimension(a_shape, -1);

  if (m != n) {
    return return_error(InvalidArgument(
        "Arguments to symmetric eigendecomposition must be square matrices: "
        "got shape (%d, %d).",
        m, n));
  }

  const int64 num_batch_dims = num_dims - 2;
  std::vector<int64> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }

  if (m <= 1) {
    SelfAdjointEigResult result;
    result.v = FullLike(a, 1);
    result.w = GetMatrixDiagonal(a);
    return result;
  }

  auto tol = ScalarLike(a, epsilon);

  auto eye = Broadcast(IdentityMatrix(builder, type, m, m), batch_dims);
  a = Triangle(a, lower);
  a = a + TransposeInMinorDims(a) - a * eye;

  const int64 k = CeilOfRatio(n, int64{2});
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

  auto output_with_status = Sweeps(
      {
          Zero(builder, S32),
          tol,
          tl,
          tr,
          bl,
          br,
          v_tl,
          v_tr,
          v_bl,
          v_br,
      },
      k * 2, max_iter, S32, builder);
  if (!output_with_status.ok()) {
    return return_error(output_with_status.status());
  }

  auto output = output_with_status.ValueOrDie();
  std::tie(tl, tr, bl, br) =
      std::make_tuple(output[2], output[3], output[4], output[5]);
  std::tie(v_tl, v_tr, v_bl, v_br) =
      std::make_tuple(output[6], output[7], output[8], output[9]);

  SelfAdjointEigResult result;
  auto w = ConcatInDim(builder,
                       {ConcatInDim(builder, {tl, tr}, num_dims - 1),
                        ConcatInDim(builder, {bl, br}, num_dims - 1)},
                       num_dims - 2);
  auto v = ConcatInDim(builder,
                       {ConcatInDim(builder, {v_tl, v_tr}, num_dims - 1),
                        ConcatInDim(builder, {v_bl, v_br}, num_dims - 1)},
                       num_dims - 2);
  if (n % 2) {
    w = SliceInMinorDims(w, {0, 0}, {n, n});
    v = SliceInMinorDims(v, {0, 0}, {n, n});
  }
  result.v = TransposeInMinorDims(v);
  result.w = GetMatrixDiagonal(w);

  auto result_or = SortByEigenvalues(result);
  if (!result_or.ok()) {
    return return_error(result_or.status());
  }
  return result_or.ValueOrDie();
}

}  // namespace xla
