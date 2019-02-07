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

#include "tensorflow/compiler/xla/client/lib/self_adjoint_eigen.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
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
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// Jacobi rotation (also known as Givens rotation):
// G = [[ c, s],
//      [-s, c]]
// matmul(G_T, G) = I
struct SymmetricSchurDecomposition {
  XlaOp c;          // cosine.
  XlaOp s;          // sine.
  XlaOp reduction;  // Reduction in the off diagonal after applying G.
};

// JacobiUpdate holds the intermediate orthogonal matrix, Jacobi-rotated matrix
// and the off-diagonal norm of the rotated matrix. After each Jacobi iteration,
// off-diagonal norm is reduced.
struct JacobiUpdate {
  XlaOp v;
  XlaOp w;
  XlaOp off_diagonal_norm;
};

// Given an n-by-n symmetric A and integers p and q that satisfy 0 <= p < q < n,
// it computes a rotation matrix G = [[c, s], [-s, c]], such that
//                        G_T * A[[p, q], [p, q]] * G
// is diagonalized.
//
//  def sym_schur2x2(A, p, q):
//      if np.abs(A[p, q]) > 1e-6:
//          tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
//          if tau >= 0:
//              t = 1.0 / (tau + np.sqrt(1 + tau ** 2))
//          else:
//              t = -1.0 / (-tau + np.sqrt(1 + tau ** 2))
//          c = 1.0 / np.sqrt(1.0 + t ** 2)
//          s = t * c
//      else:
//          c = 1.0
//          s = 0.0
//      return c, s
StatusOr<SymmetricSchurDecomposition> SymmetricShurDecomposition2x2(XlaOp a,
                                                                    XlaOp p,
                                                                    XlaOp q,
                                                                    XlaOp tol) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));

  PrimitiveType type = a_shape.element_type();

  const int64 num_dims = a_shape.rank();

  auto zero = ScalarLike(a, 0.0);
  auto one = ScalarLike(a, 1.0);
  auto two = ScalarLike(a, 2.0);

  auto pqs = DynamicSliceInMinorDims(a, {p, q}, {1, 1});

  auto ps = DynamicSliceInMinorDims(a, {p, p}, {1, 1});
  auto qs = DynamicSliceInMinorDims(a, {q, q}, {1, 1});

  auto tau = (qs - ps) / (pqs * two);
  auto t_pos = one / (tau + Sqrt(one + Square(tau)));
  auto t_neg = -one / (-tau + Sqrt(one + Square(tau)));
  auto t = Select(Ge(tau, zero), t_pos, t_neg);

  auto c_temp = Rsqrt(one + Square(t));
  auto s_temp = t * c_temp;

  auto c = Select(Ge(Abs(pqs), tol), c_temp, ZerosLike(c_temp) + one);
  auto s = Select(Ge(Abs(pqs), tol), s_temp, ZerosLike(s_temp));
  // Renormalize c and s to compensate for low precision arithmetic, this step
  // is redundant if high precision float is used, like float64.
  auto rnorm = Rsqrt(Square(c) + Square(s));

  SymmetricSchurDecomposition schur;

  schur.c = c * rnorm;
  schur.s = s * rnorm;
  schur.reduction =
      Reduce(two * Square(pqs), zero, CreateScalarAddComputation(type, builder),
             {num_dims - 2, num_dims - 1});
  return schur;
}

StatusOr<JacobiUpdate> Update(JacobiUpdate jacobi_update, XlaOp p, XlaOp q,
                              XlaOp tol, int64 n) {
  XlaBuilder* builder = jacobi_update.w.builder();
  TF_ASSIGN_OR_RETURN(
      SymmetricSchurDecomposition schur,
      SymmetricShurDecomposition2x2(jacobi_update.w, p, q, tol));

  TF_ASSIGN_OR_RETURN(Shape w_shape, builder->GetShape(jacobi_update.w));
  const std::vector<int64> batch_dims(w_shape.dimensions().begin(),
                                      w_shape.dimensions().end() - 2);
  const int64 num_dims = w_shape.rank();

  auto zero = ScalarLike(p, 0);

  XlaOp c = schur.c;
  XlaOp s = schur.s;

  auto slice_p = DynamicSliceInMinorDims(jacobi_update.w, {p, zero}, {1, n});
  auto slice_q = DynamicSliceInMinorDims(jacobi_update.w, {q, zero}, {1, n});

  auto slice_p_new = c * slice_p - s * slice_q;
  auto slice_q_new = s * slice_p + c * slice_q;

  jacobi_update.w =
      DynamicUpdateSliceInMinorDims(jacobi_update.w, slice_p_new, {p, zero});
  jacobi_update.w =
      DynamicUpdateSliceInMinorDims(jacobi_update.w, slice_q_new, {q, zero});

  slice_p = DynamicSliceInMinorDims(jacobi_update.w, {zero, p}, {n, 1});
  slice_q = DynamicSliceInMinorDims(jacobi_update.w, {zero, q}, {n, 1});

  slice_p_new = c * slice_p - s * slice_q;
  slice_q_new = s * slice_p + c * slice_q;

  jacobi_update.w =
      DynamicUpdateSliceInMinorDims(jacobi_update.w, slice_p_new, {zero, p});
  jacobi_update.w =
      DynamicUpdateSliceInMinorDims(jacobi_update.w, slice_q_new, {zero, q});

  // Zero out a_{pq} explicitly.
  std::vector<int64> pq_dims(batch_dims.begin(), batch_dims.end());
  pq_dims.push_back(1);
  pq_dims.push_back(1);
  auto pq_zero = ScalarLike(jacobi_update.w, 0.0);
  auto pq_zeros = Broadcast(pq_zero, pq_dims);
  jacobi_update.w =
      DynamicUpdateSliceInMinorDims(jacobi_update.w, pq_zeros, {p, q});
  jacobi_update.w =
      DynamicUpdateSliceInMinorDims(jacobi_update.w, pq_zeros, {q, p});

  slice_p = DynamicSliceInMinorDims(jacobi_update.v, {zero, p}, {n, 1});
  slice_q = DynamicSliceInMinorDims(jacobi_update.v, {zero, q}, {n, 1});

  std::vector<int64> broadcast_dims(batch_dims.size());
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims.push_back(num_dims - 1);

  // Renormalize the p-th and q-th columns. This step is redundant if high
  // precision floats are used, like 64-bit float. But for 32-bit float, it
  // becomes necessary. This step will not increase the overall complexity.
  slice_p_new = c * slice_p - s * slice_q;
  slice_p_new = Mul(
      slice_p_new,
      Rsqrt(Reduce(Square(slice_p_new), pq_zero,
                   CreateScalarAddComputation(w_shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);
  slice_q_new = s * slice_p + c * slice_q;
  slice_q_new = Mul(
      slice_q_new,
      Rsqrt(Reduce(Square(slice_q_new), pq_zero,
                   CreateScalarAddComputation(w_shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  jacobi_update.v =
      DynamicUpdateSliceInMinorDims(jacobi_update.v, slice_p_new, {zero, p});
  jacobi_update.v =
      DynamicUpdateSliceInMinorDims(jacobi_update.v, slice_q_new, {zero, q});

  jacobi_update.off_diagonal_norm = Sqrt(
      Max(Square(jacobi_update.off_diagonal_norm) - schur.reduction, pq_zero));

  return jacobi_update;
}

StatusOr<std::vector<XlaOp>> WhileLoopFn(
    absl::Span<const XlaOp> initial_values,  //
    int matrix_dimension,                    //
    int max_sweep_updates,                   //
    PrimitiveType index_type,                //
    absl::string_view name,                  //
    XlaBuilder* builder) {
  auto while_cond_fn = [&](absl::Span<const XlaOp> values,
                           XlaBuilder* cond_builder) -> StatusOr<XlaOp> {
    auto k = values[0];
    auto off_diagonal_norm = values[5];
    // tol = frobenius_norm * epsilon.
    auto tol = values[6] * values[7];

    auto max_sweeps = ScalarLike(k, max_sweep_updates);

    auto sweep_update_cond = Gt(max_sweeps, k);

    auto tol_cond = ReduceAll(Lt(tol, off_diagonal_norm),
                              xla::ConstantR0<bool>(cond_builder, false),
                              CreateScalarOrComputation(PRED, cond_builder));
    return And(tol_cond, sweep_update_cond);
  };

  auto while_body_fn =
      [&](absl::Span<const XlaOp> values,
          XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
    auto zero = Zero(body_builder, index_type);
    auto one = One(body_builder, index_type);
    auto end_index = ScalarLike(one, matrix_dimension);

    // Indexes.
    XlaOp k = values[0];
    XlaOp p = values[1];
    XlaOp q = values[2];

    JacobiUpdate jacobi_update;
    jacobi_update.v = values[3];
    jacobi_update.w = values[4];
    jacobi_update.off_diagonal_norm = values[5];

    XlaOp frobenius_norm = values[6];
    XlaOp tol = values[7];

    TF_ASSIGN_OR_RETURN(jacobi_update,
                        Update(jacobi_update, p, q, tol, matrix_dimension));

    std::vector<XlaOp> updated_values;
    updated_values.reserve(values.size());

    q = q + one;
    p = Select(Eq(q, end_index), p + one, p);
    k = Select(Eq(p, end_index - one), k + one, k);
    p = Select(Eq(p, end_index - one), zero, p);
    q = Select(Eq(q, end_index), p + one, q);

    updated_values.push_back(k);
    updated_values.push_back(p);
    updated_values.push_back(q);

    updated_values.push_back(jacobi_update.v);
    updated_values.push_back(jacobi_update.w);
    updated_values.push_back(jacobi_update.off_diagonal_norm);

    updated_values.push_back(frobenius_norm);
    updated_values.push_back(tol);

    return updated_values;
  };
  std::vector<XlaOp> values;
  TF_ASSIGN_OR_RETURN(values, WhileLoopHelper(while_cond_fn, while_body_fn,
                                              initial_values, name, builder));

  return values;
}

}  // namespace

// This is the cyclic Jacobi iteration. Please note that the eigenvalues are
// possibly not ordered.
//
//  def jacobi(A):
//      n, _ = A.shape
//      V = np.eye(n)
//      nfrob = np.sum(A ** 2)
//      ndiag = np.sum(np.diag(A) ** 2)
//      off = nfrob - ndiag
//      while off > 1e-6 * nfrob:
//          for p in range(n - 1):
//              for q in range(p + 1, n):
//                  if off > 1e-6 * nfrob:
//                      c, s = sym_schur2x2(A, p, q)
//                      off = off - 2 * A[p, q] ** 2
//                      A[[p, q], :] = np.matmul(np.array([[c, -s], [s, c]]),
//                                               A[[p, q], :])
//                      A[:, [p, q]] = np.matmul(A[:, [p, q]],
//                                               np.array([[c, s], [-s, c]]))
//                      V[:, [p, q]] = np.matmul(V[:, [p, q]],
//                                               np.array([[c, s], [-s, c]]))
//
//      return A, V
//
// TODO(kuny): Implement parallel order Jacobi.
//
SelfAdjointEigenResult SelfAdjointEigen(XlaOp a, bool lower, int64 max_iter,
                                        float epsilon) {
  XlaBuilder* builder = a.builder();
  auto return_error = [&](const Status& status) {
    SelfAdjointEigenResult result;
    result.v = builder->ReportError(status);
    result.w = builder->ReportError(status);
    return result;
  };
  auto shape_with_status = builder->GetShape(a);
  if (!shape_with_status.status().ok()) {
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
        "Arguments to Eigen decomposition must be square matrices: got shape "
        "(%d, %d).",
        m, n));
  }

  const int64 num_batch_dims = num_dims - 2;
  std::vector<int64> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }

  auto zero = ScalarLike(a, 0.0);
  auto tol = ScalarLike(a, epsilon);

  auto v_init = Broadcast(IdentityMatrix(builder, type, m, m), batch_dims);
  auto w_init = Triangle(a, lower);
  w_init = w_init + TransposeInMinorDims(w_init) - w_init * v_init;

  auto frobenius_norm = Sqrt(Reduce(Square(w_init), zero,
                                    CreateScalarAddComputation(type, builder),
                                    {num_dims - 2, num_dims - 1}));
  auto diag = GetMatrixDiagonal(w_init);
  auto diag_square =
      Reduce(Square(diag), zero, CreateScalarAddComputation(type, builder),
             {num_dims - 2});

  auto off_diagonal_init =
      Sqrt(Max(Square(frobenius_norm) - diag_square, zero));

  auto output_with_status = WhileLoopFn(
      {
          Zero(builder, S32),  // k
          Zero(builder, S32),  // p
          One(builder, S32),   // q
          v_init,              //
          w_init,              //
          off_diagonal_init,   //
          frobenius_norm,      //
          tol,                 //
      },                       //
      n,                       //
      max_iter,                //
      S32,                     //
      "CyclicJacobi",          //
      builder);
  if (!output_with_status.status().ok()) {
    return return_error(output_with_status.status());
  }

  auto output = output_with_status.ValueOrDie();

  SelfAdjointEigenResult result;
  result.v = output[3];
  result.w = GetMatrixDiagonal(output[4]);

  return result;
}

}  // namespace xla
