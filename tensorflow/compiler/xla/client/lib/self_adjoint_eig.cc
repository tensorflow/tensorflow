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
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// Jacobi rotation (also known as Givens rotation):
// G = [[ c, s],
//      [-s, c]]
// matmul(G_T, G) = I
struct JacobiRotation {
  XlaOp c;          // cosine.
  XlaOp s;          // sine.
};

// JacobiUpdate holds the intermediate orthogonal matrix, Jacobi-rotated matrix.
struct JacobiUpdate {
  XlaOp v;
  XlaOp w;
};

struct FrobeniusNorms {
  XlaOp off_diagonal_norm;
  XlaOp total_norm;
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
StatusOr<JacobiRotation> SymmetricShurDecomposition2x2(XlaOp a, XlaOp p,
                                                       XlaOp q, XlaOp tol) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));

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

  JacobiRotation schur;

  schur.c = c * rnorm;
  schur.s = s * rnorm;

  return schur;
}

StatusOr<JacobiUpdate> Update(JacobiUpdate jacobi_update, XlaOp p, XlaOp q,
                              XlaOp tol, int64 n) {
  XlaBuilder* builder = jacobi_update.w.builder();
  TF_ASSIGN_OR_RETURN(JacobiRotation schur, SymmetricShurDecomposition2x2(
                                                jacobi_update.w, p, q, tol));

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

  return jacobi_update;
}

StatusOr<FrobeniusNorms> ComputeFrobeniusNorms(XlaOp w) {
  XlaBuilder* builder = w.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(w));
  const int64 num_dims = shape.rank();
  auto frobenius_norm =
      Sqrt(Reduce(Square(w), ScalarLike(w, 0.0),
                  CreateScalarAddComputation(shape.element_type(), builder),
                  {num_dims - 2, num_dims - 1}));
  auto diag = GetMatrixDiagonal(w);
  auto diag_square =
      Reduce(Square(diag), ScalarLike(w, 0.0),
             CreateScalarAddComputation(shape.element_type(), builder),
             {num_dims - 2});

  FrobeniusNorms frobenius_norms;

  frobenius_norms.off_diagonal_norm =
      Sqrt(Max(Square(frobenius_norm) - diag_square, ScalarLike(w, 0.0)));
  frobenius_norms.total_norm = frobenius_norm;

  return frobenius_norms;
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
    auto max_sweeps = ScalarLike(k, max_sweep_updates);
    auto sweep_update_cond = Gt(max_sweeps, k);

    auto norms = ComputeFrobeniusNorms(values[2]).ValueOrDie();
    auto tol = norms.total_norm * values[3];
    auto tol_cond = ReduceAll(Lt(tol, norms.off_diagonal_norm),
                              xla::ConstantR0<bool>(cond_builder, false),
                              CreateScalarOrComputation(PRED, cond_builder));

    return And(sweep_update_cond, tol_cond);
  };

  auto while_body_fn =
      [&](absl::Span<const XlaOp> values,
          XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
    auto while_cond_fn_inner =
        [&](absl::Span<const XlaOp> values_inner,
            XlaBuilder* inner_cond_builder) -> StatusOr<XlaOp> {
      auto p = values_inner[0];
      return Lt(p, ScalarLike(p, matrix_dimension - 1));
    };

    auto while_body_fn_inner =
        [&](absl::Span<const XlaOp> values_inner,
            XlaBuilder* inner_body_builder) -> StatusOr<std::vector<XlaOp>> {
      auto while_cond_fn_innermost =
          [&](absl::Span<const XlaOp> values_innermost,
              XlaBuilder* innermost_cond_builder) -> StatusOr<XlaOp> {
        auto q = values_innermost[1];
        return Lt(q, ScalarLike(q, matrix_dimension));
      };
      auto while_body_fn_innermost =
          [&](absl::Span<const XlaOp> values_innermost,
              XlaBuilder* innermost_body_builder)
          -> StatusOr<std::vector<XlaOp>> {
        auto p = values_innermost[0];
        auto q = values_innermost[1];

        JacobiUpdate jacobi_update;
        jacobi_update.v = values_innermost[2];
        jacobi_update.w = values_innermost[3];

        auto tol = values_innermost[4];

        TF_ASSIGN_OR_RETURN(jacobi_update,
                            Update(jacobi_update, p, q, tol, matrix_dimension));

        std::vector<XlaOp> updated_values_innermost;
        updated_values_innermost.reserve(values_innermost.size());

        updated_values_innermost.push_back(p);
        updated_values_innermost.push_back(q + ScalarLike(q, 1));
        updated_values_innermost.push_back(jacobi_update.v);
        updated_values_innermost.push_back(jacobi_update.w);
        updated_values_innermost.push_back(tol);

        return updated_values_innermost;
      };

      std::vector<XlaOp> values_innermost(5);
      auto p = values_inner[0];
      auto q = p + ScalarLike(p, 1);
      values_innermost[0] = p;                // index p.
      values_innermost[1] = q;                // index q.
      values_innermost[2] = values_inner[1];  // v.
      values_innermost[3] = values_inner[2];  // w.
      values_innermost[4] = values_inner[3];  // tol.
      TF_ASSIGN_OR_RETURN(
          values_innermost,
          WhileLoopHelper(while_cond_fn_innermost, while_body_fn_innermost,
                          values_innermost, absl::StrCat(name, "-Innermost"),
                          inner_body_builder));

      std::vector<XlaOp> updated_values_inner;
      updated_values_inner.reserve(values_inner.size());

      updated_values_inner.push_back(p + ScalarLike(p, 1));
      updated_values_inner.push_back(values_innermost[2]);
      updated_values_inner.push_back(values_innermost[3]);
      updated_values_inner.push_back(values_innermost[4]);
      return updated_values_inner;
    };
    // Indexes.
    XlaOp k = values[0];

    std::vector<XlaOp> values_inner(4);
    values_inner[0] = ScalarLike(k, 0);  // index p.
    values_inner[1] = values[1];         // v.
    values_inner[2] = values[2];         // w.
    values_inner[3] = values[3];         // tol.
    TF_ASSIGN_OR_RETURN(
        values_inner,
        WhileLoopHelper(while_cond_fn_inner, while_body_fn_inner, values_inner,
                        absl::StrCat(name, "-Inner"), body_builder));

    std::vector<XlaOp> updated_values;
    updated_values.reserve(values_inner.size());

    updated_values.push_back(k + ScalarLike(k, 1));
    updated_values.push_back(values_inner[1]);
    updated_values.push_back(values_inner[2]);
    updated_values.push_back(values_inner[3]);

    return updated_values;
  };
  std::vector<XlaOp> values;
  TF_ASSIGN_OR_RETURN(values, WhileLoopHelper(while_cond_fn, while_body_fn,
                                              initial_values, name, builder));

  return values;
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

// This is the cyclic Jacobi iteration. Please note that the eigenvalues are
// possibly not ordered.
//
//  def jacobi(A):
//      n, _ = A.shape
//      V = np.eye(n)
//      frobenius_norm = np.linalg.norm(A)
//      diag_norm = np.linalg.norm(np.diag(A))
//      off_diag_norm = np.sqrt(
//          frobenius_norm - diag_norm) * np.sqrt(frobenius_norm + diag_norm)
//      while off_diag_norm > 1e-6 * frobenius_norm:
//          for p in range(n - 1):
//              for q in range(p + 1, n):
//                  c, s = sym_schur2x2(A, p, q)
//                  A[[p, q], :] = np.matmul(np.array([[c, -s], [s, c]]),
//                                           A[[p, q], :])
//                  A[:, [p, q]] = np.matmul(A[:, [p, q]],
//                                           np.array([[c, s], [-s, c]]))
//                  V[:, [p, q]] = np.matmul(V[:, [p, q]],
//                                               np.array([[c, s], [-s, c]]))
//          frobenius_norm = np.linalg.norm(A)
//          diag_norm = np.linalg.norm(np.diag(A))
//          off_diag_norm = np.sqrt(
//              frobenius_norm - diag_norm) * np.sqrt(
//                  frobenius_norm + diag_norm)
//
//      return A, V
//
// TODO(kuny): Implement parallel order Jacobi.
//
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

  auto tol = ScalarLike(a, epsilon);

  auto v_init = Broadcast(IdentityMatrix(builder, type, m, m), batch_dims);
  auto w_init = Triangle(a, lower);
  w_init = w_init + TransposeInMinorDims(w_init) - w_init * v_init;

  auto output_with_status = WhileLoopFn(
      {
          Zero(builder, S32),  // k
          v_init,              // v
          w_init,              // w
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

  SelfAdjointEigResult result;
  result.v = output[1];
  result.w = GetMatrixDiagonal(output[2]);

  return SortByEigenvalues(result).ValueOrDie();
}

}  // namespace xla
