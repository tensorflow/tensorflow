/* Copyright 2019 The OpenXLA Authors.

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
#include "xla/hlo/builder/lib/svd.h"

#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/loops.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Given a matrix A, define H,
//   H = A * (I - beta * v_T * v) if v is a row vector, or
//   H = (I - beta * v * v_T) if v is column vector.
// A * H or H * A zeros out trailing part of some row or column of A.
//
// [x0, ..., x_{k-1}, xk, x_{k+1}, ..., x_{n-1}] * H
//       = [x0, ..., x_{k-1}, xnorm, 0, ..., 0]
//
// Here xnorm = norm([x_k, x_{k+1}, ..., x_{n - 1}])
struct HouseHolderResult {
  XlaOp v;
  XlaOp beta;
  XlaOp a;
};

// Jacobi rotation (also known as Givens rotation):
// G = [[ c, s],
//      [-s, c]]
// matmul(G_T, G) = I
struct JacobiRotation {
  XlaOp c;  // cosine.
  XlaOp s;  // sine.
};

// JacobiUpdate holds the intermediate orthogonal matrix, Jacobi-rotated matrix.
struct JacobiUpdate {
  XlaOp v;
  XlaOp w;
};

// OneSidedJacobiRotation holds the left and right Jacobi rotations. Refer to
// GetOneSidedJacobiRotation for the effect of applying OneSidedJacobiRotation
// to a matrix.
struct OneSidedJacobiRotation {
  JacobiRotation rot_l;
  JacobiRotation rot_r;
};

// Householder reflection on the trailing elements of a vector.
//
// H = I - beta * [1, v]' * [1, v]
//
// H * x = [..., xnorm, 0, ..., 0]
//          ..., j, j + 1, ..., n
//
// def house(x, j, eps):
//    sigma = np.linalg.norm(x[(j + 1):])
//    v = np.zeros_like(x)
//    v[(j + 1):] = x[(j + 1):]
//    if sigma < eps:
//        beta = 0
//    else:
//        mu = sigma * np.sqrt((x[j]/sigma)**2 + 1)
//        if x[j] <= 0:
//            v[j] = x[j] - mu
//        else:
//            v[j] = -sigma / (x[j] + mu) * sigma
//        beta = 2 / ((sigma / v[j])**2 + 1)
//        v = v / v[j]
//    v[j] = 1
//    return v, beta
//
// Householder reflection on the trailing elements of a row of a matrix. After
// applying it on the matrix, all elements in [i, (j+1):] become zeros, i.e.,
//
// H = I - beta * [1, v]' * [1, v], then,
//
// A[i, j:] * H = [sigma, 0, 0, ..., 0]
//
absl::StatusOr<HouseHolderResult> HouseRow(
    XlaOp a, XlaOp i, XlaOp j, XlaOp eps,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int64_t num_dims = a_shape.dimensions().size();
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  XlaOp zero = ScalarLike(i, 0);
  XlaOp x = DynamicSliceInMinorDims(a, {i, zero}, {1, n});

  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int k = 0; k < num_batch_dims; ++k) {
    batch_dims[k] = ShapeUtil::GetDimension(a_shape, k);
  }

  TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
  auto idx = Iota(builder, ShapeUtil::MakeShape(S32, x_shape.dimensions()),
                  num_dims - 1);
  auto zeros = ZerosLike(x);
  auto v = Select(Gt(idx, j), x, zeros);

  auto one = ScalarLike(v, 1.0);

  auto sigma =
      Sqrt(Reduce(Square(v), ScalarLike(v, 0.0),
                  CreateScalarAddComputation(x_shape.element_type(), builder),
                  {num_dims - 1}));

  std::vector<int64_t> broadcast_dims(num_dims - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  auto x_0j = DynamicSliceInMinorDims(x, {zero, j}, {1, 1});
  auto mu = Mul(sigma, Sqrt(Square(Div(x_0j, sigma, broadcast_dims)) + one),
                broadcast_dims);

  auto v_0j = Select(
      Le(x_0j, ScalarLike(x_0j, 0.0)), Sub(x_0j, mu),
      -Mul(sigma, Div(sigma, Add(x_0j, mu), broadcast_dims), broadcast_dims));

  auto beta = Div(ScalarLike(v_0j, 2.0),
                  (Square(Div(sigma, v_0j, broadcast_dims)) + one));

  v = Select(
      BroadcastInDim(Lt(sigma, eps), x_shape.dimensions(), broadcast_dims), v,
      v / v_0j);
  v = Select(Eq(idx, j), zeros + one, v);

  beta = Select(Lt(Add(sigma, ZerosLike(beta), broadcast_dims), eps),
                ZerosLike(beta), beta);

  HouseHolderResult result;
  result.v = v;
  result.beta = beta;
  result.a = Sub(a, Mul(beta, BatchDot(BatchDot(a, false, v, true, precision),
                                       v, precision)));

  return result;
}

// Householder reflection on the trailing elements of a col of a matrix. After
// applying it on the matrix, all elements in [(i+1):, j] become zeros, i.e.,
//
// H = I - beta * [1; v] * [1; v]', then,
//
// H * A[i:, j] = [xnorm, 0, 0, ..., 0]
//
absl::StatusOr<HouseHolderResult> HouseCol(
    XlaOp a, XlaOp i, XlaOp j, XlaOp eps,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int64_t num_dims = a_shape.dimensions().size();
  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  XlaOp zero = ScalarLike(i, 0);
  XlaOp x = DynamicSliceInMinorDims(a, {zero, j}, {m, 1});

  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int k = 0; k < num_batch_dims; ++k) {
    batch_dims[k] = ShapeUtil::GetDimension(a_shape, k);
  }

  TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
  auto idx = Iota(builder, ShapeUtil::MakeShape(S32, x_shape.dimensions()),
                  num_dims - 2);
  auto zeros = ZerosLike(x);
  auto v = Select(Gt(idx, i), x, zeros);

  auto one = ScalarLike(v, 1.0);

  auto sigma =
      Sqrt(Reduce(Square(v), ScalarLike(v, 0.0),
                  CreateScalarAddComputation(x_shape.element_type(), builder),
                  {num_dims - 2}));

  std::vector<int64_t> broadcast_dims(num_dims - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims[num_dims - 2] = num_dims - 1;
  auto x_0i = DynamicSliceInMinorDims(x, {i, zero}, {1, 1});
  auto mu = Mul(sigma, Sqrt(Square(Div(x_0i, sigma, broadcast_dims)) + one),
                broadcast_dims);

  auto v_0i = Select(
      Le(x_0i, ScalarLike(x_0i, 0.0)), Sub(x_0i, mu),
      -Mul(sigma, Div(sigma, Add(x_0i, mu), broadcast_dims), broadcast_dims));

  auto beta = Div(ScalarLike(v_0i, 2.0),
                  (Square(Div(sigma, v_0i, broadcast_dims)) + one));

  v = Select(
      BroadcastInDim(Lt(sigma, eps), x_shape.dimensions(), broadcast_dims), v,
      v / v_0i);
  v = Select(Eq(idx, i), zeros + one, v);

  beta = Select(Lt(Add(sigma, ZerosLike(beta), broadcast_dims), eps),
                ZerosLike(beta), beta);

  HouseHolderResult result;
  result.v = v;
  result.beta = beta;
  result.a = Sub(
      a, Mul(beta, BatchDot(v, false, BatchDot(v, true, a, false, precision),
                            false, precision)));

  return result;
}

// Apply column and row householder reflections for bidiagonalization.
//
// def house_bidiag(A):
//    xz, yz = A.shape
//    LL = np.eye(xz)
//    RR = np.eye(yz)
//    for i in range(yz - 1):
//        v, beta = house_col(A, i, i, 1e-8)
//        L = np.eye(xz) - beta * np.outer(v, v)
//        LL = np.matmul(LL, L)
//        A = np.matmul(L, A)
//        if i < yz - 2:
//            v, beta = house_row(A, i, i + 1, 1e-8)
//            R = np.eye(yz) - beta * np.outer(v, v)
//            RR = np.matmul(RR, R)
//            A = np.matmul(A, R)
//    return LL, A, RR
//
absl::StatusOr<SVDResult> HouseHolderBidiagonalization(
    XlaOp a, XlaOp eps, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int64_t num_dims = a_shape.dimensions().size();
  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }
  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  XlaOp u_init = Broadcast(
      IdentityMatrix(builder, a_shape.element_type(), m, m), batch_dims);
  XlaOp v_init = Broadcast(
      IdentityMatrix(builder, a_shape.element_type(), n, n), batch_dims);

  auto while_cond_fn = [&](absl::Span<const XlaOp> values,
                           XlaBuilder* cond_builder) -> absl::StatusOr<XlaOp> {
    auto i = values[0];
    return Lt(i, ScalarLike(i, n - 2));
  };
  auto while_body_fn =
      [&](absl::Span<const XlaOp> values,
          XlaBuilder* body_builder) -> absl::StatusOr<std::vector<XlaOp>> {
    auto i = values[0];
    auto one = ScalarLike(i, 1);

    auto u = values[1];
    auto v = values[2];
    auto a = values[3];
    auto eps = values[4];

    TF_ASSIGN_OR_RETURN(HouseHolderResult house_col,
                        HouseCol(a, i, i, eps, precision));
    u = Sub(u,
            Mul(house_col.beta, BatchDot(BatchDot(u, house_col.v, precision),
                                         false, house_col.v, true, precision)));
    a = house_col.a;

    TF_ASSIGN_OR_RETURN(HouseHolderResult house_row,
                        HouseRow(a, i, i + one, eps, precision));
    v = Sub(v, Mul(house_row.beta,
                   BatchDot(BatchDot(v, false, house_row.v, true, precision),
                            house_row.v, precision)));
    a = house_row.a;

    std::vector<XlaOp> updated_values;
    updated_values.reserve(values.size());

    updated_values.push_back(i + one);
    updated_values.push_back(u);
    updated_values.push_back(v);
    updated_values.push_back(a);
    updated_values.push_back(eps);
    return updated_values;
  };

  std::vector<XlaOp> values(5);
  values[0] = Zero(builder, S32);
  values[1] = u_init;
  values[2] = v_init;
  values[3] = a;
  values[4] = eps;

  TF_ASSIGN_OR_RETURN(values,
                      WhileLoopHelper(while_cond_fn, while_body_fn, values,
                                      "HouseHolderBidiagonalization", builder));

  for (int k = 2; k > 0; --k) {
    if (n - k >= 0) {
      XlaOp index = ScalarLike(values[0], n - k);
      TF_ASSIGN_OR_RETURN(HouseHolderResult house_col,
                          HouseCol(values[3], index, index, eps, precision));
      values[1] = Sub(values[1],
                      Mul(house_col.beta,
                          BatchDot(BatchDot(values[1], house_col.v, precision),
                                   false, house_col.v, true, precision)));
      values[3] = house_col.a;
    }
  }

  SVDResult result;
  result.u = values[1];
  result.v = values[2];
  result.d = values[3];
  return result;
}

// MakeJacobi computes a rotation matrix G = [[c, s], [-s, c]], such that
//                        G_T * [[ps, pqs], [pqs, qs]] * G
// is diagonalized.
//
//  def make_jacobi(ps, qs, pqs, eps):
//     if np.abs(a_pq) > eps:
//         tau = (a_qq - a_pp) / (2 * a_pq)
//         if tau >= 0:
//             t = 1.0 / (tau + np.sqrt(1 + tau ** 2))
//         else:
//             t = -1.0 / (-tau + np.sqrt(1 + tau ** 2))
//         c = 1.0 / np.sqrt(1.0 + t ** 2)
//         s = t * c
//     else:
//         c = 1.0
//         s = 0.0
//     return c, s
//
absl::StatusOr<JacobiRotation> MakeJacobi(XlaOp ps, XlaOp qs, XlaOp pqs,
                                          XlaOp eps) {
  auto zero = ScalarLike(ps, 0.0);
  auto one = ScalarLike(ps, 1.0);
  auto two = ScalarLike(ps, 2.0);

  auto tau = (qs - ps) / (pqs * two);
  auto t_pos = one / (tau + Sqrt(one + Square(tau)));
  auto t_neg = -one / (-tau + Sqrt(one + Square(tau)));
  auto t = Select(Ge(tau, zero), t_pos, t_neg);

  auto c_temp = Rsqrt(one + Square(t));
  auto s_temp = t * c_temp;

  auto c = Select(Ge(Abs(pqs), eps), c_temp, ZerosLike(c_temp) + one);
  auto s = Select(Ge(Abs(pqs), eps), s_temp, ZerosLike(s_temp));
  // Renormalize c and s to compensate for low precision arithmetic, this step
  // is redundant if high precision float is used, like float64.
  auto rnorm = Rsqrt(Square(c) + Square(s));

  JacobiRotation rot;

  rot.c = c * rnorm;
  rot.s = s * rnorm;

  return rot;
}

// One sided Jacobi rotations. For a matrix,
//  [a_pp, a_pq]
//  [a_qp, a_qq]
// After applying Jacobi rotations on both sides, the matrix is diagonalized.
//  [b_pp, 0]
//  [0, b_qq]
//
// def jacobi_rot(a, p, q, eps):
//     t = a[p, p] + a[q, q]
//     d = a[q, p] - a[p, q]
//
//     if np.abs(d) < eps:
//         s = 0.0
//         c = 1.0
//     else:
//         u = t / d
//         tmp = np.sqrt(1.0 + u**2)
//         s = -1.0 / tmp
//         c = u / tmp
//
//     rot = np.array([[c, s], [-s, c]])
//     m_tmp = rot.T @ a[[p, q], [p, q]]
//     c_r, s_r = make_jacobi(m_tmp[0, 0], m_tmp[1, 1], m_tmp[0, 1])
//     rot_r = np.array([[c_r, s_r], [-s_r, c_r]])
//     rot_l = rot @ rot_r
//    return rot_l, rot_r
//
absl::StatusOr<OneSidedJacobiRotation> GetOneSidedJacobiRotation(XlaOp a,
                                                                 XlaOp p,
                                                                 XlaOp q,
                                                                 XlaOp eps) {
  XlaOp a_pp = DynamicSliceInMinorDims(a, {p, p}, {1, 1});
  XlaOp a_pq = DynamicSliceInMinorDims(a, {p, q}, {1, 1});
  XlaOp a_qp = DynamicSliceInMinorDims(a, {q, p}, {1, 1});
  XlaOp a_qq = DynamicSliceInMinorDims(a, {q, q}, {1, 1});

  XlaOp one = ScalarLike(a, 1.0);

  XlaOp t = a_pp + a_qq;
  XlaOp d = a_qp - a_pq;

  XlaOp u = Div(t, d);
  XlaOp tmp = Rsqrt(one + Square(u));

  JacobiRotation rot;

  XlaOp zeros = ZerosLike(tmp);
  XlaOp ones = zeros + one;

  rot.s = Select(Lt(Abs(d), eps), zeros, -tmp);
  rot.c = Select(Lt(Abs(d), eps), ones, Mul(u, tmp));

  XlaOp a_pp_new = rot.c * a_pp - rot.s * a_qp;
  XlaOp a_pq_new = rot.c * a_pq - rot.s * a_qq;
  XlaOp a_qq_new = rot.s * a_pq + rot.c * a_qq;

  OneSidedJacobiRotation rots;
  TF_ASSIGN_OR_RETURN(rots.rot_r,
                      MakeJacobi(a_pp_new, a_qq_new, a_pq_new, eps));

  rots.rot_l.c = rot.c * rots.rot_r.c - rot.s * rots.rot_r.s;
  rots.rot_l.s = rot.s * rots.rot_r.c + rot.c * rots.rot_r.s;

  return rots;
}

// Apply one-sided Jacobi on elements at indices pp, pq, qp, qq.
absl::StatusOr<SVDResult> OneSidedJacobiUpdate(SVDResult svd_result, XlaOp p,
                                               XlaOp q, XlaOp eps) {
  XlaOp u = svd_result.u;
  XlaOp v = svd_result.v;
  XlaOp d = svd_result.d;
  XlaBuilder* builder = d.builder();
  TF_ASSIGN_OR_RETURN(Shape d_shape, builder->GetShape(d));
  const int64_t num_dims = d_shape.dimensions().size();
  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(d_shape, i);
  }
  const int64_t m = ShapeUtil::GetDimension(d_shape, -2);
  const int64_t n = ShapeUtil::GetDimension(d_shape, -1);

  TF_ASSIGN_OR_RETURN(OneSidedJacobiRotation onesided_jacobi,
                      GetOneSidedJacobiRotation(d, p, q, eps));

  auto zero = ScalarLike(p, 0);

  // Zero out a_{pq} explicitly.
  std::vector<int64_t> pq_dims(batch_dims.begin(), batch_dims.end());
  pq_dims.push_back(1);
  pq_dims.push_back(1);
  auto pq_zero = ScalarLike(d, 0.0);
  auto pq_zeros = Broadcast(pq_zero, pq_dims);

  std::vector<int64_t> broadcast_dims(batch_dims.size());
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims.push_back(num_dims - 1);

  // Apply Jacobi Rotation on the left.
  auto slice_p = DynamicSliceInMinorDims(d, {p, zero}, {1, n});
  auto slice_q = DynamicSliceInMinorDims(d, {q, zero}, {1, n});
  auto slice_p_new =
      onesided_jacobi.rot_l.c * slice_p - onesided_jacobi.rot_l.s * slice_q;
  auto slice_q_new =
      onesided_jacobi.rot_l.s * slice_p + onesided_jacobi.rot_l.c * slice_q;
  d = DynamicUpdateSliceInMinorDims(d, slice_p_new, {p, zero});
  d = DynamicUpdateSliceInMinorDims(d, slice_q_new, {q, zero});

  // Apply Jacobi Rotation on the right.
  slice_p = DynamicSliceInMinorDims(d, {zero, p}, {m, 1});
  slice_q = DynamicSliceInMinorDims(d, {zero, q}, {m, 1});
  slice_p_new =
      onesided_jacobi.rot_r.c * slice_p - onesided_jacobi.rot_r.s * slice_q;
  slice_q_new =
      onesided_jacobi.rot_r.s * slice_p + onesided_jacobi.rot_r.c * slice_q;
  d = DynamicUpdateSliceInMinorDims(d, slice_p_new, {zero, p});
  d = DynamicUpdateSliceInMinorDims(d, slice_q_new, {zero, q});

  d = DynamicUpdateSliceInMinorDims(d, pq_zeros, {p, q});
  d = DynamicUpdateSliceInMinorDims(d, pq_zeros, {q, p});

  // Apply left Jacobi Rotation on U.
  slice_p = DynamicSliceInMinorDims(u, {zero, p}, {m, 1});
  slice_q = DynamicSliceInMinorDims(u, {zero, q}, {m, 1});
  slice_p_new =
      onesided_jacobi.rot_l.c * slice_p - onesided_jacobi.rot_l.s * slice_q;

  slice_p_new = Mul(
      slice_p_new,
      Rsqrt(Reduce(Square(slice_p_new), pq_zero,
                   CreateScalarAddComputation(d_shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  slice_q_new =
      onesided_jacobi.rot_l.s * slice_p + onesided_jacobi.rot_l.c * slice_q;

  slice_q_new = Mul(
      slice_q_new,
      Rsqrt(Reduce(Square(slice_q_new), pq_zero,
                   CreateScalarAddComputation(d_shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  u = DynamicUpdateSliceInMinorDims(u, slice_p_new, {zero, p});
  u = DynamicUpdateSliceInMinorDims(u, slice_q_new, {zero, q});

  // Apply right Jacobi Rotation on V.
  slice_p = DynamicSliceInMinorDims(v, {zero, p}, {n, 1});
  slice_q = DynamicSliceInMinorDims(v, {zero, q}, {n, 1});
  slice_p_new =
      onesided_jacobi.rot_r.c * slice_p - onesided_jacobi.rot_r.s * slice_q;

  slice_p_new = Mul(
      slice_p_new,
      Rsqrt(Reduce(Square(slice_p_new), pq_zero,
                   CreateScalarAddComputation(d_shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  slice_q_new =
      onesided_jacobi.rot_r.s * slice_p + onesided_jacobi.rot_r.c * slice_q;

  slice_q_new = Mul(
      slice_q_new,
      Rsqrt(Reduce(Square(slice_q_new), pq_zero,
                   CreateScalarAddComputation(d_shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  v = DynamicUpdateSliceInMinorDims(v, slice_p_new, {zero, p});
  v = DynamicUpdateSliceInMinorDims(v, slice_q_new, {zero, q});

  svd_result.d = d;
  svd_result.u = u;
  svd_result.v = v;

  return svd_result;
}

absl::StatusOr<XlaOp> ComputeToleranceComparison(XlaOp w, XlaOp epsilon) {
  XlaBuilder* builder = w.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(w));
  auto num_dims = static_cast<int32_t>(shape.dimensions().size());
  int64_t n = shape.dimensions(num_dims - 1);
  shape.set_dimensions(num_dims - 2, n);
  auto w_sliced = SliceInMinorDims(w, {0, 0}, {n, n});
  auto diag = GetMatrixDiagonal(w_sliced);
  diag = Select(Lt(diag, ZerosLike(diag)), -diag, diag);
  std::vector<int64_t> broadcasted_dims(num_dims - 1);
  std::iota(broadcasted_dims.begin(), broadcasted_dims.end(), 0);
  auto broadcast_to_rows =
      BroadcastInDim(diag, shape.dimensions(), broadcasted_dims);
  broadcasted_dims.back() = num_dims - 1;
  auto broadcast_to_columns =
      BroadcastInDim(diag, shape.dimensions(), broadcasted_dims);
  // Compute tolerance = w_{i,i} * w_{j,j} * epsilon^2
  // Use at least F32 precision to avoid precision issues with small denormal.
  XlaOp tolerance;
  if (builder->GetShape(epsilon)->element_type() == BF16 ||
      builder->GetShape(epsilon)->element_type() == F16) {
    auto upscale_eps = ConvertElementType(epsilon, F32);
    tolerance = ConvertElementType(broadcast_to_rows, F32) *
                ConvertElementType(broadcast_to_columns, F32) * upscale_eps *
                upscale_eps;
    // Convert back into the original precision.
    tolerance = ConvertElementType(tolerance,
                                   builder->GetShape(epsilon)->element_type());
  } else {
    tolerance = broadcast_to_rows * broadcast_to_columns * epsilon * epsilon;
  }
  // tolerance < (w_{i,j})^2
  return Lt(tolerance, Square(Select(GetDiagonalMask(w_sliced),
                                     ZerosLike(w_sliced), w_sliced)));
}

// Main boby of One-sided Jacobi Method.
absl::StatusOr<std::vector<XlaOp>> WhileLoopFn(
    absl::Span<const XlaOp> initial_values,  //
    int matrix_dimension,                    //
    int max_sweep_updates,                   //
    absl::string_view name,                  //
    XlaBuilder* builder) {
  auto while_cond_fn = [&](absl::Span<const XlaOp> values,
                           XlaBuilder* cond_builder) -> absl::StatusOr<XlaOp> {
    auto k = values[0];
    auto max_sweeps = ScalarLike(k, max_sweep_updates);
    auto sweep_update_cond = Gt(max_sweeps, k);

    TF_ASSIGN_OR_RETURN(auto tolerance_comparison,
                        ComputeToleranceComparison(values[3], values[4]));
    auto tolerance_cond = ReduceAll(
        tolerance_comparison, xla::ConstantR0<bool>(cond_builder, false),
        CreateScalarOrComputation(PRED, cond_builder));

    return And(sweep_update_cond, tolerance_cond);
  };

  auto while_body_fn =
      [&](absl::Span<const XlaOp> values,
          XlaBuilder* body_builder) -> absl::StatusOr<std::vector<XlaOp>> {
    auto while_cond_fn_inner =
        [&](absl::Span<const XlaOp> values_inner,
            XlaBuilder* inner_cond_builder) -> absl::StatusOr<XlaOp> {
      auto p = values_inner[0];
      return Lt(p, ScalarLike(p, matrix_dimension - 1));
    };

    auto while_body_fn_inner = [&](absl::Span<const XlaOp> values_inner,
                                   XlaBuilder* inner_body_builder)
        -> absl::StatusOr<std::vector<XlaOp>> {
      auto while_cond_fn_innermost =
          [&](absl::Span<const XlaOp> values_innermost,
              XlaBuilder* innermost_cond_builder) -> absl::StatusOr<XlaOp> {
        auto q = values_innermost[1];
        return Lt(q, ScalarLike(q, matrix_dimension));
      };
      auto while_body_fn_innermost =
          [&](absl::Span<const XlaOp> values_innermost,
              XlaBuilder* innermost_body_builder)
          -> absl::StatusOr<std::vector<XlaOp>> {
        auto p = values_innermost[0];
        auto q = values_innermost[1];

        SVDResult onesided_jacobi_update;
        onesided_jacobi_update.u = values_innermost[2];
        onesided_jacobi_update.v = values_innermost[3];
        onesided_jacobi_update.d = values_innermost[4];

        auto eps = values_innermost[5];

        TF_ASSIGN_OR_RETURN(
            onesided_jacobi_update,
            OneSidedJacobiUpdate(onesided_jacobi_update, p, q, eps));

        std::vector<XlaOp> updated_values_innermost;
        updated_values_innermost.reserve(values_innermost.size());

        updated_values_innermost.push_back(p);
        updated_values_innermost.push_back(q + ScalarLike(q, 1));
        updated_values_innermost.push_back(onesided_jacobi_update.u);
        updated_values_innermost.push_back(onesided_jacobi_update.v);
        updated_values_innermost.push_back(onesided_jacobi_update.d);
        updated_values_innermost.push_back(eps);

        return updated_values_innermost;
      };

      std::vector<XlaOp> values_innermost(6);
      auto p = values_inner[0];
      auto q = p + ScalarLike(p, 1);
      values_innermost[0] = p;                // index p.
      values_innermost[1] = q;                // index q.
      values_innermost[2] = values_inner[1];  // u.
      values_innermost[3] = values_inner[2];  // v.
      values_innermost[4] = values_inner[3];  // d.
      values_innermost[5] = values_inner[4];  // eps.
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
      updated_values_inner.push_back(values_innermost[5]);
      return updated_values_inner;
    };
    // Indexes.
    XlaOp k = values[0];

    std::vector<XlaOp> values_inner(5);
    values_inner[0] = ScalarLike(k, 0);  // index p.
    values_inner[1] = values[1];         // u.
    values_inner[2] = values[2];         // v.
    values_inner[3] = values[3];         // d.
    values_inner[4] = values[4];         // eps.
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
    updated_values.push_back(values_inner[4]);

    return updated_values;
  };
  std::vector<XlaOp> values;
  TF_ASSIGN_OR_RETURN(values, WhileLoopHelper(while_cond_fn, while_body_fn,
                                              initial_values, name, builder));

  return values;
}

// Sort singular values in descending order, and make sure they are non-negative
// by flipping the signs of negative diagonal values and transferring the signs
// to V. And for numeric stability, renormalize U and V.
absl::StatusOr<SVDResult> SortBySingularValuesAndPostProcessing(
    SVDResult result) {
  XlaBuilder* builder = result.d.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(result.d));
  const int64_t num_dims = shape.dimensions().size();
  auto dimensions = shape.dimensions();
  const int64_t m = ShapeUtil::GetDimension(shape, -2);
  const int64_t n = ShapeUtil::GetDimension(shape, -1);

  std::vector<int64_t> broadcast_dims(num_dims - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims[num_dims - 2] = num_dims - 1;

  auto d = GetMatrixDiagonal(result.d);

  auto zeros = ZerosLike(d);
  auto one = ScalarLike(d, 1.0);

  // Make all the singular values to be non-negative by transferring the signs
  // to V.
  auto sign = Select(Ge(d, zeros), zeros + one, zeros - one);
  d = Select(Ge(d, zeros), d, -d);
  result.v = Mul(result.v, sign, broadcast_dims);

  d = BroadcastInDim(d, dimensions, broadcast_dims);

  // As m >= n, only first n column vectors need to be permuted, and the rest of
  // m - n vectors are appended after the sorting is done.
  XlaOp sort_u_result =
      Sort({d, SliceInMinorDims(result.u, {0, 0}, {m, n})},
           CreateScalarGtComputation(
               {shape.element_type(), shape.element_type()}, builder),
           num_dims - 1);

  XlaOp sort_v_result =
      Sort({SliceInMinorDims(d, {0, 0}, {n, n}), result.v},
           CreateScalarGtComputation(
               {shape.element_type(), shape.element_type()}, builder),
           num_dims - 1);
  result.d = GetMatrixDiagonal(GetTupleElement(sort_v_result, 0));

  result.v = GetTupleElement(sort_v_result, 1);
  result.v = Mul(
      result.v,
      Rsqrt(Reduce(Square(result.v), ScalarLike(d, 0.0),
                   CreateScalarAddComputation(shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  // Append the rest of m - n vectors.
  result.u = ConcatInDim(builder,
                         {GetTupleElement(sort_u_result, 1),
                          SliceInMinorDims(result.u, {0, n}, {m, m})},
                         num_dims - 1);
  result.u = Mul(
      result.u,
      Rsqrt(Reduce(Square(result.u), ScalarLike(d, 0.0),
                   CreateScalarAddComputation(shape.element_type(), builder),
                   {num_dims - 2})),
      broadcast_dims);

  return result;
}

}  // namespace

// def jacobi_svd(A):
//    U, D, V = house_bidiag(A)
//    m, n = D.shape
//    iter, max_iter = 0, 100
//    frobenius_norm = np.linalg.norm(D)
//    diag_norm = np.linalg.norm(np.diag(D))
//    off_diag_norm = np.sqrt(
//        frobenius_norm - diag_norm) * np.sqrt(frobenius_norm + diag_norm)
//    while off_diag_norm > 1e-6 * frobenius_norm and iter < max_iter:
//        iter += 1
//        for p in range(m - 1):
//            for q in range(p + 1, n):
//                rot_l, rot_r = jacobi_rot(D[p][p], D[p][q], D[q][p], D[q][q])
//                D[[p, q], :] = np.matmul(rot_l.T, D[[p, q], :])
//                D[:, [p, q]] = np.matmul(D[:, [p, q]], rot_r)
//                U[:, [p, q]] = np.matmul(U[:, [p, q]], rot_l)
//                V[:, [p, q]] = np.matmul(V[:, [p, q]], rot_r)
//        frobenius_norm = np.linalg.norm(D)
//        diag_norm = np.linalg.norm(np.diag(D))
//        off_diag_norm = np.sqrt(
//            frobenius_norm - diag_norm) * np.sqrt(frobenius_norm + diag_norm)
//
//    return U, np.diag(D), V
//
SVDResult SVD(XlaOp a, int64_t max_iter, float epsilon,
              PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  auto return_error = [&](const absl::Status& status) {
    SVDResult result;
    result.u = builder->ReportError(status);
    result.v = builder->ReportError(status);
    result.d = builder->ReportError(status);
    return result;
  };
  auto shape_with_status = builder->GetShape(a);
  if (!shape_with_status.status().ok()) {
    return return_error(shape_with_status.status());
  }
  Shape a_shape = shape_with_status.value();
  const int64_t num_dims = a_shape.dimensions().size();
  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }
  int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  bool maybe_transpose = m < n;

  if (maybe_transpose) {
    a = TransposeInMinorDims(a);
    std::swap(m, n);
  }

  auto eps = ScalarLike(a, epsilon);

  auto svd_result_or = HouseHolderBidiagonalization(a, eps, precision);
  if (!svd_result_or.ok()) {
    return return_error(svd_result_or.status());
  }
  SVDResult svd_result = svd_result_or.value();

  auto output_with_status = WhileLoopFn(
      {
          Zero(builder, S32),  // k
          svd_result.u,        // u
          svd_result.v,        // v
          svd_result.d,        // d
          eps,                 // epsilon
      },                       //
      n,                       //
      max_iter,                //
      "CyclicOneSidedJacobi",  //
      builder);
  if (!output_with_status.status().ok()) {
    return return_error(output_with_status.status());
  }

  auto output = output_with_status.value();

  svd_result.u = output[1];
  svd_result.v = output[2];
  svd_result.d = output[3];

  svd_result_or = SortBySingularValuesAndPostProcessing(svd_result);
  if (!svd_result_or.ok()) {
    return return_error(svd_result_or.status());
  }
  svd_result = svd_result_or.value();

  if (maybe_transpose) {
    std::swap(svd_result.u, svd_result.v);
  }
  return svd_result;
}

}  // namespace xla
