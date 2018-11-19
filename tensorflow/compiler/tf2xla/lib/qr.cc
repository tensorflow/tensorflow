/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/qr.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/lib/while_loop.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

// Computes a Householder reflection of the form:
// H = I - tau v v.T.
// such that
// H . ( x1  ) = ( x1   )
//     ( x2  ) = ( x2   )
//     ( ... ) = ( ...  )
//     ( xk  ) = ( beta )
//     ( ... )   ( 0    )
//     ( ... )   ( 0    )
// Unlike the usual formulation, we allow the caller to supply 'k' rather than
// only providing the relevant part of 'x' to maintain XLA's static shape
// invariant. In addition, the implementation supports batching.
// Pseudo-code, without batching:
//   alpha = x[k]
//   x_copy = np.copy(x)
//   x_copy[:k+1] = 0
//   xnorm = norm2(x_copy)
//   if xnorm == 0:
//     beta = alpha
//     tau = 0
//     v = np.zeros_like(x)
//   else:
//     beta = - np.sign(alpha) * dlapy2(alpha, xnorm)
//     tau = (beta - alpha) / beta
//     v = x / (alpha - beta)
//   v[k] = 1
//   return (v, tau, beta)
// TODO(phawkins): LAPACK's xLARFG implementation has code for handling
// overflows in the norm/beta calculations. Perhaps do the same here.
xla::Status House(xla::XlaOp x, xla::XlaOp k,
                  absl::Span<const int64> batch_dims, const int64 m,
                  xla::XlaOp* v, xla::XlaOp* tau, xla::XlaOp* beta) {
  xla::XlaBuilder* const builder = x.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape x_shape, builder->GetShape(x));
  const xla::PrimitiveType type = x_shape.element_type();

  std::vector<int64> batch_dim_ids(batch_dims.size());
  std::iota(batch_dim_ids.begin(), batch_dim_ids.end(), 0);
  const int64 minor_dim = batch_dims.size();

  xla::XlaOp zero = xla::ScalarLike(x, 0.0);
  xla::XlaOp one = xla::ScalarLike(x, 1.0);

  // alpha = x[k]
  xla::XlaOp alpha =
      xla::Reshape(DynamicSliceInMinorDims(x, {k}, {1}), batch_dims);

  // Compute x[k+1:] (padded with zeros in elements 0..k)
  xla::XlaOp iota = xla::Iota(builder, xla::S32, m);
  xla::XlaOp x_after_k =
      xla::Mul(x, xla::ConvertElementType(xla::Gt(iota, k), type),
               /*broadcast_dimensions=*/{minor_dim});

  // sigma = np.dot(x[k+1:], x[k+1:])
  auto sigma =
      xla::Reduce(x_after_k * x_after_k, zero,
                  xla::CreateScalarAddComputation(type, builder), {minor_dim});
  // mu = np.sqrt(x[k]*x[k] + sigma)
  auto mu = xla::Sqrt(xla::Square(alpha) + sigma);

  auto sigma_is_zero = xla::Eq(sigma, zero);

  *beta = xla::Select(sigma_is_zero, alpha, -xla::Sign(alpha) * mu);
  *tau = xla::Select(sigma_is_zero, xla::Broadcast(zero, batch_dims),
                     (*beta - alpha) / *beta);
  auto divisor = xla::Select(sigma_is_zero, xla::Broadcast(one, batch_dims),
                             alpha - *beta);

  auto e_k = xla::Broadcast(xla::ConvertElementType(xla::Eq(iota, k), type),
                            std::vector<int64>(batch_dims.size(), 1));

  // Form v as [0, 0, ..., 1] ++ x[k+1:] / divisor
  // If sigma is zero, x[k+1:] is zero, so use any non-zero divisor.
  *v = e_k +
       xla::Div(x_after_k, divisor, /*broadcast_dimensions=*/batch_dim_ids);
  return Status::OK();
}

// Householder QR decomposition. Algorithm 5.2.1 from Golub and Van
// Loan "Matrix Computations", 4th Edition. This is an unblocked implementation
// used as an inner routine of the blocked implementation.
// Algorithm is adapted slightly so the shapes inside the loop are static, at
// the cost of some redundant computation. Since this is used as an inner block
// kernel, accumulates the Householder transformations (vs, taus) rather than
// the matrix q.
// Equivalent Python code, without batching:
// def qr(a):
//   m = a.shape[0]
//   n = a.shape[1]
//   vs = np.zeros([m, n])
//   taus = np.zeros([n])
//   for j in xrange(min(m, n)):
//     v, tau, beta = house(a[:, j], j)
//     # Unusually, we apply the Householder transformation to the entirety of
//     # a, wasting FLOPs to maintain the static shape invariant that XLA
//     # requires. For columns that precede j this has no effect.
//     a[:, :] -= tau * np.dot(v[:, np.newaxis],
//                              np.dot(v[np.newaxis, :], a[:, :]))
//     # Form column j explicitly rather than relying on the precision of the
//     # Householder update.
//     a[j, j] = beta
//     a[j+1:, j] = np.zeros([m - j - 1], dtype=a.dtype)
//     vs[:, j] = v
//     taus[j] = tau
//   return (q, vs, taus)
struct QRBlockResult {
  // The factored R value
  xla::XlaOp r;

  // Representation of the Householder matrices I - beta v v.T
  xla::XlaOp taus;  // Shape: [..., n]
  xla::XlaOp vs;    // Shape: [..., m, n]
};
xla::StatusOr<QRBlockResult> QRBlock(
    xla::XlaOp a, xla::PrecisionConfig::Precision precision) {
  xla::XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
  const int num_dims = xla::ShapeUtil::Rank(a_shape);
  if (num_dims < 2) {
    return errors::InvalidArgument("Arguments to QR must have rank >= 2: ",
                                   num_dims);
  }
  xla::PrimitiveType type = a_shape.element_type();

  const int64 m = xla::ShapeUtil::GetDimension(a_shape, -2);
  const int64 n = xla::ShapeUtil::GetDimension(a_shape, -1);

  const int64 num_batch_dims = num_dims - 2;
  std::vector<int64> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = xla::ShapeUtil::GetDimension(a_shape, i);
  }

  std::vector<int64> batch_dim_indices(num_batch_dims);
  std::iota(batch_dim_indices.begin(), batch_dim_indices.end(), 0);

  auto qr_body_fn =
      [&](xla::XlaOp j, absl::Span<const xla::XlaOp> values,
          xla::XlaBuilder* builder) -> xla::StatusOr<std::vector<xla::XlaOp>> {
    auto a = values[0];
    auto vs = values[1];
    auto taus = values[2];

    // v, beta = house(a[:, j], j)
    auto x = DynamicSliceInMinorDims(a, {j}, {1});
    xla::XlaOp v, tau, beta;
    TF_RETURN_IF_ERROR(House(xla::Collapse(x, {num_dims - 2, num_dims - 1}), j,
                             batch_dims, m, &v, &tau, &beta));

    std::vector<int64> shape = batch_dims;
    shape.push_back(1);
    shape.push_back(m);
    auto v_broadcast = xla::Reshape(v, shape);
    // a[:, :] -= tau * np.dot(v[:, np.newaxis],
    //                          np.dot(v[np.newaxis, :], a[:, :]))
    auto vva =
        BatchDot(v_broadcast, a, /*transpose_x=*/false, /*transpose_y=*/false,
                 /*conjugate_x=*/false, /*conjugate_y=*/false, precision);
    vva =
        BatchDot(v_broadcast, vva, /*transpose_x=*/true, /*transpose_y=*/false,
                 /*conjugate_x=*/false, /*conjugate_y=*/false, precision);
    a = a - xla::Mul(tau, vva,
                     /*broadcast_dimensions=*/batch_dim_indices);

    // It is more precise to populate column 'k' explicitly, rather than
    // computing it implicitly by applying the Householder transformation.
    // a[k,k] = beta
    // a[k+1:,k] = np.zeros([m-k-1], dtype=a.dtype)
    auto iota = xla::Reshape(xla::Iota(a.builder(), xla::S32, m), {m, 1});
    auto predecessor_mask = xla::ConvertElementType(xla::Lt(iota, j), type);
    auto mask = xla::Broadcast(xla::ConvertElementType(xla::Eq(iota, j), type),
                               std::vector<int64>(batch_dims.size(), 1));
    auto new_x =
        xla::Mul(x, predecessor_mask,
                 /*broadcast_dimensions=*/{num_dims - 2, num_dims - 1}) +
        xla::Mul(beta, mask, /*broadcast_dimensions=*/batch_dim_indices);
    a = DynamicUpdateSliceInMinorDims(a, new_x, {j});

    // vs[:, j] = v
    vs = DynamicUpdateSliceInMinorDims(
        vs, xla::Reshape(v, ConcatVectors(batch_dims, {m, 1})), {j});
    // taus[j] = tau
    taus = DynamicUpdateSliceInMinorDims(
        taus, xla::Reshape(tau, ConcatVectors(batch_dims, {1})), {j});
    return std::vector<xla::XlaOp>{a, vs, taus};
  };

  auto vs = xla::Zeros(builder, xla::ShapeUtil::MakeShape(
                                    type, ConcatVectors(batch_dims, {m, n})));
  auto taus = xla::Zeros(
      builder, xla::ShapeUtil::MakeShape(type, ConcatVectors(batch_dims, {n})));

  TF_ASSIGN_OR_RETURN(auto values,
                      XlaForEachIndex(std::min(m, n), xla::S32, qr_body_fn,
                                      {a, vs, taus}, "qr", builder));

  QRBlockResult result;
  result.r = values[0];
  result.vs = values[1];
  result.taus = values[2];
  return result;
}

// Computes W and Y such that I-WY is equivalent to the sequence of Householder
// transformations given by vs and taus.
// Golub and van Loan, "Matrix Computations", algorithm 5.1.2.
// Y = np.zeros([m, n])
// W = np.zeros([m, n])
// Y[:, 0] = vs[:, 0]
// W[:, 0] = -taus[0] * vs[:, 0]
// for j in xrange(1, n):
//   v = vs[:, j]
//   z = -taus[j] * v - taus[j] * np.dot(W, np.dot(Y.T, v))
//   W[:, j] = z
//   Y[:, j] = v
// return W
// There is no need to return Y since at termination of the loop it is equal to
// vs.
xla::StatusOr<xla::XlaOp> ComputeWYRepresentation(
    xla::PrimitiveType type, absl::Span<const int64> batch_dims, xla::XlaOp vs,
    xla::XlaOp taus, int64 m, int64 n,
    xla::PrecisionConfig::Precision precision) {
  std::vector<int64> batch_dim_indices(batch_dims.size());
  std::iota(batch_dim_indices.begin(), batch_dim_indices.end(), 0);
  int64 n_index = batch_dims.size() + 1;

  auto body_fn =
      [&](xla::XlaOp j, absl::Span<const xla::XlaOp> values,
          xla::XlaBuilder* builder) -> xla::StatusOr<std::vector<xla::XlaOp>> {
    auto w = values[0];
    auto y = values[1];
    const auto vs = values[2];
    const auto taus = values[3];

    // Want j values in range [1, ... n).
    j = j + xla::ConstantR0<int32>(builder, 1);
    // vs has shape [..., m, 1]
    auto v = DynamicSliceInMinorDims(vs, {j}, {1});
    // beta has shape [..., 1]
    auto beta = DynamicSliceInMinorDims(taus, {j}, {1});

    // yv has shape [..., n, 1]
    auto yv = BatchDot(y, v, /*transpose_x=*/true, /*transpose_y=*/false,
                       /*conjugate_x=*/false, /*conjugate_y=*/false, precision);
    // wyv has shape [..., m, 1]
    auto wyv =
        BatchDot(w, yv, /*transpose_x=*/false, /*transpose_y=*/false,
                 /*conjugate_x=*/false, /*conjugate_y=*/false, precision);

    auto z = xla::Mul(
        -beta, v + wyv,
        /*broadcast_dimensions=*/ConcatVectors(batch_dim_indices, {n_index}));

    w = DynamicUpdateSliceInMinorDims(w, z, {j});
    y = DynamicUpdateSliceInMinorDims(y, v, {j});

    return std::vector<xla::XlaOp>{w, y, vs, taus};
  };

  xla::XlaBuilder* builder = vs.builder();
  auto w = xla::Zeros(builder, xla::ShapeUtil::MakeShape(
                                   type, ConcatVectors(batch_dims, {m, n})));
  auto y = w;
  auto v = SliceInMinorDims(vs, {0}, {1});
  auto beta = SliceInMinorDims(taus, {0}, {1});
  y = UpdateSliceInMinorDims(y, v, {0});
  auto bv = xla::Mul(
      -beta, v,
      /*broadcast_dimensions=*/ConcatVectors(batch_dim_indices, {n_index}));
  w = UpdateSliceInMinorDims(w, bv, {0});

  TF_ASSIGN_OR_RETURN(
      auto values, XlaForEachIndex(n - 1, xla::S32, body_fn, {w, y, vs, taus},
                                   "wy", builder));
  return values[0];
}

}  // namespace

// Block Householder QR Factorization. Algorithm 5.2.2 of Golub and van Loan.
// def qr_blocked(a, block_size):
//   m = a.shape[0]
//   n = a.shape[1]
//   q = np.eye(m)
//   for i in xrange(0, min(m, n), block_size):
//     k = min(block_size, min(m, n) - s)
//     (a, vs, taus) = qr(a[i:, i:i+k])
//     y = vs
//     w = ComputeWYRepresentation(vs, taus, m-i, k)
//     a[i:, i+r:] += np.dot(y, np.dot(w.T, a[i:, i+k:]))
//     q[:, i:] += np.dot(q[:, i:], np.dot(w, y.T))
//   return (q, a)
// TODO(phawkins): consider using UT transformations (in the form I - V U V')
// rather than WY transformations.
xla::StatusOr<QRDecompositionResult> QRDecomposition(
    xla::XlaOp a, bool full_matrices, int64 block_size,
    xla::PrecisionConfig::Precision precision) {
  xla::XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
  const int num_dims = xla::ShapeUtil::Rank(a_shape);
  if (num_dims < 2) {
    return errors::InvalidArgument("Arguments to QR must have rank >= 2: ",
                                   num_dims);
  }
  xla::PrimitiveType type = a_shape.element_type();

  const int64 m = xla::ShapeUtil::GetDimension(a_shape, -2);
  const int64 n = xla::ShapeUtil::GetDimension(a_shape, -1);
  const int64 p = std::min(m, n);

  if (block_size < 1) {
    return errors::InvalidArgument(
        "block_size argument to QR must be >= 1; got ", block_size);
  }

  const int64 num_batch_dims = num_dims - 2;
  std::vector<int64> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = xla::ShapeUtil::GetDimension(a_shape, i);
  }

  auto q = xla::Broadcast(xla::IdentityMatrix(builder, type, m, m), batch_dims);
  for (int64 i = 0; i < p; i += block_size) {
    int64 k = std::min(block_size, p - i);

    auto a_block = SliceInMinorDims(a, {i, i}, {m, i + k});
    TF_ASSIGN_OR_RETURN(auto qr_block, QRBlock(a_block, precision));

    a = UpdateSliceInMinorDims(a, qr_block.r, {i, i});

    // Compute the I-WY block representation of a product of Householder
    // matrices.
    TF_ASSIGN_OR_RETURN(
        auto w, ComputeWYRepresentation(type, batch_dims, qr_block.vs,
                                        qr_block.taus, m - i, k, precision));
    auto y = qr_block.vs;

    // a[i:, i+k:] += np.dot(Y, np.dot(W.T, a[i:, i+k:]))
    auto a_panel = SliceInMinorDims(a, {i, i + k}, {m, n});
    auto a_update =
        BatchDot(w, a_panel, /*transpose_x=*/true, /*transpose_y=*/false,
                 /*conjugate_x=*/false, /*conjugate_y=*/false, precision);
    a_update =
        BatchDot(y, a_update, /*transpose_x=*/false, /*transpose_y=*/false,
                 /*conjugate_x=*/false, /*conjugate_y=*/false, precision);
    a_panel = a_panel + a_update;
    a = UpdateSliceInMinorDims(a, a_panel, {i, i + k});

    // q[:, i:] += np.dot(np.dot(q[:, i:], W), Y.T))
    auto q_panel = SliceInMinorDims(q, {0, i}, {m, m});
    auto q_update =
        BatchDot(q_panel, w, /*transpose_x=*/false, /*transpose_y=*/false,
                 /*conjugate_x=*/false, /*conjugate_y=*/false, precision);
    q_update = BatchDot(q_update, y, /*transpose_x=*/false,
                        /*transpose_y=*/true, /*conjugate_x=*/false,
                        /*conjugate_y=*/false, precision);
    q_panel = q_panel + q_update;
    q = UpdateSliceInMinorDims(q, q_panel, {0, i});
  }
  QRDecompositionResult result;

  // full_matrices is false when only a partial result in needed. Slice to the
  // needed dimensions here.
  if (!full_matrices) {
    q = SliceInMinorDims(q, {0, 0}, {m, p});
    a = SliceInMinorDims(a, {0, 0}, {p, n});
  }
  result.q = q;
  result.r = a;
  return result;
}

}  // namespace tensorflow
