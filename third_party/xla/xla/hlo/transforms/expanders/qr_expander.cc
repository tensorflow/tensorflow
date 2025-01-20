/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/qr_expander.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/loops.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/qr.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

std::vector<int64_t> ConcatVectors(absl::Span<const int64_t> xs,
                                   absl::Span<const int64_t> ys) {
  std::vector<int64_t> output;
  output.reserve(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), std::back_inserter(output));
  std::copy(ys.begin(), ys.end(), std::back_inserter(output));
  return output;
}

// Computes sqrt(x^2 + y^2 + ...), avoiding overflow/underflow.
// e.g. for 3 arguments:
// def norm(x, y, z):
//   xabs = np.abs(x)
//   yabs = np.abs(y)
//   zabs = np.abs(z)
//   w = np.maximum(np.maximum(xabs, yabs), zabs)
//   if w == 0:
//     return 0
//   else:
//     return w * np.sqrt((xabs / w)**2 + (yabs / w) ** 2 + (zabs / w) ** 2)
XlaOp Norm(std::vector<XlaOp> xs) {
  CHECK(!xs.empty());
  XlaOp w;
  for (size_t i = 0; i < xs.size(); ++i) {
    xs[i] = Abs(xs[i]);
    w = i == 0 ? xs[i] : xla::Max(w, xs[i]);
  }

  XlaOp out;
  for (size_t i = 0; i < xs.size(); ++i) {
    XlaOp t = Square(xs[i] / w);
    out = i == 0 ? t : xla::Add(out, t);
  }
  return Select(Eq(w, ZerosLike(w)), ZerosLike(w), w * Sqrt(out));
}

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
//   if xnorm == 0 and np.imag(alpha) == 0:
//     beta = alpha
//     tau = 0
//     v = np.zeros_like(x)
//   else:
//     beta = -np.sign(np.real(alpha)) * np.sqrt(alpha * np.conj(alpha) + xnorm)
//     if np.issubdtype(x.dtype, np.complexfloating):
//       tau = (beta - alpha) / beta
//     else:
//       tau = (beta - np.real(alpha) / beta) + (-np.imag(alpha) / beta) * 1j
//     v = x / (alpha - beta)
//   v[k] = 1
//   return (v, tau, beta)
// TODO(phawkins): LAPACK's xLARFG implementation has code for handling
// overflows in the norm/beta calculations. Perhaps do the same here.
absl::Status House(XlaOp x, XlaOp k, absl::Span<const int64_t> batch_dims,
                   const int64_t m, XlaOp* v, XlaOp* tau, XlaOp* beta) {
  XlaBuilder* const builder = x.builder();
  TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
  const PrimitiveType type = x_shape.element_type();

  std::vector<int64_t> batch_dim_ids(batch_dims.size());
  std::iota(batch_dim_ids.begin(), batch_dim_ids.end(), 0);
  const int64_t minor_dim = batch_dims.size();

  XlaOp zero = ScalarLike(x, 0.0);

  // alpha = x[k]
  XlaOp alpha = Reshape(DynamicSliceInMinorDims(x, {k}, {1}), batch_dims);

  // Compute x[k+1:] (padded with zeros in elements 0..k)
  XlaOp iota = Iota(builder, S32, m);
  XlaOp x_after_k = Mul(x, ConvertElementType(Gt(iota, k), type),
                        /*broadcast_dimensions=*/{minor_dim});

  XlaOp sigma_is_zero;
  if (primitive_util::IsComplexType(type)) {
    // sigma = np.dot(x[k+1:], np.conj(x[k+1:]))
    auto x_squared = Real(x_after_k * Conj(x_after_k));
    auto sigma =
        Reduce(x_squared, ScalarLike(x_squared, 0.0),
               CreateScalarAddComputation(
                   primitive_util::ComplexComponentType(type), builder),
               {minor_dim});
    auto mu = Norm({Real(alpha), Imag(alpha), Sqrt(sigma)});

    sigma_is_zero = Eq(sigma, ScalarLike(sigma, 0));
    sigma_is_zero = And(sigma_is_zero, Eq(Imag(alpha), ScalarLike(sigma, 0)));

    *beta = Select(Lt(Real(alpha), ScalarLike(sigma, 0)), ScalarLike(mu, 1),
                   ScalarLike(mu, -1)) *
            mu;
    *beta = Select(sigma_is_zero, Real(alpha), *beta);
    *tau = Complex((*beta - Real(alpha)) / *beta, -Imag(alpha) / *beta);
  } else {
    // sigma = np.dot(x[k+1:], x[k+1:])
    auto sigma = Reduce(x_after_k * x_after_k, zero,
                        CreateScalarAddComputation(type, builder), {minor_dim});
    auto mu = Norm({alpha, Sqrt(sigma)});
    sigma_is_zero = Eq(sigma, zero);

    XlaOp one = ScalarLike(x, 1.0);
    *beta = Select(Lt(alpha, zero), one, -one) * mu;
    *beta = Select(sigma_is_zero, alpha, *beta);
    *tau = (*beta - alpha) / *beta;
  }
  *tau = Select(sigma_is_zero, ZerosLike(*tau), *tau);

  auto divisor =
      Select(sigma_is_zero, Broadcast(ScalarLike(alpha, 1), batch_dims),
             alpha - ConvertElementType(*beta, type));

  auto e_k = Broadcast(ConvertElementType(Eq(iota, k), type),
                       std::vector<int64_t>(batch_dims.size(), 1));

  // Form v as [0, 0, ..., 1] ++ x[k+1:] / divisor
  // If sigma is zero, x[k+1:] is zero, so use any non-zero divisor.
  *v = e_k + Div(x_after_k, divisor, /*broadcast_dimensions=*/batch_dim_ids);
  return absl::OkStatus();
}

}  // namespace

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
//   taus = np.zeros([n])
//   for j in xrange(min(m, n)):
//     v, tau, beta = house(a[:, j], j)
//     a[:, j+1:] -= np.conj(tau) * np.dot(v[:, np.newaxis],
//                                np.dot(np.conj(v[np.newaxis, :]), a[:, j+1:]))
//     # Form column j explicitly rather than relying on the precision of the
//     # Householder update.
//     a[j, j] = beta
//     a[j+1:, j] = v[j+1:]
//     taus[j] = tau
//   return (a, taus)
absl::StatusOr<QrDecomposition> QrExpander::QrBlock(
    XlaOp a, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int num_dims = a_shape.rank();
  if (num_dims < 2) {
    return InvalidArgument("Argument to QR must have rank >= 2; got shape %s",
                           a_shape.ToString());
  }
  PrimitiveType type = a_shape.element_type();

  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);

  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }

  std::vector<int64_t> batch_dim_indices(num_batch_dims);
  std::iota(batch_dim_indices.begin(), batch_dim_indices.end(), 0);

  auto qr_body_fn =
      [&](XlaOp j, absl::Span<const XlaOp> values,
          XlaBuilder* builder) -> absl::StatusOr<std::vector<XlaOp>> {
    auto a = values[0];
    auto taus = values[1];

    // v, tau, beta = house(a[:, j], j)
    auto x = DynamicSliceInMinorDims(a, {j}, {1});
    XlaOp v, tau, beta;
    TF_RETURN_IF_ERROR(House(Collapse(x, {num_dims - 2, num_dims - 1}), j,
                             batch_dims, m, &v, &tau, &beta));

    const int64_t minor_dim = batch_dims.size();
    auto iota_mn = Iota(
        builder, ShapeUtil::MakeShape(S32, ConcatVectors(batch_dims, {m, n})),
        minor_dim + 1);

    std::vector<int64_t> shape = batch_dims;
    shape.push_back(1);
    shape.push_back(m);
    auto v_broadcast = Reshape(v, shape);
    // a[:, j+1:] -= np.conj(tau) * (v[:, np.newaxis] @
    //     (np.conj(v[np.newaxis, :]) @ a[:, j+1:]))
    // We use masking rather than a loop-variant shape to handle the j+1:
    // indexing.
    auto vva = BatchDot(MaybeConjugate(v_broadcast, true),
                        Select(Lt(j, iota_mn), a, ZerosLike(a)), precision);
    vva = BatchDot(v_broadcast, true, vva, false, precision);
    a = a - Mul(MaybeConjugate(tau, true), vva,
                /*broadcast_dimensions=*/batch_dim_indices);

    // a[j, j] = beta
    // a[j+1:,j] = v[j+1:]
    auto iota = Reshape(Iota(a.builder(), S32, m), {m, 1});
    auto predecessor_mask = ConvertElementType(Lt(iota, j), type);
    auto mask = Broadcast(ConvertElementType(Eq(iota, j), type),
                          std::vector<int64_t>(batch_dims.size(), 1));
    auto successor_mask = Gt(Iota(a.builder(), S32, m), j);
    auto new_x = Mul(x, predecessor_mask,
                     /*broadcast_dimensions=*/{num_dims - 2, num_dims - 1}) +
                 Mul(ConvertElementType(beta, type), mask,
                     /*broadcast_dimensions=*/batch_dim_indices);
    new_x = Add(
        new_x, Select(Broadcast(successor_mask, batch_dims), v, ZerosLike(v)),
        /*broadcast_dimensions=*/ConcatVectors(batch_dim_indices, {minor_dim}));
    // Update a[:,j]
    std::vector<int64_t> dim_ids(num_dims);
    std::iota(dim_ids.begin(), dim_ids.end(), 0);
    new_x = BroadcastInDim(new_x, ConcatVectors(batch_dims, {m, n}),
                           /*broadcast_dimensions=*/dim_ids);
    a = Select(Eq(iota_mn, j), new_x, a);

    // taus[j] = tau
    std::vector<int64_t> tau_broadcast_dims(batch_dims.size());
    std::iota(tau_broadcast_dims.begin(), tau_broadcast_dims.end(), 0);

    auto iota_n =
        Iota(builder, ShapeUtil::MakeShape(S32, ConcatVectors(batch_dims, {n})),
             minor_dim);
    auto taus_zeros = ZerosLike(taus);
    auto taus_update = Select(
        Eq(iota_n, j),
        Add(taus_zeros, tau, /*broadcast_dimensions=*/tau_broadcast_dims),
        taus_zeros);
    taus = taus + taus_update;
    return std::vector<XlaOp>{a, taus};
  };

  auto taus = Zeros(
      builder,
      ShapeUtil::MakeShape(type, ConcatVectors(batch_dims, {std::min(m, n)})));

  TF_ASSIGN_OR_RETURN(auto values, ForEachIndex(std::min(m, n), S32, qr_body_fn,
                                                {a, taus}, "qr", builder));

  QrDecomposition result;
  result.q_and_r = values[0];
  result.taus = values[1];
  return result;
}

// Computes an upper triangular matrix T such that (I - Y @ T @ Y^t) is a
// product of the elementary Householder reflectors given by `vs` and `taus`.
//
// Schreiber, Robert, and Charles Van Loan. "A storage-efficient WY
// representation for products of Householder transformations." SIAM Journal on
// Scientific and Statistical Computing 10.1 (1989): 53-57.
//
// def compact_wy(vs, taus):
//   m, n = vs.shape[-2:]
//   t = np.eye(n) * -taus
//   # We premultiply Y.T @ vs, since we would prefer to compute a single matrix
//   # multiplication to many matrix-vector products.
//   vtv = -taus[None, :] * np.triu(np.conj(vs.T) @ vs, 1) + np.eye(n)
//   for i in range(1, n):
//     t[:, i] = scipy.linalg.blas.strmm(t, vtv[:, i])
//   return t
absl::StatusOr<XlaOp> QrExpander::CompactWYRepresentation(
    PrimitiveType type, absl::Span<const int64_t> batch_dims, XlaOp vs,
    XlaOp taus, int64_t m, int64_t n, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = vs.builder();

  std::vector<int64_t> batch_dim_indices(batch_dims.size());
  std::iota(batch_dim_indices.begin(), batch_dim_indices.end(), 0);
  int64_t n_index = batch_dims.size() + 1;

  auto body_fn =
      [&](XlaOp j, absl::Span<const XlaOp> values,
          XlaBuilder* builder) -> absl::StatusOr<std::vector<XlaOp>> {
    // w has shape [..., m, n]
    auto t = values[0];
    const auto vtv = values[1];

    // yv has shape [..., n, 1]
    auto yv = DynamicSliceInMinorDims(vtv, {j}, {1});

    // z has shape [..., n, 1]
    auto z = BatchDot(t, yv, precision);

    t = DynamicUpdateSliceInMinorDims(t, z, {j});

    return std::vector<XlaOp>{t, vtv};
  };

  auto tau_scale = BroadcastInDim(-taus, ConcatVectors(batch_dims, {1, n}),
                                  ConcatVectors(batch_dim_indices, {n_index}));

  auto eye = Broadcast(IdentityMatrix(builder, type, n, n), batch_dims);
  auto t = eye;

  auto vtv = BatchDot(MaybeConjugate(vs, true), /*transpose_x=*/true, vs,
                      /*transpose_y=*/false, precision);
  vtv = Select(TriangleMask(vtv, 0), ZerosLike(vtv), vtv);
  vtv = (vtv + eye) * tau_scale;

  TF_ASSIGN_OR_RETURN(auto values,
                      ForEachIndex(n, S32, body_fn, {t, vtv}, "wy", builder));
  return values[0];
}

// Block Householder QR Factorization. Algorithm 5.2.2 of Golub and van Loan.
// def qr_blocked(a, block_size):
//   m = a.shape[0]
//   n = a.shape[1]
//   q = np.eye(m)
//   for i in xrange(0, min(m, n), block_size):
//     k = min(block_size, min(m, n) - s)
//     (a, taus) = qr(a[i:, i:i+k])
//     y = np.eye(m, n) + np.tril(a, -1)
//     t = CompactWYRepresentation(vs, taus, m-i, k)
//     a[i:, i+k:] += (y @ np.conj(t.T)) @ (np.conj(y.T) @ a[i:, i+k:])
//     q[:, i:] += (q[:, i:] @ y) @ np.conj((y @ np.conj(t.T)).T)
//   return (q, a)
absl::StatusOr<XlaOp> QrExpander::BuildQrDecomposition(
    XlaOp a, int64_t block_size, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int num_dims = a_shape.rank();
  if (num_dims < 2) {
    return InvalidArgument("Arguments to QR must have rank >= 2: got shape %s",
                           a_shape.ToString());
  }
  PrimitiveType type = a_shape.element_type();

  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  const int64_t p = std::min(m, n);

  if (block_size < 1) {
    return InvalidArgument("block_size argument to QR must be >= 1; got %d",
                           block_size);
  }

  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }

  std::vector<int64_t> taus_dims = batch_dims;
  taus_dims.push_back(p);
  auto taus = Zeros(builder, ShapeUtil::MakeShape(type, taus_dims));
  for (int64_t i = 0; i < p; i += block_size) {
    int64_t k = std::min(block_size, p - i);

    auto a_block = SliceInMinorDims(a, {i, i}, {m, i + k});
    TF_ASSIGN_OR_RETURN(auto qr_block, QrBlock(a_block, precision));
    auto y = Add(IdentityMatrix(builder, type, m - i, k),
                 Select(TriangleMask(qr_block.q_and_r, -1), qr_block.q_and_r,
                        ZerosLike(qr_block.q_and_r)),
                 /*broadcast_dimensions=*/{num_dims - 2, num_dims - 1});

    a = UpdateSliceInMinorDims(a, qr_block.q_and_r, {i, i});
    taus = UpdateSliceInMinorDims(taus, qr_block.taus, {i});

    // Compute the I + Y @ T @ Y^t block representation of a product of
    // Householder matrices.
    TF_ASSIGN_OR_RETURN(
        auto t, CompactWYRepresentation(type, batch_dims, y, qr_block.taus,
                                        m - i, k, precision));

    // a[i:, i+k:] += (y @ np.conj(t.T)) @ (np.conj(y.T) @ a[i:, i+k:])
    auto yt = BatchDot(y, /*transpose_x=*/false, MaybeConjugate(t, true),
                       /*transpose_y=*/true, precision);
    auto a_panel = SliceInMinorDims(a, {i, i + k}, {m, n});
    auto a_update =
        BatchDot(MaybeConjugate(y, true), /*transpose_x=*/true, a_panel,
                 /*transpose_y=*/false, precision);
    a_update = BatchDot(yt, a_update, precision);
    a_panel = a_panel + a_update;
    a = UpdateSliceInMinorDims(a, a_panel, {i, i + k});
  }

  return Tuple(builder, {a, taus});
}

absl::StatusOr<XlaOp> QrExpander::ProductOfElementaryHouseholderReflectors(
    XlaOp a, XlaOp taus, int64_t block_size,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  TF_ASSIGN_OR_RETURN(Shape taus_shape, builder->GetShape(taus));
  const int num_dims = a_shape.rank();
  if (num_dims < 2) {
    return InvalidArgument("Arguments to QR must have rank >= 2: got shape %s",
                           a_shape.ToString());
  }
  PrimitiveType type = a_shape.element_type();

  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  const int64_t p = ShapeUtil::GetDimension(taus_shape, -1);
  if (m < n) {
    return InvalidArgument(
        "Argument to product of elementary Householder "
        "reflectors must have m >= n, got shape %s",
        a_shape.ToString());
  }

  if (block_size < 1) {
    return InvalidArgument("block_size argument to QR must be >= 1; got %d",
                           block_size);
  }

  const int64_t num_batch_dims = num_dims - 2;
  std::vector<int64_t> batch_dims(num_batch_dims);
  for (int i = 0; i < num_batch_dims; ++i) {
    batch_dims[i] = ShapeUtil::GetDimension(a_shape, i);
  }

  auto q = Broadcast(IdentityMatrix(builder, type, m, n), batch_dims);
  for (int64_t i = RoundDownTo(p - 1, block_size); i >= 0; i -= block_size) {
    int64_t k = std::min(block_size, p - i);

    auto a_block = SliceInMinorDims(a, {i, i}, {m, i + k});
    auto y = Add(IdentityMatrix(builder, type, m - i, k),
                 Select(TriangleMask(a_block, -1), a_block, ZerosLike(a_block)),
                 /*broadcast_dimensions=*/{num_dims - 2, num_dims - 1});

    // Compute the I + Y @ T @ Y^t block representation of a product of
    // Householder matrices.
    auto taus_block = SliceInMinorDims(taus, {i}, {i + k});

    TF_ASSIGN_OR_RETURN(
        auto t, CompactWYRepresentation(type, batch_dims, y, taus_block, m - i,
                                        k, precision));
    // q[i:, i:] += y @ t @ (np.conj(y.T) @ q[i:, i:])
    auto q_panel = SliceInMinorDims(q, {i, i}, {m, n});
    auto ytq = BatchDot(MaybeConjugate(y, true), /*transpose_x=*/true, q_panel,
                        /*transpose_y=*/false, precision);
    auto q_update = BatchDot(y, BatchDot(t, ytq, precision), precision);
    q_panel = q_panel + q_update;
    q = UpdateSliceInMinorDims(q, q_panel, {i, i});
  }
  return q;
}

static const char* kQrCustomCallName = "Qr";
static const char* kHouseholderProductCustomCallName =
    "ProductOfElementaryHouseholderReflectors";

bool QrExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCustomCall &&
         (instruction->custom_call_target() == kQrCustomCallName ||
          instruction->custom_call_target() ==
              kHouseholderProductCustomCallName);
}

absl::StatusOr<HloInstruction*> QrExpander::ExpandInstruction(
    HloInstruction* instruction) {
  std::string name =
      absl::StrFormat("xla.%s_%s", instruction->custom_call_target(),
                      instruction->operand(0)->shape().ToString());
  if (instruction->custom_call_target() == kHouseholderProductCustomCallName) {
    name += "_" + instruction->operand(1)->shape().ToString();
  }

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
    TF_RET_CHECK(instruction->operand_count() >= 1);
    XlaOp a = Parameter(&builder, 0, instruction->operand(0)->shape(), "a");
    XlaOp result;
    if (instruction->custom_call_target() == kQrCustomCallName) {
      TF_RET_CHECK(instruction->operand_count() == 1);
      TF_ASSIGN_OR_RETURN(
          result, BuildQrDecomposition(a,
                                       /*block_size=*/128,
                                       /*precision=*/PrecisionConfig::HIGHEST));
    } else {
      TF_RET_CHECK(instruction->operand_count() == 2);
      XlaOp taus =
          Parameter(&builder, 1, instruction->operand(1)->shape(), "taus");
      TF_ASSIGN_OR_RETURN(result, ProductOfElementaryHouseholderReflectors(
                                      a, taus, /*block_size=*/128,
                                      /*precision=*/PrecisionConfig::HIGHEST));
    }

    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build(result));
    TF_ASSIGN_OR_RETURN(
        computation, XlaComputationToHloComputation(xla_computation, module));
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

}  // namespace xla
