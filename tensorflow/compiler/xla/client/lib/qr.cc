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

#include "tensorflow/compiler/xla/client/lib/qr.h"

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



}  // namespace

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
//     a[i:, i+k:] += (y @ t.T) @ (y.T @ a[i:, i+k:])
//     q[:, i:] += (q[:, i:] @ y) @ (y @ t.T).T
//   return (q, a)
StatusOr<QRDecompositionResult> QRDecomposition(
    XlaOp a, bool full_matrices, int64 block_size,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int num_dims = a_shape.rank();
  if (num_dims < 2) {
    return InvalidArgument("Arguments to QR must have rank >= 2: got shape %s",
                           a_shape.ToString());
  }
  const int64 m = ShapeUtil::GetDimension(a_shape, -2);
  const int64 n = ShapeUtil::GetDimension(a_shape, -1);
  const int64 p = std::min(m, n);

  if (block_size < 1) {
    return InvalidArgument("block_size argument to QR must be >= 1; got %d",
                           block_size);
  }

  Shape q_shape = a_shape;
  q_shape.mutable_dimensions().back() = m;

  Shape qr_shape = ShapeUtil::MakeTupleShape({q_shape, a_shape});
  auto qr = CustomCall(a.builder(), "QrDecomposition", {a}, qr_shape);
  auto q = GetTupleElement(qr, 0);
  auto r = GetTupleElement(qr, 1);

  // full_matrices is false when only a partial result in needed. Slice to the
  // needed dimensions here.
  if (!full_matrices) {
    q = SliceInMinorDims(q, {0, 0}, {m, p});
    r = SliceInMinorDims(r, {0, 0}, {p, n});
  }
  return QRDecompositionResult{q, r};
}

}  // namespace xla
