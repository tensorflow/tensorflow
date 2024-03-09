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

#ifndef XLA_CLIENT_LIB_LU_DECOMPOSITION_H_
#define XLA_CLIENT_LIB_LU_DECOMPOSITION_H_

#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Computes the LU decomposition with partial pivoting of a batch of matrices.
//
// Given a (batched) matrix a with shape [..., m, n], computes the matrix
// decomposition A = P @ L @ U where P is a permutation matrix, L is a
// lower-triangular matrix with unit diagonal entries, and U is an
// upper-triangular matrix.
//
// L and U are returned as a single matrix [..., m, n] containing both L and U
// packed in the same array. The unit diagonal of L is not represented
// explicitly.
//
// The permutation matrix P is returned in two forms, both as `pivots`, which is
// an s32[..., min(m, n)] array that describes a sequence of row-swaps in the
// style of LAPACK's xGETRF API, and `permutation`, which is a s32[..., m] array
// which gives the permutation to apply to the rows. We return both
// representations because they are each useful for different purposes; `pivots`
// is useful for computing the sign of a determinant, whereas `permutation` can
// be used via a Gather operation to permute the rows of a matrix.
//
// This method is only implemented on TPU at the moment.
// TODO(b/168208200): the implementation only supports F32 arrays. Handle the
// complex case.
struct LuDecompositionResult {
  // The LU decomposition, with both L and U packed into an array with shape
  // [..., m, n].
  XlaOp lu;
  // An array of shape s32[..., min(m, n)] containing the pivot rows.
  XlaOp pivots;
  // An array of shape s32[..., m], containing an another representation of the
  // pivots as a permutation.
  XlaOp permutation;
};

LuDecompositionResult LuDecomposition(XlaOp a);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_LU_DECOMPOSITION_H_
