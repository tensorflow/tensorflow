/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRANSPOSE_OPS_H_
#define XLA_PJRT_TRANSPOSE_OPS_H_

#include "xla/pjrt/transpose_kernels.h"

namespace xla {
namespace transpose_internal {

// Operations that combine, offset, repeat, or allocate scratch memory for child
// operations.

// Executes two child operations in order on the same `(a, b, scratch)`:
// first the child at `op + 1`, then the child at `op + op->right_off`.
void OpSeq(const TransposeOp* __restrict op, const char* __restrict a,
           char* __restrict b, char* __restrict scratch);

// Advances `a` by `op->da` bytes and `b` by `op->db` bytes, then executes the
// child operation at `op + 1`.
void OpOffset(const TransposeOp* __restrict op, const char* __restrict a,
              char* __restrict b, char* __restrict scratch);

// Executes the child operation at `op + 1` `op->count` times; iteration `i`
// receives `(a + i * op->da, b + i * op->db, scratch)`.
void OpLoop(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict scratch);

// Pipes `step1` into `step2` through a temporary buffer of `op->lda` bytes at
// the start of `scratch`, passing `scratch + op->lda` as the remaining scratch
// space to both children. Repeats `op->count` times with strides
// `(op->da, op->db)`: each iteration runs the first child at `op + 1` reading
// from `a + i * op->da` and writing into the temporary buffer, then runs the
// second child at `op + op->right_off` reading from the temporary buffer and
// writing into `b + i * op->db`.
void OpPipe(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict scratch);

// Operations that read and write array data.

// Returns the scalar 2D transpose op for `elem_size`-byte elements (1, 2, 4, 8,
// or 16). Used when the tile is smaller than the vectorized microkernel size
// along either axis or when `inner_block_elems == 1`.
TransposeOp::Fn GetScalarTranspose2DOp(int elem_size);

// Copies `op->count` contiguous runs of `op->lda` bytes from `a + i*op->da` to
// `b + i*op->db`.
void OpCopy(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict scratch);

// Copies `op->count` 2D slices (strides `op->da`, `op->db`), each consisting of
// `inner->count` contiguous runs of `inner->lda` bytes (strides `inner->da`,
// `inner->db`), optionally followed by a trailing copy of `inner->ldb` bytes at
// `(inner->count * inner->da, inner->count * inner->db)`. The inner-loop
// parameters are stored inline at `inner = op + 1`.
void OpCopy2D(const TransposeOp* __restrict op, const char* __restrict a,
              char* __restrict b, char* __restrict scratch);

// Fills `op->count` contiguous runs of `op->lda` bytes at `b + i*op->db` with
// the byte `op->a_elems`.
void OpFill(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict scratch);

// Converts `op->count` rows of `op->lda` contiguous `double` values at
// `a + i*op->da` into `2 * op->lda` contiguous `float` values in TPU ef57
// format at `b + i*op->db`.
void OpConvertEf57(const TransposeOp* __restrict op, const char* __restrict a,
                   char* __restrict b, char* __restrict scratch);

}  // namespace transpose_internal
}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_OPS_H_
