/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRANSPOSE_KERNELS_H_
#define XLA_PJRT_TRANSPOSE_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/optimization.h"

namespace xla {

// Maximum byte size along each dimension of a macroblock.
inline constexpr int kMaxOuterBlockSizeBytes = ABSL_CACHELINE_SIZE;

// Maximum input byte stride (lda) for which the 128-bit square microkernel
// is preferred over the 256-bit rectangular microkernel when lda >= ldb.
inline constexpr int kMaxSquare128StrideBytes = ABSL_CACHELINE_SIZE;

// A single operation in a chunk's flattened execution plan.
//
// Each chunk's plan is a tree of operations stored in a flat vector
// (`chunk_ops_[chunk_id]`) in depth-first order:
//   - Operations with one child (`OpOffset`, `OpLoop`) place their child
//     immediately after themselves at `op + 1`.
//   - Operations with two children (`OpSeq`, `OpPipe`) place their first
//     child at `op + 1` and their second child at `op + op->right_off`.
//   - `OpCopy2D` is a two-entry leaf occupying `op` (outer loop) and `op + 1`
//     (inner loop and optional trailing copy payload).
//
// To execute the subtree rooted at `op`, call:
//   op->fn(op, a, b, scratch);
// where:
//   - `op` points to this `TransposeOp` inside `chunk_ops_[chunk_id]`.
//   - `a` points to the current position in the input buffer (or a temporary
//     scratch buffer inside the second step of `OpPipe`).
//   - `b` points to the current position in the output buffer (or a temporary
//     scratch buffer inside the first step of `OpPipe`).
//   - `scratch` is a 64-byte-aligned scratch buffer with at least as many bytes
//     as the subtree requires (computed by `FlattenPlan`), or `nullptr` if the
//     subtree requires no scratch memory.
//
// The buffers referenced by `op`, `a`, `b`, and `scratch` must not overlap.
struct TransposeOp {
  using Fn = void (*)(const TransposeOp* __restrict op,
                      const char* __restrict a, char* __restrict b,
                      char* __restrict scratch);

  // Function implementing this operation.
  Fn fn = nullptr;
  // Trip count for `OpLoop`, `OpPipe`, `OpCopy2D`, and fused leaf loops
  // (`>= 1`; `1` for an unlooped op).
  int64_t count = 0;
  // Byte stride or offset in `a` (`OpLoop`, `OpOffset`, `OpPipe`, `OpCopy2D`,
  // and fused leaf loops).
  int64_t da = 0;
  // Byte stride or offset in `b` (`OpLoop`, `OpOffset`, `OpPipe`, `OpCopy2D`,
  // and fused leaf loops).
  int64_t db = 0;
  // Inner row stride `lda` in bytes for 2D transpose/interleave operations,
  // temporary buffer size in bytes (rounded up to a multiple of 64) for
  // `OpPipe`, or per-iteration element/byte count for 1D/2D leaf operations
  // (`OpCopy`, `OpCopy2D` payload, `OpFill`, `OpPackSubbyteHwy`,
  // `OpConvertEf57`).
  int64_t lda = 0;
  // Inner row stride `ldb` in bytes for 2D transpose/deinterleave operations,
  // or trailing copy byte count (`0` when absent) for `OpCopy2D`'s payload.
  int64_t ldb = 0;
  // Extent along A's stride-1 dimension for 2D transpose/interleave operations,
  // or the 1-byte fill value for `OpFill`.
  uint16_t a_elems = 0;
  // Extent along B's stride-1 dimension for 2D transpose/deinterleave
  // operations.
  uint16_t b_elems = 0;
  // Offset (in `TransposeOp` entries from `op`) to the second child for `OpSeq`
  // and `OpPipe`.
  uint32_t right_off = 0;
};
static_assert(sizeof(TransposeOp) == 56);

// Defined in transpose_kernels.cc. Called once when building a transpose plan
// so `op.fn` points directly to the right CPU specific function for
// `elem_size_in_bytes` and `bs`.
TransposeOp::Fn GetTransposeMacroKernelOp(int elem_size_in_bytes, int bs);

template <int kBits>
TransposeOp::Fn GetPackSubbyteOp();

// Returns the maximum vectorized inner block row size in bytes (16, 32, or 64)
// for `elem_size_in_bytes` on the active Highway target.
int GetMaxTransposeInnerBlockSizeBytes(int elem_size_in_bytes);

// Returns whether deinterleaving `k` rows of `elem_size`-byte elements is
// faster than the default 2D transpose, based on the BM_TransposeSmallDim_*
// benchmarks in transpose_test.cc across Arm and x86 CPUs:
// - k = 2, 3, and 4 are vectorized with Highway for 1- to 8-byte elements.
// - Larger k up to the limits below use a plain loop; beyond those limits
//   (and for all 16-byte elements), deinterleave is slower on at least one CPU.
constexpr bool DeinterleaveKernelIsFaster(size_t elem_size, int64_t k) {
  switch (elem_size) {
    case 1:
    case 2:
      return k >= 2 && k <= 7;
    case 4:
      return k >= 2 && k <= 5;
    case 8:
      return k >= 2 && k <= 4;
  }
  return false;
}

// Returns the deinterleave function for `k` rows of `elem_size`-byte elements
// (matching `DeinterleaveKernelIsFaster`), or nullptr if unsupported. Defined
// in transpose_kernels_interleave.cc.
TransposeOp::Fn GetDeinterleaveOp(size_t elem_size, int64_t k);
template <typename T, int bs>
void TransposeMacroKernelDispatch(const char* a, int64_t lda, int outer_bs_a,
                                  char* b, int64_t ldb, int outer_bs_b) {
  static_assert(sizeof(T) * bs <= kMaxOuterBlockSizeBytes);
  TransposeOp op;
  op.count = 1;
  op.lda = lda;
  op.ldb = ldb;
  op.a_elems = static_cast<uint16_t>(outer_bs_a * bs);
  op.b_elems = static_cast<uint16_t>(outer_bs_b * bs);
  GetTransposeMacroKernelOp(sizeof(T), bs)(&op, a, b, /*scratch=*/nullptr);
}

template <typename T, int bs>
void TransposeMicroKernelDispatch(const char* a, int64_t lda, char* b,
                                  int64_t ldb) {
  TransposeMacroKernelDispatch<T, bs>(a, lda, /*outer_bs_a=*/1, b, ldb,
                                      /*outer_bs_b=*/1);
}

// Plain loop that zips (interleaves) `n` elements across `K` rows.
template <typename T, int K>
void InterleaveKernel(const char* __restrict a, int64_t lda, char* __restrict b,
                      int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k) {
      std::memcpy(b + k * sizeof(T), a + k * lda + i * sizeof(T), sizeof(T));
    }
    b += K * sizeof(T);
  }
}

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_H_
