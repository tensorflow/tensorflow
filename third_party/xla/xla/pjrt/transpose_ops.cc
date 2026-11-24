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

#include "xla/pjrt/transpose_ops.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/ef57.h"
#include "xla/pjrt/transpose_kernels.h"

namespace xla {
namespace transpose_internal {

namespace {

struct uint128 {
  uint64_t lo;
  uint64_t hi;
};
static_assert(sizeof(uint128) == 16, "uint128 should be 16 bytes in size");

// Transposes a 2D tile of shape `[b_elems, a_elems]` of element type `T` in `a`
// (row stride `lda` bytes, contiguous along `a_elems`) into shape
// `[a_elems, b_elems]` in `b` (row stride `ldb` bytes, contiguous along
// `b_elems`) using a scalar loop.
template <typename T>
void TransposeScalar(const char* __restrict a, int64_t lda, int a_elems,
                     char* __restrict b, int64_t ldb, int b_elems) {
  for (int ia = 0; ia < a_elems; ++ia) {
    for (int ib = 0; ib < b_elems; ++ib) {
      std::memcpy(b + ia * ldb + ib * sizeof(T), a + ib * lda + ia * sizeof(T),
                  sizeof(T));
    }
  }
}

// Transposes `op->count` `[op->b_elems, op->a_elems]` tiles of `T` from
// `a + i*op->da` (row stride `op->lda` bytes) to `[op->a_elems, op->b_elems]`
// in `b + i*op->db` (row stride `op->ldb` bytes) using a scalar loop. Used when
// the tile is smaller than the vectorized microkernel size along either axis or
// when `inner_block_elems == 1`.
template <typename T>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpScalarTranspose2D(const TransposeOp* __restrict op,
                         const char* __restrict a, char* __restrict b,
                         char* __restrict /*scratch*/) {
  const int64_t count = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const int64_t lda = op->lda;
  const int64_t ldb = op->ldb;
  const int a_elems = op->a_elems;
  const int b_elems = op->b_elems;
  for (int64_t i = 0; i < count; ++i) {
    TransposeScalar<T>(a + i * da, lda, a_elems, b + i * db, ldb, b_elems);
  }
}

}  // namespace

// Operations that combine, offset, repeat, or allocate scratch memory for child
// operations.

#if ABSL_HAVE_CPP_ATTRIBUTE(clang::musttail) && \
    (defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64))
#define XLA_PJRT_TRANSPOSE_MUSTTAIL [[clang::musttail]]
#else
#define XLA_PJRT_TRANSPOSE_MUSTTAIL
#endif

// Executes two child operations in order on the same `(a, b, scratch)`:
// first the child at `op + 1`, then the child at `op + op->right_off`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpSeq(const TransposeOp* __restrict op, const char* __restrict a,
           char* __restrict b, char* __restrict scratch) {
  const TransposeOp* left = op + 1;
  left->fn(left, a, b, scratch);
  const TransposeOp* right = op + op->right_off;
  XLA_PJRT_TRANSPOSE_MUSTTAIL return right->fn(right, a, b, scratch);
}

// Advances `a` by `op->da` bytes and `b` by `op->db` bytes, then executes the
// child operation at `op + 1`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpOffset(const TransposeOp* __restrict op, const char* __restrict a,
              char* __restrict b, char* __restrict scratch) {
  const TransposeOp* body = op + 1;
  XLA_PJRT_TRANSPOSE_MUSTTAIL return body->fn(body, a + op->da, b + op->db,
                                              scratch);
}

// Executes the child operation at `op + 1` `op->count` times; iteration `i`
// receives `(a + i * op->da, b + i * op->db, scratch)`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpLoop(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict scratch) {
  const TransposeOp* body = op + 1;
  const TransposeOp::Fn body_fn = body->fn;
  const int64_t n = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  for (int64_t i = 0; i < n; ++i) {
    body_fn(body, a + i * da, b + i * db, scratch);
  }
}

// Pipes `step1` into `step2` through a temporary buffer of `op->lda` bytes at
// the start of `scratch`, passing `scratch + op->lda` as the remaining scratch
// space to both children. Repeats `op->count` times with strides
// `(op->da, op->db)`: each iteration runs the first child at `op + 1` reading
// from `a + i * op->da` and writing into the temporary buffer, then runs the
// second child at `op + op->right_off` reading from the temporary buffer and
// writing into `b + i * op->db`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpPipe(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict scratch) {
  const int64_t n = op->count;
  char* __restrict next_scratch = scratch + op->lda;
  const TransposeOp* step1 = op + 1;
  const TransposeOp* step2 = op + op->right_off;
  const TransposeOp::Fn fn1 = step1->fn;
  const TransposeOp::Fn fn2 = step2->fn;
  if (n == 1) {
    fn1(step1, a, scratch, next_scratch);
    XLA_PJRT_TRANSPOSE_MUSTTAIL return fn2(step2, scratch, b, next_scratch);
  }
  const int64_t da = op->da;
  const int64_t db = op->db;
  for (int64_t i = 0; i < n; ++i) {
    fn1(step1, a + i * da, scratch, next_scratch);
    fn2(step2, scratch, b + i * db, next_scratch);
  }
}

#undef XLA_PJRT_TRANSPOSE_MUSTTAIL

// Operations that read and write array data.

TransposeOp::Fn GetScalarTranspose2DOp(int elem_size) {
  switch (elem_size) {
    case 1:
      return &OpScalarTranspose2D<uint8_t>;
    case 2:
      return &OpScalarTranspose2D<uint16_t>;
    case 4:
      return &OpScalarTranspose2D<uint32_t>;
    case 8:
      return &OpScalarTranspose2D<uint64_t>;
    case 16:
      return &OpScalarTranspose2D<uint128>;
    default:
      LOG(FATAL) << "Unimplemented element size " << elem_size;
  }
}

// Copies `op->count` contiguous runs of `op->lda` bytes from `a + i*op->da` to
// `b + i*op->db`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpCopy(const TransposeOp* __restrict op, const char* __restrict a,
            char* __restrict b, char* __restrict /*scratch*/) {
  const int64_t count = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const size_t bytes = static_cast<size_t>(op->lda);
  for (int64_t i = 0; i < count; ++i) {
    std::memcpy(b + i * db, a + i * da, bytes);
  }
}

// Copies `op->count` 2D slices (strides `op->da`, `op->db`), each consisting of
// `inner->count` contiguous runs of `inner->lda` bytes (strides `inner->da`,
// `inner->db`), optionally followed by a trailing copy of `inner->ldb` bytes at
// `(inner->count * inner->da, inner->count * inner->db)`. The inner-loop
// parameters are stored inline at `inner = op + 1`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpCopy2D(const TransposeOp* __restrict op, const char* __restrict a,
              char* __restrict b, char* __restrict /*scratch*/) {
  const TransposeOp* inner = op + 1;
  const int64_t n0 = op->count;
  const int64_t da0 = op->da;
  const int64_t db0 = op->db;
  const int64_t n1 = inner->count;
  const int64_t da1 = inner->da;
  const int64_t db1 = inner->db;
  const size_t bytes = static_cast<size_t>(inner->lda);
  const size_t tail_bytes = static_cast<size_t>(inner->ldb);
  if (tail_bytes == 0) {
    for (int64_t i = 0; i < n0; ++i) {
      const char* row_a = a + i * da0;
      char* row_b = b + i * db0;
      for (int64_t j = 0; j < n1; ++j) {
        std::memcpy(row_b + j * db1, row_a + j * da1, bytes);
      }
    }
  } else {
    for (int64_t i = 0; i < n0; ++i) {
      const char* row_a = a + i * da0;
      char* row_b = b + i * db0;
      for (int64_t j = 0; j < n1; ++j) {
        std::memcpy(row_b + j * db1, row_a + j * da1, bytes);
      }
      std::memcpy(row_b + n1 * db1, row_a + n1 * da1, tail_bytes);
    }
  }
}

// Fills `op->count` contiguous runs of `op->lda` bytes at `b + i*op->db` with
// the byte `op->a_elems`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpFill(const TransposeOp* __restrict op, const char* __restrict /*a*/,
            char* __restrict b, char* __restrict /*scratch*/) {
  const int64_t count = op->count;
  const int64_t db = op->db;
  const int value = static_cast<int>(op->a_elems);
  const size_t bytes = static_cast<size_t>(op->lda);
  for (int64_t i = 0; i < count; ++i) {
    std::memset(b + i * db, value, bytes);
  }
}

// Converts `op->count` rows of `op->lda` contiguous `double` values at
// `a + i*op->da` into `2 * op->lda` contiguous `float` values in TPU ef57
// format at `b + i*op->db`.
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpConvertEf57(const TransposeOp* __restrict op, const char* __restrict a,
                   char* __restrict b, char* __restrict /*scratch*/) {
  const int64_t count = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const int64_t doubles = op->lda;
  if (doubles == 1) {
    for (int64_t i = 0; i < count; ++i) {
      double x;
      std::memcpy(&x, a + i * da, sizeof(double));
      const auto [hi, lo] = SplitF64ToF32(x);
      std::memcpy(b + i * db, &hi, sizeof(float));
      std::memcpy(b + i * db + sizeof(float), &lo, sizeof(float));
    }
    return;
  }
  for (int64_t i = 0; i < count; ++i) {
    ConvertF64ToEf57(
        absl::MakeConstSpan(reinterpret_cast<const double*>(a + i * da),
                            doubles),
        absl::MakeSpan(reinterpret_cast<float*>(b + i * db), doubles * 2));
  }
}

}  // namespace transpose_internal
}  // namespace xla
