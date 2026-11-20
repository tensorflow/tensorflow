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

// Sketch of algorithm:
// A good description can be found in the HPTT paper, which is linked from the
// header file.
//
// The algorithm is divided into two parts: plan creation, that chooses an
// execution plan for the transpose, and execution. Plans may be cached and
// reused multiple times.
//
// We use a two level blocking scheme:
//
// The inner "microkernel" level is the unit of vectorization. A microkernel
// transposes a vector-unit sized tile, e.g., an 8x8 tile of floats for AVX2.
// The microkernels require one of the input dimensions and one of the
// output dimensions to have stride 1, while the other dimension in each case
// may have a non-trivial element stride. The kernels load N vectors of N
// stride-1 elements, and performs an NxN transpose, swapping the role of the
// stride-1 dimension. The N vectors are then written to the output. To perform
// a complete tensor transpose, we simply need apply the microkernel over all
// blocks of the matrix.
//
// In the event that the stride-1 dimensions of the input and output are the
// same, we use a simpler kernel which is a memcpy().
//
// To improve cache locality, we use another level of blocking, namely
// "macrokernels". The outer "macrokernel" level is a block of, for example,
// 4x4 microkernels. Macrokernels are the basic unit of work of the loop nest
// plan. For dimensions that aren't exactly divisible by the macrokernel size,
// we repeatedly halve the kernel size for trailing elements. For dimensions
// that aren't exactly divisible by the microkernel size, we use a scalar
// transpose for the trailing elements.
//
// A transpose plan iterates over the array's index space, applying macro-kernel
// sized blocks. Any iteration order is possible, although some orders give
// better locality than others. Currently we always use a default iteration
// order.
//
// A plan contains the data structures that describe how to perform a transpose.
// Plan creation chooses such things as a loop iteration order and kernel sizes.
// Plan creation also performs a handful of optimizations, such as
// coalescing adjacent dimensions that do not change their order and removing
// trivial dimensions.
//
// TODO(phawkins):
// * we don't incorporate a number of optimizations from HPTT, notably explicit
//   prefetching, and manual loop unrolling.
// * we could use vector-aligned stores for some arrays, which might
//   be worth something. We could also use nontemporal stores in the aligned
//   case.
// * we don't yet search for a good loop ordering. This probably matters less
//   for arrays that fit entirely in cache.
// * we could do a better job of vectorizing where the stride-1 dimensions are
//   small (e.g., inner dimensions of size [..., 3] are not uncommon in some
//   use cases.)

#include "xla/pjrt/transpose.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/ef57.h"
#include "xla/permutation_util.h"
#include "xla/pjrt/transpose_kernels.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

namespace {
#ifdef __AVX__
static constexpr int kMaxInnerBlockSizeBytes = 32;
#else
static constexpr int kMaxInnerBlockSizeBytes = 16;
#endif

struct uint128 {
  uint64_t lo;
  uint64_t hi;
};
static_assert(sizeof(uint128) == 16, "uint128 should be 16 bytes in size");

// Returns whether to use the `kind` interleave/deinterleave kernel for `k` and
// `elem_size`-byte elements instead of the default 2D transpose, based on the
// BM_TransposeSmallDim_* benchmarks in transpose_test.cc across Arm and x86
// CPUs:
// - k >= 8 is slower on some CPUs.
// - Interleave: k = 4 is only faster for 1- and 2-byte elements. Among the
//   selected cases, 16-byte interleave at k = 6 and 7 is up to 6% slower on
//   some x86 CPUs.
// - Deinterleave: see `DeinterleaveKernelIsFaster` in transpose_kernels.h.
constexpr bool InterleaveKernelIsFaster(TransposePlan::InnerKernelKind kind,
                                        int64_t k, int64_t elem_size) {
  if (kind == TransposePlan::InnerKernelKind::kInterleave) {
    return k >= 2 && k < 8 && (k != 4 || elem_size <= 2);
  }
  return DeinterleaveKernelIsFaster(elem_size, k);
}

// Maps a runtime K to a compile-time constant for `fn`, instantiating `fn` only
// for the K that InterleaveKernelIsFaster can select for `kind` and T, so
// kernels that are never used are not compiled.
template <typename T, TransposePlan::InnerKernelKind kind, typename Fn>
TransposeOp::Fn WithInterleaveSize(int64_t k, Fn fn) {
  const auto call = [&](auto const_k) -> TransposeOp::Fn {
    if constexpr (InterleaveKernelIsFaster(kind, const_k, sizeof(T))) {
      return fn(const_k);
    } else {
      LOG(FATAL) << "Unreachable: interleave kernel with K=" << k
                 << " sizeof(T)=" << sizeof(T);
    }
  };
  switch (k) {
    case 2:
      return call(std::integral_constant<int, 2>{});
    case 3:
      return call(std::integral_constant<int, 3>{});
    case 4:
      return call(std::integral_constant<int, 4>{});
    case 5:
      return call(std::integral_constant<int, 5>{});
    case 6:
      return call(std::integral_constant<int, 6>{});
    case 7:
      return call(std::integral_constant<int, 7>{});
    default:
      LOG(FATAL) << "Invalid interleave size " << k;
  }
}

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

// Interleaves `op->count` slices of `K` rows of `op->a_elems` elements of `T`
// spaced `op->lda` bytes apart in `a + i*op->da` into contiguous runs of
// `op->a_elems * K` elements in `b + i*op->db`, computing
// `b[K * j + k] = a[k][j]`.
template <typename T, int K>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpInterleave(const TransposeOp* __restrict op, const char* __restrict a,
                  char* __restrict b, char* __restrict /*scratch*/) {
  const int64_t count = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const int64_t lda = op->lda;
  const int a_elems = op->a_elems;
  for (int64_t i = 0; i < count; ++i) {
    InterleaveKernel<T, K>(a + i * da, lda, b + i * db, a_elems);
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

// Picks the execution function for a 2D transpose of element type `T` with
// extents (`a_elems`, `b_elems`), given the plan's kernel kind, vectorized
// microkernel block size `inner_block_elems`, and interleave factor
// `interleave_size`. Partially-divisible tiles are decomposed into full SIMD
// blocks and scalar remainder strips by `BuildChunkPlan` before reaching here.
//
// Examples (with `T = uint32_t`):
//   - `kind = kDefault, inner_block_elems = 4`:
//     - `(16, 16)` -> `GetTransposeMacroKernelOp<4 | (4 << 16)>()`
//     - `(2, 16)`  -> `OpScalarTranspose2D<uint32_t>` (smaller than 4)
//   - `kind = kInterleave, interleave_size = 3`:
//     - `(64, 3)`  -> `OpInterleave<uint32_t, 3>`
//   - `kind = kDeinterleave, interleave_size = 3`:
//     - `(3, 64)`  -> `GetDeinterleaveOp(4, 3)`
template <typename T>
TransposeOp::Fn SelectTranspose2DOpTyped(TransposePlan::InnerKernelKind kind,
                                         int inner_block_elems,
                                         int interleave_size, int a_elems,
                                         int b_elems) {
  if (kind == TransposePlan::InnerKernelKind::kInterleave) {
    DCHECK_EQ(b_elems, interleave_size);
    return WithInterleaveSize<T, TransposePlan::InnerKernelKind::kInterleave>(
        interleave_size, [](auto const_k) -> TransposeOp::Fn {
          return &OpInterleave<T, const_k>;
        });
  }
  if (kind == TransposePlan::InnerKernelKind::kDeinterleave) {
    DCHECK_EQ(a_elems, interleave_size);
    TransposeOp::Fn fn = GetDeinterleaveOp(sizeof(T), interleave_size);
    CHECK(fn != nullptr) << "Unsupported deinterleave kernel: elem_size="
                         << sizeof(T) << " k=" << interleave_size;
    return fn;
  }
  DCHECK(kind == TransposePlan::InnerKernelKind::kDefault);
  if (inner_block_elems == 1 || a_elems < inner_block_elems ||
      b_elems < inner_block_elems) {
    return &OpScalarTranspose2D<T>;
  }
  auto pick = [&](auto const_bs) -> TransposeOp::Fn {
    constexpr int kBs = decltype(const_bs)::value;
    if constexpr (sizeof(T) < 16 &&
                  sizeof(T) * kBs <= kMaxInnerBlockSizeBytes) {
      DCHECK_EQ(a_elems % kBs, 0);
      DCHECK_EQ(b_elems % kBs, 0);
      constexpr int kPacked = static_cast<int>(sizeof(T)) | (kBs << 16);
      return GetTransposeMacroKernelOp<kPacked>();
    } else {
      LOG(FATAL) << "Unreachable: inner_block_elems=" << inner_block_elems
                 << " sizeof(T)=" << sizeof(T);
    }
  };
  switch (inner_block_elems) {
    case 2:
      return pick(std::integral_constant<int, 2>{});
    case 4:
      return pick(std::integral_constant<int, 4>{});
    case 8:
      return pick(std::integral_constant<int, 8>{});
    case 16:
      return pick(std::integral_constant<int, 16>{});
    case 32:
      return pick(std::integral_constant<int, 32>{});
    default:
      LOG(FATAL) << "Invalid inner_block_elems " << inner_block_elems;
  }
}

// Dispatches `SelectTranspose2DOpTyped` on the runtime `elem_size_in_bytes`
// (1, 2, 4, 8, or 16 bytes).
TransposeOp::Fn SelectTranspose2DOp(int64_t elem_size_in_bytes,
                                    TransposePlan::InnerKernelKind kind,
                                    int inner_block_elems, int interleave_size,
                                    int a_elems, int b_elems) {
  switch (elem_size_in_bytes) {
    case 1:
      return SelectTranspose2DOpTyped<uint8_t>(
          kind, inner_block_elems, interleave_size, a_elems, b_elems);
    case 2:
      return SelectTranspose2DOpTyped<uint16_t>(
          kind, inner_block_elems, interleave_size, a_elems, b_elems);
    case 4:
      return SelectTranspose2DOpTyped<uint32_t>(
          kind, inner_block_elems, interleave_size, a_elems, b_elems);
    case 8:
      return SelectTranspose2DOpTyped<uint64_t>(
          kind, inner_block_elems, interleave_size, a_elems, b_elems);
    case 16:
      return SelectTranspose2DOpTyped<uint128>(
          kind, inner_block_elems, interleave_size, a_elems, b_elems);
    default:
      LOG(FATAL) << "Unimplemented element size " << elem_size_in_bytes;
  }
}

// Returns the `OpPackSubbyteHwy` specialization for `bits_per_element` (1, 2,
// or 4 bits per element).
TransposeOp::Fn SelectPackSubbyteOp(int bits_per_element) {
  switch (bits_per_element) {
    case 1:
      return GetPackSubbyteOp<1>();
    case 2:
      return GetPackSubbyteOp<2>();
    case 4:
      return GetPackSubbyteOp<4>();
    default:
      LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

}  // namespace

// Tree representation of a chunk's execution plan used during plan building
// before flattening into `std::vector<TransposeOp>`.
//
// Each node describes an action parameterized by the current input pointer `a`
// and output pointer `b`. The `Make*` constructors fold trivial cases on the
// fly (zero-work leaves, empty or single-trip loops, zero or nested offsets,
// inner offsets inside loops, contiguous nested `Loop`/`Fill1D`/`Copy1D`/
// `ConvertEf57`/`PackSubbyte` runs, and `Noop` children inside `Seq`).
struct TransposePlan::Plan {
  // Does nothing. Removed by `Make*` everywhere except at the root of an empty
  // chunk.
  struct Noop {};

  // Runs each child plan in `parts` in order on the same `(a, b)` pointers.
  struct Seq {
    std::vector<Plan> parts;
  };

  // Adds `da` bytes to `a` and `db` bytes to `b`, then executes `body`.
  struct Offset {
    int64_t da;
    int64_t db;
    std::unique_ptr<Plan> body;
  };

  // Executes `body` `count` times; iteration `i` runs with input `a + i * da`
  // and output `b + i * db` (`da` and `db` are byte strides).
  struct Loop {
    int64_t count;
    int64_t da;
    int64_t db;
    std::unique_ptr<Plan> body;
  };

  // Pipes `step1` into `step2` through a `bytes`-byte temporary buffer carved
  // from scratch memory: runs `step1` from `a` into that buffer, then runs
  // `step2` from that buffer into `b`.
  struct Pipe {
    int64_t bytes;
    std::unique_ptr<Plan> step1;
    std::unique_ptr<Plan> step2;
  };

  // Transposes a `[b_elems, a_elems]` tile in `a` (row stride `lda` bytes,
  // contiguous along `a_elems`) into `[a_elems, b_elems]` in `b` (row stride
  // `ldb` bytes, contiguous along `b_elems`).
  struct Transpose2D {
    int a_elems;
    int b_elems;
    int64_t lda;
    int64_t ldb;
  };

  // Copies `bytes` contiguous bytes from `a` to `b`.
  struct Copy1D {
    int64_t bytes;
  };

  // Fills `bytes` contiguous bytes at `b` with byte `value`.
  struct Fill1D {
    int64_t bytes;
    uint8_t value;
  };

  // Packs `elems` 1-byte elements from `a` into `bits_per_element`-bit packed
  // values at `b`.
  struct PackSubbyte {
    int bits_per_element;
    int64_t elems;
  };

  // Converts `doubles` contiguous `double` values at `a` into `2 * doubles`
  // contiguous `float` values in TPU ef57 format at `b`.
  struct ConvertEf57 {
    int64_t doubles;
  };

  std::variant<Noop, Seq, Offset, Loop, Pipe, Transpose2D, Copy1D, Fill1D,
               PackSubbyte, ConvertEf57>
      node;

  bool IsNoop() const { return std::holds_alternative<Noop>(node); }

  static Plan MakeNoop() {
    return Plan{Noop{}};
  }

  static Plan MakeSeq(std::vector<Plan> parts) {
    std::vector<Plan> non_empty;
    non_empty.reserve(parts.size());
    for (Plan& p : parts) {
      if (p.IsNoop()) {
        continue;
      }
      if (auto* sub = std::get_if<Seq>(&p.node)) {
        for (Plan& sp : sub->parts) {
          non_empty.push_back(std::move(sp));
        }
      } else {
        non_empty.push_back(std::move(p));
      }
    }
    if (non_empty.empty()) {
      return MakeNoop();
    }
    if (non_empty.size() == 1) {
      return std::move(non_empty[0]);
    }
    return Plan{Seq{std::move(non_empty)}};
  }

  static Plan MakeSeq(Plan first, Plan second) {
    std::vector<Plan> parts;
    parts.reserve(2);
    parts.push_back(std::move(first));
    parts.push_back(std::move(second));
    return MakeSeq(std::move(parts));
  }

  static Plan MakeOffset(int64_t da, int64_t db, Plan body) {
    if (body.IsNoop()) {
      return body;
    }
    if (auto* inner = std::get_if<Offset>(&body.node)) {
      da += inner->da;
      db += inner->db;
      std::unique_ptr<Plan> inner_body = std::move(inner->body);
      body = std::move(*inner_body);
    }
    if (da == 0 && db == 0) {
      return body;
    }
    return Plan{Offset{da, db, std::make_unique<Plan>(std::move(body))}};
  }

  static Plan MakeLoop(int64_t count, int64_t da, int64_t db, Plan body) {
    if (count <= 0 || body.IsNoop()) {
      return MakeNoop();
    }
    if (count == 1) {
      return body;
    }
    if (auto* off = std::get_if<Offset>(&body.node)) {
      const int64_t oda = off->da;
      const int64_t odb = off->db;
      std::unique_ptr<Plan> off_body = std::move(off->body);
      return MakeOffset(oda, odb,
                        MakeLoop(count, da, db, std::move(*off_body)));
    }
    if (auto* inner = std::get_if<Loop>(&body.node)) {
      if (da == inner->count * inner->da && db == inner->count * inner->db) {
        count *= inner->count;
        da = inner->da;
        db = inner->db;
        std::unique_ptr<Plan> inner_body = std::move(inner->body);
        body = std::move(*inner_body);
      }
    } else if (const auto* f = std::get_if<Fill1D>(&body.node)) {
      if (da == 0 && db == f->bytes) {
        return MakeFill1D(count * f->bytes, f->value);
      }
    } else if (const auto* c = std::get_if<Copy1D>(&body.node)) {
      if (da == c->bytes && db == c->bytes) {
        return MakeCopy1D(count * c->bytes);
      }
    } else if (const auto* e = std::get_if<ConvertEf57>(&body.node)) {
      if (da == e->doubles * static_cast<int64_t>(sizeof(double)) &&
          db == e->doubles * 2 * static_cast<int64_t>(sizeof(float))) {
        return MakeConvertEf57(count * e->doubles);
      }
    } else if (const auto* p = std::get_if<PackSubbyte>(&body.node)) {
      const int64_t epb = 8 / p->bits_per_element;
      if (p->elems % epb == 0 && da == p->elems && db == p->elems / epb) {
        return MakePackSubbyte(p->bits_per_element, count * p->elems);
      }
    }
    return Plan{Loop{count, da, db, std::make_unique<Plan>(std::move(body))}};
  }

  static Plan MakePipe(int64_t bytes, Plan step1, Plan step2) {
    if (bytes <= 0 || step1.IsNoop() || step2.IsNoop()) {
      DCHECK_EQ(step1.IsNoop(), step2.IsNoop());
      return MakeNoop();
    }
    return Plan{Pipe{bytes, std::make_unique<Plan>(std::move(step1)),
                     std::make_unique<Plan>(std::move(step2))}};
  }

  static Plan MakeTranspose2D(int a_elems, int b_elems, int64_t lda,
                              int64_t ldb) {
    if (a_elems <= 0 || b_elems <= 0) {
      return MakeNoop();
    }
    return Plan{Transpose2D{a_elems, b_elems, lda, ldb}};
  }

  static Plan MakeCopy1D(int64_t bytes) {
    if (bytes <= 0) {
      return MakeNoop();
    }
    return Plan{Copy1D{bytes}};
  }

  static Plan MakeFill1D(int64_t bytes, uint8_t value) {
    if (bytes <= 0) {
      return MakeNoop();
    }
    return Plan{Fill1D{bytes, value}};
  }

  static Plan MakePackSubbyte(int bits_per_element, int64_t elems) {
    if (elems <= 0) {
      return MakeNoop();
    }
    return Plan{PackSubbyte{bits_per_element, elems}};
  }

  static Plan MakeConvertEf57(int64_t doubles) {
    if (doubles <= 0) {
      return MakeNoop();
    }
    return Plan{ConvertEf57{doubles}};
  }
};

namespace {

struct CopyRowMatch {
  int64_t count = 0;
  int64_t da = 0;
  int64_t db = 0;
  int64_t bytes = 0;
  int64_t tail_bytes = 0;
};

// Matches a 2-part `Seq` representing a row copy with a trailing partial chunk:
//  - `Seq(Loop(n1, da1, db1, Copy1D(bytes)),
//         Offset(n1 * da1, n1 * db1, Copy1D(tail_bytes)))`, or
//  - `Seq(Copy1D(bytes), Offset(da1, db1, Copy1D(tail_bytes)))` (`n1 = 1`).
std::optional<CopyRowMatch> MatchCopySeqRow(
    const TransposePlan::Plan::Seq& seq) {
  using Plan = TransposePlan::Plan;
  if (seq.parts.size() != 2) {
    return std::nullopt;
  }
  const auto* off = std::get_if<Plan::Offset>(&seq.parts[1].node);
  if (off == nullptr) {
    return std::nullopt;
  }
  const auto* tail = std::get_if<Plan::Copy1D>(&off->body->node);
  if (tail == nullptr) {
    return std::nullopt;
  }
  if (const auto* loop = std::get_if<Plan::Loop>(&seq.parts[0].node)) {
    const auto* copy = std::get_if<Plan::Copy1D>(&loop->body->node);
    if (copy != nullptr && off->da == loop->count * loop->da &&
        off->db == loop->count * loop->db) {
      return CopyRowMatch{loop->count, loop->da, loop->db, copy->bytes,
                          tail->bytes};
    }
    return std::nullopt;
  }
  if (const auto* copy = std::get_if<Plan::Copy1D>(&seq.parts[0].node)) {
    return CopyRowMatch{/*count=*/1, off->da, off->db, copy->bytes,
                        tail->bytes};
  }
  return std::nullopt;
}

struct PlanFlattener {
  using Plan = TransposePlan::Plan;

  int64_t elem_size_in_bytes;
  TransposePlan::InnerKernelKind inner_kernel_kind;
  int inner_block_elems;
  int interleave_size;
  std::vector<TransposeOp>& ops;

  void EmitPendingLoop(int64_t count, int64_t da, int64_t db) {
    if (count > 1) {
      TransposeOp op;
      op.fn = &OpLoop;
      op.count = count;
      op.da = da;
      op.db = db;
      ops.push_back(op);
    }
  }

  void EmitCopy2D(int64_t count0, int64_t da0, int64_t db0,
                  const CopyRowMatch& inner) {
    TransposeOp outer_op;
    outer_op.fn = &OpCopy2D;
    outer_op.count = count0;
    outer_op.da = da0;
    outer_op.db = db0;
    TransposeOp inner_op;
    inner_op.count = inner.count;
    inner_op.da = inner.da;
    inner_op.db = inner.db;
    inner_op.lda = inner.bytes;
    inner_op.ldb = inner.tail_bytes;
    ops.push_back(outer_op);
    ops.push_back(inner_op);
  }

  // Emits `plan` with an optional pending outer loop `(count, da, db)` from an
  // enclosing `Plan::Loop` (`count == 1` when unlooped). Ops with a built-in
  // loop (all 5 leaves, `Pipe`, and `Copy2D` patterns) absorb it directly;
  // other ops flush an `OpLoop` first. Returns the peak scratch memory in bytes
  // needed by this subtree.
  int64_t Emit(const Plan& plan, int64_t count = 1, int64_t da = 0,
               int64_t db = 0) {
    return std::visit(
        [&](const auto& node) -> int64_t {
          using NodeT = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<NodeT, Plan::Noop>) {
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Transpose2D>) {
            DCHECK_LE(node.a_elems, std::numeric_limits<uint16_t>::max());
            DCHECK_LE(node.b_elems, std::numeric_limits<uint16_t>::max());
            TransposeOp op;
            op.fn = SelectTranspose2DOp(elem_size_in_bytes, inner_kernel_kind,
                                        inner_block_elems, interleave_size,
                                        node.a_elems, node.b_elems);
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.lda;
            op.ldb = node.ldb;
            op.a_elems = static_cast<uint16_t>(node.a_elems);
            op.b_elems = static_cast<uint16_t>(node.b_elems);
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Copy1D>) {
            TransposeOp op;
            op.fn = &OpCopy;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.bytes;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Fill1D>) {
            TransposeOp op;
            op.fn = &OpFill;
            op.count = count;
            op.db = db;
            op.lda = node.bytes;
            op.a_elems = node.value;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::PackSubbyte>) {
            TransposeOp op;
            op.fn = SelectPackSubbyteOp(node.bits_per_element);
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.elems;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::ConvertEf57>) {
            TransposeOp op;
            op.fn = &OpConvertEf57;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.doubles;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Pipe>) {
            DCHECK(!node.step1->IsNoop() && !node.step2->IsNoop());
            const int64_t aligned_bytes = RoundUpTo<int64_t>(node.bytes, 64);
            const size_t idx = ops.size();
            TransposeOp op;
            op.fn = &OpPipe;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = aligned_bytes;
            ops.push_back(op);
            const int64_t t1 = Emit(*node.step1);
            const size_t right_off = ops.size() - idx;
            DCHECK_LE(right_off, std::numeric_limits<uint32_t>::max());
            ops[idx].right_off = static_cast<uint32_t>(right_off);
            const int64_t t2 = Emit(*node.step2);
            return aligned_bytes + std::max(t1, t2);
          } else if constexpr (std::is_same_v<NodeT, Plan::Seq>) {
            if (std::optional<CopyRowMatch> row = MatchCopySeqRow(node)) {
              EmitCopy2D(count, da, db, *row);
              return 0;
            }
            EmitPendingLoop(count, da, db);
            DCHECK_GE(node.parts.size(), 2);
            // Emit binary OpSeq nodes right-associatively so OpSeq tail-calls
            // down the chain in O(1) stack space.
            int64_t max_temp = 0;
            for (size_t i = 0; i + 1 < node.parts.size(); ++i) {
              const size_t seq_idx = ops.size();
              TransposeOp op;
              op.fn = &OpSeq;
              ops.push_back(op);
              max_temp = std::max(max_temp, Emit(node.parts[i]));
              const size_t right_off = ops.size() - seq_idx;
              DCHECK_LE(right_off, std::numeric_limits<uint32_t>::max());
              ops[seq_idx].right_off = static_cast<uint32_t>(right_off);
            }
            return std::max(max_temp, Emit(node.parts.back()));
          } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
            EmitPendingLoop(count, da, db);
            TransposeOp op;
            op.fn = &OpOffset;
            op.da = node.da;
            op.db = node.db;
            ops.push_back(op);
            return Emit(*node.body);
          } else {
            static_assert(std::is_same_v<NodeT, Plan::Loop>);
            if (count > 1) {
              if (const auto* copy =
                      std::get_if<Plan::Copy1D>(&node.body->node)) {
                EmitCopy2D(count, da, db,
                           CopyRowMatch{node.count, node.da, node.db,
                                        copy->bytes, /*tail_bytes=*/0});
                return 0;
              }
              EmitPendingLoop(count, da, db);
            }
            return Emit(*node.body, node.count, node.da, node.db);
          }
        },
        plan.node);
  }
};

// Appends a flattened depth-first encoding of `plan` into `ops`, binding each
// leaf node to its concrete execution function (fusing immediately enclosing
// `Loop` nodes into leaves, `Pipe`, and 2D `Copy1D` patterns with optional
// trailing partial tiles), and returns the peak scratch memory in bytes needed
// to execute `plan` (with each `Pipe` reservation rounded up to 64 bytes to
// keep nested scratch pointers cache-line aligned).
int64_t FlattenPlan(const TransposePlan::Plan& plan, int64_t elem_size_in_bytes,
                    TransposePlan::InnerKernelKind inner_kernel_kind,
                    int inner_block_elems, int interleave_size,
                    std::vector<TransposeOp>& ops) {
  PlanFlattener flattener{elem_size_in_bytes, inner_kernel_kind,
                          inner_block_elems, interleave_size, ops};
  return flattener.Emit(plan);
}

// Appends a human-readable indented tree representation of `plan` to `*out`
// (used by `TransposePlan::ToString`).
void FormatPlan(const TransposePlan::Plan& plan, int indent,
                int64_t elem_size_in_bytes,
                TransposePlan::InnerKernelKind inner_kernel_kind,
                int interleave_size, std::string* out) {
  using Plan = TransposePlan::Plan;
  auto pad = [&](int n) { out->append(n, ' '); };
  std::visit(
      [&](const auto& node) {
        using NodeT = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<NodeT, Plan::Noop>) {
          pad(indent);
          absl::StrAppend(out, "Noop");
        } else if constexpr (std::is_same_v<NodeT, Plan::Seq>) {
          pad(indent);
          absl::StrAppend(out, "Seq(\n");
          for (size_t i = 0; i < node.parts.size(); ++i) {
            if (i > 0) {
              absl::StrAppend(out, ",\n");
            }
            FormatPlan(node.parts[i], indent + 2, elem_size_in_bytes,
                       inner_kernel_kind, interleave_size, out);
          }
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
          pad(indent);
          absl::StrAppendFormat(out, "Offset(da=%d, db=%d,\n", node.da,
                                node.db);
          FormatPlan(*node.body, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Loop>) {
          pad(indent);
          absl::StrAppendFormat(out, "Loop(count=%d, da=%d, db=%d,\n",
                                node.count, node.da, node.db);
          FormatPlan(*node.body, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Pipe>) {
          pad(indent);
          absl::StrAppendFormat(out, "Pipe(bytes=%d,\n", node.bytes);
          FormatPlan(*node.step1, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ",\n");
          FormatPlan(*node.step2, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Copy1D>) {
          pad(indent);
          absl::StrAppendFormat(out, "Copy1D(bytes=%d)", node.bytes);
        } else if constexpr (std::is_same_v<NodeT, Plan::Fill1D>) {
          pad(indent);
          absl::StrAppendFormat(out, "Fill1D(bytes=%d, value=0x%02x)",
                                node.bytes, node.value);
        } else if constexpr (std::is_same_v<NodeT, Plan::PackSubbyte>) {
          pad(indent);
          absl::StrAppendFormat(out, "PackSubbyte<%d>(elems=%d)",
                                node.bits_per_element, node.elems);
        } else if constexpr (std::is_same_v<NodeT, Plan::ConvertEf57>) {
          pad(indent);
          absl::StrAppendFormat(out, "ConvertEf57(doubles=%d)", node.doubles);
        } else {
          static_assert(std::is_same_v<NodeT, Plan::Transpose2D>);
          pad(indent);
          if (inner_kernel_kind ==
              TransposePlan::InnerKernelKind::kInterleave) {
            absl::StrAppendFormat(
                out, "Interleave<elem_size=%d,K=%d>(a_elems=%d, lda=%d)",
                elem_size_in_bytes, interleave_size, node.a_elems, node.lda);
          } else if (inner_kernel_kind ==
                     TransposePlan::InnerKernelKind::kDeinterleave) {
            absl::StrAppendFormat(
                out, "Deinterleave<elem_size=%d,K=%d>(b_elems=%d, ldb=%d)",
                elem_size_in_bytes, interleave_size, node.b_elems, node.ldb);
          } else {
            absl::StrAppendFormat(
                out,
                "Transpose2D<elem_size=%d>(a_elems=%d, b_elems=%d, lda=%d, "
                "ldb=%d)",
                elem_size_in_bytes, node.a_elems, node.b_elems, node.lda,
                node.ldb);
          }
        }
      },
      plan.node);
}

// Scales output byte offsets, strides, and fill sizes in a padding `Plan` from
// unpacked 1-byte element units to packed sub-byte output bytes.
void ScalePlanOutputBytes(TransposePlan::Plan& plan, int elements_per_byte) {
  using Plan = TransposePlan::Plan;
  std::visit(
      [&](auto& node) {
        using NodeT = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<NodeT, Plan::Noop>) {
          return;
        } else if constexpr (std::is_same_v<NodeT, Plan::Seq>) {
          for (Plan& p : node.parts) {
            ScalePlanOutputBytes(p, elements_per_byte);
          }
        } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
          DCHECK_EQ(node.db % elements_per_byte, 0);
          node.db /= elements_per_byte;
          ScalePlanOutputBytes(*node.body, elements_per_byte);
        } else if constexpr (std::is_same_v<NodeT, Plan::Loop>) {
          DCHECK_EQ(node.db % elements_per_byte, 0);
          node.db /= elements_per_byte;
          ScalePlanOutputBytes(*node.body, elements_per_byte);
        } else if constexpr (std::is_same_v<NodeT, Plan::Fill1D>) {
          DCHECK_EQ(node.bytes % elements_per_byte, 0);
          node.bytes /= elements_per_byte;
        } else {
          LOG(FATAL) << "Unexpected node in padding plan";
        }
      },
      plan.node);
}

ABSL_ATTRIBUTE_NOINLINE void ExecuteRootWithScratch(const TransposeOp* root,
                                                    const char* ac, char* bc,
                                                    int64_t scratch_size) {
  constexpr int64_t kStackScratchBytes = 4096;
  if (ABSL_PREDICT_TRUE(scratch_size <= kStackScratchBytes)) {
    alignas(64) char stack_scratch[kStackScratchBytes];
    root->fn(root, ac, bc, stack_scratch);
    return;
  }
  std::unique_ptr<void, void (*)(void*)> heap_scratch(
      tsl::port::AlignedMalloc(static_cast<size_t>(scratch_size),
                               std::align_val_t{64}),
      &tsl::port::AlignedFree);
  CHECK(heap_scratch != nullptr) << "AlignedMalloc failed";
  root->fn(root, ac, bc, static_cast<char*>(heap_scratch.get()));
}

}  // namespace

ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void TransposePlan::ExecuteChunk(int chunk_id, const void* a, void* b,
                                 bool input_is_global,
                                 bool output_is_global) const {
  CHECK(!use_fallback_pack_)
      << "ExecuteChunk does not support unaligned sub-byte packing; use "
         "Execute().";
  ExecuteChunkInternal(chunk_id, a, b, input_is_global, output_is_global);
}

ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void TransposePlan::ExecuteChunkInternal(int chunk_id, const void* a, void* b,
                                         bool input_is_global,
                                         bool output_is_global) const {
  if (num_elems_ == 0) {
    return;
  }
  absl::Span<TransposeOp const> ops = chunk_ops_[chunk_id];
  if (ops.empty()) {
    return;
  }
  tsl::profiler::TraceMe traceme("Transpose::ExecuteChunk", /*level=*/2);

  const char* ac = static_cast<const char*>(a);
  if (input_is_global) {
    ac += input_chunk_offset_bytes_[chunk_id] +
          input_chunk_iteration_offsets_[chunk_id];
  } else {
    ac += input_chunk_iteration_offsets_[chunk_id];
  }
  char* bc = static_cast<char*>(b);
  if (output_is_global) {
    bc += output_chunk_offset_bytes_[chunk_id] +
          output_chunk_iteration_offsets_[chunk_id];
  } else {
    bc += output_chunk_iteration_offsets_[chunk_id];
  }

  const TransposeOp* root = ops.data();
  if (ABSL_PREDICT_TRUE(scratch_size_ == 0)) {
    root->fn(root, ac, bc, /*scratch=*/nullptr);
    return;
  }
  ExecuteRootWithScratch(root, ac, bc, scratch_size_);
}

void TransposePlan::ExecuteInternal(
    const void* a, void* b,
    std::optional<absl::FunctionRef<void(std::function<void()>)>> schedule_work)
    const {
  if (!schedule_work || Parallelism() <= 1) {
    for (int i = 0; i < Parallelism(); ++i) {
      ExecuteChunkInternal(i, a, b, /*input_is_global=*/true,
                           /*output_is_global=*/true);
    }
  } else {
    absl::BlockingCounter counter(Parallelism() - 1);
    for (size_t i = 1; i < chunk_ops_.size(); ++i) {
      (*schedule_work)([&, i]() {
        ExecuteChunkInternal(i, a, b, /*input_is_global=*/true,
                             /*output_is_global=*/true);
        counter.DecrementCount();
      });
    }
    ExecuteChunkInternal(0, a, b, /*input_is_global=*/true,
                         /*output_is_global=*/true);
    counter.Wait();
  }
}

ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void TransposePlan::Execute(
    const void* a, void* b,
    std::optional<absl::FunctionRef<void(std::function<void()>)>> schedule_work)
    const {
  if (num_elems_ == 0) {
    return;
  }
  if (use_fallback_pack_) {
    tsl::profiler::TraceMe traceme("Transpose::ExecuteAndSubbytePack",
                                   /*level=*/2);
    const int64_t output_num_elems = OutputNumElems();
    const int elements_per_byte = 8 / bits_per_element_;
    const int64_t buf_elems =
        tile_padding_ ? RoundUpTo<int64_t>(output_num_elems, elements_per_byte)
                      : output_num_elems;
    std::unique_ptr<char[]> temp_buf =
        tile_padding_ ? std::unique_ptr<char[]>(new char[buf_elems])
                      : std::make_unique<char[]>(buf_elems);
    if (tile_padding_ && buf_elems > output_num_elems) {
      std::memset(temp_buf.get() + output_num_elems, *tile_padding_,
                  buf_elems - output_num_elems);
    }
    ExecuteInternal(a, temp_buf.get(), schedule_work);
    PackIntN(bits_per_element_, absl::MakeConstSpan(temp_buf.get(), buf_elems),
             absl::MakeSpan(
                 static_cast<char*>(b),
                 CeilOfRatio<int64_t>(output_num_elems, elements_per_byte)));
  } else {
    tsl::profiler::TraceMe traceme("Transpose::Execute", /*level=*/2);
    ExecuteInternal(a, b, schedule_work);
  }
}

// Everything above this point pertains to executing plans.
// Everything below this point pertains to building plans.

TransposePlan::TransposePlan() = default;
TransposePlan::~TransposePlan() = default;

static void ComputeStrides(
    int64_t elem_size_in_bytes, absl::Span<const int64_t> dims,
    absl::Span<const int64_t> tiling,
    absl::InlinedVector<int64_t, 4>& outer_tile_strides,
    absl::InlinedVector<int64_t, 4>& inner_tile_strides) {
  inner_tile_strides.resize(dims.size());
  int64_t acc = elem_size_in_bytes;
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    inner_tile_strides[d] = acc;
    acc *= tiling[d];
  }
  outer_tile_strides.resize(dims.size());
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    outer_tile_strides[d] = acc;
    acc *= CeilOfRatio(dims[d], tiling[d]);
  }
}

int64_t TransposePlan::InputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims_.size(); ++i) {
    size *= RoundUpTo(a_dims_[i], a_tiling_[i]);
  }
  return size;
}

int64_t TransposePlan::OutputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims_.size(); ++i) {
    size *= RoundUpTo(a_dims_[permutation_[i]], b_tiling_[i]);
  }
  return size;
}

// Parses and validates a tiling specification, and populates `tiling`.
static absl::Status ParseTilingSpecification(
    int ndim, const std::optional<TransposePlan::Tiling>& tiling_opt,
    bool nonstandard_layout, absl::InlinedVector<int64_t, 4>& tiling) {
  tiling.resize(ndim, 1);
  if (!tiling_opt) {
    return absl::OkStatus();
  }
  absl::Span<int64_t const> tiling_spec = tiling_opt->tiling;
  if (tiling_spec.size() > ndim) {
    return InvalidArgument(
        "Tiling (%s) must have at most as many dimensions as the array (%d)",
        absl::StrJoin(tiling_spec, ","), ndim);
  }
  if (absl::c_find_if(tiling_spec, [](int64_t d) { return d < 1; }) !=
      tiling_spec.end()) {
    return InvalidArgument("Tiling sizes (%s) must be >= 1",
                           absl::StrJoin(tiling_spec, ","));
  }
  if (ndim == 1 && !nonstandard_layout) {
    // Tiling doesn't do anything for a rank-1 array with default strides,
    // except add padding. Since we're not going to touch any padding elements,
    // we can ignore it.
    // TODO(phawkins): this seems like it should be a loop optimization, not
    // done here.
    return absl::OkStatus();
  }
  int offset = ndim;
  offset -= tiling_spec.size();
  absl::c_copy(tiling_spec, tiling.begin() + offset);

  return absl::OkStatus();
}

std::string TransposePlan::Loop::ToString() const {
  return absl::StrFormat(
      "%d%s[dim_size=%d,tile_size=%d,start=%d,end=%d,is_inner_dim_in_a=%d,is_"
      "inner_dim_in_b=%d,lda=%d,ldb=%d,parallelism=%d,contiguity=%d,has_"
      "partial_tile=%d)",
      dim_in_a, tile_interior ? "[tile]" : "", dim_size, tile_size, start, end,
      is_inner_dim_in_a, is_inner_dim_in_b, lda, ldb, parallelism, contiguity,
      has_partial_tile);
}

bool TransposePlan::Loop::operator==(const Loop& other) const {
  return dim_in_a == other.dim_in_a && tile_interior == other.tile_interior &&
         dim_size == other.dim_size && tile_size == other.tile_size &&
         lda == other.lda && ldb == other.ldb &&
         is_inner_dim_in_a == other.is_inner_dim_in_a &&
         is_inner_dim_in_b == other.is_inner_dim_in_b &&
         parallelism == other.parallelism && start == other.start &&
         end == other.end;
}

struct TransposePlan::ChunkPlanBuilder {
  const TransposePlan& plan;
  absl::Span<const Loop> chunk_loops;
  int elements_per_byte;

  ChunkPlanBuilder(const TransposePlan& plan, int chunk_id)
      : plan(plan),
        chunk_loops(plan.chunk_loops_[chunk_id]),
        elements_per_byte((plan.transformation_ == Transformation::kPackSubbyte)
                              ? (8 / plan.bits_per_element_)
                              : 1) {}

  int64_t StepB(const Loop& l, int64_t n) const {
    DCHECK_EQ((n * l.ldb) % elements_per_byte, 0);
    return (n * l.ldb) / elements_per_byte;
  }

  // Decomposes partially-divisible default-kernel tiles (both sides at least
  // `inner_block_elems_`, not both multiples) into a full SIMD block plus 1 or
  // 2 scalar remainder strips so every `Plan::Transpose2D` leaf maps 1-to-1 to
  // a single `TransposeOp`.
  Plan MakeTranspose2D(int a_elems, int b_elems, int64_t lda,
                       int64_t ldb) const {
    if (plan.inner_kernel_kind_ == InnerKernelKind::kDefault &&
        plan.inner_block_elems_ > 1 && a_elems >= plan.inner_block_elems_ &&
        b_elems >= plan.inner_block_elems_ &&
        (a_elems % plan.inner_block_elems_ != 0 ||
         b_elems % plan.inner_block_elems_ != 0)) {
      const int done_a =
          (a_elems / plan.inner_block_elems_) * plan.inner_block_elems_;
      const int done_b =
          (b_elems / plan.inner_block_elems_) * plan.inner_block_elems_;
      std::vector<Plan> pieces;
      pieces.reserve(3);
      pieces.push_back(Plan::MakeTranspose2D(done_a, done_b, lda, ldb));
      pieces.push_back(Plan::MakeOffset(
          done_b * lda, done_b * plan.elem_size_in_bytes_,
          Plan::MakeTranspose2D(done_a, b_elems - done_b, lda, ldb)));
      pieces.push_back(Plan::MakeOffset(
          done_a * plan.elem_size_in_bytes_, done_a * ldb,
          Plan::MakeTranspose2D(a_elems - done_a, b_elems, lda, ldb)));
      return Plan::MakeSeq(std::move(pieces));
    }
    return Plan::MakeTranspose2D(a_elems, b_elems, lda, ldb);
  }

  Plan BuildLeaf(int a_elems, int b_elems) const {
    switch (plan.transformation_) {
      case Transformation::kNone:
        return MakeTranspose2D(a_elems, b_elems, plan.sentinel_lda_,
                               plan.sentinel_ldb_);
      case Transformation::kF64ToEf57: {
        DCHECK_EQ(a_elems % 2, 0);
        const int64_t tmp_lda = int64_t{a_elems} * sizeof(float);
        Plan step1 = Plan::MakeLoop(b_elems, plan.sentinel_lda_, tmp_lda,
                                    Plan::MakeConvertEf57(a_elems / 2));
        Plan step2 =
            MakeTranspose2D(a_elems, b_elems, tmp_lda, plan.sentinel_ldb_);
        return Plan::MakePipe(int64_t{b_elems} * tmp_lda, std::move(step1),
                              std::move(step2));
      }
      case Transformation::kPackSubbyte: {
        DCHECK_EQ(plan.elem_size_in_bytes_, 1);
        DCHECK_EQ(b_elems % elements_per_byte, 0);
        DCHECK_EQ(plan.sentinel_ldb_ % elements_per_byte, 0);
        Plan step1 = MakeTranspose2D(a_elems, b_elems, plan.sentinel_lda_,
                                     /*ldb=*/b_elems);
        Plan step2 = Plan::MakeLoop(
            a_elems, /*da=*/b_elems, plan.sentinel_ldb_ / elements_per_byte,
            Plan::MakePackSubbyte(plan.bits_per_element_, b_elems));
        return Plan::MakePipe(int64_t{a_elems} * b_elems, std::move(step1),
                              std::move(step2));
      }
    }
    LOG(FATAL) << "Unhandled transformation "
               << static_cast<int>(plan.transformation_);
  }

  Plan BuildTransfer(size_t idx, int a_elems, int b_elems,
                     absl::InlinedVector<bool, 4>& partial) const {
    if (idx == chunk_loops.size()) {
      DCHECK(!plan.inner_kernel_is_memcpy());
      DCHECK_GT(a_elems, 0);
      DCHECK_GT(b_elems, 0);
      return BuildLeaf(a_elems, b_elems);
    }

    const Loop& loop = chunk_loops[idx];
    const int a_dim = loop.dim_in_a;

    if (!loop.tile_interior && loop.tile_size > 1) {
      // Exterior loop over tiles of a dimension that has a trailing partial
      // tile. Split into complete tiles and (if covered by this chunk) the
      // trailing partial tile.
      DCHECK_NE(loop.dim_size % loop.tile_size, 0);
      DCHECK_EQ(loop.inc, 1);
      DCHECK(!loop.is_inner_dim_in_a && !loop.is_inner_dim_in_b);
      const int64_t num_complete = loop.dim_size / loop.tile_size;
      const int64_t complete_count =
          std::min<int64_t>(num_complete, loop.end) - loop.start;
      const bool saved_partial = partial[a_dim];
      partial[a_dim] = false;
      Plan full_plan =
          (complete_count > 0)
              ? Plan::MakeLoop(
                    complete_count, loop.lda, StepB(loop, 1),
                    BuildTransfer(idx + 1, a_elems, b_elems, partial))
              : Plan::MakeNoop();
      Plan rem_plan = Plan::MakeNoop();
      if (loop.start <= num_complete && num_complete < loop.end) {
        partial[a_dim] = true;
        const int64_t offset = num_complete - loop.start;
        rem_plan =
            Plan::MakeOffset(offset * loop.lda, StepB(loop, offset),
                             BuildTransfer(idx + 1, a_elems, b_elems, partial));
      }
      partial[a_dim] = saved_partial;
      return Plan::MakeSeq(std::move(full_plan), std::move(rem_plan));
    }

    if (loop.tile_interior) {
      DCHECK_EQ(loop.start, 0);
      DCHECK_EQ(loop.end, loop.tile_size);
    }
    const int64_t count =
        loop.tile_interior
            ? (partial[a_dim] ? (loop.dim_size % loop.tile_size)
                              : loop.tile_size)
            : std::max<int64_t>(
                  0, std::min<int64_t>(loop.dim_size, loop.end) - loop.start);
    if (count <= 0) {
      return Plan::MakeNoop();
    }

    if (plan.inner_kernel_is_memcpy() && loop.is_inner_dim_in_a) {
      DCHECK_EQ(idx + 1, chunk_loops.size());
      if (plan.transformation_ == Transformation::kPackSubbyte) {
        DCHECK_EQ(count % elements_per_byte, 0);
        return Plan::MakePackSubbyte(plan.bits_per_element_, count);
      }
      return Plan::MakeCopy1D(count * plan.elem_size_in_bytes_);
    }

    if (!plan.inner_kernel_is_memcpy() &&
        (loop.is_inner_dim_in_a || loop.is_inner_dim_in_b)) {
      const int64_t block = loop.inc;
      DCHECK_GT(block, 0);
      const int64_t full_blocks = count / block;
      const int64_t rem = count % block;
      auto build_child = [&](int width) -> Plan {
        const int next_a = loop.is_inner_dim_in_a ? width : a_elems;
        const int next_b = loop.is_inner_dim_in_b ? width : b_elems;
        return BuildTransfer(idx + 1, next_a, next_b, partial);
      };
      Plan full_plan =
          (full_blocks > 0)
              ? Plan::MakeLoop(full_blocks, block * loop.lda,
                               StepB(loop, block),
                               build_child(static_cast<int>(block)))
              : Plan::MakeNoop();
      const int64_t rem_start = full_blocks * block;
      Plan rem_plan =
          (rem > 0)
              ? Plan::MakeOffset(rem_start * loop.lda, StepB(loop, rem_start),
                                 build_child(static_cast<int>(rem)))
              : Plan::MakeNoop();
      return Plan::MakeSeq(std::move(full_plan), std::move(rem_plan));
    }

    DCHECK_EQ(loop.inc, 1);
    return Plan::MakeLoop(count, loop.lda, StepB(loop, 1),
                          BuildTransfer(idx + 1, a_elems, b_elems, partial));
  }

  Plan PadFrom(absl::Span<const Loop> loops, size_t idx, bool is_padding,
               absl::InlinedVector<bool, 4>& in_partial) const {
    if (idx == loops.size()) {
      return is_padding ? Plan::MakeFill1D(plan.elem_size_in_bytes_,
                                           *plan.tile_padding_)
                        : Plan::MakeNoop();
    }
    const Loop& loop = loops[idx];
    auto sub_loop = [&](int64_t count, bool sub_padding) {
      if (count <= 0) {
        return Plan::MakeNoop();
      }
      return Plan::MakeLoop(count, /*da=*/0, loop.ldb,
                            PadFrom(loops, idx + 1, sub_padding, in_partial));
    };
    if (!loop.tile_interior) {
      const int64_t full = loop.dim_size / loop.tile_size;
      const bool saved = in_partial[loop.dim_in_a];
      in_partial[loop.dim_in_a] = false;
      Plan full_plan =
          sub_loop(std::min(loop.end, full) - loop.start, is_padding);
      Plan rem_plan = Plan::MakeNoop();
      if (loop.start <= full && full < loop.end) {
        in_partial[loop.dim_in_a] = true;
        rem_plan =
            Plan::MakeOffset(/*da=*/0, (full - loop.start) * loop.ldb,
                             PadFrom(loops, idx + 1, is_padding, in_partial));
      }
      in_partial[loop.dim_in_a] = saved;
      return Plan::MakeSeq(std::move(full_plan), std::move(rem_plan));
    }
    DCHECK_EQ(loop.start, 0);
    DCHECK_EQ(loop.end, loop.tile_size);
    if (!in_partial[loop.dim_in_a] || is_padding) {
      return sub_loop(loop.tile_size, is_padding);
    }
    const int64_t rem = loop.dim_size % loop.tile_size;
    return Plan::MakeSeq(
        sub_loop(rem, /*sub_padding=*/false),
        Plan::MakeOffset(
            /*da=*/0, rem * loop.ldb,
            sub_loop(loop.tile_size - rem, /*sub_padding=*/true)));
  }

  // Builds the padding fills for any partial output tiles in `chunk_id`.
  //
  // For example, a partial row-major 8x128 tile with 3 valid rows and 5 valid
  // columns has an L-shaped padding region. We cut the L horizontally so the
  // bottom rectangle spans full rows and coalesces into a single Fill1D:
  //
  //        col 0    col 5              col 128
  //          +--------+-------------------+
  //    row 0 | valid  |    padding (1)    |  Loop(3, Fill1D(123))
  //          |  3x5   |       3x123       |
  //    row 3 +--------+-------------------+
  //          |        padding (2)         |  Fill1D(5*128)
  //          |           5x128            |
  //    row 8 +----------------------------+
  //
  // Sorting the chunk's loops from largest to smallest output stride gives this
  // cut in N dimensions: at each partial-tile interior loop, `[0, rem)`
  // recurses with `is_padding=false` to pad any inner partial dimensions while
  // `[rem, tile_size)` recurses with `is_padding=true`. Recursing through inner
  // loops ensures that any inner loop partitioned across chunks only fills its
  // `[loop.start, loop.end)` slice, while `Plan::MakeLoop` coalesces unsplit
  // inner loops into a single `Fill1D`.
  Plan BuildPadding() const {
    if (!plan.tile_padding_.has_value() || !plan.b_is_tiled_ ||
        !absl::c_any_of(chunk_loops,
                        [](const Loop& l) { return l.has_partial_tile; })) {
      return Plan::MakeNoop();
    }

    std::vector<Loop> loops;
    loops.reserve(chunk_loops.size());
    for (const Loop& l : chunk_loops) {
      if (l.ldb != 0) {
        loops.push_back(l);
      }
    }
    absl::c_sort(loops, [](const Loop& a, const Loop& b) {
      return std::make_tuple(a.ldb, !a.tile_interior) >
             std::make_tuple(b.ldb, !b.tile_interior);
    });

    absl::InlinedVector<bool, 4> in_partial(plan.a_dims_.size(), false);
    Plan pad_plan = PadFrom(loops, /*idx=*/0, /*is_padding=*/false, in_partial);
    if (elements_per_byte > 1) {
      ScalePlanOutputBytes(pad_plan, elements_per_byte);
    }
    return pad_plan;
  }
};

// Builds the execution `Plan` for `chunk_id`.
TransposePlan::Plan TransposePlan::BuildChunkPlan(int chunk_id) const {
  const int ndim = static_cast<int>(a_dims_.size());
  DCHECK_GT(ndim, 0);
  ChunkPlanBuilder builder(*this, chunk_id);
  absl::InlinedVector<bool, 4> partial(ndim, false);
  return Plan::MakeSeq(
      builder.BuildTransfer(/*idx=*/0, /*a_elems=*/0, /*b_elems=*/0, partial),
      builder.BuildPadding());
}

absl::StatusOr<std::unique_ptr<TransposePlan>> TransposePlan::Create(
    Options o) {
  if (o.input_layout.has_value()) {
    if (const auto* t = std::get_if<Tiling>(&*o.input_layout)) {
      o.input_tiling = *t;
    } else if (const auto* s = std::get_if<Striding>(&*o.input_layout)) {
      o.input_striding = *s;
    }
  }

  auto is_negative = [](int64_t d) { return d < 0; };
  if (absl::c_find_if(o.dims, is_negative) != o.dims.end()) {
    return InvalidArgument("dims must be non-negative, got %s",
                           absl::StrJoin(o.dims, ","));
  }
  if (o.permutation.size() != o.dims.size()) {
    return InvalidArgument(
        "dims and permutation must have equal sizes, got %d and %d",
        o.dims.size(), o.permutation.size());
  }
  if (!IsPermutation(o.permutation)) {
    return InvalidArgument("permutation argument is not valid, got: %s",
                           absl::StrJoin(o.permutation, ","));
  }
  if (o.num_threads < 1) {
    return InvalidArgument("num_threads argument must be >= 1, got: %d",
                           o.num_threads);
  }

  int ndim = o.dims.size();

  auto plan = std::make_unique<TransposePlan>();
  plan->num_chunks_requested_ = o.num_threads;
  plan->elem_size_in_bytes_ = o.elem_size_in_bytes;
  switch (o.elem_size_in_bytes) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
      break;
    default:
      return InvalidArgument("Unsupported elem_size_in_bytes=%d",
                             o.elem_size_in_bytes);
  }
  plan->num_elems_ = std::accumulate(o.dims.begin(), o.dims.end(), int64_t{1},
                                     std::multiplies<int64_t>());
  plan->original_a_dims_.resize(ndim);
  absl::c_copy(o.dims, plan->original_a_dims_.begin());
  plan->original_b_dims_ = Permute(o.dims, o.permutation);

  bool output_contiguity =
      o.chunk_contiguity == TransposePlan::ChunkContiguity::kOutput;
  bool input_contiguity =
      o.chunk_contiguity == TransposePlan::ChunkContiguity::kInput;

  ABSL_RETURN_IF_ERROR(ParseTilingSpecification(
      ndim, o.output_tiling,
      /*nonstandard_layout=*/output_contiguity || o.tile_padding.has_value(),
      plan->b_tiling_));

  // Temporary vectors to hold un-permuted attributes
  absl::InlinedVector<int64_t, 4> temp_lda, temp_lda_tile, temp_a_tiling;

  // Parse the tile and stride specifications.
  ABSL_RETURN_IF_ERROR(ParseTilingSpecification(
      ndim, o.input_tiling,
      /*nonstandard_layout=*/o.input_striding.has_value() || input_contiguity,
      temp_a_tiling));
  ComputeStrides(plan->elem_size_in_bytes_, o.dims, temp_a_tiling, temp_lda,
                 temp_lda_tile);

  // Determine tile (outer) strides
  absl::InlinedVector<int64_t, 4> input_outer_strides;
  if (o.input_striding) {
    absl::Span<int64_t const> input_strides_in_bytes =
        o.input_striding->strides_in_bytes;
    if (input_strides_in_bytes.size() != o.dims.size()) {
      return InvalidArgument(
          "dims and input_striding must have equal sizes, "
          "got %d and %d",
          o.dims.size(), input_strides_in_bytes.size());
    }
    input_outer_strides.assign(input_strides_in_bytes.begin(),
                               input_strides_in_bytes.end());
    // Also save original strides if explicit
    plan->original_a_strides_.resize(ndim);
    absl::c_copy(input_strides_in_bytes, plan->original_a_strides_.begin());
  } else {
    input_outer_strides = temp_lda;
  }

  // Sort the dimensions from slowest-varying (largest strides) to
  // fastest-varying (smallest strides).
  // Maps new input dim -> old input dim
  std::vector<int64_t> dim_order(ndim);
  absl::c_iota(dim_order, 0);

  auto cost = [&](int k) {
    int64_t stride = input_outer_strides.at(k);
    // If there is a dimension with size equal to the element size, sort it
    // last. This ensures that we place any stride-1 dimension last.
    bool is_stride1 = stride == o.elem_size_in_bytes;
    // If there are multiple stride-1 dimensions, we'd prefer the one that
    // matches the stride-1 dimension of the output.
    // Failing that, we'd just prefer the largest stride-1 dimension last.
    bool is_trailing_dim_in_b = o.permutation.back() == k;

    // If we are applying ef57 conversion, we want a size-2 stride-1
    // dimension last.
    bool ef57_even =
        (is_stride1 && o.transformation == Transformation::kF64ToEf57 &&
         o.dims[k] == 2);

    return std::make_tuple(is_stride1, -std::abs(stride), ef57_even,
                           is_trailing_dim_in_b, o.dims[k]);
  };
  absl::c_stable_sort(dim_order,
                      [&cost](int i, int j) { return cost(i) < cost(j); });

  // Apply permutation to all plan attributes
  // dim_order maps new input dim -> old input dim, we need its inverse to
  // compute the new permutation.
  auto inv_dim_order = InversePermutation(dim_order);
  plan->lda_.reserve(ndim);
  plan->lda_tile_.reserve(ndim);
  plan->a_dims_.reserve(ndim);
  plan->permutation_.reserve(ndim);
  plan->a_tiling_.reserve(ndim);

  for (int i = 0; i < ndim; ++i) {
    int old_idx = dim_order[i];
    plan->lda_.push_back(input_outer_strides.at(old_idx));
    plan->lda_tile_.push_back(temp_lda_tile.at(old_idx));
    plan->a_dims_.push_back(o.dims[old_idx]);
    plan->permutation_.push_back(inv_dim_order[o.permutation[i]]);
    plan->a_tiling_.push_back(temp_a_tiling[old_idx]);
  }

  auto is_not_one = [](int64_t x) { return x != 1; };
  plan->a_is_tiled_ =
      (absl::c_find_if(plan->a_tiling_, is_not_one) != plan->a_tiling_.end());
  plan->b_is_tiled_ =
      (absl::c_find_if(plan->b_tiling_, is_not_one) != plan->b_tiling_.end());
  if (plan->a_is_tiled_ && plan->b_is_tiled_) {
    return Unimplemented(
        "Only one of the input and output may have a non-trivial tiling, "
        "got tilings: %s and %s",
        absl::StrJoin(plan->a_tiling_, ","),
        absl::StrJoin(plan->b_tiling_, ","));
  }

  plan->bits_per_element_ = o.elem_size_in_bytes * 8;
  TransposePlan::Transformation transformation = o.transformation;
  if (o.dest_bits_per_element.has_value() &&
      transformation == Transformation::kNone) {
    transformation = Transformation::kPackSubbyte;
  }
  plan->transformation_ = transformation;

  plan->chunk_contiguity_ = o.chunk_contiguity;
  plan->tile_padding_ = o.tile_padding;

  switch (plan->transformation_) {
    case Transformation::kNone:
      break;
    case Transformation::kF64ToEf57:
      if (o.dest_bits_per_element.has_value()) {
        return InvalidArgument(
            "EF57 conversion cannot be combined with dest_bits_per_element");
      }
      if (o.elem_size_in_bytes != sizeof(float)) {
        return InvalidArgument(
            "EF57 conversion requires a element size of %d bytes, got %d",
            sizeof(float), o.elem_size_in_bytes);
      }
      if (plan->a_dims_.empty() || plan->a_dims_.back() % 2 != 0 ||
          plan->lda_.back() != sizeof(float)) {
        return InvalidArgument(
            "EF57 conversion requires a stride-%d dimension whose size is a "
            "multiple of 2",
            sizeof(float));
      }
      break;
    case Transformation::kPackSubbyte:
      if (o.elem_size_in_bytes != 1) {
        return InvalidArgument(
            "Pack conversion requires an element size of 1 byte, "
            "got %d",
            o.elem_size_in_bytes);
      }
      if (!o.dest_bits_per_element.has_value()) {
        return InvalidArgument(
            "Pack conversion requires dest_bits_per_element to be set");
      }
      {
        int bits = *o.dest_bits_per_element;
        if (bits != 1 && bits != 2 && bits != 4) {
          return InvalidArgument(
              "Pack conversion requires dest_bits_per_element to be 1, 2, or "
              "4, "
              "got %d",
              bits);
        }
        if (plan->tile_padding_.has_value()) {
          const uint8_t mask = (1u << bits) - 1u;
          plan->tile_padding_ = static_cast<uint8_t>(
              (*plan->tile_padding_ & mask) * (0xffu / mask));
        }
        std::vector<int64_t> output_dims = Permute(o.dims, o.permutation);
        int elements_per_byte = 8 / bits;
        plan->bits_per_element_ = bits;
        if (output_dims.empty() ||
            output_dims.back() % elements_per_byte != 0 ||
            (plan->b_is_tiled_ &&
             plan->b_tiling_.back() % elements_per_byte != 0)) {
          plan->transformation_ = Transformation::kNone;
          plan->use_fallback_pack_ = true;
        }
      }
      break;
  }

  ABSL_RETURN_IF_ERROR(plan->Initialize());
  return plan;
}

absl::Status TransposePlan::Initialize() {
  if (num_elems_ == 0) {
    return absl::OkStatus();
  }
  // permutation maps dimensions of b to a
  // inverse_permutation maps dimensions of a to b
  std::vector<int64_t> inverse_permutation = InversePermutation(permutation_);

  int ndim = a_dims_.size();

  // Returns the inner of the strides for dimension `d`. If the dimension is
  // tiled, that's the tile stride, and if it is not, that is the untiled
  // stride.
  auto inner_stride = [&](int d, absl::Span<int64_t const> strides,
                          absl::Span<int64_t const> tile_strides,
                          absl::Span<int64_t const> tiling) {
    return tiling[d] > 1 ? tile_strides[d] : strides[d];
  };

  // If the plan is 0-dimensional, or the innermost dimension of A is not of
  // stride 1, adds a trivial size 1 dimension. The transpose kernels rely on
  // the presence of a stride-1 innermost dimension in the input.
  int pos_stride1a_in_a = ndim - 1;
  if (lda_.empty() || inner_stride(ndim - 1, lda_, lda_tile_, a_tiling_) !=
                          elem_size_in_bytes_) {
    int dim = static_cast<int>(a_dims_.size());
    permutation_.push_back(dim);
    inverse_permutation.push_back(dim);
    a_dims_.push_back(1);
    lda_.push_back(elem_size_in_bytes_);
    lda_tile_.push_back(1);
    a_tiling_.push_back(1);
    b_tiling_.push_back(1);
    ++ndim;
    ++pos_stride1a_in_a;
  }

  is_identity_permutation_ = true;
  for (int i = 0; i < permutation_.size(); ++i) {
    if (permutation_[i] != i) {
      is_identity_permutation_ = false;
      break;
    }
  }

  b_dims_ = Permute(a_dims_, permutation_);

  ComputeStrides(elem_size_in_bytes_, b_dims_, b_tiling_, ldb_, ldb_tile_);

  // Find the innermost dimension of B that is stride 1 element. We know such a
  // dimension exists, because we do not accept arbitrary stridings for B, but
  // it may not be the last dimension in the presence of tiling.
  int pos_stride1b_in_b = -1;
  for (int i = ndim - 1; i >= 0; --i) {
    if (inner_stride(i, ldb_, ldb_tile_, b_tiling_) == elem_size_in_bytes_) {
      pos_stride1b_in_b = i;
      break;
    }
  }
  CHECK_GE(pos_stride1b_in_b, 0);

  int pos_stride1b_in_a = permutation_[pos_stride1b_in_b];
  if (transformation_ == Transformation::kPackSubbyte) {
    int elements_per_byte = 8 / bits_per_element_;
    int64_t b_stride1_tile_size =
        std::max(a_tiling_[pos_stride1b_in_a], b_tiling_[pos_stride1b_in_b]);
    if (b_dims_[pos_stride1b_in_b] % elements_per_byte != 0 ||
        (b_stride1_tile_size > 1 &&
         b_stride1_tile_size % elements_per_byte != 0)) {
      transformation_ = Transformation::kNone;
      use_fallback_pack_ = true;
    }
  }

  int64_t stride_pos1a =
      inner_stride(pos_stride1a_in_a, lda_, lda_tile_, a_tiling_);
  int64_t stride_pos1b =
      inner_stride(pos_stride1b_in_b, ldb_, ldb_tile_, b_tiling_);

  inner_kernel_kind_ = (pos_stride1b_in_a == pos_stride1a_in_a) &&
                               (stride_pos1a == elem_size_in_bytes_) &&
                               (stride_pos1b == elem_size_in_bytes_)
                           ? InnerKernelKind::kMemcpy
                           : InnerKernelKind::kDefault;

  if (inner_kernel_is_memcpy() && transformation_ != Transformation::kNone &&
      transformation_ != Transformation::kPackSubbyte) {
    return InvalidArgument(
        "Memcpy-based plans cannot be used with transformations.");
  }

  // Calculate sentinel strides.
  if (!inner_kernel_is_memcpy()) {
    int pos_stride1a_in_b = inverse_permutation[pos_stride1a_in_a];
    sentinel_lda_ = inner_stride(pos_stride1b_in_a, lda_, lda_tile_, a_tiling_);
    sentinel_ldb_ = inner_stride(pos_stride1a_in_b, ldb_, ldb_tile_, b_tiling_);
  }

  // Order to traverse dimensions, from slowest-varying to fastest-varying.
  std::vector<Loop> loop_order;

  loop_order.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    Loop loop;
    loop.dim_in_a = i;
    loop.tile_interior = false;
    int64_t dim_size = a_dims_[i];
    int64_t tile_size =
        std::max(a_tiling_[i], b_tiling_[inverse_permutation[i]]);
    bool has_partial_tile = (dim_size % tile_size != 0);
    if (has_partial_tile) {
      // We only need to track tile interiors correctly if there are partial
      // tiles.
      loop.dim_size = dim_size;
      loop.tile_size = tile_size;
    } else {
      loop.dim_size = dim_size / tile_size;
      loop.tile_size = 1;
    }

    loop.lda = lda_[i];
    if (a_tiling_[i] == 1) {
      loop.lda *= tile_size;
    }
    loop.ldb = ldb_[inverse_permutation[i]];
    if (b_tiling_[inverse_permutation[i]] == 1) {
      loop.ldb *= tile_size;
    }
    loop.is_inner_dim_in_a = (tile_size == 1) && (i == pos_stride1a_in_a);
    loop.is_inner_dim_in_b = (tile_size == 1) && (i == pos_stride1b_in_a);
    loop_order.push_back(loop);

    if (tile_size > 1) {
      loop.tile_interior = has_partial_tile;
      if (!has_partial_tile) {
        loop.dim_size = tile_size;
        loop.tile_size = 1;
      }
      loop.lda = a_is_tiled_ ? lda_tile_[i] : lda_[i];
      loop.ldb = b_is_tiled_ ? ldb_tile_[inverse_permutation[i]]
                             : ldb_[inverse_permutation[i]];
      loop.is_inner_dim_in_a = (i == pos_stride1a_in_a);
      loop.is_inner_dim_in_b = (i == pos_stride1b_in_a);
      loop_order.push_back(loop);
    }
  }

  auto loops_to_string = [](absl::Span<const Loop> loops) {
    return absl::StrJoin(loops, "\n  ", [](std::string* out, const Loop& l) {
      absl::StrAppend(out, l.ToString());
    });
  };

  VLOG(5) << "Before RemoveTrivialLoops: " << loops_to_string(loop_order);
  RemoveTrivialLoops(loop_order);
  VLOG(5) << "After RemoveTrivialLoops: " << loops_to_string(loop_order);
  CoalesceLoops(loop_order);
  VLOG(5) << "After CoalesceLoops: " << loops_to_string(loop_order);

  // Bound the block sizes so they are smaller than the stride-1 dimension
  // size.
  int64_t a_stride1_size =
      std::max(a_tiling_[pos_stride1a_in_a],
               b_tiling_[inverse_permutation[pos_stride1a_in_a]]);
  if (a_stride1_size == 1) {
    a_stride1_size = a_dims_[pos_stride1a_in_a];
  } else {
    // If there's only one tile, we should use the dimension size.
    a_stride1_size = std::min(a_dims_[pos_stride1a_in_a], a_stride1_size);
  }
  int64_t b_stride1_size = std::max(a_tiling_[permutation_[pos_stride1b_in_b]],
                                    b_tiling_[pos_stride1b_in_b]);
  if (b_stride1_size == 1) {
    b_stride1_size = b_dims_[pos_stride1b_in_b];
  } else {
    b_stride1_size = std::min(b_stride1_size, b_dims_[pos_stride1b_in_b]);
  }

  // For kInterleave and kDeinterleave, the trip count of the large side's
  // stride-1 loop.
  int64_t large_loop_size = 0;
  if (!inner_kernel_is_memcpy() && transformation_ == Transformation::kNone) {
    const auto loop_a_it = absl::c_find_if(
        loop_order, [](const Loop& l) { return l.is_inner_dim_in_a; });
    const auto loop_b_it = absl::c_find_if(
        loop_order, [](const Loop& l) { return l.is_inner_dim_in_b; });
    DCHECK(loop_a_it != loop_order.end());
    DCHECK(loop_b_it != loop_order.end());
    // loop_a and loop_b iterate over A's and B's stride-1 dimensions (or their
    // tile interiors).
    const Loop& loop_a = *loop_a_it;
    const Loop& loop_b = *loop_b_it;
    DCHECK_EQ(loop_a.lda, elem_size_in_bytes_);
    DCHECK_EQ(loop_b.ldb, elem_size_in_bytes_);
    DCHECK_EQ(loop_a.ldb, sentinel_ldb_);
    DCHECK_EQ(loop_b.lda, sentinel_lda_);
    const auto trip_count = [](const Loop& l) {
      return l.tile_interior ? std::min(l.tile_size, l.dim_size) : l.dim_size;
    };
    const int64_t loop_a_size = trip_count(loop_a);
    const int64_t loop_b_size = trip_count(loop_b);
    // Returns whether to use the `kind` kernel with K = small.dim_size, where
    // `small` is the stride-1 loop of the small side.
    const auto use_kernel = [&](InnerKernelKind kind, const Loop& small,
                                int64_t group_stride, int64_t large_size) {
      DCHECK_EQ(small.tile_size == 1, !small.tile_interior);
      const int64_t k = small.dim_size;
      // With a partial tile, the trip count is not a constant K.
      const bool no_partial_tile = !small.tile_interior;
      // The K-groups are back to back, so the small side is one contiguous run.
      const bool contiguous = group_stride == k * elem_size_in_bytes_;
      // Faster than the default kernel for this K and element size.
      const bool faster =
          InterleaveKernelIsFaster(kind, k, elem_size_in_bytes_);
      const bool large_side_is_long = large_size >= 8;
      return no_partial_tile && contiguous && faster && large_side_is_long;
    };
    if (use_kernel(InnerKernelKind::kInterleave, loop_b, sentinel_ldb_,
                   loop_a_size)) {
      inner_kernel_kind_ = InnerKernelKind::kInterleave;
      interleave_size_ = loop_b.dim_size;
      large_loop_size = loop_a_size;
    } else if (use_kernel(InnerKernelKind::kDeinterleave, loop_a, sentinel_lda_,
                          loop_b_size)) {
      inner_kernel_kind_ = InnerKernelKind::kDeinterleave;
      interleave_size_ = loop_a.dim_size;
      large_loop_size = loop_b_size;
    }
  }

  if (inner_kernel_is_memcpy()) {
    inner_block_elems_ = -1;
    outer_block_elems_a_ = -1;
    outer_block_elems_b_ = -1;
  } else {
    // What are the smallest and largest block sizes for which we have a
    // vectorized kernel for this element size?
    const int max_outer_block_elems =
        kMaxOuterBlockSizeBytes / elem_size_in_bytes_;
    int min_inner_block_elems;
    int max_inner_block_elems;
    switch (elem_size_in_bytes_) {
      case 1:
      case 2:
      case 4:
      case 8:
        min_inner_block_elems = 1;
        max_inner_block_elems =
            std::min<int>(max_outer_block_elems,
                          kMaxInnerBlockSizeBytes / elem_size_in_bytes_);
        break;
      case 16:
        min_inner_block_elems = 1;
        max_inner_block_elems = 1;
        break;
      default:
        LOG(FATAL) << "Unreachable: element size " << elem_size_in_bytes_;
    }
    if (inner_kernel_kind_ == InnerKernelKind::kInterleave) {
      inner_block_elems_ = 1;
      outer_block_elems_a_ = std::min<int64_t>(
          large_loop_size, max_inner_block_elems * max_outer_block_elems);
      outer_block_elems_b_ = 1;
    } else if (inner_kernel_kind_ == InnerKernelKind::kDeinterleave) {
      inner_block_elems_ = 1;
      outer_block_elems_a_ = 1;
      outer_block_elems_b_ = std::min<int64_t>(
          large_loop_size, max_inner_block_elems * max_outer_block_elems);
    } else {
      inner_block_elems_ = max_inner_block_elems;
      while (inner_block_elems_ > std::min(a_stride1_size, b_stride1_size)) {
        inner_block_elems_ /= 2;
      }
      if (inner_block_elems_ < min_inner_block_elems) {
        // Size is smaller than our smallest vectorized kernel. Use the scalar
        // path.
        inner_block_elems_ = 1;
      }
      outer_block_elems_a_ = FloorOfRatio<int64_t>(
          std::min<int64_t>(max_outer_block_elems, a_stride1_size),
          inner_block_elems_);
      outer_block_elems_a_ = std::max<int64_t>(outer_block_elems_a_, 1);
      outer_block_elems_b_ = FloorOfRatio<int64_t>(
          std::min<int64_t>(max_outer_block_elems, b_stride1_size),
          inner_block_elems_);
      outer_block_elems_b_ = std::max<int64_t>(outer_block_elems_b_, 1);
    }
  }

  // Identify contiguous loops for chunk scheduling.
  // Loops with smallest strides in the contiguous buffer form the contiguous
  // region. Accumulate loops until their combined iteration count reaches
  // the target chunk size.
  IdentifyContiguousLoops(loop_order);
  ChooseLoopOrder(loop_order);

  for (Loop& loop : loop_order) {
    if (loop.tile_interior || loop.tile_size == 1) {
      if (inner_kernel_is_memcpy()) {
        if (transformation_ == Transformation::kPackSubbyte &&
            loop.is_inner_dim_in_a) {
          loop.inc = 8 / bits_per_element_;
        }
      } else if (loop.is_inner_dim_in_a) {
        loop.inc = (inner_kernel_kind_ == InnerKernelKind::kDeinterleave)
                       ? loop.dim_size
                       : inner_block_elems_ * outer_block_elems_a_;
      } else if (loop.is_inner_dim_in_b) {
        loop.inc = (inner_kernel_kind_ == InnerKernelKind::kInterleave)
                       ? loop.dim_size
                       : inner_block_elems_ * outer_block_elems_b_;
      }
    }
  }

  // It is a required invariant of the loop order that tile interiors always
  // appear after the corresponding tile exterior. This is a consequence of the
  // heuristic above, because the tile interior must have smaller strides in
  // both input and output.

  // The stride-1 loop must be innermost for a memcpy loop.
  DCHECK(!inner_kernel_is_memcpy() || loop_order.back().is_inner_dim_in_a)
      << ToString();

  int num_chunks = ChooseParallelizationStrategy(loop_order);
  VLOG(5) << "After ChooseParallelizationStrategy num_chunks=" << num_chunks
          << " loops: " << loops_to_string(loop_order);
  PartitionLoops(a_dims_.size(), num_chunks, loop_order, chunk_loops_,
                 input_chunk_iteration_offsets_,
                 output_chunk_iteration_offsets_);
  ComputeChunkSizes();
  VLOG(8) << "Before plan build: " << ToString();
  chunk_ops_.resize(num_chunks);
  scratch_size_ = 0;
  for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    Plan plan = BuildChunkPlan(chunk_id);
    chunk_ops_[chunk_id].clear();
    int64_t chunk_temp_bytes =
        FlattenPlan(plan, elem_size_in_bytes_, inner_kernel_kind_,
                    inner_block_elems_, interleave_size_, chunk_ops_[chunk_id]);
    scratch_size_ = std::max(scratch_size_, chunk_temp_bytes);
  }

  VLOG(5) << "Final plan: " << ToString();
  return absl::OkStatus();
}

void TransposePlan::ChooseLoopOrder(std::vector<Loop>& loop_order) const {
  // Loop order heuristic: try to make loops with small strides innermost.
  // We use a greedy selection to pick loops from slowest-varying to
  // fastest-varying, subject to hard constraints:
  // 1. If the inner kernel is a memcpy, the stride-1 dimension MUST be
  //    innermost.
  // 2. A tile interior MUST come after its corresponding tile exterior.
  std::vector<Loop> remaining = std::move(loop_order);
  loop_order.clear();
  loop_order.reserve(remaining.size());

  // Computes a sort key for each candidate loop, returning a 3-tuple:
  //   <contiguity, -miss_cost, -stride_tie_breaker>
  // Lexicographical ordering selects the smallest tuple first, placing that
  // loop into an outer position (from slowest-varying to fastest-varying):
  // 1. Contiguity category (non-contiguous loops placed in outer loops).
  // 2. Primary cost: cache-line misses (higher miss cost placed in outer
  // loops).
  // 3. Secondary tie-breaker: when cache misses tie (e.g. both loops have
  //    strides >= 128B), the loop with larger minimum stride goes outer to
  //    favor better locality in at least one buffer for the inner loop.
  auto soft_cost = [&](const Loop& l) -> std::tuple<int, double, double> {
    int64_t a_stride = std::abs(l.lda);
    if (!inner_kernel_is_memcpy() && l.is_inner_dim_in_a) {
      a_stride *= inner_block_elems_ * outer_block_elems_a_;
    }
    int64_t b_stride = std::abs(l.ldb);
    if (!inner_kernel_is_memcpy() && l.is_inner_dim_in_b) {
      b_stride *= inner_block_elems_ * outer_block_elems_b_;
    }

    switch (chunk_contiguity_) {
      case ChunkContiguity::kOutput:
        return std::make_tuple(0, -b_stride, 0.0);
      case ChunkContiguity::kInput:
        return std::make_tuple(0, -a_stride, 0.0);
      case ChunkContiguity::kNone: {
        constexpr double kOutputPenalty = 1.01;
        const double min_stride =
            std::min<double>(a_stride * kOutputPenalty, b_stride);

        // Pure tile-packing identity linearizers/delinearizers preserve the
        // baseline std::min cost to keep 2D hardware tile pairings intact.
        if (is_identity_permutation_) {
          return std::make_tuple(l.contiguity, -min_stride, 0.0);
        }

        // Model the memory traffic of placing this loop in an inner position
        // based on the count of cache lines missed (bytes pulled from L2 to
        // L1). Memory transfers occur at cache-line granularity (128-byte cache
        // line pairs on modern CPUs like AMD Zen 3/4).
        // - Strides < 128 bytes share cache lines across iterations (spatial
        //   reuse).
        // - Strides >= 128 bytes incur a full cache-line miss per iteration.
        constexpr double kCacheLineBytes = 128.0;

        const double misses_a = std::min(1.0, a_stride / kCacheLineBytes);
        const double misses_b = std::min(1.0, b_stride / kCacheLineBytes);
        const double base_cost = misses_a + misses_b * kOutputPenalty;

        // Break ties among loops with full cache-line misses using the minimum
        // stride across both buffers (favoring loops with better locality in
        // at least one buffer, such as tile interior loops over tile exterior
        // loops).
        return std::make_tuple(l.contiguity, -base_cost, -min_stride);
      }
    }
  };

  while (!remaining.empty()) {
    int best_idx = -1;
    for (int i = 0; i < remaining.size(); ++i) {
      const Loop& l = remaining[i];

      // Hard constraint 1: memcpy kernel requirement.
      if (inner_kernel_is_memcpy() && l.is_inner_dim_in_a &&
          remaining.size() > 1) {
        continue;
      }

      // Hard constraint 2: tile ordering.
      // A tile interior MUST come after its corresponding tile exterior.
      if (l.tile_interior) {
        auto is_exterior = [&](const Loop& r) {
          return !r.tile_interior && r.dim_in_a == l.dim_in_a;
        };
        if (absl::c_any_of(remaining, is_exterior)) {
          continue;
        }
      }

      if (best_idx == -1 || soft_cost(l) < soft_cost(remaining[best_idx])) {
        best_idx = i;
      }
    }
    CHECK_NE(best_idx, -1);
    loop_order.push_back(std::move(remaining[best_idx]));
    remaining.erase(remaining.begin() + best_idx);
  }

  VLOG(5) << "After loop ordering sort: "
          << absl::StrJoin(loop_order, ", ",
                           [](std::string* out, const Loop& l) {
                             absl::StrAppend(out, l.ToString());
                           });
}
int TransposePlan::ChooseParallelizationStrategy(
    std::vector<Loop>& loop_order) const {
  int available_parallelism = num_chunks_requested_;

  // Compute the number of iterations in `loop`.
  auto loop_iterations = [&](const Loop& loop) {
    int64_t size = loop.tile_interior
                       ? ((tile_padding_.has_value() && b_is_tiled_)
                              ? loop.tile_size
                              : std::min(loop.tile_size, loop.dim_size))
                       : (CeilOfRatio(loop.dim_size, loop.tile_size));
    return CeilOfRatio<int64_t>(size, loop.inc);
  };

  // Estimate the number of bytes each iteration of each loop processes.
  absl::InlinedVector<int64_t, 4> work_in_bytes(loop_order.size());
  int64_t acc = elem_size_in_bytes_;
  if (inner_kernel_kind_ == InnerKernelKind::kInterleave ||
      inner_kernel_kind_ == InnerKernelKind::kDeinterleave) {
    acc *= interleave_size_ * inner_block_elems_ * outer_block_elems_a_ *
           outer_block_elems_b_;
  } else if (!inner_kernel_is_memcpy()) {
    acc *= inner_block_elems_ * inner_block_elems_ * outer_block_elems_a_ *
           outer_block_elems_b_;
  } else if (transformation_ == Transformation::kPackSubbyte) {
    acc *= 8 / bits_per_element_;
  }
  auto work_it = work_in_bytes.rbegin();
  for (auto it = loop_order.rbegin(); it != loop_order.rend(); ++it) {
    *work_it++ = acc;
    acc *= loop_iterations(*it);
  }
  VLOG(7) << "Per-loop iteration work in bytes: "
          << absl::StrJoin(work_in_bytes, ",");

  // Heuristic that attempts to parallelize the outermost loops, down to a
  // minimum per-thread number of bytes processed.
  int num_chunks = 1;
  for (size_t i = 0; i < loop_order.size(); ++i) {
    Loop& loop = loop_order[i];
    CHECK_GE(available_parallelism, 1);

    // Initialize loop iteration bounds to full range in element units.
    loop.start = 0;
    loop.end = loop.tile_interior ? loop.tile_size
                                  : CeilOfRatio(loop.dim_size, loop.tile_size);

    int64_t iterations_without_blocking = loop.end - loop.start;
    int64_t iterations = loop_iterations(loop);

    // Contiguous loops must not be parallelized to maintain chunk contiguity.
    // Tile interiors with partial tiles must not be parallelized since
    // otherwise we will not account for things like padding correctly when
    // calculating chunk sizes.
    if (loop.contiguity > 1 || loop.tile_interior) {
      loop.parallelism = 1;
      if (chunk_contiguity_ != ChunkContiguity::kNone &&
          loop.parallelism < iterations_without_blocking) {
        available_parallelism = 1;
      }
      VLOG(8) << "loop " << i << " restricted to parallelism=1"
              << " iterations=" << iterations_without_blocking
              << " available_parallelism=" << available_parallelism;
      continue;
    }

    int kMinBytesPerThread =
        chunk_contiguity_ != ChunkContiguity::kNone ? 1 : (1 << 26);
    int64_t min_iterations_per_thread =
        CeilOfRatio<int64_t>(kMinBytesPerThread, work_in_bytes[i]);
    int64_t parallel_work = CeilOfRatio(iterations, min_iterations_per_thread);

    VLOG(8) << "loop " << i << ": iterations=" << iterations
            << " parallel_work=" << parallel_work
            << " available_parallelism=" << available_parallelism;
    int parallelism = std::min<int64_t>(available_parallelism, parallel_work);
    if (parallelism > 1) {
      // If we use CeilOfRatio(iterations, parallelism) as the chunk size, we
      // might end up with fewer chunks than parallelism if the chunk size is
      // large. For example, if iterations=17 and parallelism=16,
      // chunk_size=2. Then useful_tasks=9. We should reduce parallelism to 9.
      int64_t chunk_size =
          CeilOfRatio(iterations, static_cast<int64_t>(parallelism));
      int64_t useful_tasks = CeilOfRatio(iterations, chunk_size);
      parallelism = useful_tasks;
    }
    loop.parallelism = parallelism;
    num_chunks *= loop.parallelism;

    // Once a loop is sequential (parallelism < iterations), we must not
    // parallelize any inner loops if we want to maintain chunk contiguity.
    // This is because parallelizing an inner loop while a sequential outer
    // loop has multiple iterations would interleave the inner splits across
    // the outer iterations.
    if (chunk_contiguity_ != ChunkContiguity::kNone &&
        loop.parallelism < iterations_without_blocking) {
      available_parallelism = 1;
    } else {
      available_parallelism /= loop.parallelism;
    }
  }
  return num_chunks;
}

/*static*/ void TransposePlan::PartitionLoops(
    int num_dims, int num_chunks, const std::vector<Loop>& loop_order,
    std::vector<std::vector<TransposePlan::Loop>>& result,
    std::vector<int64_t>& input_chunk_iteration_offsets,
    std::vector<int64_t>& output_chunk_iteration_offsets) {
  // Copy the base loop order for each chunk.
  result.resize(num_chunks, loop_order);
  input_chunk_iteration_offsets.resize(num_chunks);
  output_chunk_iteration_offsets.resize(num_chunks);
  for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    // For each loop, narrow the start/end bounds to this chunk's portion.
    int task_id_remaining = chunk_id;
    int num_tasks_remaining = num_chunks;

    for (size_t i = 0; i < loop_order.size(); ++i) {
      Loop& chunk_loop = result[chunk_id][i];
      const Loop& base_loop = loop_order[i];

      num_tasks_remaining /= base_loop.parallelism;
      int task_id = task_id_remaining / num_tasks_remaining;
      task_id_remaining = task_id_remaining % num_tasks_remaining;

      // Divide this loop's iterations (in element units) among parallelism
      // tasks.
      int64_t iterations = base_loop.end - base_loop.start;
      int64_t blocks = CeilOfRatio(iterations, base_loop.inc);
      int64_t blocks_per_task =
          CeilOfRatio<int64_t>(blocks, base_loop.parallelism);

      chunk_loop.start =
          base_loop.start +
          std::min(iterations, task_id * blocks_per_task * base_loop.inc);
      chunk_loop.end =
          base_loop.start +
          std::min(iterations, (task_id + 1) * blocks_per_task * base_loop.inc);
      input_chunk_iteration_offsets[chunk_id] +=
          chunk_loop.start * chunk_loop.lda;
      output_chunk_iteration_offsets[chunk_id] +=
          chunk_loop.start * chunk_loop.ldb;
    }

    // Set has_partial_tile for tile interior loops.
    // First pass: find which dimensions' exterior loops include the last tile.
    std::vector<bool> dims_with_partial(num_dims, false);
    for (const Loop& loop : result[chunk_id]) {
      if (!loop.tile_interior && loop.dim_size % loop.tile_size != 0) {
        int64_t full_tiles = CeilOfRatio(loop.dim_size, loop.tile_size);
        if (loop.end >= full_tiles) {
          dims_with_partial[loop.dim_in_a] = true;
        }
      }
    }
    // Second pass: set has_partial_tile on interior loops.
    for (Loop& loop : result[chunk_id]) {
      if (loop.tile_interior && dims_with_partial[loop.dim_in_a]) {
        loop.has_partial_tile = true;
      }
    }
  }
}

void TransposePlan::ComputeChunkSizes() {
  input_chunk_size_bytes_.resize(chunk_loops_.size(), 0);
  output_chunk_size_bytes_.resize(chunk_loops_.size(), 0);
  input_chunk_offset_bytes_.resize(chunk_loops_.size(), 0);
  output_chunk_offset_bytes_.resize(chunk_loops_.size(), 0);
  max_input_chunk_size_bytes_ = 0;
  max_output_chunk_size_bytes_ = 0;
  // Compute chunk sizes from the pre-computed chunk loop bounds.
  // For each loop, accumulate the offset range based on the iteration bounds
  // and strides. Strides can be negative, so track both min and max.
  for (size_t chunk_id = 0; chunk_id < chunk_loops_.size(); ++chunk_id) {
    absl::Span<const Loop> loops = chunk_loops_[chunk_id];

    int64_t input_min = 0, input_max = 0;
    int64_t output_min = 0, output_max = 0;

    for (const Loop& loop : loops) {
      if (loop.start >= loop.end) {
        continue;  // Empty iteration range
      }

      int64_t first = loop.start;
      int64_t last = loop.end - 1;  // Last iteration (exclusive end)

      // For tile interior loops with partial tiles, use the full tile size.
      int64_t input_last = last;
      int64_t output_last = last;
      if (loop.tile_interior && loop.has_partial_tile) {
        int64_t partial_size = loop.dim_size % loop.tile_size;
        if (a_is_tiled_) {
          input_last = loop.tile_size - 1;
        } else {
          input_last = partial_size - 1;
        }
        // Output: if tiled, use full tile extent (no adjustment)
        // If untiled, use partial size
        if (b_is_tiled_) {
          output_last = loop.tile_size - 1;
        } else {
          output_last = partial_size - 1;
        }
      }

      // Accumulate min/max (strides can be negative)
      int64_t input_first_offset = first * loop.lda;
      int64_t input_last_offset = input_last * loop.lda;
      input_min += std::min(input_first_offset, input_last_offset);
      input_max += std::max(input_first_offset, input_last_offset);

      int64_t output_first_offset = first * loop.ldb;
      int64_t output_last_offset = output_last * loop.ldb;
      output_min += std::min(output_first_offset, output_last_offset);
      output_max += std::max(output_first_offset, output_last_offset);
    }

    input_chunk_offset_bytes_[chunk_id] = input_min;
    output_chunk_offset_bytes_[chunk_id] = output_min;

    // Adjust iteration offsets to be relative to the physical start of the
    // chunk.
    input_chunk_iteration_offsets_[chunk_id] -= input_min;
    output_chunk_iteration_offsets_[chunk_id] -= output_min;

    // Extent = range of offsets + element size at the max position.
    input_chunk_size_bytes_[chunk_id] =
        input_max - input_min + elem_size_in_bytes_;
    output_chunk_size_bytes_[chunk_id] =
        output_max - output_min + elem_size_in_bytes_;
    if (transformation_ == Transformation::kPackSubbyte) {
      const int elements_per_byte = 8 / bits_per_element_;
      DCHECK_EQ(output_chunk_offset_bytes_[chunk_id] % elements_per_byte, 0);
      DCHECK_EQ(output_chunk_iteration_offsets_[chunk_id] % elements_per_byte,
                0);
      DCHECK_EQ(output_chunk_size_bytes_[chunk_id] % elements_per_byte, 0);
      output_chunk_offset_bytes_[chunk_id] /= elements_per_byte;
      output_chunk_iteration_offsets_[chunk_id] /= elements_per_byte;
      output_chunk_size_bytes_[chunk_id] /= elements_per_byte;
    }

    max_input_chunk_size_bytes_ = std::max(max_input_chunk_size_bytes_,
                                           input_chunk_size_bytes_[chunk_id]);
    max_output_chunk_size_bytes_ = std::max(max_output_chunk_size_bytes_,
                                            output_chunk_size_bytes_[chunk_id]);
  }
}

std::string TransposePlan::ToString() const {
  std::vector<std::string> plan_strings;
  plan_strings.reserve(chunk_ops_.size());
  for (size_t i = 0; i < chunk_ops_.size(); ++i) {
    std::string s = "thread:\n";
    FormatPlan(BuildChunkPlan(static_cast<int>(i)), /*indent=*/4,
               elem_size_in_bytes_, inner_kernel_kind_, interleave_size_, &s);
    plan_strings.push_back(std::move(s));
  }
  std::string plans_str = absl::StrJoin(plan_strings, "\n");
  auto format_loop = [](std::string* out, const Loop& loop) {
    absl::StrAppend(out, loop.ToString());
  };
  std::vector<std::string> chunk_strings;
  chunk_strings.reserve(chunk_loops_.size());
  for (int i = 0; i < chunk_loops_.size(); ++i) {
    chunk_strings.push_back(absl::StrFormat(
        "    chunk %d: physical_input_offset=%d physical_output_offset=%d "
        "input_size=%d output_size=%d "
        "logical_input_offset=%d logical_output_offset=%d loops:\n      %s",
        i, input_chunk_offset_bytes_[i], output_chunk_offset_bytes_[i],
        input_chunk_size_bytes_[i], output_chunk_size_bytes_[i],
        input_chunk_iteration_offsets_[i], output_chunk_iteration_offsets_[i],
        absl::StrJoin(chunk_loops_[i], "\n      ", format_loop)));
  }
  std::string chunk_loops_str = absl::StrJoin(chunk_strings, "\n");
  std::string transformation_str;
  switch (transformation_) {
    case Transformation::kNone:
      transformation_str = "none";
      break;
    case Transformation::kF64ToEf57:
      transformation_str = "ef57";
      break;
    case Transformation::kPackSubbyte:
      transformation_str = absl::StrCat("pack(", bits_per_element_, ")");
      break;
  }
  std::string inner_kernel_str;
  switch (inner_kernel_kind_) {
    case InnerKernelKind::kDefault:
      inner_kernel_str = "default";
      break;
    case InnerKernelKind::kMemcpy:
      inner_kernel_str = "memcpy";
      break;
    case InnerKernelKind::kInterleave:
      inner_kernel_str = absl::StrCat("interleave(", interleave_size_, ")");
      break;
    case InnerKernelKind::kDeinterleave:
      inner_kernel_str = absl::StrCat("deinterleave(", interleave_size_, ")");
      break;
  }
  std::string tile_padding_str = tile_padding_.has_value()
                                     ? absl::StrFormat("0x%02x", *tile_padding_)
                                     : "none";
  return absl::StrFormat(
      "elem_size=%d a_dims=%s b_dims=%s permutation=%s a_tiling=%s b_tiling=%s "
      "lda=%s lda_tile=%s ldb=%s ldb_tile=%s "
      "outer_bs=[%d,%d] inner_bs=%d inner_kernel=%s "
      "transformation=%s tile_padding=%s scratch_size=%d "
      "num_chunks_requested=%d\n"
      "chunk_loops:\n%s\n"
      "plans:\n%s",
      elem_size_in_bytes_, absl::StrJoin(a_dims_, ","),
      absl::StrJoin(Permute(a_dims_, permutation_), ","),
      absl::StrJoin(permutation_, ","), absl::StrJoin(a_tiling_, ","),
      absl::StrJoin(b_tiling_, ","), absl::StrJoin(lda_, ","),
      absl::StrJoin(lda_tile_, ","), absl::StrJoin(ldb_, ","),
      absl::StrJoin(ldb_tile_, ","), outer_block_elems_a_, outer_block_elems_b_,
      inner_block_elems_, inner_kernel_str, transformation_str,
      tile_padding_str, scratch_size_, num_chunks_requested_, chunk_loops_str,
      plans_str);
}

bool TransposePlanCacheKey::operator==(
    const TransposePlanCacheKey& other) const {
  return elem_size_in_bytes == other.elem_size_in_bytes && dims == other.dims &&
         permutation == other.permutation &&
         input_tiling == other.input_tiling &&
         input_striding == other.input_striding &&
         output_tiling == other.output_tiling &&
         transformation == other.transformation &&
         dest_bits_per_element == other.dest_bits_per_element &&
         num_threads == other.num_threads &&
         chunk_contiguity == other.chunk_contiguity &&
         tile_padding == other.tile_padding;
}

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key) {
  return H::combine(std::move(h), key.elem_size_in_bytes, key.num_threads,
                    key.chunk_contiguity, key.tile_padding, key.transformation,
                    key.dest_bits_per_element, key.dims, key.permutation,
                    key.input_tiling, key.input_striding, key.output_tiling);
}

TransposePlanCache::TransposePlanCache(int capacity)
    : lru_list_(capacity), cache_(&lru_list_) {}

TransposePlanCache::~TransposePlanCache() = default;

absl::StatusOr<std::shared_ptr<TransposePlan>> TransposePlanCache::GetOrCreate(
    const TransposePlan::Options& o) {
  TransposePlanCacheKey key;
  key.elem_size_in_bytes = o.elem_size_in_bytes;
  key.dims.resize(o.dims.size());
  absl::c_copy(o.dims, key.dims.begin());
  key.permutation.resize(o.permutation.size());
  absl::c_copy(o.permutation, key.permutation.begin());
  if (o.input_tiling) {
    key.input_tiling.emplace(o.input_tiling->tiling.begin(),
                             o.input_tiling->tiling.end());
  }
  if (o.input_striding) {
    key.input_striding.emplace(o.input_striding->strides_in_bytes.begin(),
                               o.input_striding->strides_in_bytes.end());
  }
  if (o.output_tiling) {
    key.output_tiling.emplace(o.output_tiling->tiling.begin(),
                              o.output_tiling->tiling.end());
  }
  TransposePlan::Transformation transformation = o.transformation;
  if (o.dest_bits_per_element.has_value() &&
      transformation == TransposePlan::Transformation::kNone) {
    transformation = TransposePlan::Transformation::kPackSubbyte;
  }
  key.transformation = transformation;
  key.dest_bits_per_element = o.dest_bits_per_element;
  key.num_threads = o.num_threads;
  key.chunk_contiguity = o.chunk_contiguity;
  key.tile_padding = o.tile_padding;
  return cache_.GetOrCreateIfAbsent(
      key,
      [&](const TransposePlanCacheKey& key)
          -> absl::StatusOr<std::shared_ptr<TransposePlan>> {
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<TransposePlan> plan,
                         TransposePlan::Create(o));
        return std::shared_ptr<TransposePlan>(std::move(plan));
      });
}
void TransposePlan::IdentifyContiguousLoops(
    std::vector<Loop>& loop_order) const {
  if (chunk_contiguity_ != ChunkContiguity::kNone) {
    int64_t target_chunk_bytes = CeilOfRatio<int64_t>(
        num_elems_ * elem_size_in_bytes_, num_chunks_requested_);

    // Sort loops by stride in the contiguous buffer (ascending).
    auto contiguous_stride = [&](const Loop& l) {
      return chunk_contiguity_ == ChunkContiguity::kInput ? std::abs(l.lda)
                                                          : std::abs(l.ldb);
    };

    // Create index vector and sort by contiguous stride.
    std::vector<size_t> indices(loop_order.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto cost = [&](const Loop& l) {
      int64_t stride = contiguous_stride(l);
      return std::make_tuple(!l.tile_interior, stride);
    };
    absl::c_stable_sort(indices, [&](size_t a, size_t b) {
      return cost(loop_order[a]) < cost(loop_order[b]);
    });

    int prev = -1;
    for (size_t idx : indices) {
      Loop& loop = loop_order[idx];
      int64_t stride = contiguous_stride(loop);
      if (stride == 0 ||
          (!loop.tile_interior && loop.dim_size == loop.tile_size)) {
        continue;
      }
      if (stride >= target_chunk_bytes) {
        break;
      }
      loop.contiguity = 1;
      if (prev != -1) {
        loop_order[prev].contiguity = 2;
      }
      prev = idx;
    }
  }
}

/*static*/ void TransposePlan::RemoveTrivialLoops(std::vector<Loop>& loops) {
  auto it = std::remove_if(loops.begin(), loops.end(), [](const Loop& loop) {
    // We must preserve the loop if it corresponds to the innermost dimension
    // of the layout, because the kernels (especially TransposeConstStride1)
    // rely on finding a node with is_inner_dim_in_a/b set to true.
    if (loop.is_inner_dim_in_a || loop.is_inner_dim_in_b) {
      return false;
    }
    if (loop.tile_interior) {
      return loop.tile_size == 1;
    }
    // Exterior loop.
    // Trivial if dim_size == tile_size (1 complete tile, no partials). This
    // also accounts for the case where the dimension is of size 1, since in
    // that case the tile size is also 1.
    return loop.dim_size == loop.tile_size;
  });
  loops.erase(it, loops.end());
}

/*static*/ void TransposePlan::CoalesceLoops(std::vector<Loop>& loops) {
  if (loops.empty()) {
    return;
  }

  // Coalesce from slow-varying to fast-varying (outer to inner).
  // loops[0] is slowest.
  int write_pos = 0;
  for (int read_pos = 1; read_pos < loops.size(); ++read_pos) {
    Loop& outer = loops[write_pos];
    const Loop& inner = loops[read_pos];

    int64_t inner_iter_size = inner.tile_interior
                                  ? inner.tile_size
                                  : (inner.dim_size / inner.tile_size);

    // Two loops can be coalesced if:
    // * neither has a partial tile
    // * the inner loop is a multiple of the outer loop.
    bool coalescable = (outer.dim_size % outer.tile_size == 0) &&
                       (inner.dim_size % inner.tile_size == 0) &&
                       (outer.lda == inner.lda * inner_iter_size) &&
                       (outer.ldb == inner.ldb * inner_iter_size);
    if (coalescable) {
      if (outer.tile_interior) {
        outer.tile_size *= inner.tile_size;
        outer.dim_size *= inner.dim_size;
      } else {
        outer.dim_size *= inner_iter_size;
      }

      outer.lda = inner.lda;
      outer.ldb = inner.ldb;

      outer.is_inner_dim_in_a =
          inner.is_inner_dim_in_a || outer.is_inner_dim_in_a;
      outer.is_inner_dim_in_b =
          inner.is_inner_dim_in_b || outer.is_inner_dim_in_b;

      // Don't advance write_pos, so we can merge more into 'outer'.
    } else {
      ++write_pos;
      if (write_pos != read_pos) {
        loops[write_pos] = inner;
      }
    }
  }
  loops.resize(write_pos + 1);
}

}  // namespace xla
