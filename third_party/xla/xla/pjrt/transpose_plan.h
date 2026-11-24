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

#ifndef XLA_PJRT_TRANSPOSE_PLAN_H_
#define XLA_PJRT_TRANSPOSE_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/transpose_kernels.h"

namespace xla {
namespace transpose_internal {

enum class Transformation {
  kNone = 0,
  kF64ToEf57 = 1,
  kPackSubbyte = 2,
};

enum class ChunkContiguity {
  kNone,
  kInput,
  kOutput,
};

// The kernel used for the innermost (stride-1) dimensions of the plan.
enum class InnerKernelKind {
  // Blocked transpose using TransposeMicroKernel.
  kDefault,
  // A and B have the same stride-1 dimension, so the inner loop is a memcpy.
  kMemcpy,
  // B's stride-1 dimension has a small size K and A's is large, each after
  // being coalesced with any dimensions contiguous with it in both A and B:
  // K rows of A, `lda` bytes apart, are merged element by element into one
  // contiguous run of B, i.e. b[K * i + k] = a[k][i].
  kInterleave,
  // The reverse of kInterleave: one contiguous run of A is split into K rows
  // of B, `ldb` bytes apart, i.e. b[k][i] = a[K * i + k].
  kDeinterleave,
};

// Represents one loop level in the planner's loop nest.
struct Loop {
  // Dimension number in A from which this loop originated. This is mostly
  // for debugging the plan.
  int dim_in_a;

  // If true, the loop iterates over the interior of a tiled dimension that
  // may have a trailing partial tile.
  // For an untiled dimension or a tiled dimension whose tile size evenly
  // divides the dimension size, this is always false. For a tiled dimension
  // with a trailing partial tile, we will have two loops: one over the tile
  // exteriors and one over the tile interiors.
  bool tile_interior;

  // Size of the iteration space.
  int64_t dim_size;

  // Size of the tiles, if this a tiled dimension with a trailing partial
  // tile.
  // We do not set this for tiled dimensions without partial tiles, since in
  // that case we can just use the dimension size, which affords more
  // opportunities for loop optimizations.
  int64_t tile_size;

  int64_t lda;  // Stride in A for this loop.
  int64_t ldb;  // Stride in B for this loop.

  // Is this the innermost (stride 1) dimension in A or B? These dimensions
  // are special for the kernels.
  bool is_inner_dim_in_a;
  bool is_inner_dim_in_b;

  // Number of parallel threads to use for this loop.
  int64_t parallelism = -1;

  // The unit of work for this loop. For loops that are not the innermost
  // dimension in A or B, this is 1. For the innermost dimension of a
  // transpose kernel, it is the kernel size.
  int64_t inc = 1;

  // Iteration bounds for this chunk. Initially [0, full_iterations).
  // After chunk splitting, each chunk's loops have narrowed bounds.
  int64_t start = 0;  // Inclusive start of iteration range
  int64_t end = 0;    // Exclusive end of iteration range

  // Is this loop over a tiled dimension where the iteration space touches
  // the partial tile at the end?
  bool has_partial_tile = false;

  // Is this loop which should remain contiguous to ensure the requested
  // chunk contiguity?
  // 0 = not contiguous
  // 1 = boundary loop
  // 2 = deep inner loop
  int contiguity = 0;

  bool operator==(const Loop& other) const;
  std::string ToString() const;
};

struct PlanOptions {
  size_t elem_size_in_bytes = 0;
  absl::Span<int64_t const> dims;
  absl::Span<int64_t const> permutation;
  std::optional<absl::Span<int64_t const>> input_tiling;
  std::optional<absl::Span<int64_t const>> input_striding;
  std::optional<absl::Span<int64_t const>> output_tiling;
  Transformation transformation = Transformation::kNone;
  std::optional<int> dest_bits_per_element;
  int num_threads = 1;
  ChunkContiguity chunk_contiguity = ChunkContiguity::kNone;
  std::optional<uint8_t> tile_padding;
};

// Flattened transpose plan returned by `BuildTransposePlan`: contains only the
// flattened per-chunk `TransposeOp` sequences, scratch-buffer sizing, and chunk
// metadata needed for execution.
struct FlattenedPlan {
  // Size of each element in bytes.
  int64_t elem_size_in_bytes = 0;

  // Number of elements in the input array.
  int64_t num_elems = 0;

  // Input and output size, in number of elements. Ignores any input striding,
  // but accounts for tiling.
  int64_t input_num_elems = 0;
  int64_t output_num_elems = 0;

  // Per-chunk byte offsets and sizes into the input and output arrays.
  // `offset_bytes` are the physical offsets of the start of each chunk (minimum
  // byte addresses touched by that chunk) relative to the logical start
  // (index 0) of the array. Note: if the input strides are negative, these
  // offsets can also be negative.
  std::vector<int64_t> input_chunk_offset_bytes;
  std::vector<int64_t> output_chunk_offset_bytes;

  // `iteration_offsets` are the byte offsets of the first logical element of
  // each chunk's iteration space, relative to the physical start of that chunk
  // (i.e. relative to `input_chunk_offset_bytes`). These are non zero for
  // negative strides, where the chunk may start before the logical start of the
  // array.
  std::vector<int64_t> input_chunk_iteration_offsets;
  std::vector<int64_t> output_chunk_iteration_offsets;
  std::vector<int64_t> input_chunk_size_bytes;
  std::vector<int64_t> output_chunk_size_bytes;
  int64_t max_input_chunk_size_bytes = 0;
  int64_t max_output_chunk_size_bytes = 0;

  // Flattened preorder plan ops per chunk (indexed by chunk_id).
  absl::InlinedVector<std::vector<TransposeOp>, 1> chunk_ops;

  // Maximum per-chunk scratch buffer size in bytes (64-byte aligned). 0 means
  // no scratch buffer is required.
  int64_t scratch_size = 0;

  int bits_per_element = 0;
  // If the inner dim is not divisible by the packing ratio, we cannot
  // pack into the destination in parallel, so we fallback to transposing
  // into a temporary buffer and then packing into the destination.
  bool use_fallback_pack = false;
  bool inner_kernel_is_memcpy = false;

  std::optional<uint8_t> tile_padding;

  // Exposed for unit tests verifying inner kernel selection.
  InnerKernelKind inner_kernel_kind = InnerKernelKind::kDefault;
  int interleave_size = 0;
};

// Exposed for testing.
void RemoveTrivialLoops(std::vector<Loop>& loops);
void CoalesceLoops(std::vector<Loop>& loops);
void ChooseLoopOrder(std::vector<Loop>& loop_order, bool is_memcpy,
                     ChunkContiguity chunk_contiguity = ChunkContiguity::kNone,
                     bool is_identity_permutation = false,
                     int inner_block_elems = 1, int outer_block_elems_a = 4,
                     int outer_block_elems_b = 4);

// Builds a flattened transpose plan from `options`.
absl::StatusOr<FlattenedPlan> BuildTransposePlan(const PlanOptions& options);

}  // namespace transpose_internal
}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_PLAN_H_
