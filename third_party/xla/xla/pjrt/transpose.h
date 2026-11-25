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

// This file implements a out-of-place multidimensional array transpose
// inspired by the paper:
//
// Springer, P., Su, T. and Bientinesi, P., 2017, June. HPTT: A high-performance
// tensor transposition C++ library. In Proceedings of the 4th ACM SIGPLAN
// International Workshop on Libraries, Languages, and Compilers for Array
// Programming (pp. 56-62).
// https://arxiv.org/abs/1704.04374
//

#ifndef XLA_PJRT_TRANSPOSE_H_
#define XLA_PJRT_TRANSPOSE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/lru_cache.h"
#include "xla/pjrt/transpose_kernels.h"

namespace xla {

class TransposePlan {
 public:
  // elem_size_in_bytes: size of each element in bytes.
  // dims: the input shape, in elements.
  // permutation: for each output dimension, gives the number of the
  //   corresponding input dimension. Must be a permutation of [0..dims.size())
  // input_tiling: optional input tiling.
  // input_striding: optional input byte strides.
  // output_tiling: optional output tiling.
  //
  // A Striding represents the strides of the input array in bytes. (N.B. not
  // elements).
  //
  // A Tiling is a tiling specification for the input or output array. May
  // have fewer dimensions than `dims`, in which case the tiling applies to the
  // minormost dimensions and any remaining dimensions are left untiled (i.e.,
  // tile size 1). An empty tiling vector corresponds to an untiled dense
  // major-to-minor layout.
  //
  // Multiple tiling levels may be listed from largest (outermost) to smallest
  // (innermost), where each level evenly divides the one around it (matching
  // XLA's `Layout::tiles()`). For example, `{Tiling{{8, 128}}, Tiling{{2, 1}}}`
  // splits a 2D array into 8x128 blocks, and then splits each 8x128 block into
  // four 2x128 strips of 2x1 pairs so vertically adjacent pairs of elements sit
  // next to each other in memory. At most one of the input and output may be
  // tiled.
  //
  // For more information about tiling, see
  // https://www.tensorflow.org/xla/tiled_layout
  //
  // The size of the plan may be exponential in the number of non-trivial
  // tiled dimensions. This is acceptable because in the intended use case for
  // this code we expect at most 2 tiled dimensions on input and output.
  //
  // The input may have both a tiling and a striding. If both are present,
  // the striding determines the strides between outermost tiles (in bytes).
  //
  //
  // num_threads: is the number of threads requested. The actual number of
  //   threads used may be smaller if there isn't enough work per thread.
  struct Tiling {
    absl::Span<int64_t const> tiling;
  };
  struct Striding {
    absl::Span<int64_t const> strides_in_bytes;
  };
  enum class Transformation {
    // Apply no transformations to the data.
    kNone = 0,

    // Convert doubles into the ef57 extended precision pair-of-floats
    // representation used on TPU.
    kF64ToEf57 = 1,

    // Pack 8-bit integers into sub-byte packed representation.
    kPackSubbyte = 2,
  };

  // Requested contiguity of chunks.
  enum class ChunkContiguity {
    // We don't care about contiguity (default).
    kNone,

    // We want input chunks to be contiguous. Note that input contiguity is best
    // effort if there are input strides. We do not guarantee in this case that
    // the input chunks are non-overlapping. If there are no input strides, we
    // guarantee that the input chunks are non-overlapping.
    kInput,

    // We want output chunks to be contiguous. Output contiguity is guaranteed,
    // since we do not support arbitrary output strides.
    kOutput,
  };

  struct Options {
    size_t elem_size_in_bytes;
    absl::Span<int64_t const> dims;
    absl::Span<int64_t const> permutation;
    std::vector<Tiling> input_tiling;
    std::optional<Striding> input_striding = std::nullopt;
    std::vector<Tiling> output_tiling;
    Transformation transformation = Transformation::kNone;
    // If set, controls the subbyte packing ratio. Only 1, 2, 4 is supported.
    // nullopt is equivalent to 8 (no subbyte packing). Packing happens after
    // transpose is complete and packs 8 / dest_bits_per_element adjacent
    // values together into a single byte.
    std::optional<int> dest_bits_per_element = std::nullopt;

    // Requested number of chunks (threads). We will attempt to create
    // approximately this many chunks. The actual number of chunks may be
    // smaller or larger.
    // TODO(phawkins): rename to num_chunks.
    int num_threads = 1;

    ChunkContiguity chunk_contiguity = ChunkContiguity::kNone;

    // If set, sets the padding bytes in the tiled output buffer to this value.
    // When `dest_bits_per_element` is set, the low `dest_bits_per_element` bits
    // are repeated for each padded sub-byte element and any unused high bits in
    // the final byte. Zero-element transposes remain a no-op.
    std::optional<uint8_t> tile_padding = std::nullopt;
  };

  static absl::StatusOr<std::unique_ptr<TransposePlan>> Create(Options options);

  ~TransposePlan() = default;

  // Executes the transposition.
  // `a` is the input array and `b` is the output array. The input and output
  // arrays must not overlap.
  // Currently there are no alignment requirements on either `a` or `b`. However
  // performance may be better if either or both are aligned.
  void Execute(const void* a, void* b,
               std::optional<absl::FunctionRef<void(std::function<void()>)>>
                   schedule_work = std::nullopt) const;

  // Executes a single chunk of the transposition. To perform a complete
  // transposition, call ExecuteChunk for each chunk ID from 0 to Parallelism()
  // - 1. It is legal to call ExecuteChunk for independent chunks in parallel.
  // This is useful for callers that want to manage their own threading.
  // If input_is_global, `a` points to the base address of the input array,
  // exactly as Execute() accepts.
  // Otherwise, `a` points to the a chunk of that array, starting at offset
  // InputChunkOffsetBytes() and of size InputChunkSizeBytes(). Note that in the
  // presence of negative strides, this may be a negative offset.
  //
  // The same applies to `b` and `output_is_global`.
  void ExecuteChunk(int chunk_id, const void* a, void* b, bool input_is_global,
                    bool output_is_global) const;

  int64_t InputChunkOffsetBytes(int chunk_id) const {
    return input_chunk_offset_bytes_[chunk_id];
  }
  int64_t InputChunkSizeBytes(int chunk_id) const {
    return input_chunk_size_bytes_[chunk_id];
  }
  int64_t OutputChunkOffsetBytes(int chunk_id) const {
    return output_chunk_offset_bytes_[chunk_id];
  }
  int64_t OutputChunkSizeBytes(int chunk_id) const {
    return output_chunk_size_bytes_[chunk_id];
  }

  // Maximum chunk size in bytes across all chunks in the plan.
  int64_t MaxInputChunkSizeBytes() const { return max_input_chunk_size_bytes_; }
  int64_t MaxOutputChunkSizeBytes() const {
    return max_output_chunk_size_bytes_;
  }

  size_t ElemSizeInBytes() const { return elem_size_in_bytes_; }

  // Input and output size, in number of elements. Ignores any input striding,
  // but accounts for tiling.
  int64_t InputNumElems() const { return input_num_elems_; }
  int64_t OutputNumElems() const { return output_num_elems_; }

  absl::Span<int64_t const> InputDims() const { return original_a_dims_; }
  absl::Span<int64_t const> OutputDims() const { return original_b_dims_; }

  absl::Span<int64_t const> InputStrides() const { return original_a_strides_; }

  // Returns the number of items of parallel work in the plan.
  int Parallelism() const { return chunk_ops_.size(); }

  bool inner_kernel_is_memcpy() const { return inner_kernel_is_memcpy_; }

 private:
  TransposePlan() = default;

  void ExecuteChunkInternal(int chunk_id, const void* a, void* b,
                            bool input_is_global, bool output_is_global) const;

  void ExecuteInternal(
      const void* a, void* b,
      std::optional<absl::FunctionRef<void(std::function<void()>)>>
          schedule_work) const;

  // Size of each element in bytes.
  int64_t elem_size_in_bytes_ = 0;

  // Number of elements in the input array.
  int64_t num_elems_ = 0;

  // Input and output size, in number of elements. Ignores any input striding,
  // but accounts for tiling.
  int64_t input_num_elems_ = 0;
  int64_t output_num_elems_ = 0;

  // Description of the transpose, before any optimizations such as coalescing
  // dimensions have been applied.
  absl::InlinedVector<int64_t, 4> original_a_dims_;
  absl::InlinedVector<int64_t, 4> original_a_strides_;
  std::vector<int64_t> original_b_dims_;

  // Per-chunk byte offsets and sizes into the input and output arrays.
  // `offset_bytes` are the physical offsets of the start of each chunk (minimum
  // byte addresses touched by that chunk) relative to the logical start
  // (index 0) of the array. Note: if the input strides are negative, these
  // offsets can also be negative.
  std::vector<int64_t> input_chunk_offset_bytes_;
  std::vector<int64_t> output_chunk_offset_bytes_;

  // `iteration_offsets` are the byte offsets of the first logical element of
  // each chunk's iteration space, relative to the physical start of that chunk
  // (i.e. relative to `input_chunk_offset_bytes_`). These are non zero for
  // negative strides, where the chunk may start before the logical start of the
  // array.
  std::vector<int64_t> input_chunk_iteration_offsets_;
  std::vector<int64_t> output_chunk_iteration_offsets_;
  std::vector<int64_t> input_chunk_size_bytes_;
  std::vector<int64_t> output_chunk_size_bytes_;
  int64_t max_input_chunk_size_bytes_ = 0;
  int64_t max_output_chunk_size_bytes_ = 0;

  // Flattened preorder plan ops per chunk (indexed by chunk_id).
  absl::InlinedVector<std::vector<TransposeOp>, 1> chunk_ops_;

  // Maximum per-chunk scratch buffer size in bytes (64-byte aligned). 0 means
  // no scratch buffer is required.
  int64_t scratch_size_ = 0;

  int bits_per_element_ = 0;
  // If the inner dim is not divisible by the packing ratio, we cannot
  // pack into the destination in parallel, so we fallback to transposing
  // into a temporary buffer and then packing into the destination.
  bool use_fallback_pack_ = false;
  bool inner_kernel_is_memcpy_ = false;

  std::optional<uint8_t> tile_padding_;
};

struct TransposePlanCacheKey {
  template <typename H>
  friend H AbslHashValue(H h, const TransposePlanCacheKey& key);

  size_t elem_size_in_bytes;
  absl::InlinedVector<int64_t, 4> dims;
  absl::InlinedVector<int64_t, 4> permutation;
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2> input_tiling;
  std::optional<absl::InlinedVector<int64_t, 4>> input_striding;
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2> output_tiling;
  TransposePlan::Transformation transformation;
  std::optional<int> dest_bits_per_element;
  int num_threads;
  TransposePlan::ChunkContiguity chunk_contiguity;
  std::optional<uint8_t> tile_padding;

  bool operator==(const TransposePlanCacheKey& other) const;
};

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key);

// An LRU cache for transpose plans. Not thread-safe.
// Transpose plans aren't cheap to build, but once computed for a particular set
// of inputs can be cached and reused for arrays. TransposePlanCache implements
// such a cache.
class TransposePlanCache {
 public:
  explicit TransposePlanCache(int capacity);
  ~TransposePlanCache();

  TransposePlanCache(const TransposePlanCache&) = delete;
  TransposePlanCache(TransposePlanCache&&) = delete;
  TransposePlanCache& operator=(const TransposePlanCache&) = delete;
  TransposePlanCache& operator=(TransposePlanCache&&) = delete;

  // Creates or returns a cached copy of a transpose plan.
  absl::StatusOr<std::shared_ptr<TransposePlan>> GetOrCreate(
      const TransposePlan::Options& options);

 private:
  LRUCache<TransposePlanCacheKey,
           absl::StatusOr<std::shared_ptr<TransposePlan>>>::LRUList lru_list_;
  LRUCache<TransposePlanCacheKey,
           absl::StatusOr<std::shared_ptr<TransposePlan>>>
      cache_;
};

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_H_
