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
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/lru_cache.h"

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
  // have fewer dimensions that `dims`, in which case the tiling applies to the
  // minormost dimensions and any remaining dimensions are left untiled (i.e.,
  // tile size 1). An empty tiling corresponds to an untiled dense
  // major-to-minor layout.
  //
  // For more information about tiling, see
  // https://www.tensorflow.org/xla/tiled_layout
  // This class supports a single level of tiling. In addition, the same
  // dimension currently cannot have different non-trivial tiling values in
  // both the input and output.
  //
  // The size of the plan may be exponential in the number of non-trivial
  // tiled dimensions. This is acceptable because in the intended use case for
  // this code we expect at most 2 tiled dimensions on input and output.
  //
  // The input may have both a tiling and a striding. If both are present,
  // the striding determines the strides between tiles (in bytes).
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
    std::optional<Tiling> input_tiling = std::nullopt;
    std::optional<Striding> input_striding = std::nullopt;
    std::optional<Tiling> output_tiling = std::nullopt;
    Transformation transformation = Transformation::kNone;

    // Requested number of chunks (threads). We will attempt to create
    // approximately this many chunks. The actual number of chunks may be
    // smaller or larger.
    // TODO(phawkins): rename to num_chunks.
    int num_threads = 1;

    ChunkContiguity chunk_contiguity = ChunkContiguity::kNone;

    // DEPRECATED: Use input_tiling or input_striding instead.
    // This field is only present for backward compatibility.
    // TODO(phawkins): remove me.
    std::optional<std::variant<Tiling, Striding>> input_layout = std::nullopt;
  };

  static absl::StatusOr<std::unique_ptr<TransposePlan>> Create(Options options);

  TransposePlan();
  ~TransposePlan();

  // Executes the transposition.
  // `a` is the input array and `b` is the output array. The input and output
  // arrays must not overlap.
  // Currently there are no alignment requirements on either `a` or `b`. However
  // performance may be better if either or both are aligned.
  void Execute(const void* a, void* b,
               std::optional<absl::FunctionRef<void(std::function<void(void)>)>>
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

  // Returns a human-readable description of the plan.
  std::string ToString() const;

  size_t ElemSizeInBytes() const { return elem_size_in_bytes_; }

  // Input and output size, in number of elements. Ignores any input striding,
  // but accounts for tiling.
  int64_t InputNumElems() const;
  int64_t OutputNumElems() const;

  absl::Span<int64_t const> InputDims() const { return original_a_dims_; }
  absl::Span<int64_t const> OutputDims() const { return original_b_dims_; }

  absl::Span<int64_t const> InputStrides() const { return original_a_strides_; }

  // Returns the number of items of parallel work in the plan.
  int Parallelism() const { return nodes_.size(); }

  struct Node;

 protected:
  // Methods protected so they can be accessed by tests.

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

  // Exposed for testing.
  static void RemoveTrivialLoops(std::vector<Loop>& loops);
  static void CoalesceLoops(std::vector<Loop>& loops);

  // Reorders loops to optimize for locality.
  void ChooseLoopOrder(std::vector<Loop>& loop_order) const;

  void set_inner_kernel_is_memcpy(bool is_memcpy) {
    inner_kernel_is_memcpy_ = is_memcpy;
  }

 private:
  // Performs plan initialization that cannot fail.
  void Initialize();

  void BuildPlanNodes(int chunk_id, std::vector<Node>& nodes);

  // Chooses a parallelism for each loop. Returns the number of separate chunks
  // in the plan, and populates the `parallelism` field of each loop.
  int ChooseParallelizationStrategy(std::vector<Loop>& loop_order) const;

  // Identifies the innermost loops that should remain contiguous to achieve
  // the requested chunk contiguity.
  void IdentifyContiguousLoops(std::vector<Loop>& loop_order) const;

  // Creates per-chunk loop vectors by splitting loop_order_ into per-chunk
  // loops. Returns a vector of loop vectors, one per chunk. Each chunk's
  // loops have their start/end bounds narrowed to represent that chunk's work.
  static void PartitionLoops(
      int num_dims, int num_chunks, const std::vector<Loop>& loop_order,
      std::vector<std::vector<TransposePlan::Loop>>& result,
      std::vector<int64_t>& input_chunk_iteration_offsets,
      std::vector<int64_t>& output_chunk_iteration_offsets);

  void ComputeChunkSizes();

  // The signature of ExecuteTyped uses char* pointers because we perform
  // address calculations with strides in bytes; the strides need not be
  // multiples of the element size.
  template <typename T, Transformation transformation>
  void ExecuteTyped(const char* a, char* b, absl::Span<Node const> nodes) const;

  // Number of chunks requested.
  int num_chunks_requested_ = 1;

  // Size of each element in bytes.
  int64_t elem_size_in_bytes_ = 0;

  // Number of elements in the input array.
  int64_t num_elems_ = 0;

  // Description of the transpose, before any optimizations such as coalescing
  // dimensions have been applied.
  absl::InlinedVector<int64_t, 4> original_a_dims_;
  absl::InlinedVector<int64_t, 4> original_a_strides_;
  std::vector<int64_t> original_b_dims_;

  // Dimensions of the input array A.
  absl::InlinedVector<int64_t, 4> a_dims_;
  absl::InlinedVector<int64_t, 4> a_strides_;

  // Dimensions of the output array B.
  std::vector<int64_t> b_dims_;

  // Dimension permutation to apply to form B. For each dimension of B, what is
  // the corresponding dimension of A?
  absl::InlinedVector<int64_t, 4> permutation_;

  // Leading-dimension sizes (byte strides) of each dimension.
  absl::InlinedVector<int64_t, 4> lda_;       // Strides for tiles
  absl::InlinedVector<int64_t, 4> lda_tile_;  // Strides for tile interiors
  absl::InlinedVector<int64_t, 4> ldb_;       // Strides for tiles
  absl::InlinedVector<int64_t, 4> ldb_tile_;  // Strides for tile interiors

  // Tile sizes in each dimension. Has size equal to the number of dimensions.
  // A 1 entry means that dimension is not tiled.
  absl::InlinedVector<int64_t, 4> a_tiling_;
  absl::InlinedVector<int64_t, 4> b_tiling_;
  bool a_is_tiled_ = false;
  bool b_is_tiled_ = false;

  // Per-chunk loop nests. Each loop nest has its own start/end bounds
  // representing one chunk of the work.
  std::vector<std::vector<Loop>> chunk_loops_;

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

  // Root nodes of the plan, i.e., pointing to the outermost loops in the loop
  // nest. The outer vector is indexed on the thread ID.
  absl::InlinedVector<std::vector<Node>, 1> nodes_;

  // Are the innermost (stride-1) dimensions the same dimension? This determines
  // whether the inner kernel is a transpose or a memcpy.
  bool inner_kernel_is_memcpy_ = false;

  // Size of the inner (microkernel) block size. This is the unit of work for
  // our vectorized kernels.
  int inner_block_elems_ = 1;
  // Size of the outer (macrokernel) block size. This is the unit of work for
  // cache blocking and need not be equal between input and output.
  int outer_block_elems_a_ = 4;
  int outer_block_elems_b_ = 4;

  // Strides used by an inner transpose kernel. Unused for memcpy kernels.
  int64_t sentinel_lda_ = -1;
  int64_t sentinel_ldb_ = -1;

  // Transformations to apply to the input before transposition.
  // Currently the only supported transformation is EF57 conversion, which is
  // a pair-of-floats extended precision representation used on TPU. We
  // support fusing transformations with the transpose for two reasons:
  // (a) it makes sense to fuse cheap computations with a memory-bandwidth
  //     bound transformation, and
  // (b) it allows us to support non-trivial striding.
  Transformation transformation_ = Transformation::kNone;

  ChunkContiguity chunk_contiguity_ = ChunkContiguity::kNone;

  // Size of the per-thread scratch buffer. 0 means "no scratch buffer required"
  int64_t scratch_size_ = 0;
};

struct TransposePlanCacheKey {
  template <typename H>
  friend H AbslHashValue(H h, const TransposePlanCacheKey& key);

  size_t elem_size_in_bytes;
  absl::InlinedVector<int64_t, 4> dims;
  absl::InlinedVector<int64_t, 4> permutation;
  std::optional<absl::InlinedVector<int64_t, 4>> input_tiling;
  std::optional<absl::InlinedVector<int64_t, 4>> input_striding;
  std::optional<absl::InlinedVector<int64_t, 4>> output_tiling;
  TransposePlan::Transformation transformation;
  int num_threads;

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
