/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/pjrt/lru_cache.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class TransposePlan {
 public:
  // elem_size_in_bytes: size of each element in bytes.
  // dims: the input shape, in elements.
  // permutation: for each output dimension, gives the number of the
  //   corresponding input dimension. Must be a permutation of [0..dims.size())
  // input_layout: either byte strides or an input tiling.
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
  // The input may have either a striding or a tiling but not both.
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

  static StatusOr<std::unique_ptr<TransposePlan>> Create(
      size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
      absl::Span<int64_t const> permutation,
      std::variant<Tiling, Striding> input_layout = Tiling{},
      Tiling output_tiling = Tiling{},
      Transformation transformation = Transformation::kNone,
      int num_threads = 1);

  TransposePlan();
  ~TransposePlan();

  // Executes the transposition.
  // `a` is the input array and `b` is the output array. The input and output
  // arrays must not overlap.
  // Currently there are no alignment requirements on either `a` or `b`. However
  // performance may be better if either or both are aligned.
  void Execute(const void* a, void* b,
               const std::function<void(std::function<void(void)>)>&
                   schedule_work = {}) const;

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

  // Removes any size-1 dimensions.
  static void RemoveTrivialDimensions(
      absl::InlinedVector<int64_t, 4>& a_dims,
      absl::InlinedVector<int64_t, 4>& permutation,
      absl::InlinedVector<int64_t, 4>& lda,
      absl::InlinedVector<int64_t, 4>& lda_tile,
      absl::InlinedVector<int64_t, 4>& a_tiling,
      absl::InlinedVector<int64_t, 4>& b_tiling);

  // Collapses together dimensions that are adjacent both in `dims` and
  // `permutation`.
  static void CoalesceDimensions(absl::InlinedVector<int64_t, 4>& a_dims,
                                 absl::InlinedVector<int64_t, 4>& permutation,
                                 absl::InlinedVector<int64_t, 4>& lda,
                                 absl::InlinedVector<int64_t, 4>& lda_tile,
                                 absl::InlinedVector<int64_t, 4>& a_tiling,
                                 absl::InlinedVector<int64_t, 4>& b_tiling);

 private:
  // Performs plan initialization that cannot fail.
  void Initialize();

  void BuildPlanNodes(absl::Span<int64_t const> inverse_permutation,
                      int thread_id, std::vector<Node>& output_nodes);

  std::vector<int> ChooseParallelizationStrategy(
      absl::Span<int64_t const> inverse_permutation);

  // The signature of ExecuteTyped uses char* pointers because we perform
  // address calculations with strides in bytes; the strides need not be
  // multiples of the element size.
  template <typename T, Transformation transformation>
  void ExecuteTyped(const char* a, char* b, absl::Span<Node const> nodes) const;

  // Number of threads requested.
  int num_threads_requested_ = 1;

  // Size of each element in bytes.
  int64_t elem_size_in_bytes_;

  // Number of elements in the input array.
  int64_t num_elems_;

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
  absl::InlinedVector<int64_t, 4> lda_;
  absl::InlinedVector<int64_t, 4> lda_tile_;
  absl::InlinedVector<int64_t, 4> ldb_;
  absl::InlinedVector<int64_t, 4> ldb_tile_;

  // Tile sizes in each dimension. Has size equal to the number of dimensions.
  // A 1 entry means that dimension is not tiled.
  absl::InlinedVector<int64_t, 4> a_tiling_;
  absl::InlinedVector<int64_t, 4> b_tiling_;
  bool a_is_tiled_;
  bool b_is_tiled_;

  // Order to traverse dimensions, from slowest-varying to fastest-varying.
  struct Loop {
    // The integers are dimension numbers in A.
    int dim_in_a;
    // If true, the loop iterates over the interior of a tile.
    bool tile_interior;
  };
  std::vector<Loop> loop_order_;
  std::vector<int> loop_parallelism_;

  // Root nodes of the plan, i.e., pointing to the outermost loops in the loop
  // nest. The outer vector is indexed on the thread ID.
  absl::InlinedVector<std::vector<Node>, 1> nodes_;

  // Are the innermost (stride-1) dimensions the same dimension? This determines
  // whether the inner kernel is a transpose or a memcpy.
  bool inner_kernel_is_memcpy_;

  // Size of the inner (microkernel) block size. This is the unit of work for
  // our vectorized kernels.
  int inner_block_elems_ = 1;
  // Size of the outer (macrokernel) block size. This is the unit of work for
  // cache blocking and need not be equal between input and output.
  int outer_block_elems_a_ = 4;
  int outer_block_elems_b_ = 4;

  // Transformations to apply to the input before transposition.
  // Currently the only supported transformation is EF57 conversion, which is
  // a pair-of-floats extended precision representation used on TPU. We
  // support fusing transformations with the transpose for two reasons:
  // (a) it makes sense to fuse cheap computations with a memory-bandwidth
  //     bound transformation, and
  // (b) it allows us to support non-trivial striding.
  Transformation transformation_;

  // Size of the per-thread scratch buffer. 0 means "no scratch buffer required"
  int64_t scratch_size_ = 0;
};

struct TransposePlanCacheKey {
  template <typename H>
  friend H AbslHashValue(H h, const TransposePlanCacheKey& key);

  size_t elem_size_in_bytes;
  absl::InlinedVector<int64_t, 4> dims;
  absl::InlinedVector<int64_t, 4> permutation;
  bool input_layout_is_tiling;
  absl::InlinedVector<int64_t, 4> input_layout;
  absl::InlinedVector<int64_t, 4> output_tiling;
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
  StatusOr<std::shared_ptr<TransposePlan>> GetOrCreate(
      size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
      absl::Span<int64_t const> permutation,
      std::variant<TransposePlan::Tiling, TransposePlan::Striding>
          input_layout = TransposePlan::Tiling{},
      TransposePlan::Tiling output_tiling = TransposePlan::Tiling{},
      TransposePlan::Transformation transformation =
          TransposePlan::Transformation::kNone,
      int num_threads = 1);

 private:
  LRUCache<TransposePlanCacheKey,
           StatusOr<std::shared_ptr<TransposePlan>>>::LRUList lru_list_;
  LRUCache<TransposePlanCacheKey, StatusOr<std::shared_ptr<TransposePlan>>>
      cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_H_
