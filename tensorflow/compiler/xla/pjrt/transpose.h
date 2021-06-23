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
#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/pjrt/lru_cache.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class TransposePlan {
 public:
  // elem_size_in_bytes: size of each element in bytes.
  // dims: the input shape, in elements.
  // permutation: for each output dimension, gives the number of the
  //   corresponding input dimension. Must be a permutation of [0..dims.size())
  // input_strides_in_bytes: optional; the strides of the input array in
  //   bytes. (N.B. not elements). If omitted, the array is assumed to be in
  //   a dense major-to-minor layout.
  static StatusOr<std::unique_ptr<TransposePlan>> Create(
      size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
      absl::Span<int64_t const> permutation,
      absl::optional<absl::Span<int64_t const>> input_strides_in_bytes =
          absl::nullopt);

  TransposePlan();
  ~TransposePlan();

  // Executes the transposition.
  // `a` is the input array and `b` is the output array. The input and output
  // arrays must not overlap.
  // Currently there are no alignment requirements on either `a` or `b`. However
  // performance may be better if either or both are aligned.
  void Execute(const void* a, void* b) const;

  // Returns a human-readable description of the plan.
  std::string ToString() const;

  size_t ElemSizeInBytes() const { return elem_size_in_bytes_; }

  // Input and output size, in number of elements.
  int64_t NumElems() const;

  absl::Span<int64_t const> InputDims() const { return original_a_dims_; }
  absl::Span<int64_t const> OutputDims() const { return original_b_dims_; }

  absl::Span<int64_t const> InputStrides() const { return original_a_strides_; }
  absl::Span<int64_t const> OutputStrides() const {
    return original_b_strides_;
  }

  struct Node;

 protected:
  // Methods protected so they can be accessed by tests.

  // Removes any size-1 dimensions.
  static void RemoveTrivialDimensions(
      absl::InlinedVector<int64_t, 4>& a_dims,
      absl::InlinedVector<int64_t, 4>& permutation,
      absl::InlinedVector<int64_t, 4>& lda);

  // Collapses together dimensions that are adjacent both in `dims` and
  // `permutation`.
  static void CoalesceDimensions(absl::InlinedVector<int64_t, 4>& a_dims,
                                 absl::InlinedVector<int64_t, 4>& permutation,
                                 absl::InlinedVector<int64_t, 4>& lda);

 private:
  Node* BuildPlanNode(absl::Span<int64_t const> inverse_permutation, int i);

  // The signature of ExecuteTyped uses char* pointers because we perform
  // address calculations with strides in bytes; the strides need not be
  // multiples of the element size.
  template <typename T>
  void ExecuteTyped(const char* a, char* b) const;

  // Size of each element in bytes.
  int64_t elem_size_in_bytes_;

  // Number of elements in the input array.
  int64_t num_elems_;

  absl::InlinedVector<int64_t, 4> original_a_dims_;
  absl::InlinedVector<int64_t, 4> original_a_strides_;
  std::vector<int64_t> original_b_dims_;
  absl::InlinedVector<int64_t, 4> original_b_strides_;

  // Dimensions of the input array A.
  absl::InlinedVector<int64_t, 4> a_dims_;
  absl::InlinedVector<int64_t, 4> a_strides_;

  // Dimensions of the output array B.
  std::vector<int64_t> b_dims_;

  // Dimension permutation to apply to form B. For each dimension of B, what is
  // the corresponding dimension of A?
  absl::InlinedVector<int64_t, 4> permutation_;

  // Leading-dimension sizes (strides) of each dimension.
  absl::InlinedVector<int64_t, 4> lda_;
  absl::InlinedVector<int64_t, 4> ldb_;

  // Order to traverse dimensions, from slowest-varying to fastest-varying.
  // The integers are dimension numbers in A.
  std::vector<int> loop_order_;

  // Nodes of the plan, in no particular order. Holds ownership of the plan
  // nodes.
  // TODO(phawkins): pack nodes into a dense array to minimize the effects of
  // pointer jumping.
  std::vector<std::unique_ptr<Node>> nodes_;

  // Root node of the plan, i.e., pointing to the outermost loop in the loop
  // nest.
  Node* root_node_;

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
};

// An LRU cache for transpose plans. Not thread-safe.
struct TransposePlanCacheKey;

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key);

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
      absl::optional<absl::Span<int64_t const>> input_strides_in_bytes =
          absl::nullopt);

 private:
  LRUCache<TransposePlanCacheKey,
           StatusOr<std::shared_ptr<TransposePlan>>>::LRUList lru_list_;
  LRUCache<TransposePlanCacheKey, StatusOr<std::shared_ptr<TransposePlan>>>
      cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_H_
