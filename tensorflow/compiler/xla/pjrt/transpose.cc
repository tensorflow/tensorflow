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
// * we don't yet parallelize.
// * we could do a better job of vectorizing where the stride-1 dimensions are
//   small (e.g., inner dimensions of size [..., 3] are not uncommon in some
//   use cases.)
// * consider adding a TransposePlanCache.

#include "tensorflow/compiler/xla/pjrt/transpose.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/pjrt/transpose_kernels.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// A plan is a linked data structure that describes a loop nest.
// TODO(phawkins): consider shrinking Node so it fits in a cache line.
struct TransposePlan::Node {
  // The loop should iterate over the index space range(start, end, inc).
  // These fields are ignored by the macrokernel.
  int64_t start;
  int64_t end;
  int64_t inc;

  // Strides of this dimension in A and B.
  int64_t lda;
  int64_t ldb;

  // Next node in the plan. nullptr means this node represents the macrokernel.
  // TODO(phawkins): we may wish to avoid pointer jumping.
  Node* next;

  // Is this dimension the innermost dimension in either A or B, and hence may
  // have non-trivial blocking?
  bool is_inner_dim_in_a = false;
  bool is_inner_dim_in_b = false;
};

template <typename T, int inner_bs>
void MacroKernel(const char* __restrict a, int64_t lda, int outer_bs_a,
                 char* __restrict b, int64_t ldb, int outer_bs_b) {
  DVLOG(10) << "MacroKernel lda=" << lda << " ldb=" << ldb
            << " outer_bs_a=" << outer_bs_a << " outer_bs_b=" << outer_bs_b
            << " inner_bs=" << inner_bs;

  // TODO(phawkins): consider adding prefetching and streaming stores.
  for (int i = 0; i < outer_bs_a; ++i) {
    for (int j = 0; j < outer_bs_b; ++j) {
      TransposeMicroKernel<T, inner_bs>::Apply(
          a + inner_bs * j * lda + i * inner_bs * sizeof(T), lda,
          b + inner_bs * i * ldb + j * inner_bs * sizeof(T), ldb);
    }
  }
}

// Transpose() is a driver function that implements a multidimensional loop nest
// following by iterating over the linked Node data structure.
template <typename T, int inner_bs>
void Transpose(const char* __restrict a, int outer_bs_a, char* __restrict b,
               int outer_bs_b, TransposePlan::Node const* node) {
  DVLOG(10) << "Transpose " << outer_bs_a << " " << outer_bs_b;
  DCHECK_GT(outer_bs_a, 0);
  DCHECK_GT(outer_bs_b, 0);
  const int64_t start = node->start;
  const int64_t end = node->end;
  const int64_t stop = node->end - (node->inc - 1);
  const int64_t lda = node->lda;
  const int64_t ldb = node->ldb;
  const int64_t inc = node->inc;
  if (node->next->next == nullptr) {
    // This is the last loop in the nested loops. The next node is a sentinel
    // plan node that describes how to invoke the macrokernels.
    const int64_t lda_block = node->next->lda;
    const int64_t ldb_block = node->next->ldb;
    int64_t i;
    for (i = start; i < stop; i += inc) {
      MacroKernel<T, inner_bs>(a + i * lda, lda_block, outer_bs_a, b + i * ldb,
                               ldb_block, outer_bs_b);
    }
    // Handle trailing elements that didn't fit in a complete macrokernel.
    // Only the innermost dimensions have non-trivial outer_bs blocking.
    if (i < end) {
      DCHECK(node->is_inner_dim_in_a || node->is_inner_dim_in_b);
      if (node->is_inner_dim_in_a) {
        outer_bs_a = (end - i) / inner_bs;
        if (outer_bs_a > 0) {
          MacroKernel<T, inner_bs>(a + i * lda, lda_block, outer_bs_a,
                                   b + i * ldb, ldb_block, outer_bs_b);
          i += outer_bs_a * inner_bs;
        }
        // If there are still trailing elements left over that don't fit in the
        // inner block size, handle them via an unvectorized transpose.
        if (i < end) {
          MacroKernel<T, 1>(a + i * lda, lda_block, end - i, b + i * ldb,
                            ldb_block, outer_bs_b * inner_bs);
        }
      } else if (node->is_inner_dim_in_b) {
        outer_bs_b = (end - i) / inner_bs;
        if (outer_bs_b > 0) {
          MacroKernel<T, inner_bs>(a + i * lda, lda_block, outer_bs_a,
                                   b + i * ldb, ldb_block, outer_bs_b);
          i += outer_bs_b * inner_bs;
        }
        if (i < end) {
          MacroKernel<T, 1>(a + i * lda, lda_block, outer_bs_a * inner_bs,
                            b + i * ldb, ldb_block, end - i);
        }
      }
    }
  } else {
    // This is not the last loop in the nested loops. Recursively visit the
    // inner loops. Structurally this code is identical to the previous case,
    // but we call Transpose() recursively instead of MacroKernel().
    int64_t i;
    for (i = start; i < stop; i += inc) {
      Transpose<T, inner_bs>(a + i * lda, outer_bs_a, b + i * ldb, outer_bs_b,
                             node->next);
    }
    if (i < end) {
      DCHECK(node->is_inner_dim_in_a || node->is_inner_dim_in_b);
      if (node->is_inner_dim_in_a) {
        outer_bs_a = (end - i) / inner_bs;
        if (outer_bs_a > 0) {
          Transpose<T, inner_bs>(a + i * lda, outer_bs_a, b + i * ldb,
                                 outer_bs_b, node->next);
          i += outer_bs_a * inner_bs;
        }
        if (i < end) {
          Transpose<T, 1>(a + i * lda, end - i, b + i * ldb,
                          outer_bs_b * inner_bs, node->next);
        }
      } else if (node->is_inner_dim_in_b) {
        outer_bs_b = (end - i) / inner_bs;
        if (outer_bs_b > 0) {
          Transpose<T, inner_bs>(a + i * lda, outer_bs_a, b + i * ldb,
                                 outer_bs_b, node->next);
          i += outer_bs_b * inner_bs;
        }
        if (i < end) {
          Transpose<T, 1>(a + i * lda, outer_bs_a * inner_bs, b + i * ldb,
                          end - i, node->next);
        }
      }
    }
  }
}

template <typename T>
void TransposeConstStride1(const char* __restrict a, char* __restrict b,
                           TransposePlan::Node const* node) {
  const int64_t start = node->start;
  const int64_t end = node->end;
  const int64_t lda = node->lda;
  const int64_t ldb = node->ldb;
  if (node->next == nullptr) {
    DCHECK_EQ(lda, sizeof(T));
    DCHECK_EQ(ldb, sizeof(T));
    std::memcpy(b + start * sizeof(T), a + start * sizeof(T),
                (end - start) * sizeof(T));
  } else {
    DCHECK_EQ(node->inc, 1);
    int64_t i;
    for (i = start; i < end; ++i) {
      TransposeConstStride1<T>(a + i * lda, b + i * ldb, node->next);
    }
  }
}

template <typename T>
void TransposePlan::ExecuteTyped(
    const char* a, char* b, absl::Span<const Node* const> root_nodes) const {
  if (inner_kernel_is_memcpy_) {
    for (Node const* node : root_nodes) {
      TransposeConstStride1<T>(a, b, node);
    }
  } else {
    switch (inner_block_elems_) {
      case 1:
        for (Node const* node : root_nodes) {
          if (node->next) {
            Transpose<T, 1>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                            node);
          } else {
            MacroKernel<T, 1>(a, node->lda, outer_block_elems_a_, b, node->ldb,
                              outer_block_elems_b_);
          }
        }
        break;
      case 2:
        for (Node const* node : root_nodes) {
          if (node->next) {
            Transpose<T, 2>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                            node);
          } else {
            MacroKernel<T, 2>(a, node->lda, outer_block_elems_a_, b, node->ldb,
                              outer_block_elems_b_);
          }
        }
        break;
      case 4:
        for (Node const* node : root_nodes) {
          if (node->next) {
            Transpose<T, 4>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                            node);
          } else {
            MacroKernel<T, 4>(a, node->lda, outer_block_elems_a_, b, node->ldb,
                              outer_block_elems_b_);
          }
        }
        break;
      case 8:
        for (Node const* node : root_nodes) {
          if (node->next) {
            Transpose<T, 8>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                            node);
          } else {
            MacroKernel<T, 8>(a, node->lda, outer_block_elems_a_, b, node->ldb,
                              outer_block_elems_b_);
          }
        }
        break;
      case 16:
        for (Node const* node : root_nodes) {
          if (node->next) {
            Transpose<T, 16>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                             node);
          } else {
            MacroKernel<T, 16>(a, node->lda, outer_block_elems_a_, b, node->ldb,
                               outer_block_elems_b_);
          }
        }
        break;
      default:
        LOG(FATAL) << "Invalid inner_block_size " << inner_block_elems_;
    }
  }
}

struct uint128 {
  uint64_t lo;
  uint64_t hi;
};
static_assert(sizeof(uint128) == 16, "uint128 should be 16 bytes in size");

void TransposePlan::Execute(
    const void* a, void* b,
    const std::function<void(std::function<void(void)>)>& schedule_work) const {
  if (num_elems_ == 0) {
    return;
  }

  const char* ac = static_cast<const char*>(a);
  char* bc = static_cast<char*>(b);
  DCHECK((ac + elem_size_in_bytes_ * num_elems_ <= b ||
          bc + elem_size_in_bytes_ * num_elems_ <= a));

  auto execute_by_type = [&](absl::Span<const Node* const> nodes) {
    switch (elem_size_in_bytes_) {
      case 1:
        ExecuteTyped<uint8_t>(ac, bc, nodes);
        break;
      case 2:
        ExecuteTyped<uint16_t>(ac, bc, nodes);
        break;
      case 4:
        ExecuteTyped<uint32_t>(ac, bc, nodes);
        break;
      case 8:
        ExecuteTyped<uint64_t>(ac, bc, nodes);
        break;
      case 16:
        ExecuteTyped<uint128>(ac, bc, nodes);
        break;
      default:
        LOG(FATAL) << "Unimplemented element size " << elem_size_in_bytes_;
    }
  };

  if (!schedule_work || root_nodes_.size() <= 1) {
    for (const auto& nodes : root_nodes_) {
      execute_by_type(nodes);
    }
  } else {
    absl::BlockingCounter counter(root_nodes_.size());
    for (const auto& nodes : root_nodes_) {
      absl::Span<const Node* const> nodes_span = nodes;
      schedule_work([&, nodes_span]() {
        execute_by_type(nodes_span);
        counter.DecrementCount();
      });
    }
    counter.Wait();
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

void TransposePlan::RemoveTrivialDimensions(
    absl::InlinedVector<int64_t, 4>& a_dims,
    absl::InlinedVector<int64_t, 4>& permutation,
    absl::InlinedVector<int64_t, 4>& lda,
    absl::InlinedVector<int64_t, 4>& lda_tile,
    absl::InlinedVector<int64_t, 4>& a_tiling,
    absl::InlinedVector<int64_t, 4>& b_tiling) {
  int ndim = a_dims.size();
  // How many positions has the i-th dimension of 'a' been moved to the left?
  // -1 if the dimension is to be removed.
  std::vector<int> shift(ndim);
  absl::InlinedVector<int64_t, 4> updated_a_dims;
  absl::InlinedVector<int64_t, 4> updated_lda;
  absl::InlinedVector<int64_t, 4> updated_lda_tile;
  absl::InlinedVector<int64_t, 4> updated_a_tiling;
  updated_a_dims.reserve(ndim);
  updated_lda.reserve(ndim);
  updated_lda_tile.reserve(ndim);
  updated_a_tiling.reserve(ndim);
  std::vector<int64_t> inv_permutation = InversePermutation(permutation);
  for (int a_dim = 0; a_dim < ndim; ++a_dim) {
    int b_dim = inv_permutation[a_dim];
    // A dimension is trivial if it has size 1 and is not tiled.
    if (a_dims[a_dim] == 1 && a_tiling[a_dim] == 1 && b_tiling[b_dim] == 1) {
      shift[a_dim] = -1;
    } else {
      updated_a_dims.push_back(a_dims[a_dim]);
      updated_lda.push_back(lda[a_dim]);
      updated_lda_tile.push_back(lda_tile[a_dim]);
      updated_a_tiling.push_back(a_tiling[a_dim]);
      shift[a_dim] = a_dim + 1 - updated_a_dims.size();
    }
  }

  // Updates the permutation and tiling of b.
  absl::InlinedVector<int64_t, 4> updated_permutation;
  absl::InlinedVector<int64_t, 4> updated_b_tiling;
  updated_permutation.reserve(updated_a_dims.size());
  updated_b_tiling.reserve(updated_a_dims.size());
  for (int b_dim = 0; b_dim < ndim; ++b_dim) {
    int a_dim = permutation[b_dim];
    if (shift[a_dim] >= 0) {
      updated_permutation.push_back(a_dim - shift[a_dim]);
      updated_b_tiling.push_back(b_tiling[b_dim]);
    }
  }

  DCHECK(IsPermutation(updated_permutation));
  a_dims = std::move(updated_a_dims);
  permutation = std::move(updated_permutation);
  lda = std::move(updated_lda);
  lda_tile = std::move(updated_lda_tile);
  a_tiling = std::move(updated_a_tiling);
  b_tiling = std::move(updated_b_tiling);
}

void TransposePlan::CoalesceDimensions(
    absl::InlinedVector<int64_t, 4>& a_dims,
    absl::InlinedVector<int64_t, 4>& permutation,
    absl::InlinedVector<int64_t, 4>& lda,
    absl::InlinedVector<int64_t, 4>& lda_tile,
    absl::InlinedVector<int64_t, 4>& a_tiling,
    absl::InlinedVector<int64_t, 4>& b_tiling) {
  int ndim = a_dims.size();
  // How many positions has the i-th dimension of 'a' been moved to the left?
  // -1 if the dimension is to be removed.
  std::vector<int> shift(ndim, 0);
  absl::InlinedVector<int64_t, 4> updated_a_dims;
  absl::InlinedVector<int64_t, 4> updated_lda;
  absl::InlinedVector<int64_t, 4> updated_lda_tile;
  absl::InlinedVector<int64_t, 4> updated_a_tiling;
  updated_a_dims.reserve(ndim);
  updated_lda.reserve(ndim);
  updated_lda_tile.reserve(ndim);
  updated_a_tiling.reserve(ndim);
  std::vector<int64_t> inv_permutation = InversePermutation(permutation);
  for (int a_dim = 0; a_dim < ndim; ++a_dim) {
    // We can coalesce two dimensions if they appear consecutively
    // in both the input dimensions and the output dimensions, and the stride
    // of the outer dimension is the usual multiple of the inner dimension.
    if (a_dim > 0 && inv_permutation[a_dim - 1] + 1 == inv_permutation[a_dim] &&
        lda[a_dim - 1] == lda[a_dim] * a_dims[a_dim] &&
        a_tiling[a_dim - 1] == 1 && a_tiling[a_dim] == 1 &&
        b_tiling[inv_permutation[a_dim]] == 1 &&
        b_tiling[inv_permutation[a_dim - 1]] == 1) {
      updated_a_dims.back() *= a_dims[a_dim];
      updated_lda.back() = lda[a_dim];
      shift[a_dim] = -1;
    } else {
      updated_a_dims.push_back(a_dims[a_dim]);
      updated_lda.push_back(lda[a_dim]);
      updated_lda_tile.push_back(lda_tile[a_dim]);
      updated_a_tiling.push_back(a_tiling[a_dim]);
      shift[a_dim] = a_dim + 1 - updated_a_dims.size();
    }
  }

  // Updates the permutation.
  absl::InlinedVector<int64_t, 4> updated_permutation;
  absl::InlinedVector<int64_t, 4> updated_b_tiling;
  updated_permutation.reserve(updated_a_dims.size());
  updated_b_tiling.reserve(updated_a_dims.size());
  for (int b_dim = 0; b_dim < ndim; ++b_dim) {
    int a_dim = permutation[b_dim];
    if (shift[a_dim] >= 0) {
      updated_permutation.push_back(a_dim - shift[a_dim]);
      updated_b_tiling.push_back(b_tiling[b_dim]);
    }
  }
  DCHECK(IsPermutation(updated_permutation));
  a_dims = std::move(updated_a_dims);
  permutation = std::move(updated_permutation);
  lda = std::move(updated_lda);
  lda_tile = std::move(updated_lda_tile);
  a_tiling = std::move(updated_a_tiling);
  b_tiling = std::move(updated_b_tiling);
}

int64_t TransposePlan::InputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims_.size(); ++i) {
    size *= RoundUpToNearest(a_dims_[i], a_tiling_[i]);
  }
  return size;
}

int64_t TransposePlan::OutputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims_.size(); ++i) {
    size *= RoundUpToNearest(a_dims_[permutation_[i]], b_tiling_[i]);
  }
  return size;
}

// Parses and validates a tiling specification, and populates `tiling`.
static Status ParseTilingSpecification(int ndim,
                                       absl::Span<int64_t const> tiling_spec,
                                       absl::InlinedVector<int64, 4>& tiling) {
  tiling.resize(ndim, 1);
  if (tiling_spec.size() > ndim) {
    return InvalidArgument(
        "Tiling (%s) must have at as many dimensions as the array (%d)",
        absl::StrJoin(tiling_spec, ","), ndim);
  }
  if (absl::c_find_if(tiling_spec, [](int64_t d) { return d < 1; }) !=
      tiling_spec.end()) {
    return InvalidArgument("Tiling sizes (%s) must be >= 1",
                           absl::StrJoin(tiling_spec, ","));
  }
  int offset = ndim;
  offset -= tiling_spec.size();
  absl::c_copy(tiling_spec, tiling.begin() + offset);
  return Status::OK();
}

// Recursive helper function that builds a plan.
void TransposePlan::BuildPlanNodes(
    absl::Span<int64_t const> inverse_permutation, int thread_id,
    absl::InlinedVector<TransposePlan::Node*, 1>& output_nodes) {
  const int ndim = a_dims_.size();
  DCHECK_GT(ndim, 0);
  const int pos_stride1a = ndim - 1;
  const int pos_stride1b_in_a = permutation_.back();
  const int pos_stride1a_in_b = inverse_permutation[pos_stride1a];

  struct Agendum {
    TransposePlan::Node* node;
    // For which dimensions of a does `node` visit the partial trailing tile in
    // an inner loop?
    absl::InlinedVector<bool, 4> partial_tiles;
  };
  std::vector<Agendum> current_agenda;
  // Builds a sentinel node that says that we should invoke the kernel.
  Node* node = nullptr;
  if (!inner_kernel_is_memcpy_) {
    nodes_.push_back(std::make_unique<Node>());
    node = nodes_.back().get();
    node->next = nullptr;
    node->start = node->end = node->inc = -1;
    node->lda = a_tiling_[pos_stride1b_in_a] > 1 ? lda_tile_[pos_stride1b_in_a]
                                                 : lda_[pos_stride1b_in_a];
    node->ldb = b_tiling_[pos_stride1a_in_b] > 1 ? ldb_tile_[pos_stride1a_in_b]
                                                 : ldb_[pos_stride1a_in_b];
  }
  current_agenda = {Agendum{node, absl::InlinedVector<bool, 4>(ndim, false)}};
  auto loop_has_trivial_iteration_space = [](const Node* node) {
    return node->start == 0 && node->start + node->inc == node->end;
  };

  // Number of tasks to be assigned to the current loop
  int num_tasks_at_loop =
      absl::c_accumulate(loop_parallelism_, int{1}, std::multiplies<int>());
  // ID of the current task within the tasks at the current loop.
  int task_id_at_loop = thread_id;

  std::vector<Agendum> new_agenda;
  for (int loop_id = static_cast<int>(loop_order_.size()) - 1; loop_id >= 0;
       --loop_id) {
    const Loop& loop = loop_order_[loop_id];
    int a_dim = loop.dim_in_a;
    int b_dim = inverse_permutation[a_dim];
    DCHECK(a_tiling_[a_dim] == 1 || b_tiling_[b_dim] == 1 ||
           a_tiling_[a_dim] == b_tiling_[b_dim]);
    int64_t tile_size = std::max(a_tiling_[a_dim], b_tiling_[b_dim]);

    // Compute the number of tasks for the next loop iteration.
    num_tasks_at_loop /= loop_parallelism_[loop_id];

    new_agenda.clear();

    DCHECK_GE(tile_size, 1);
    absl::InlinedVector<TransposePlan::Node*, 1> partial_tile_nodes;
    // If the dimension is tiled, generate two nested loops, one for inside the
    // tile, one for outside.
    if (loop.tile_interior) {
      bool has_partial_tile = (a_dims_[a_dim] % tile_size != 0);

      auto make_tile_node = [&](const Agendum& agendum,
                                bool partial) -> absl::optional<Agendum> {
        nodes_.push_back(std::make_unique<Node>());
        Node* node = nodes_.back().get();
        node->lda = a_tiling_[a_dim] > 1 ? lda_tile_[a_dim] : lda_[a_dim];
        node->ldb = b_tiling_[b_dim] > 1 ? ldb_tile_[b_dim] : ldb_[b_dim];
        node->inc = 1;
        if (a_dim == pos_stride1a) {
          node->inc = inner_block_elems_ * outer_block_elems_a_;
          node->is_inner_dim_in_a = true;
        } else if (a_dim == pos_stride1b_in_a) {
          node->inc = inner_block_elems_ * outer_block_elems_b_;
          node->is_inner_dim_in_b = true;
        }
        node->next = agendum.node;
        int task_id = task_id_at_loop / num_tasks_at_loop;
        if (partial) {
          // Only the last task handles the trailing tile.
          if (task_id != loop_parallelism_[loop_id] - 1) {
            return absl::nullopt;
          }
          node->start = 0;
          node->end = a_dims_[a_dim] % tile_size;
        } else {
          int64_t num_iterations = CeilOfRatio(tile_size, node->inc);
          int64_t num_iterations_per_task =
              CeilOfRatio<int64_t>(num_iterations, loop_parallelism_[loop_id]);
          node->start = std::min(tile_size,
                                 task_id * num_iterations_per_task * node->inc);
          node->end = std::min(
              tile_size, (task_id + 1) * num_iterations_per_task * node->inc);
        }

        DCHECK(partial || node->start + node->inc <= node->end)
            << node->start << " " << node->inc << " " << node->end;
        // If this loop has a trivial iteration space, drop it.
        if (node->start >= node->end) {
          return absl::nullopt;
        }
        if (loop_has_trivial_iteration_space(node) && node->next) {
          nodes_.pop_back();
          node = agendum.node;
        }
        Agendum new_agendum;
        new_agendum.node = node;
        new_agendum.partial_tiles = agendum.partial_tiles;
        new_agendum.partial_tiles[a_dim] = partial;
        return new_agendum;
      };

      for (const Agendum& agendum : current_agenda) {
        // If the dimension contains a complete tile, add a loop over the entire
        // tile.
        if (a_dims_[a_dim] >= tile_size) {
          auto new_agendum = make_tile_node(agendum, /*partial=*/false);
          if (new_agendum) {
            new_agenda.push_back(std::move(*new_agendum));
          }
        }

        // If the dimension size is not exactly divisible by the tile size,
        // then add an additional loop that handles just the trailing partial
        // tile.
        if (has_partial_tile) {
          auto new_agendum = make_tile_node(agendum, /*partial=*/true);
          if (new_agendum) {
            new_agenda.push_back(std::move(*new_agendum));
          }
        }
      }
    } else {
      auto make_node = [&](const Agendum& agendum,
                           bool partial) -> absl::optional<Agendum> {
        nodes_.push_back(std::make_unique<Node>());
        Node* node = nodes_.back().get();
        node->lda = lda_[a_dim] * tile_size / a_tiling_[a_dim];
        node->ldb = ldb_[b_dim] * tile_size / b_tiling_[b_dim];
        node->inc = 1;
        if (tile_size == 1 && a_dim == ndim - 1) {
          node->inc = inner_block_elems_ * outer_block_elems_a_;
          node->is_inner_dim_in_a = true;
        } else if (tile_size == 1 && a_dim == pos_stride1b_in_a) {
          node->inc = inner_block_elems_ * outer_block_elems_b_;
          node->is_inner_dim_in_b = true;
        }
        node->next = agendum.node;
        int task_id = task_id_at_loop / num_tasks_at_loop;
        int64_t num_complete_tiles = a_dims_[a_dim] / tile_size;
        if (partial) {
          // For trailing partial tiles, we only need visit the single entry at
          // the end.
          DCHECK_NE(a_dims_[a_dim] % tile_size, 0);
          // Only the last task handles the trailing tile.
          if (task_id != loop_parallelism_[loop_id] - 1) {
            return absl::nullopt;
          }
          node->start = num_complete_tiles;
          node->end = num_complete_tiles + 1;
        } else {
          // Evenly divide the loop iterations amongst the threads.
          int64_t num_iterations = CeilOfRatio(num_complete_tiles, node->inc);
          int64_t num_iterations_per_task =
              CeilOfRatio<int64_t>(num_iterations, loop_parallelism_[loop_id]);
          node->start = std::min(num_complete_tiles,
                                 task_id * num_iterations_per_task * node->inc);
          node->end =
              std::min(num_complete_tiles,
                       (task_id + 1) * num_iterations_per_task * node->inc);
        }
        // If this loop has a trivial iteration space, drop it.
        if (node->start >= node->end) {
          return absl::nullopt;
        }
        if (loop_has_trivial_iteration_space(node) && node->next) {
          nodes_.pop_back();
          node = agendum.node;
        }
        Agendum new_agendum;
        new_agendum.node = node;
        new_agendum.partial_tiles = agendum.partial_tiles;
        return new_agendum;
      };
      for (const Agendum& agendum : current_agenda) {
        if (tile_size > 1 && agendum.partial_tiles[a_dim]) {
          if (a_dims_[a_dim] / tile_size == 0) {
            DCHECK_EQ(loop_parallelism_[loop_id], 1);
            new_agenda.push_back(agendum);
          } else {
            auto new_agendum = make_node(agendum, /*partial=*/true);
            if (new_agendum) {
              new_agenda.push_back(std::move(*new_agendum));
            }
          }
        } else {
          if (tile_size > 1 && a_dims_[a_dim] == tile_size) {
            DCHECK_EQ(loop_parallelism_[loop_id], 1);
            new_agenda.push_back(agendum);
          } else {
            auto new_agendum = make_node(agendum, /*partial=*/false);
            if (new_agendum) {
              new_agenda.push_back(std::move(*new_agendum));
            }
          }
        }
      }
    }
    std::swap(current_agenda, new_agenda);

    task_id_at_loop = task_id_at_loop % num_tasks_at_loop;
  }
  DCHECK_EQ(num_tasks_at_loop, 1);
  output_nodes.reserve(current_agenda.size());
  for (const Agendum& agendum : current_agenda) {
    output_nodes.push_back(agendum.node);
  }
}

StatusOr<std::unique_ptr<TransposePlan>> TransposePlan::Create(
    size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
    absl::Span<int64_t const> permutation,
    absl::variant<Tiling, Striding> input_layout, Tiling output_tiling,
    int num_threads) {
  auto is_negative = [](int d) { return d < 0; };
  if (absl::c_find_if(dims, is_negative) != dims.end()) {
    return InvalidArgument("dims must be non-negative, got %s",
                           absl::StrJoin(dims, ","));
  }
  if (permutation.size() != dims.size()) {
    return InvalidArgument(
        "dims and permutation must have equal sizes, got %d and %d",
        dims.size(), permutation.size());
  }
  if (!IsPermutation(permutation)) {
    return InvalidArgument("permutation argument is not valid, got: %s",
                           absl::StrJoin(permutation, ","));
  }
  if (num_threads < 1) {
    return InvalidArgument("num_threads argument must be >= 1, got: %d",
                           num_threads);
  }

  int ndim = dims.size();

  auto plan = std::make_unique<TransposePlan>();
  plan->num_threads_requested_ = num_threads;
  plan->elem_size_in_bytes_ = elem_size_in_bytes;
  switch (elem_size_in_bytes) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
      break;
    default:
      return InvalidArgument("Unsupported elem_size_in_bytes=%d",
                             elem_size_in_bytes);
  }
  plan->num_elems_ = std::accumulate(dims.begin(), dims.end(), int64_t{1},
                                     std::multiplies<int64_t>());
  plan->original_a_dims_.resize(ndim);
  absl::c_copy(dims, plan->original_a_dims_.begin());
  plan->original_b_dims_ = Permute(dims, permutation);

  TF_RETURN_IF_ERROR(
      ParseTilingSpecification(ndim, output_tiling.tiling, plan->b_tiling_));

  // Handles strides.
  if (absl::holds_alternative<Striding>(input_layout)) {
    absl::Span<int64_t const> input_strides_in_bytes =
        absl::get<Striding>(input_layout).strides_in_bytes;
    if (input_strides_in_bytes.size() != dims.size()) {
      return InvalidArgument(
          "dims and input_strides_in_bytes must have equal sizes, got %d "
          "and %d",
          dims.size(), input_strides_in_bytes.size());
    }
    if (absl::c_find_if(input_strides_in_bytes, is_negative) !=
        input_strides_in_bytes.end()) {
      return InvalidArgument(
          "input_strides_in_bytes must be non-negative, got %s",
          absl::StrJoin(dims, ","));
    }
    plan->original_a_strides_.resize(ndim);
    absl::c_copy(input_strides_in_bytes, plan->original_a_strides_.begin());
    // Sort the dimensions from slowest-varying (largest strides) to
    // fastest-varying (smallest strides).
    std::vector<int64_t> dim_order(ndim);
    absl::c_iota(dim_order, 0);
    absl::c_stable_sort(dim_order, [&](int i, int j) {
      int64_t stride_i = input_strides_in_bytes.at(i);
      int64_t stride_j = input_strides_in_bytes.at(j);
      // If there is a dimension with size equal to the element size, sort it
      // last. This ensures that we place any stride-1 dimension last.
      if (stride_i != elem_size_in_bytes && stride_j == elem_size_in_bytes) {
        return true;
      }
      return stride_i > stride_j;
    });
    // dim_order maps new input dim -> old input dim, we need its inverse to
    // compute the new permutation.
    auto inv_dim_order = InversePermutation(dim_order);
    plan->lda_.reserve(ndim);
    plan->a_dims_.reserve(ndim);
    plan->permutation_.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      plan->lda_.push_back(input_strides_in_bytes.at(dim_order[i]));
      plan->a_dims_.push_back(dims[dim_order[i]]);
      plan->permutation_.push_back(inv_dim_order[permutation[i]]);
    }
    plan->lda_tile_.resize(ndim, 1);
    plan->a_tiling_.resize(ndim, 1);
  } else {
    TF_RETURN_IF_ERROR(ParseTilingSpecification(
        ndim, absl::get<Tiling>(input_layout).tiling, plan->a_tiling_));

    plan->a_dims_ = plan->original_a_dims_;
    plan->permutation_.resize(ndim);
    absl::c_copy(permutation, plan->permutation_.begin());
    ComputeStrides(plan->elem_size_in_bytes_, plan->a_dims_, plan->a_tiling_,
                   plan->lda_, plan->lda_tile_);
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

  plan->Initialize();
  VLOG(5) << plan->ToString();
  return plan;
}

void TransposePlan::Initialize() {
  if (num_elems_ == 0) {
    return;
  }
  RemoveTrivialDimensions(a_dims_, permutation_, lda_, lda_tile_, a_tiling_,
                          b_tiling_);
  CoalesceDimensions(a_dims_, permutation_, lda_, lda_tile_, a_tiling_,
                     b_tiling_);

  // permutation maps dimensions of b to a
  // inverse_permutation maps dimensions of a to b
  std::vector<int64_t> inverse_permutation = InversePermutation(permutation_);

  int ndim = a_dims_.size();

  int64_t stride_pos1a =
      lda_.empty()
          ? -1
          : (a_tiling_[ndim - 1] > 1 ? lda_tile_[ndim - 1] : lda_[ndim - 1]);
  // We don't accept arbitrary stridings for B, so we know B always has a stride
  // 1 dimension innermost.

  // If the plan is 0-dimensional, or the innermost dimension of A is not of
  // stride 1, adds a trivial size 1 dimension. The transpose kernels rely on
  // the presence of a stride-1 innermost dimension in the input.
  if (lda_.empty() || stride_pos1a != elem_size_in_bytes_) {
    int dim = static_cast<int>(a_dims_.size());
    permutation_.push_back(dim);
    inverse_permutation.push_back(dim);
    a_dims_.push_back(1);
    lda_.push_back(elem_size_in_bytes_);
    lda_tile_.push_back(1);
    a_tiling_.push_back(1);
    b_tiling_.push_back(1);
    ++ndim;
  }
  const int pos_stride1a = ndim - 1;
  inner_kernel_is_memcpy_ = (permutation_[ndim - 1] == pos_stride1a);

  loop_order_.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    loop_order_.push_back(Loop{i, /*tile_interior=*/false});
    if (a_tiling_[i] != 1 || b_tiling_[inverse_permutation[i]] != 1) {
      loop_order_.push_back(Loop{i, /*tile_interior=*/true});
    }
  }
  // Loop order heuristic: try to make loops with small strides innermost.
  auto cost = [&](const Loop& l) -> double {
    if (inner_kernel_is_memcpy_ && l.dim_in_a == pos_stride1a) {
      return l.tile_interior ? -2 : -1;
    }
    int64_t a_stride = (l.tile_interior && a_is_tiled_) ? lda_tile_[l.dim_in_a]
                                                        : lda_[l.dim_in_a];
    int b_dim = inverse_permutation[l.dim_in_a];
    int64_t b_stride =
        (l.tile_interior && b_is_tiled_) ? lda_tile_[b_dim] : lda_[b_dim];
    // Add a small penalty to the input strides: given the choice between
    // consecutive writes and consecutive reads, we would prefer consecutive
    // writes.
    double penalty = 1.01;
    return a_stride * penalty + b_stride;
  };
  absl::c_stable_sort(loop_order_, [&](const Loop& a, const Loop& b) {
    return cost(a) > cost(b);
  });

  b_dims_ = Permute(a_dims_, permutation_);
  ComputeStrides(elem_size_in_bytes_, b_dims_, b_tiling_, ldb_, ldb_tile_);

  if (inner_kernel_is_memcpy_) {
    // The stride-1 loop must be innermost.
    CHECK_EQ(loop_order_.back().dim_in_a, ndim - 1);
  } else {
    switch (elem_size_in_bytes_) {
      case 1:
        inner_block_elems_ = 16;
        break;
      case 2:
        inner_block_elems_ = 8;
        break;
      case 4:
        inner_block_elems_ = 8;
        break;
      case 8:
        inner_block_elems_ = 4;
        break;
      case 16:
        inner_block_elems_ = 4;
        break;
      default:
        LOG(FATAL) << "Unreachable: element size " << elem_size_in_bytes_;
    }
  }

  // Bound the block sizes so they are smaller than the stride-1 dimension size.
  int64_t a_stride1_size = std::max(
      a_tiling_[pos_stride1a], b_tiling_[inverse_permutation[pos_stride1a]]);
  if (a_stride1_size == 1) {
    a_stride1_size = a_dims_[pos_stride1a];
  }
  int64_t b_stride1_size =
      std::max(a_tiling_[permutation_.back()], b_tiling_.back());
  if (b_stride1_size == 1) {
    b_stride1_size = b_dims_.back();
  }

  if (inner_kernel_is_memcpy_) {
    inner_block_elems_ = -1;
    outer_block_elems_a_ = -1;
    outer_block_elems_b_ = -1;
  } else {
    while (inner_block_elems_ > std::min(a_stride1_size, b_stride1_size)) {
      inner_block_elems_ /= 2;
      outer_block_elems_a_ *= 2;
      outer_block_elems_b_ *= 2;
    }
    while (outer_block_elems_a_ > 1 &&
           inner_block_elems_ * outer_block_elems_a_ > a_stride1_size) {
      outer_block_elems_a_ /= 2;
    }
    while (outer_block_elems_b_ > 1 &&
           inner_block_elems_ * outer_block_elems_b_ > b_stride1_size) {
      outer_block_elems_b_ /= 2;
    }
  }

  loop_parallelism_ = ChooseParallelizationStrategy(inverse_permutation);
  int num_threads =
      absl::c_accumulate(loop_parallelism_, int{1}, std::multiplies<int>());
  root_nodes_.resize(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    BuildPlanNodes(inverse_permutation, thread_id, root_nodes_[thread_id]);
  }
}

std::vector<int> TransposePlan::ChooseParallelizationStrategy(
    absl::Span<int64_t const> inverse_permutation) {
  std::vector<int> parallelism;
  int available_parallelism = num_threads_requested_;
  parallelism.reserve(loop_order_.size());

  int ndim = permutation_.size();
  const int pos_stride1a = ndim - 1;
  const int pos_stride1b_in_a = permutation_.back();
  // Compute the number of iterations in `loop`.
  auto loop_iterations = [&](const Loop& loop) {
    int a_dim = loop.dim_in_a;
    int b_dim = inverse_permutation[a_dim];
    int64_t tile_size = std::max(a_tiling_[a_dim], b_tiling_[b_dim]);
    int64_t size = loop.tile_interior
                       ? tile_size
                       : (CeilOfRatio(a_dims_[loop.dim_in_a], tile_size));
    if (!inner_kernel_is_memcpy_ && (loop.tile_interior || tile_size == 1)) {
      if (loop.dim_in_a == pos_stride1a) {
        size = CeilOfRatio<int64_t>(size,
                                    inner_block_elems_ * outer_block_elems_a_);
      } else if (loop.dim_in_a == pos_stride1b_in_a) {
        size = CeilOfRatio<int64_t>(size,
                                    inner_block_elems_ * outer_block_elems_b_);
      }
    }
    return size;
  };

  // Estimate the number of bytes each iteration of each loop processes.
  absl::InlinedVector<int64_t, 4> work_in_bytes(loop_order_.size());
  int64_t acc = elem_size_in_bytes_;
  if (!inner_kernel_is_memcpy_) {
    acc *= inner_block_elems_ * inner_block_elems_ * outer_block_elems_a_ *
           outer_block_elems_b_;
  }
  auto work_it = work_in_bytes.rbegin();
  for (auto it = loop_order_.rbegin(); it != loop_order_.rend(); ++it) {
    *work_it++ = acc;
    acc *= loop_iterations(*it);
  }
  VLOG(7) << "Per-loop iteration work in bytes: "
          << absl::StrJoin(work_in_bytes, ",");

  // Heuristic that attempts to parallelize the outermost loops, down to a
  // minimum per-thread number of bytes processed.
  for (size_t i = 0; i < loop_order_.size(); ++i) {
    const Loop& loop = loop_order_[i];
    CHECK_GE(available_parallelism, 1);
    int64_t iterations = loop_iterations(loop);
    int kMinBytesPerThread = inner_kernel_is_memcpy_ ? (1 << 20) : (1 << 17);
    int64_t min_iterations_per_thread =
        CeilOfRatio<int64_t>(kMinBytesPerThread, work_in_bytes[i]);
    int64_t parallel_work = CeilOfRatio(iterations, min_iterations_per_thread);

    VLOG(8) << "iterations=" << iterations << " parallel_work=" << parallel_work
            << " available_parallelism=" << available_parallelism;
    if (parallel_work >= available_parallelism) {
      parallelism.push_back(available_parallelism);
      available_parallelism = 1;
    } else {
      parallelism.push_back(parallel_work);
      available_parallelism /= parallel_work;
    }
  }
  return parallelism;
}

static void PrintPlan(TransposePlan::Node const* node, int indent,
                      std::string* out) {
  std::string indent_str(indent, ' ');
  absl::StrAppendFormat(out, "%sNode(start=%d,end=%d,inc=%d,lda=%d,ldb=%d)\n",
                        indent_str, node->start, node->end, node->inc,
                        node->lda, node->ldb);
  if (node->next) {
    PrintPlan(node->next, indent, out);
  }
}

std::string TransposePlan::ToString() const {
  std::string nodes = absl::StrJoin(
      root_nodes_, "\n",
      [](std::string* out, const absl::InlinedVector<Node*, 1>& nodes) {
        absl::StrAppend(
            out, "thread:\n",
            absl::StrJoin(nodes, "\n", [](std::string* out, Node const* node) {
              absl::StrAppend(out, "  root:\n");
              PrintPlan(node, /*indent=*/4, out);
            }));
      });
  auto format_loop_order = [](std::string* out, const Loop& loop) {
    return absl::StrAppend(out, loop.dim_in_a,
                           loop.tile_interior ? "[tile]" : "");
  };
  return absl::StrFormat(
      "a_dims=%s b_dims=%s permutation=%s a_tiling=%s b_tiling=%s "
      "lda=%s lda_tile=%s ldb=%s ldb_tile=%s loop_order=%s loop_parallelism=%s "
      "outer_bs=[%d,%d] inner_bs=%d\n"
      "nodes:\n%s",
      absl::StrJoin(a_dims_, ","),
      absl::StrJoin(Permute(a_dims_, permutation_), ","),
      absl::StrJoin(permutation_, ","), absl::StrJoin(a_tiling_, ","),
      absl::StrJoin(b_tiling_, ","), absl::StrJoin(lda_, ","),
      absl::StrJoin(lda_tile_, ","), absl::StrJoin(ldb_, ","),
      absl::StrJoin(ldb_tile_, ","),
      absl::StrJoin(loop_order_, ",", format_loop_order),
      absl::StrJoin(loop_parallelism_, ","), outer_block_elems_a_,
      outer_block_elems_b_, inner_block_elems_, nodes);
}

struct TransposePlanCacheKey {
  size_t elem_size_in_bytes;
  absl::InlinedVector<int64_t, 4> dims;
  absl::InlinedVector<int64_t, 4> permutation;
  bool input_layout_is_tiling;
  absl::InlinedVector<int64_t, 4> input_layout;
  absl::InlinedVector<int64_t, 4> output_tiling;
  int num_threads;

  bool operator==(const TransposePlanCacheKey& other) const;
};

bool TransposePlanCacheKey::operator==(
    const TransposePlanCacheKey& other) const {
  return elem_size_in_bytes == other.elem_size_in_bytes && dims == other.dims &&
         permutation == other.permutation &&
         input_layout_is_tiling == other.input_layout_is_tiling &&
         input_layout == other.input_layout &&
         output_tiling == other.output_tiling &&
         num_threads == other.num_threads;
}

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key) {
  h = H::combine(std::move(h), key.elem_size_in_bytes,
                 key.input_layout_is_tiling, key.num_threads);
  h = H::combine_contiguous(std::move(h), key.dims.data(), key.dims.size());
  h = H::combine_contiguous(std::move(h), key.permutation.data(),
                            key.permutation.size());
  h = H::combine_contiguous(std::move(h), key.input_layout.data(),
                            key.input_layout.size());
  h = H::combine_contiguous(std::move(h), key.output_tiling.data(),
                            key.output_tiling.size());
  return h;
}

TransposePlanCache::TransposePlanCache(int capacity)
    : lru_list_(capacity), cache_(&lru_list_) {}

TransposePlanCache::~TransposePlanCache() = default;

StatusOr<std::shared_ptr<TransposePlan>> TransposePlanCache::GetOrCreate(
    size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
    absl::Span<int64_t const> permutation,
    absl::variant<TransposePlan::Tiling, TransposePlan::Striding> input_layout,
    TransposePlan::Tiling output_tiling, int num_threads) {
  TransposePlanCacheKey key;
  key.elem_size_in_bytes = elem_size_in_bytes;
  key.dims.resize(dims.size());
  absl::c_copy(dims, key.dims.begin());
  key.permutation.resize(permutation.size());
  absl::c_copy(permutation, key.permutation.begin());
  if (absl::holds_alternative<TransposePlan::Striding>(input_layout)) {
    absl::Span<int64_t const> input_strides_in_bytes =
        absl::get<TransposePlan::Striding>(input_layout).strides_in_bytes;
    key.input_layout = absl::InlinedVector<int64_t, 4>(
        input_strides_in_bytes.begin(), input_strides_in_bytes.end());
    key.input_layout_is_tiling = false;
  } else {
    absl::Span<int64_t const> input_tiling =
        absl::get<TransposePlan::Tiling>(input_layout).tiling;
    key.input_layout = absl::InlinedVector<int64_t, 4>(input_tiling.begin(),
                                                       input_tiling.end());
    key.input_layout_is_tiling = true;
  }
  key.num_threads = num_threads;
  return cache_.GetOrCreateIfAbsent(
      key,
      [&](const TransposePlanCacheKey& key)
          -> StatusOr<std::shared_ptr<TransposePlan>> {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<TransposePlan> plan,
            TransposePlan::Create(elem_size_in_bytes, dims, permutation,
                                  input_layout, output_tiling, num_threads));
        return std::shared_ptr<TransposePlan>(std::move(plan));
      });
}

}  // namespace xla
