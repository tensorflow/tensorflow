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
// * we could do a better job of vectorizing where the stride-1 dimensions are
//   small (e.g., inner dimensions of size [..., 3] are not uncommon in some
//   use cases.)

#include "tensorflow/compiler/xla/pjrt/transpose.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <stack>
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
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {

// A plan is a data structure that describes a loop nest.
// TODO(phawkins): consider shrinking Node so it fits in a cache line.
struct TransposePlan::Node {
  // The loop should iterate over the index space range(start, end, inc).
  // These fields are ignored by the macrokernel.
  int64_t start;
  int64_t end;
  int64_t inc;  // The transpose sentinel node has inc < 0.

  // Strides of this dimension in A and B.
  int64_t lda;
  int64_t ldb;

  // If > 0, this loop is a loop over tile exteriors and has a trailing partial
  // tile. To handle the trailing partial tile, skip to the plan node this many
  // steps ahead in the vector of plan nodes.
  int trailing_tile_next_node_inc = 0;

  // Is this dimension the innermost dimension in either A or B, and hence may
  // have non-trivial blocking?
  bool is_inner_dim_in_a = false;
  bool is_inner_dim_in_b = false;
};

void ConvertF64ToEf57(const double* input, float* output, int n) {
  // TODO(phawkins): vectorize this transformation.
  for (int i = 0; i < n; ++i) {
    std::tie(output[0], output[1]) = SplitF64ToF32(*input);
    ++input;
    output += 2;
  }
}

template <typename T, int inner_bs,
          TransposePlan::Transformation transformation>
void MacroKernel(const char* __restrict a, int64_t lda, int outer_bs_a,
                 char* __restrict b, int64_t ldb, int outer_bs_b,
                 void* __restrict scratch) {
  DVLOG(10) << "MacroKernel lda=" << lda << " ldb=" << ldb
            << " outer_bs_a=" << outer_bs_a << " outer_bs_b=" << outer_bs_b
            << " inner_bs=" << inner_bs;

  // TODO(phawkins): consider adding prefetching and streaming stores.

  if (transformation == TransposePlan::Transformation::kF64ToEf57) {
    DCHECK_EQ(outer_bs_a * inner_bs % 2, 0);
    float* p = reinterpret_cast<float*>(scratch);
    for (int i = 0; i < outer_bs_b * inner_bs; ++i) {
      ConvertF64ToEf57(reinterpret_cast<const double*>(a + lda * i),
                       p + outer_bs_a * inner_bs * i,
                       outer_bs_a * inner_bs / 2);
    }
    a = reinterpret_cast<const char*>(scratch);
    lda = outer_bs_a * inner_bs * sizeof(float);
  }

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
template <typename T, int inner_bs,
          TransposePlan::Transformation transformation>
void Transpose(const char* __restrict a, int outer_bs_a, char* __restrict b,
               int outer_bs_b, TransposePlan::Node const* __restrict node,
               void* __restrict scratch) {
  DVLOG(10) << "Transpose " << outer_bs_a << " " << outer_bs_b;
  DCHECK_GT(outer_bs_a, 0);
  DCHECK_GT(outer_bs_b, 0);
  const int64_t start = node->start;
  const int64_t end = node->end;
  const int64_t stop = node->end - (node->inc - 1);
  const int64_t lda = node->lda;
  const int64_t ldb = node->ldb;
  const int64_t inc = node->inc;
  TransposePlan::Node const* next_node = node + 1;
  if (next_node->inc < 0) {
    // This is the last loop in the nested loops. The next node is a sentinel
    // plan node that describes how to invoke the macrokernels.

    const int64_t lda_block = next_node->lda;
    const int64_t ldb_block = next_node->ldb;
    int64_t i;
    for (i = start; i < stop; i += inc) {
      MacroKernel<T, inner_bs, transformation>(a + i * lda, lda_block,
                                               outer_bs_a, b + i * ldb,
                                               ldb_block, outer_bs_b, scratch);
    }
    // Handle trailing elements that didn't fit in a complete macrokernel.
    // Only the innermost dimensions have non-trivial outer_bs blocking.
    if (i < end) {
      DCHECK_EQ(node->trailing_tile_next_node_inc, 0);
      DCHECK(node->is_inner_dim_in_a || node->is_inner_dim_in_b);
      if (node->is_inner_dim_in_a) {
        outer_bs_a = (end - i) / inner_bs;
        if (outer_bs_a > 0) {
          MacroKernel<T, inner_bs, transformation>(
              a + i * lda, lda_block, outer_bs_a, b + i * ldb, ldb_block,
              outer_bs_b, scratch);
          i += outer_bs_a * inner_bs;
        }
        // If there are still trailing elements left over that don't fit in the
        // inner block size, handle them via an unvectorized transpose.
        if (i < end) {
          MacroKernel<T, 1, transformation>(a + i * lda, lda_block, end - i,
                                            b + i * ldb, ldb_block,
                                            outer_bs_b * inner_bs, scratch);
        }
      } else if (node->is_inner_dim_in_b) {
        outer_bs_b = (end - i) / inner_bs;
        if (outer_bs_b > 0) {
          MacroKernel<T, inner_bs, transformation>(
              a + i * lda, lda_block, outer_bs_a, b + i * ldb, ldb_block,
              outer_bs_b, scratch);
          i += outer_bs_b * inner_bs;
        }
        if (i < end) {
          MacroKernel<T, 1, transformation>(a + i * lda, lda_block,
                                            outer_bs_a * inner_bs, b + i * ldb,
                                            ldb_block, end - i, scratch);
        }
      }
    } else if (node->trailing_tile_next_node_inc) {
      // Handle the case where there is a trailing partial tile. We know
      // inc == 1 for this case, so the loop above has already left `a` and `b`
      // pointing to the start of the tile. We just need to use the alternate
      // trailing_next_node to process the interior of the tile.
      DCHECK_EQ(inc, 1);
      TransposePlan::Node const* trailing_next_node =
          node + node->trailing_tile_next_node_inc;
      if (trailing_next_node->inc < 0) {
        const int64_t lda_block = trailing_next_node->lda;
        const int64_t ldb_block = trailing_next_node->ldb;
        MacroKernel<T, inner_bs, transformation>(
            a + i * lda, lda_block, outer_bs_a, b + i * ldb, ldb_block,
            outer_bs_b, scratch);
      } else {
        Transpose<T, inner_bs, transformation>(a + i * lda, outer_bs_a,
                                               b + i * ldb, outer_bs_b,
                                               trailing_next_node, scratch);
      }
    }
  } else {
    // This is not the last loop in the nested loops. Recursively visit the
    // inner loops. Structurally this code is identical to the previous case,
    // but we call Transpose() recursively instead of MacroKernel().
    int64_t i;
    for (i = start; i < stop; i += inc) {
      Transpose<T, inner_bs, transformation>(
          a + i * lda, outer_bs_a, b + i * ldb, outer_bs_b, next_node, scratch);
    }
    if (i < end) {
      DCHECK_EQ(node->trailing_tile_next_node_inc, 0);
      DCHECK(node->is_inner_dim_in_a || node->is_inner_dim_in_b);
      if (node->is_inner_dim_in_a) {
        outer_bs_a = (end - i) / inner_bs;
        if (outer_bs_a > 0) {
          Transpose<T, inner_bs, transformation>(a + i * lda, outer_bs_a,
                                                 b + i * ldb, outer_bs_b,
                                                 next_node, scratch);
          i += outer_bs_a * inner_bs;
        }
        if (i < end) {
          Transpose<T, 1, transformation>(a + i * lda, end - i, b + i * ldb,
                                          outer_bs_b * inner_bs, next_node,
                                          scratch);
        }
      } else if (node->is_inner_dim_in_b) {
        outer_bs_b = (end - i) / inner_bs;
        if (outer_bs_b > 0) {
          Transpose<T, inner_bs, transformation>(a + i * lda, outer_bs_a,
                                                 b + i * ldb, outer_bs_b,
                                                 next_node, scratch);
          i += outer_bs_b * inner_bs;
        }
        if (i < end) {
          Transpose<T, 1, transformation>(a + i * lda, outer_bs_a * inner_bs,
                                          b + i * ldb, end - i, next_node,
                                          scratch);
        }
      }
    } else if (node->trailing_tile_next_node_inc) {
      TransposePlan::Node const* trailing_next_node =
          node + node->trailing_tile_next_node_inc;
      if (trailing_next_node->inc < 0) {
        const int64_t lda_block = trailing_next_node->lda;
        const int64_t ldb_block = trailing_next_node->ldb;
        MacroKernel<T, inner_bs, transformation>(
            a + i * lda, lda_block, outer_bs_a, b + i * ldb, ldb_block,
            outer_bs_b, scratch);
      } else {
        Transpose<T, inner_bs, transformation>(a + i * lda, outer_bs_a,
                                               b + i * ldb, outer_bs_b,
                                               trailing_next_node, scratch);
      }
    }
  }
}

template <typename T>
void TransposeConstStride1(const char* __restrict a, char* __restrict b,
                           TransposePlan::Node const* __restrict node) {
  a += node[0].start * node[0].lda;
  b += node[0].start * node[0].ldb;
  if (node[0].is_inner_dim_in_a) {
    int64_t num_bytes = (node->end - node->start) * sizeof(T);
    std::memcpy(b, a, num_bytes);
  } else if (node[1].is_inner_dim_in_a) {
    int64_t offset_a = node[1].start * node[1].lda;
    int64_t offset_b = node[1].start * node[1].ldb;
    int64_t num_bytes = (node[1].end - node[1].start) * sizeof(T);
    a += offset_a;
    b += offset_b;
    for (int64_t i = node[0].start; i < node[0].end; ++i) {
      std::memcpy(b, a, num_bytes);
      a += node[0].lda;
      b += node[0].ldb;
    }
    if (node[0].trailing_tile_next_node_inc) {
      TransposeConstStride1<T>(a - offset_a, b - offset_b,
                               node + node[0].trailing_tile_next_node_inc);
    }
  } else if (node[2].is_inner_dim_in_a) {
    int64_t num_bytes = (node[2].end - node[2].start) * sizeof(T);
    int64_t offset_a1 = node[1].start * node[1].lda;
    int64_t offset_b1 = node[1].start * node[1].ldb;
    int64_t offset_a2 = node[2].start * node[2].lda;
    int64_t offset_b2 = node[2].start * node[2].ldb;
    a += offset_a1 + offset_a2;
    b += offset_b1 + offset_b2;
    for (int64_t i = node[0].start; i < node[0].end; ++i) {
      const char* a1 = a;
      char* b1 = b;
      for (int64_t j = node[1].start; j < node[1].end; ++j) {
        std::memcpy(b1, a1, num_bytes);
        a1 += node[1].lda;
        b1 += node[1].ldb;
      }
      if (node[1].trailing_tile_next_node_inc) {
        TransposeConstStride1<T>(
            a1 - offset_a2, b1 - offset_b2,
            &node[1] + node[1].trailing_tile_next_node_inc);
      }
      a += node[0].lda;
      b += node[0].ldb;
    }
    if (node[0].trailing_tile_next_node_inc) {
      TransposeConstStride1<T>(a - offset_a1 - offset_a2,
                               b - offset_b1 - offset_b2,
                               node + node[0].trailing_tile_next_node_inc);
    }
  } else {
    for (int64_t i = node[0].start; i < node[0].end; ++i) {
      const char* a1 = a + node[1].start * node[1].lda;
      char* b1 = b + node[1].start * node[1].ldb;
      for (int64_t j = node[1].start; j < node[1].end; ++j) {
        TransposeConstStride1<T>(a1, b1, node + 2);
        a1 += node[1].lda;
        b1 += node[1].ldb;
      }
      if (node[1].trailing_tile_next_node_inc) {
        TransposeConstStride1<T>(
            a1, b1, &node[1] + node[1].trailing_tile_next_node_inc);
      }
      a += node[0].lda;
      b += node[0].ldb;
    }
    if (node[0].trailing_tile_next_node_inc) {
      TransposeConstStride1<T>(a, b,
                               node + node[0].trailing_tile_next_node_inc);
    }
  }
}

template <typename T, TransposePlan::Transformation transformation>
void TransposePlan::ExecuteTyped(const char* a, char* b,
                                 absl::Span<Node const> nodes) const {
  if (inner_kernel_is_memcpy_) {
    DCHECK(transformation_ == Transformation::kNone);
    TransposeConstStride1<T>(a, b, nodes.data());
  } else {
    std::unique_ptr<char[]> scratch;
    if (scratch_size_ > 0) {
      scratch.reset(new char[scratch_size_]);
    }
    switch (inner_block_elems_) {
      case 1:
        if (nodes.size() > 1) {
          Transpose<T, 1, transformation>(a, outer_block_elems_a_, b,
                                          outer_block_elems_b_, nodes.data(),
                                          scratch.get());
        } else {
          MacroKernel<T, 1, transformation>(
              a, nodes.back().lda, outer_block_elems_a_, b, nodes.back().ldb,
              outer_block_elems_b_, scratch.get());
        }
        break;
      case 2:
        if (nodes.size() > 1) {
          Transpose<T, 2, transformation>(a, outer_block_elems_a_, b,
                                          outer_block_elems_b_, nodes.data(),
                                          scratch.get());
        } else {
          MacroKernel<T, 2, transformation>(
              a, nodes.back().lda, outer_block_elems_a_, b, nodes.back().ldb,
              outer_block_elems_b_, scratch.get());
        }
        break;
      case 4:

        if (nodes.size() > 1) {
          Transpose<T, 4, transformation>(a, outer_block_elems_a_, b,
                                          outer_block_elems_b_, nodes.data(),
                                          scratch.get());
        } else {
          MacroKernel<T, 4, transformation>(
              a, nodes.back().lda, outer_block_elems_a_, b, nodes.back().ldb,
              outer_block_elems_b_, scratch.get());
        }
        break;
      case 8:
        if (nodes.size() > 1) {
          Transpose<T, 8, transformation>(a, outer_block_elems_a_, b,
                                          outer_block_elems_b_, nodes.data(),
                                          scratch.get());
        } else {
          MacroKernel<T, 8, transformation>(
              a, nodes.back().lda, outer_block_elems_a_, b, nodes.back().ldb,
              outer_block_elems_b_, scratch.get());
        }
        break;
      case 16:
        if (nodes.size() > 1) {
          Transpose<T, 16, transformation>(a, outer_block_elems_a_, b,
                                           outer_block_elems_b_, nodes.data(),
                                           scratch.get());
        } else {
          MacroKernel<T, 16, transformation>(
              a, nodes.back().lda, outer_block_elems_a_, b, nodes.back().ldb,
              outer_block_elems_b_, scratch.get());
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

  auto execute_by_type = [&](absl::Span<Node const> nodes) {
    switch (elem_size_in_bytes_) {
      case 1:
        ExecuteTyped<uint8_t, Transformation::kNone>(ac, bc, nodes);
        break;
      case 2:
        ExecuteTyped<uint16_t, Transformation::kNone>(ac, bc, nodes);
        break;
      case 4:
        if (transformation_ == Transformation::kNone) {
          ExecuteTyped<uint32_t, Transformation::kNone>(ac, bc, nodes);
        } else {
          DCHECK(transformation_ == Transformation::kF64ToEf57);
          ExecuteTyped<uint32_t, Transformation::kF64ToEf57>(ac, bc, nodes);
        }
        break;
      case 8:
        ExecuteTyped<uint64_t, Transformation::kNone>(ac, bc, nodes);
        break;
      case 16:
        ExecuteTyped<uint128, Transformation::kNone>(ac, bc, nodes);
        break;
      default:
        LOG(FATAL) << "Unimplemented element size " << elem_size_in_bytes_;
    }
  };

  if (!schedule_work || nodes_.size() <= 1) {
    for (const auto& nodes : nodes_) {
      execute_by_type(nodes);
    }
  } else {
    absl::BlockingCounter counter(nodes_.size());
    for (absl::Span<Node const> nodes : nodes_) {
      schedule_work([&, nodes]() {
        tensorflow::profiler::TraceMe traceme("Transpose::Execute",
                                              /*level=*/2);
        execute_by_type(nodes);
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
static Status ParseTilingSpecification(
    int ndim, absl::Span<int64_t const> tiling_spec,
    absl::InlinedVector<int64_t, 4>& tiling) {
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
  return OkStatus();
}

// Helper function that builds a plan.
void TransposePlan::BuildPlanNodes(
    absl::Span<int64_t const> inverse_permutation, int thread_id,
    std::vector<TransposePlan::Node>& nodes) {
  VLOG(8) << "Before plan build: " << ToString();
  const int ndim = a_dims_.size();
  DCHECK_GT(ndim, 0);
  const int pos_stride1a = ndim - 1;
  const int pos_stride1b_in_a = permutation_.back();
  const int pos_stride1a_in_b = inverse_permutation[pos_stride1a];

  // We builld plans in a depth-first order, visiting loops from outermost to
  // innermost. We use a stack (depth-first) order to handle trailing partial
  // tiles, which we "come back to" after handling the non-trailing case.
  struct Agendum {
    // The ID of the loop to visit in loop_order_.
    int loop_id;
    // The parent node ID whose trailing tile should be made to point to this
    // node.
    int parent_node_id;

    // The number of parallel tasks available to run this loop and its
    // successors.
    int num_tasks_at_loop;

    // The ID number of the current thread in the tasks at this loop.
    int task_id_at_loop;

    // For which dimensions of `a` are we to visit the partial trailing tile
    // a loop that visits that tile's interior?
    absl::InlinedVector<bool, 4> partial_tiles;
  };
  std::stack<Agendum> agenda;

  int total_tasks =
      absl::c_accumulate(loop_parallelism_, int{1}, std::multiplies<int>());

  agenda.push(Agendum{/*loop_id=*/0, /*parent_node_id=*/-1,
                      /*num_tasks_at_loop=*/total_tasks,
                      /*task_id_at_loop=*/thread_id,
                      absl::InlinedVector<bool, 4>(ndim, false)});

  auto loop_has_trivial_iteration_space = [](const Node& node) {
    return node.start == 0 && node.start + node.inc == node.end;
  };

  while (!agenda.empty()) {
    Agendum agendum = std::move(agenda.top());
    agenda.pop();

    int node_id = static_cast<int>(nodes.size());
    if (agendum.parent_node_id >= 0) {
      // This is a trailing partial tile node; update the parent node to
      // point to it.
      nodes[agendum.parent_node_id].trailing_tile_next_node_inc =
          node_id - agendum.parent_node_id;
    }

    if (agendum.loop_id == loop_order_.size()) {
      // We've reached the end of the loop nest.
      DCHECK_EQ(agendum.num_tasks_at_loop, 1);
      // Transpose loops have a sentinel node, indicated by a negative `inc`
      // value, that describes the striding of the inner transpose kernel.
      if (!inner_kernel_is_memcpy_) {
        Node node;
        node.start = node.end = node.inc = -1;
        node.lda = a_tiling_[pos_stride1b_in_a] > 1
                       ? lda_tile_[pos_stride1b_in_a]
                       : lda_[pos_stride1b_in_a];
        node.ldb = b_tiling_[pos_stride1a_in_b] > 1
                       ? ldb_tile_[pos_stride1a_in_b]
                       : ldb_[pos_stride1a_in_b];
        nodes.push_back(node);
      }
      DCHECK(!(inner_kernel_is_memcpy_ && agendum.parent_node_id >= 0));
      continue;
    }

    const Loop& loop = loop_order_[agendum.loop_id];
    int a_dim = loop.dim_in_a;
    int b_dim = inverse_permutation[a_dim];
    DCHECK(a_tiling_[a_dim] == 1 || b_tiling_[b_dim] == 1 ||
           a_tiling_[a_dim] == b_tiling_[b_dim]);
    int64_t tile_size = std::max(a_tiling_[a_dim], b_tiling_[b_dim]);

    // Compute the number of tasks for the next loop iteration.
    int task_id_at_loop = agendum.task_id_at_loop;
    int num_tasks_at_loop =
        agendum.num_tasks_at_loop / loop_parallelism_[agendum.loop_id];
    int task_id_at_next_loop = task_id_at_loop % num_tasks_at_loop;

    if (loop.tile_interior) {
      // We are visiting the tile interior of a tiled dimension.
      bool partial = agendum.partial_tiles[a_dim];

      Node node;
      node.lda = a_tiling_[a_dim] > 1 ? lda_tile_[a_dim] : lda_[a_dim];
      node.ldb = b_tiling_[b_dim] > 1 ? ldb_tile_[b_dim] : ldb_[b_dim];
      node.inc = 1;
      node.is_inner_dim_in_a = (a_dim == pos_stride1a);
      node.is_inner_dim_in_b = (a_dim == pos_stride1b_in_a);
      if (node.is_inner_dim_in_a) {
        node.inc = inner_block_elems_ * outer_block_elems_a_;
      } else if (node.is_inner_dim_in_b) {
        node.inc = inner_block_elems_ * outer_block_elems_b_;
      }

      int task_id = task_id_at_loop / num_tasks_at_loop;
      int64_t size = partial ? a_dims_[a_dim] % tile_size : tile_size;
      int64_t num_iterations = CeilOfRatio(size, node.inc);
      int64_t num_iterations_per_task = CeilOfRatio<int64_t>(
          num_iterations, loop_parallelism_[agendum.loop_id]);
      node.start = std::min(size, task_id * num_iterations_per_task * node.inc);
      node.end =
          std::min(size, (task_id + 1) * num_iterations_per_task * node.inc);
      if (!loop_has_trivial_iteration_space(node) ||
          (inner_kernel_is_memcpy_ && node.is_inner_dim_in_a)) {
        nodes.push_back(node);
      }
      Agendum new_agendum;
      new_agendum.loop_id = agendum.loop_id + 1;
      new_agendum.parent_node_id = -1;
      new_agendum.task_id_at_loop = task_id_at_next_loop;
      new_agendum.num_tasks_at_loop = num_tasks_at_loop;
      new_agendum.partial_tiles = agendum.partial_tiles;
      agenda.push(std::move(new_agendum));
    } else {
      // We are either visiting an untiled dimension, or the loop that iterates
      // over tile exteriors.
      int task_id = task_id_at_loop / num_tasks_at_loop;
      int64_t num_complete_tiles = a_dims_[a_dim] / tile_size;
      bool has_partial_tile = (a_dims_[a_dim] % tile_size != 0);

      // If there is a trailing partial tile as well as complete tiles, handle
      // it as a trailer on the loop over complete tiles.
      bool has_trailing_plan_node = false;
      if (num_complete_tiles > 0 && has_partial_tile &&
          task_id == loop_parallelism_[agendum.loop_id] - 1) {
        Agendum new_agendum;
        new_agendum.loop_id = agendum.loop_id + 1;
        new_agendum.parent_node_id = node_id;
        new_agendum.task_id_at_loop = task_id_at_next_loop;
        new_agendum.num_tasks_at_loop = num_tasks_at_loop;
        new_agendum.partial_tiles = agendum.partial_tiles;
        new_agendum.partial_tiles[a_dim] = true;
        agenda.push(std::move(new_agendum));
        has_trailing_plan_node = true;
      }
      Node node;
      node.lda = lda_[a_dim] * tile_size / a_tiling_[a_dim];
      node.ldb = ldb_[b_dim] * tile_size / b_tiling_[b_dim];
      node.inc = 1;
      node.is_inner_dim_in_a = (tile_size == 1 && a_dim == ndim - 1);
      node.is_inner_dim_in_b = (tile_size == 1 && a_dim == pos_stride1b_in_a);
      if (node.is_inner_dim_in_a) {
        node.inc = inner_block_elems_ * outer_block_elems_a_;
      } else if (node.is_inner_dim_in_b) {
        node.inc = inner_block_elems_ * outer_block_elems_b_;
      }

      // If this tiled dimension consists only of a single partial tile, handle
      // it here; there's no point emitting a degenerate loop and a separate
      // path to handle the trailing tile.
      bool partial = num_complete_tiles == 0 && has_partial_tile;

      // Evenly divide the loop iterations amongst the threads.
      int64_t num_tiles = partial ? 1 : num_complete_tiles;
      int64_t num_iterations = CeilOfRatio(num_tiles, node.inc);
      int64_t num_iterations_per_task = CeilOfRatio<int64_t>(
          num_iterations, loop_parallelism_[agendum.loop_id]);
      node.start =
          std::min(num_tiles, task_id * num_iterations_per_task * node.inc);
      node.end = std::min(num_tiles,
                          (task_id + 1) * num_iterations_per_task * node.inc);
      // If this loop has a trivial iteration space, drop it.
      if (!loop_has_trivial_iteration_space(node) ||
          (inner_kernel_is_memcpy_ && node.is_inner_dim_in_a) ||
          has_trailing_plan_node) {
        nodes.push_back(node);
      }
      Agendum new_agendum;
      new_agendum.loop_id = agendum.loop_id + 1;
      new_agendum.parent_node_id = -1;
      new_agendum.task_id_at_loop = task_id_at_next_loop;
      new_agendum.num_tasks_at_loop = num_tasks_at_loop;
      new_agendum.partial_tiles = agendum.partial_tiles;
      new_agendum.partial_tiles[a_dim] = partial;
      agenda.push(std::move(new_agendum));
    }
  }
}

StatusOr<std::unique_ptr<TransposePlan>> TransposePlan::Create(
    size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
    absl::Span<int64_t const> permutation,
    std::variant<Tiling, Striding> input_layout, Tiling output_tiling,
    Transformation transformation, int num_threads) {
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
  if (std::holds_alternative<Striding>(input_layout)) {
    absl::Span<int64_t const> input_strides_in_bytes =
        std::get<Striding>(input_layout).strides_in_bytes;
    if (input_strides_in_bytes.size() != dims.size()) {
      return InvalidArgument(
          "dims and input_strides_in_bytes must have equal sizes, got %d "
          "and %d",
          dims.size(), input_strides_in_bytes.size());
    }
    plan->original_a_strides_.resize(ndim);
    absl::c_copy(input_strides_in_bytes, plan->original_a_strides_.begin());
    // Sort the dimensions from slowest-varying (largest strides) to
    // fastest-varying (smallest strides).
    std::vector<int64_t> dim_order(ndim);
    absl::c_iota(dim_order, 0);

    auto cost = [&](int k) {
      int64_t stride = input_strides_in_bytes.at(k);
      // If there is a dimension with size equal to the element size, sort it
      // last. This ensures that we place any stride-1 dimension last.
      bool is_stride1 = stride == elem_size_in_bytes;
      // If there are multiple stride-1 dimensions, we'd prefer the one that
      // matches the stride-1 dimension of the output.
      // Failing that, we'd just prefer the largest stride-1 dimension last.
      bool is_trailing_dim_in_b = permutation.back() == k;

      // If we are applying ef57 conversion, we want a size-2 stride-1
      // dimension last.
      bool ef57_even =
          (is_stride1 && transformation == Transformation::kF64ToEf57 &&
           dims[k] == 2);

      return std::make_tuple(is_stride1, -std::abs(stride), ef57_even,
                             is_trailing_dim_in_b, dims[k]);
    };
    absl::c_stable_sort(dim_order,
                        [&cost](int i, int j) { return cost(i) < cost(j); });
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
        ndim, std::get<Tiling>(input_layout).tiling, plan->a_tiling_));

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

  plan->transformation_ = transformation;
  switch (transformation) {
    case Transformation::kNone:
      break;
    case Transformation::kF64ToEf57:
      if (elem_size_in_bytes != sizeof(float)) {
        return InvalidArgument(
            "EF57 conversion requires a element size of %d bytes, got %d",
            sizeof(float), elem_size_in_bytes);
      }
      if (plan->a_dims_.empty() || plan->a_dims_.back() % 2 != 0 ||
          plan->lda_.back() != sizeof(float)) {
        return InvalidArgument(
            "EF57 conversion requires a stride-%d dimension whose size is a "
            "multiple of 2",
            sizeof(float));
      }
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
  // We don't accept arbitrary stridings for B, so we know B always has a
  // stride 1 dimension innermost.

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
  b_dims_ = Permute(a_dims_, permutation_);
  ComputeStrides(elem_size_in_bytes_, b_dims_, b_tiling_, ldb_, ldb_tile_);

  const int pos_stride1a = ndim - 1;
  const int pos_stride1b_in_a = permutation_.back();
  inner_kernel_is_memcpy_ = (pos_stride1b_in_a == pos_stride1a);

  loop_order_.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    loop_order_.push_back(Loop{i, /*tile_interior=*/false});
    if (a_tiling_[i] != 1 || b_tiling_[inverse_permutation[i]] != 1) {
      loop_order_.push_back(Loop{i, /*tile_interior=*/true});
    }
  }

  // Bound the block sizes so they are smaller than the stride-1 dimension
  // size.
  int64_t a_stride1_size = std::max(
      a_tiling_[pos_stride1a], b_tiling_[inverse_permutation[pos_stride1a]]);
  if (a_stride1_size == 1) {
    a_stride1_size = a_dims_[pos_stride1a];
  } else {
    // If there's only one tile, we should use the dimension size.
    a_stride1_size = std::min(a_dims_[pos_stride1a], a_stride1_size);
  }
  int64_t b_stride1_size =
      std::max(a_tiling_[permutation_.back()], b_tiling_.back());
  if (b_stride1_size == 1) {
    b_stride1_size = b_dims_.back();
  } else {
    b_stride1_size = std::min(b_stride1_size, b_dims_.back());
  }

  if (inner_kernel_is_memcpy_) {
    inner_block_elems_ = -1;
    outer_block_elems_a_ = -1;
    outer_block_elems_b_ = -1;
  } else {
    // What are the smallest and largest block sizes for which we have a
    // vectorized kernel for this element size?
    int min_inner_block_elems;
    int max_inner_block_elems;
    switch (elem_size_in_bytes_) {
      case 1:
        min_inner_block_elems = 4;
        max_inner_block_elems = 16;
        break;
      case 2:
        min_inner_block_elems = 8;
        max_inner_block_elems = 8;
        break;
      case 4:
        min_inner_block_elems = 4;
        max_inner_block_elems = 8;
        break;
      case 8:
        min_inner_block_elems = 2;
        max_inner_block_elems = 4;
        break;
      case 16:
        min_inner_block_elems = 1;
        max_inner_block_elems = 1;
        break;
      default:
        LOG(FATAL) << "Unreachable: element size " << elem_size_in_bytes_;
    }
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
        std::min<int64_t>(16, a_stride1_size), inner_block_elems_);
    outer_block_elems_b_ = FloorOfRatio<int64_t>(
        std::min<int64_t>(16, b_stride1_size), inner_block_elems_);
  }

  // Loop order heuristic: try to make loops with small strides innermost.
  auto cost = [&](const Loop& l) {
    int64_t a_stride =
        std::abs((l.tile_interior && a_is_tiled_) ? lda_tile_[l.dim_in_a]
                                                  : lda_[l.dim_in_a]);
    bool is_inner_dim_in_a =
        (!a_is_tiled_ || l.tile_interior) && (l.dim_in_a == pos_stride1a);

    if (!inner_kernel_is_memcpy_ && is_inner_dim_in_a) {
      a_stride *= inner_block_elems_ * outer_block_elems_a_;
    }
    int b_dim = inverse_permutation[l.dim_in_a];
    int64_t b_stride =
        (l.tile_interior && b_is_tiled_) ? ldb_tile_[b_dim] : ldb_[b_dim];
    bool is_inner_dim_in_b =
        (!b_is_tiled_ || l.tile_interior) && (l.dim_in_a == pos_stride1b_in_a);
    if (!inner_kernel_is_memcpy_ && is_inner_dim_in_b) {
      b_stride *= inner_block_elems_ * outer_block_elems_b_;
    }
    // Add a small penalty to the input strides: given the choice between
    // consecutive writes and consecutive reads, we would prefer consecutive
    // writes.
    double penalty = 1.01;

    // If the inner kernel is a memcpy make sure the innermost loop is the
    // stride-1 dimension. This is a requirement of the memcpy kernel.
    bool dim_must_go_last =
        inner_kernel_is_memcpy_ && l.dim_in_a == pos_stride1a &&
        (l.tile_interior ||
         (a_tiling_[l.dim_in_a] == 1 && b_tiling_[b_dim] == 1));
    return std::make_tuple(dim_must_go_last,
                           inner_kernel_is_memcpy_ && l.tile_interior,
                           -std::min<double>(a_stride * penalty, b_stride));
  };
  absl::c_stable_sort(loop_order_, [&](const Loop& a, const Loop& b) {
    return cost(a) < cost(b);
  });
  // It is a required invariant of the loop order that tile interiors always
  // appear after the corresponding tile exterior. This is a consequence of the
  // heuristic above, because the tile interior must have smaller strides in
  // both input and output.

  // The stride-1 loop must be innermost for a memcpy loop.
  DCHECK(!inner_kernel_is_memcpy_ || loop_order_.back().dim_in_a == ndim - 1)
      << ToString();

  loop_parallelism_ = ChooseParallelizationStrategy(inverse_permutation);
  int num_threads =
      absl::c_accumulate(loop_parallelism_, int{1}, std::multiplies<int>());
  nodes_.resize(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    BuildPlanNodes(inverse_permutation, thread_id, nodes_[thread_id]);
  }

  switch (transformation_) {
    case Transformation::kNone:
      scratch_size_ = 0;
      break;
    case Transformation::kF64ToEf57:
      scratch_size_ = sizeof(float) * inner_block_elems_ * inner_block_elems_ *
                      outer_block_elems_a_ * outer_block_elems_b_;
      DCHECK(!inner_kernel_is_memcpy_);
      break;
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
    int kMinBytesPerThread = inner_kernel_is_memcpy_ ? (1 << 20) : (1 << 26);
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

std::string TransposePlan::ToString() const {
  std::string nodes_str = absl::StrJoin(
      nodes_, "\n", [](std::string* out, absl::Span<Node const> thread_nodes) {
        absl::StrAppend(
            out, "thread:\n",
            absl::StrJoin(
                thread_nodes, "\n", [](std::string* out, const Node& node) {
                  absl::StrAppendFormat(
                      out,
                      "    "
                      "Node(start=%d,end=%d,inc=%d,lda=%"
                      "d,ldb=%d,next_trailing=%d,inner_a=%s,inner_b=%s)",
                      node.start, node.end, node.inc, node.lda, node.ldb,
                      node.trailing_tile_next_node_inc,
                      node.is_inner_dim_in_a ? "y" : "n",
                      node.is_inner_dim_in_b ? "y" : "n");
                }));
      });
  auto format_loop_order = [](std::string* out, const Loop& loop) {
    return absl::StrAppend(out, loop.dim_in_a,
                           loop.tile_interior ? "[tile]" : "");
  };
  std::string transformation_str;
  switch (transformation_) {
    case Transformation::kNone:
      transformation_str = "none";
      break;
    case Transformation::kF64ToEf57:
      transformation_str = "ef57";
      break;
  }
  return absl::StrFormat(
      "elem_size=%d a_dims=%s b_dims=%s permutation=%s a_tiling=%s b_tiling=%s "
      "lda=%s lda_tile=%s ldb=%s ldb_tile=%s loop_order=%s "
      "loop_parallelism=%s outer_bs=[%d,%d] inner_bs=%d "
      "transformation=%s scratch_size=%d\n"
      "nodes:\n%s",
      elem_size_in_bytes_, absl::StrJoin(a_dims_, ","),
      absl::StrJoin(Permute(a_dims_, permutation_), ","),
      absl::StrJoin(permutation_, ","), absl::StrJoin(a_tiling_, ","),
      absl::StrJoin(b_tiling_, ","), absl::StrJoin(lda_, ","),
      absl::StrJoin(lda_tile_, ","), absl::StrJoin(ldb_, ","),
      absl::StrJoin(ldb_tile_, ","),
      absl::StrJoin(loop_order_, ",", format_loop_order),
      absl::StrJoin(loop_parallelism_, ","), outer_block_elems_a_,
      outer_block_elems_b_, inner_block_elems_, transformation_str,
      scratch_size_, nodes_str);
}

struct TransposePlanCacheKey {
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

bool TransposePlanCacheKey::operator==(
    const TransposePlanCacheKey& other) const {
  return elem_size_in_bytes == other.elem_size_in_bytes && dims == other.dims &&
         permutation == other.permutation &&
         input_layout_is_tiling == other.input_layout_is_tiling &&
         input_layout == other.input_layout &&
         output_tiling == other.output_tiling &&
         transformation == other.transformation &&
         num_threads == other.num_threads;
}

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key) {
  return H::combine(std::move(h), key.elem_size_in_bytes,
                    key.input_layout_is_tiling, key.num_threads,
                    key.transformation, key.dims, key.permutation,
                    key.input_layout, key.output_tiling);
}

TransposePlanCache::TransposePlanCache(int capacity)
    : lru_list_(capacity), cache_(&lru_list_) {}

TransposePlanCache::~TransposePlanCache() = default;

StatusOr<std::shared_ptr<TransposePlan>> TransposePlanCache::GetOrCreate(
    size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
    absl::Span<int64_t const> permutation,
    std::variant<TransposePlan::Tiling, TransposePlan::Striding> input_layout,
    TransposePlan::Tiling output_tiling,
    TransposePlan::Transformation transformation, int num_threads) {
  TransposePlanCacheKey key;
  key.elem_size_in_bytes = elem_size_in_bytes;
  key.dims.resize(dims.size());
  absl::c_copy(dims, key.dims.begin());
  key.permutation.resize(permutation.size());
  absl::c_copy(permutation, key.permutation.begin());
  if (std::holds_alternative<TransposePlan::Striding>(input_layout)) {
    absl::Span<int64_t const> input_strides_in_bytes =
        std::get<TransposePlan::Striding>(input_layout).strides_in_bytes;
    key.input_layout = absl::InlinedVector<int64_t, 4>(
        input_strides_in_bytes.begin(), input_strides_in_bytes.end());
    key.input_layout_is_tiling = false;
  } else {
    absl::Span<int64_t const> input_tiling =
        std::get<TransposePlan::Tiling>(input_layout).tiling;
    key.input_layout = absl::InlinedVector<int64_t, 4>(input_tiling.begin(),
                                                       input_tiling.end());
    key.input_layout_is_tiling = true;
  }
  key.output_tiling.resize(output_tiling.tiling.size());
  absl::c_copy(output_tiling.tiling, key.output_tiling.begin());
  key.transformation = transformation;
  key.num_threads = num_threads;
  return cache_.GetOrCreateIfAbsent(
      key,
      [&](const TransposePlanCacheKey& key)
          -> StatusOr<std::shared_ptr<TransposePlan>> {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<TransposePlan> plan,
            TransposePlan::Create(elem_size_in_bytes, dims, permutation,
                                  input_layout, output_tiling, transformation,
                                  num_threads));
        return std::shared_ptr<TransposePlan>(std::move(plan));
      });
}

}  // namespace xla
