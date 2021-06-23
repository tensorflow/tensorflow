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
#include <numeric>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
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
};

template <typename T, int inner_bs>
void MacroKernelBlocked(const char* __restrict a, int64_t lda, int outer_bs_a,
                        char* __restrict b, int64_t ldb, int outer_bs_b) {
  DVLOG(10) << "MacroKernelBlocked lda=" << lda << " ldb=" << ldb
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
               int outer_bs_b, TransposePlan::Node* node) {
  DVLOG(10) << "Transpose " << outer_bs_a << " " << outer_bs_b;
  DCHECK_GT(outer_bs_a, 0);
  DCHECK_GT(outer_bs_b, 0);
  const int64_t start = node->start;
  const int64_t end = node->end;
  int64_t stop = node->end - (node->inc - 1);
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
      MacroKernelBlocked<T, inner_bs>(a + i * lda, lda_block, outer_bs_a,
                                      b + i * ldb, ldb_block, outer_bs_b);
    }
    // Handle trailing elements that didn't fit in a complete macrokernel.
    // Only the innermost dimensions have non-trivial outer_bs blocking.
    if (lda == sizeof(T)) {
      // Only one of the input and output should have stride-1 dimensions at the
      // same location. Otherwise we would have called the TransposeConstStride1
      // case because we don't need to transpose the innermost dimensions.
      DCHECK_NE(ldb, sizeof(T));
      // Repeatedly halve the outer block size, until size 1
      while (outer_bs_a > 1) {
        outer_bs_a /= 2;
        if (i + outer_bs_a * inner_bs <= end) {
          MacroKernelBlocked<T, inner_bs>(a + i * lda, lda_block, outer_bs_a,
                                          b + i * ldb, ldb_block, outer_bs_b);
          i += outer_bs_a * inner_bs;
        }
      }
      // If there are still trailing elements left over that don't fit in the
      // inner block size, handle them via an unvectorized transpose.
      if (i < end) {
        MacroKernelBlocked<T, 1>(a + i * lda, lda_block, end - i, b + i * ldb,
                                 ldb_block, outer_bs_b * inner_bs);
      }
    } else if (ldb == sizeof(T)) {
      DCHECK_NE(lda, sizeof(T));  // Only stride-1 dimensions are blocked.
      while (outer_bs_b > 1) {
        outer_bs_b /= 2;
        if (i + outer_bs_b * inner_bs <= end) {
          MacroKernelBlocked<T, inner_bs>(a + i * lda, lda_block, outer_bs_a,
                                          b + i * ldb, ldb_block, outer_bs_b);
          i += outer_bs_b * inner_bs;
        }
      }
      if (i < end) {
        MacroKernelBlocked<T, 1>(a + i * lda, lda_block, outer_bs_a * inner_bs,
                                 b + i * ldb, ldb_block, end - i);
      }
    }
  } else {
    // This is not the last loop in the nested loops. Recursively visit the
    // inner loops. Structurally this code is identical to the previous case,
    // but we call Transpose() recursively instead of MacroKernelBlocked().
    int64_t i;
    for (i = start; i < stop; i += inc) {
      Transpose<T, inner_bs>(a + i * lda, outer_bs_a, b + i * ldb, outer_bs_b,
                             node->next);
    }
    if (lda == sizeof(T)) {
      DCHECK_GT(ldb, sizeof(T));
      while (outer_bs_a > 1) {
        outer_bs_a /= 2;
        if (i + outer_bs_a * inner_bs <= end) {
          Transpose<T, inner_bs>(a + i * lda, outer_bs_a, b + i * ldb,
                                 outer_bs_b, node->next);
          i += outer_bs_a * inner_bs;
        }
      }
      if (i < end) {
        Transpose<T, 1>(a + i * lda, end - i, b + i * ldb,
                        outer_bs_b * inner_bs, node->next);
      }
    } else if (ldb == sizeof(T)) {
      DCHECK_GT(lda, sizeof(T));  // Only stride-1 dimensions are blocked.
      while (outer_bs_b > 1) {
        outer_bs_b /= 2;
        if (i + outer_bs_b * inner_bs <= end) {
          Transpose<T, inner_bs>(a + i * lda, outer_bs_a, b + i * ldb,
                                 outer_bs_b, node->next);
          i += outer_bs_b * inner_bs;
        }
      }
      if (i < end) {
        Transpose<T, 1>(a + i * lda, outer_bs_a * inner_bs, b + i * ldb,
                        end - i, node->next);
      }
    }
  }
}

template <typename T>
void TransposeConstStride1(const char* __restrict a, char* __restrict b,
                           TransposePlan::Node* node) {
  const int64_t start = node->start;
  const int64_t end = node->end;
  const int64_t lda = node->lda;
  const int64_t ldb = node->ldb;
  if (node->next->next == nullptr) {
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

TransposePlan::TransposePlan() = default;
TransposePlan::~TransposePlan() = default;

static void ComputeStrides(int64_t elem_size_in_bytes,
                           absl::Span<const int64_t> dims,
                           absl::InlinedVector<int64_t, 4>& strides) {
  strides.resize(dims.size());
  int64_t acc = elem_size_in_bytes;
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    strides[d] = acc;
    acc *= dims[d];
  }
}

void TransposePlan::RemoveTrivialDimensions(
    absl::InlinedVector<int64_t, 4>& a_dims,
    absl::InlinedVector<int64_t, 4>& permutation,
    absl::InlinedVector<int64_t, 4>& lda) {
  int ndim = a_dims.size();
  std::vector<int> shift(ndim);
  absl::InlinedVector<int64_t, 4> updated_a_dims;
  absl::InlinedVector<int64_t, 4> updated_lda;
  updated_a_dims.reserve(ndim);
  updated_lda.reserve(ndim);
  for (int a_dim = 0; a_dim < ndim; ++a_dim) {
    if (a_dims[a_dim] != 1) {
      updated_a_dims.push_back(a_dims[a_dim]);
      updated_lda.push_back(lda[a_dim]);
      shift[a_dim] = a_dim + 1 - updated_a_dims.size();
    } else {
      shift[a_dim] = -1;
    }
  }

  // Updates the permutation.
  absl::InlinedVector<int64_t, 4> updated_permutation;
  updated_permutation.reserve(updated_a_dims.size());
  for (int b_dim = 0; b_dim < ndim; ++b_dim) {
    int a_dim = permutation[b_dim];
    if (shift[a_dim] >= 0) {
      updated_permutation.push_back(a_dim - shift[a_dim]);
    }
  }

  DCHECK(IsPermutation(updated_permutation));
  a_dims = std::move(updated_a_dims);
  permutation = std::move(updated_permutation);
  lda = std::move(updated_lda);
}

void TransposePlan::CoalesceDimensions(
    absl::InlinedVector<int64_t, 4>& a_dims,
    absl::InlinedVector<int64_t, 4>& permutation,
    absl::InlinedVector<int64_t, 4>& lda) {
  int ndim = a_dims.size();
  std::vector<int> shift(ndim, 0);
  absl::InlinedVector<int64_t, 4> updated_a_dims;
  absl::InlinedVector<int64_t, 4> updated_lda;
  updated_a_dims.reserve(ndim);
  updated_lda.reserve(ndim);
  std::vector<int64_t> inv_permutation = InversePermutation(permutation);
  for (int a_dim = 0; a_dim < ndim; ++a_dim) {
    // We can coalesce two dimensions if they appear consecutively
    // in both the input dimensions and the output dimensions, and the stride
    // of the outer dimension is the usual multiple of the inner dimension.
    if (a_dim > 0 && inv_permutation[a_dim - 1] + 1 == inv_permutation[a_dim] &&
        lda[a_dim - 1] == lda[a_dim] * a_dims[a_dim]) {
      updated_a_dims.back() *= a_dims[a_dim];
      updated_lda.back() = lda[a_dim];
      shift[a_dim] = -1;
    } else {
      updated_a_dims.push_back(a_dims[a_dim]);
      updated_lda.push_back(lda[a_dim]);
      shift[a_dim] = a_dim + 1 - updated_a_dims.size();
    }
  }

  // Updates the permutation.
  absl::InlinedVector<int64_t, 4> updated_permutation;
  updated_permutation.reserve(updated_a_dims.size());
  for (int b_dim = 0; b_dim < ndim; ++b_dim) {
    int a_dim = permutation[b_dim];
    if (shift[a_dim] >= 0) {
      updated_permutation.push_back(a_dim - shift[a_dim]);
    }
  }
  DCHECK(IsPermutation(updated_permutation));
  a_dims = std::move(updated_a_dims);
  permutation = std::move(updated_permutation);
  lda = std::move(updated_lda);
}

int64_t TransposePlan::NumElems() const { return num_elems_; }

// Recursive helper function that builds a plan.
TransposePlan::Node* TransposePlan::BuildPlanNode(
    absl::Span<int64_t const> inverse_permutation, int i) {
  const int pos_stride1b_in_a = permutation_.back();
  const int pos_stride1a_in_b = inverse_permutation.back();
  nodes_.push_back(std::make_unique<Node>());
  Node* node = nodes_.back().get();
  const int ndim = a_dims_.size();
  if (i == loop_order_.size()) {
    // Sentinel node that says that we should invoke the kernel.
    node->next = nullptr;
    node->start = node->end = node->inc = -1;
    node->lda = lda_[pos_stride1b_in_a];
    node->ldb = ldb_[pos_stride1a_in_b];
    return node;
  }
  int a_dim = loop_order_[i];
  int b_dim = inverse_permutation[a_dim];
  node->start = 0;
  node->end = a_dims_[a_dim];
  node->lda = lda_[a_dim];
  node->ldb = ldb_[b_dim];
  node->inc = 1;
  if (a_dim == ndim - 1) {
    node->inc = inner_block_elems_ * outer_block_elems_a_;
  } else if (a_dim == pos_stride1b_in_a) {
    node->inc = inner_block_elems_ * outer_block_elems_b_;
  }
  node->next = BuildPlanNode(inverse_permutation, i + 1);
  return node;
}

StatusOr<std::unique_ptr<TransposePlan>> TransposePlan::Create(
    size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
    absl::Span<int64_t const> permutation,
    absl::optional<absl::Span<int64_t const>> input_strides_in_bytes) {
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

  int ndim = dims.size();

  auto plan = std::make_unique<TransposePlan>();
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
  plan->original_b_dims_ = Permute(dims, permutation);
  ComputeStrides(plan->elem_size_in_bytes_, plan->original_b_dims_,
                 plan->original_b_strides_);

  if (input_strides_in_bytes) {
    if (input_strides_in_bytes->size() != dims.size()) {
      return InvalidArgument(
          "dims and input_strides_in_bytes must have equal sizes, got %d "
          "and %d",
          dims.size(), input_strides_in_bytes->size());
    }
    if (absl::c_find_if(*input_strides_in_bytes, is_negative) !=
        input_strides_in_bytes->end()) {
      return InvalidArgument(
          "input_strides_in_bytes must be non-negative, got %s",
          absl::StrJoin(dims, ","));
    }
    plan->original_a_strides_.resize(ndim);
    absl::c_copy(*input_strides_in_bytes, plan->original_a_strides_.begin());
    // Sort the dimensions from slowest-varying (largest strides) to
    // fastest-varying (smallest strides).
    std::vector<int64_t> dim_order(ndim);
    absl::c_iota(dim_order, 0);
    absl::c_stable_sort(dim_order, [&](int a, int b) {
      int64_t stride_a = input_strides_in_bytes->at(a);
      int64_t stride_b = input_strides_in_bytes->at(b);
      // If there is a dimension with size equal to the element size, sort it
      // last. This ensures that we place any stride-1 dimension last.
      if (stride_a != elem_size_in_bytes && stride_b == elem_size_in_bytes) {
        return true;
      }
      return stride_a > stride_b;
    });
    // dim_order maps new input dim -> old input dim, we need its inverse to
    // compute the new permutation.
    auto inv_dim_order = InversePermutation(dim_order);
    plan->lda_.reserve(ndim);
    plan->a_dims_.reserve(ndim);
    plan->permutation_.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      plan->lda_.push_back(input_strides_in_bytes->at(dim_order[i]));
      plan->a_dims_.push_back(dims[dim_order[i]]);
      plan->permutation_.push_back(inv_dim_order[permutation[i]]);
    }
    plan->original_a_dims_.resize(ndim);
    absl::c_copy(dims, plan->original_a_dims_.begin());
  } else {
    plan->original_a_dims_.resize(ndim);
    absl::c_copy(dims, plan->original_a_dims_.begin());
    plan->a_dims_.resize(ndim);
    absl::c_copy(dims, plan->a_dims_.begin());
    plan->permutation_.resize(ndim);
    absl::c_copy(permutation, plan->permutation_.begin());
    ComputeStrides(plan->elem_size_in_bytes_, plan->a_dims_, plan->lda_);
    plan->original_a_strides_ = plan->lda_;
  }
  permutation = {};

  RemoveTrivialDimensions(plan->a_dims_, plan->permutation_, plan->lda_);
  CoalesceDimensions(plan->a_dims_, plan->permutation_, plan->lda_);

  // If the plan is 0-dimensional, or the innermost dimension is not of stride
  // 1, adds a trivial size 1 dimension. The transpose kernels rely on the
  // presence of a 1-element-stride innermost dimension in the input.
  if (plan->lda_.empty() || plan->lda_.back() != plan->elem_size_in_bytes_) {
    plan->permutation_.push_back(plan->a_dims_.size());
    plan->a_dims_.push_back(1);
    plan->lda_.push_back(elem_size_in_bytes);
  }

  ndim = static_cast<int>(plan->a_dims_.size());

  plan->b_dims_ = Permute(plan->a_dims_, plan->permutation_);
  ComputeStrides(plan->elem_size_in_bytes_, plan->b_dims_, plan->ldb_);

  // permutation maps dimensions of b to a
  // inverse_permutation maps dimensions of a to b
  auto inverse_permutation = InversePermutation(plan->permutation_);

  plan->loop_order_.reserve(ndim);
  // TODO(phawkins): pick a good loop order.
  for (int i = 0; i < ndim; ++i) {
    plan->loop_order_.push_back(i);
  }

  plan->inner_kernel_is_memcpy_ = plan->permutation_[ndim - 1] == ndim - 1;
  if (plan->inner_kernel_is_memcpy_) {
    // The stride-1 loop must be innermost.
    CHECK_EQ(plan->loop_order_.back(), ndim - 1);
  } else {
    switch (plan->elem_size_in_bytes_) {
      case 1:
        plan->inner_block_elems_ = 16;
        break;
      case 2:
        plan->inner_block_elems_ = 8;
        break;
      case 4:
        plan->inner_block_elems_ = 8;
        break;
      case 8:
        plan->inner_block_elems_ = 4;
        break;
      case 16:
        plan->inner_block_elems_ = 4;
        break;
      default:
        return Unimplemented("Unimplemented element size %d",
                             plan->elem_size_in_bytes_);
    }
  }

  // Bound the block sizes so they are smaller than the stride-1 dimension size.
  int64_t a_stride1_size = plan->a_dims_.back();
  int64_t b_stride1_size = plan->a_dims_[plan->permutation_.back()];
  while (plan->inner_block_elems_ > std::min(a_stride1_size, b_stride1_size)) {
    plan->inner_block_elems_ /= 2;
    plan->outer_block_elems_a_ *= 2;
    plan->outer_block_elems_b_ *= 2;
  }
  while (plan->outer_block_elems_a_ > 1 &&
         plan->inner_block_elems_ * plan->outer_block_elems_a_ >
             a_stride1_size) {
    plan->outer_block_elems_a_ /= 2;
  }
  while (plan->outer_block_elems_b_ > 1 &&
         plan->inner_block_elems_ * plan->outer_block_elems_b_ >
             b_stride1_size) {
    plan->outer_block_elems_b_ /= 2;
  }
  plan->root_node_ = plan->BuildPlanNode(inverse_permutation, 0);
  VLOG(5) << plan->ToString();
  return plan;
}

template <typename T>
void TransposePlan::ExecuteTyped(const char* a, char* b) const {
  if (inner_kernel_is_memcpy_) {
    TransposeConstStride1<T>(a, b, root_node_);
  } else {
    switch (inner_block_elems_) {
      case 1:
        Transpose<T, 1>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                        root_node_);
        break;
      case 2:
        Transpose<T, 2>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                        root_node_);
        break;
      case 4:
        Transpose<T, 4>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                        root_node_);
        break;
      case 8:
        Transpose<T, 8>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                        root_node_);
        break;
      case 16:
        Transpose<T, 16>(a, outer_block_elems_a_, b, outer_block_elems_b_,
                         root_node_);
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

void TransposePlan::Execute(const void* a, void* b) const {
  if (num_elems_ == 0) {
    return;
  }
  DCHECK((static_cast<const char*>(a) + elem_size_in_bytes_ * num_elems_ <= b ||
          static_cast<const char*>(b) + elem_size_in_bytes_ * num_elems_ <= a));
  switch (elem_size_in_bytes_) {
    case 1:
      ExecuteTyped<uint8_t>(static_cast<const char*>(a), static_cast<char*>(b));
      break;
    case 2:
      ExecuteTyped<uint16_t>(static_cast<const char*>(a),
                             static_cast<char*>(b));
      break;
    case 4:
      ExecuteTyped<uint32_t>(static_cast<const char*>(a),
                             static_cast<char*>(b));
      break;
    case 8:
      ExecuteTyped<uint64_t>(static_cast<const char*>(a),
                             static_cast<char*>(b));
      break;
    case 16:
      ExecuteTyped<uint128>(static_cast<const char*>(a), static_cast<char*>(b));
      break;
    default:
      LOG(FATAL) << "Unimplemented element size " << elem_size_in_bytes_;
  }
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
  std::string nodes;
  PrintPlan(root_node_, /*indent=*/2, &nodes);
  return absl::StrFormat(
      "a_dims=%s b_dims=%s permutation=%s lda=%s ldb=%s loop_order=%s "
      "outer_bs=[%d,%d] inner_bs=%d "
      "nodes:\n%s",
      absl::StrJoin(a_dims_, ","),
      absl::StrJoin(Permute(a_dims_, permutation_), ","),
      absl::StrJoin(permutation_, ","), absl::StrJoin(lda_, ","),
      absl::StrJoin(ldb_, ","), absl::StrJoin(loop_order_, ","),
      outer_block_elems_a_, outer_block_elems_b_, inner_block_elems_, nodes);
}

struct TransposePlanCacheKey {
  size_t elem_size_in_bytes;
  absl::InlinedVector<int64_t, 4> dims;
  absl::InlinedVector<int64_t, 4> permutation;
  absl::optional<absl::InlinedVector<int64_t, 4>> input_strides_in_bytes;

  bool operator==(const TransposePlanCacheKey& other) const;
};

bool TransposePlanCacheKey::operator==(
    const TransposePlanCacheKey& other) const {
  return elem_size_in_bytes == other.elem_size_in_bytes && dims == other.dims &&
         permutation == other.permutation &&
         input_strides_in_bytes == other.input_strides_in_bytes;
}

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key) {
  h = H::combine(std::move(h), key.elem_size_in_bytes);
  h = H::combine_contiguous(std::move(h), key.dims.data(), key.dims.size());
  h = H::combine_contiguous(std::move(h), key.permutation.data(),
                            key.permutation.size());
  if (key.input_strides_in_bytes) {
    h = H::combine_contiguous(std::move(h), key.input_strides_in_bytes->data(),
                              key.input_strides_in_bytes->size());
  }
  return h;
}

TransposePlanCache::TransposePlanCache(int capacity)
    : lru_list_(capacity), cache_(&lru_list_) {}

TransposePlanCache::~TransposePlanCache() = default;

StatusOr<std::shared_ptr<TransposePlan>> TransposePlanCache::GetOrCreate(
    size_t elem_size_in_bytes, absl::Span<int64_t const> dims,
    absl::Span<int64_t const> permutation,
    absl::optional<absl::Span<int64_t const>> input_strides_in_bytes) {
  TransposePlanCacheKey key;
  key.elem_size_in_bytes = elem_size_in_bytes;
  key.dims.resize(dims.size());
  absl::c_copy(dims, key.dims.begin());
  key.permutation.resize(permutation.size());
  absl::c_copy(permutation, key.permutation.begin());
  if (input_strides_in_bytes) {
    key.input_strides_in_bytes = absl::InlinedVector<int64_t, 4>(
        input_strides_in_bytes->begin(), input_strides_in_bytes->end());
  }
  return cache_.GetOrCreateIfAbsent(
      key,
      [&](const TransposePlanCacheKey& key)
          -> StatusOr<std::shared_ptr<TransposePlan>> {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<TransposePlan> plan,
            TransposePlan::Create(elem_size_in_bytes, dims, permutation,
                                  input_strides_in_bytes));
        return std::shared_ptr<TransposePlan>(std::move(plan));
      });
}

}  // namespace xla
