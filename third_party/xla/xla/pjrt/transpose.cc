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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/ef57.h"
#include "xla/permutation_util.h"
#include "xla/pjrt/transpose_kernels.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

namespace {
#ifdef __AVX__
static constexpr int kMaxInnerBlockSizeBytes = sizeof(__m256i);
#elif defined(XLA_HAS_VEC128)
static constexpr int kMaxInnerBlockSizeBytes = sizeof(Vec128);
#else
static constexpr int kMaxInnerBlockSizeBytes = 16;
#endif
}  // namespace

// A plan is a data structure that describes a loop nest.
// TODO(phawkins): consider shrinking Node so it fits in a cache line.
struct TransposePlan::Node {
  // The loop should iterate over the index space range(0, end, inc).
  // These fields are ignored by the macrokernel.
  int64_t end;  // For the inner loop of a memcpy loop nest, this is the size of
                // the transfer.
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

template <typename T, int inner_bs,
          TransposePlan::Transformation transformation>
void MacroKernel(const char* __restrict a, int64_t lda, int outer_bs_a,
                 char* __restrict b, int64_t ldb, int outer_bs_b,
                 void* __restrict scratch) {
  DVLOG(10) << "MacroKernel lda=" << lda << " ldb=" << ldb
            << " outer_bs_a=" << outer_bs_a << " outer_bs_b=" << outer_bs_b
            << " inner_bs=" << inner_bs;

  // TODO(phawkins): consider adding prefetching and streaming stores.

  if constexpr (transformation == TransposePlan::Transformation::kF64ToEf57) {
    DCHECK_EQ(outer_bs_a * inner_bs % 2, 0);
    float* p = reinterpret_cast<float*>(scratch);
    if (ABSL_PREDICT_TRUE(lda == sizeof(double) &&
                          outer_bs_a * inner_bs == 2)) {
      absl::Span<const double> input = absl::MakeConstSpan(
          reinterpret_cast<const double*>(a), outer_bs_b * inner_bs);
      absl::Span<float> output =
          absl::MakeSpan(reinterpret_cast<float*>(p), input.size() * 2);
      ConvertF64ToEf57(input, output);
    } else {
      for (int i = 0; i < outer_bs_b * inner_bs; ++i) {
        absl::Span<const double> input =
            absl::MakeConstSpan(reinterpret_cast<const double*>(a + lda * i),
                                outer_bs_a * inner_bs / 2);
        absl::Span<float> output = absl::MakeSpan(
            reinterpret_cast<float*>(p + outer_bs_a * inner_bs * i),
            input.size() * 2);
        ConvertF64ToEf57(input, output);
      }
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
  tsl::profiler::TraceMe traceme([&]() {
    return tsl::profiler::TraceMeEncode("Transpose",
                                        {{"inner_bs", inner_bs},
                                         {"outer_bs_a", outer_bs_a},
                                         {"outer_bs_b", outer_bs_b}});
  });
  DVLOG(10) << "Transpose " << outer_bs_a << " " << outer_bs_b;
  DCHECK_GT(outer_bs_a, 0);
  DCHECK_GT(outer_bs_b, 0);
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
    for (i = 0; i < stop; i += inc) {
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
    for (i = 0; i < stop; i += inc) {
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

void TransposeConstStride1(const char* __restrict a, char* __restrict b,
                           TransposePlan::Node const* __restrict node) {
  if (node[0].is_inner_dim_in_a) {
    int64_t num_bytes = node->end;
    std::memcpy(b, a, num_bytes);
  } else if (node[1].is_inner_dim_in_a) {
    int64_t num_bytes = node[1].end;
    for (int64_t i = 0; i < node[0].end; ++i) {
      std::memcpy(b, a, num_bytes);
      a += node[0].lda;
      b += node[0].ldb;
    }
    if (node[0].trailing_tile_next_node_inc) {
      TransposeConstStride1(a, b, node + node[0].trailing_tile_next_node_inc);
    }
  } else if (node[2].is_inner_dim_in_a) {
    int64_t num_bytes = node[2].end;
    for (int64_t i = 0; i < node[0].end; ++i) {
      const char* a1 = a;
      char* b1 = b;
      for (int64_t j = 0; j < node[1].end; ++j) {
        std::memcpy(b1, a1, num_bytes);
        a1 += node[1].lda;
        b1 += node[1].ldb;
      }
      if (node[1].trailing_tile_next_node_inc) {
        TransposeConstStride1(a1, b1,
                              &node[1] + node[1].trailing_tile_next_node_inc);
      }
      a += node[0].lda;
      b += node[0].ldb;
    }
    if (node[0].trailing_tile_next_node_inc) {
      TransposeConstStride1(a, b, node + node[0].trailing_tile_next_node_inc);
    }
  } else {
    for (int64_t i = 0; i < node[0].end; ++i) {
      const char* a1 = a;
      char* b1 = b;
      for (int64_t j = 0; j < node[1].end; ++j) {
        TransposeConstStride1(a1, b1, node + 2);
        a1 += node[1].lda;
        b1 += node[1].ldb;
      }
      if (node[1].trailing_tile_next_node_inc) {
        TransposeConstStride1(a1, b1,
                              &node[1] + node[1].trailing_tile_next_node_inc);
      }
      a += node[0].lda;
      b += node[0].ldb;
    }
    if (node[0].trailing_tile_next_node_inc) {
      TransposeConstStride1(a, b, node + node[0].trailing_tile_next_node_inc);
    }
  }
}

template <typename T, TransposePlan::Transformation transformation>
void TransposePlan::ExecuteTyped(const char* a, char* b,
                                 absl::Span<Node const> nodes) const {
  tsl::profiler::TraceMe traceme([&]() {
    return tsl::profiler::TraceMeEncode(
        "TransposePlan::ExecuteTyped",
        {{"inner_kernel_is_memcpy", inner_kernel_is_memcpy_},
         {"inner_block_elems", inner_block_elems_}});
  });

  CHECK(!inner_kernel_is_memcpy_);
  std::unique_ptr<char[]> scratch;
  if (scratch_size_ > 0) {
    scratch.reset(new char[scratch_size_]);
  }
  DCHECK_LE(sizeof(T) * inner_block_elems_, kMaxInnerBlockSizeBytes);
  auto handle_inner_block_elems = [&](auto const_inner_block_elems) {
    if (nodes.size() > 1) {
      Transpose<T, const_inner_block_elems, transformation>(
          a, outer_block_elems_a_, b, outer_block_elems_b_, nodes.data(),
          scratch.get());
    } else {
      MacroKernel<T, const_inner_block_elems, transformation>(
          a, nodes.back().lda, outer_block_elems_a_, b, nodes.back().ldb,
          outer_block_elems_b_, scratch.get());
    }
  };
  switch (inner_block_elems_) {
    case 1:
      handle_inner_block_elems(std::integral_constant<int, 1>{});
      break;
    case 2:
      handle_inner_block_elems(std::integral_constant<int, 2>{});
      break;
    case 4:
      handle_inner_block_elems(std::integral_constant<int, 4>{});
      break;
    case 8:
      handle_inner_block_elems(std::integral_constant<int, 8>{});
      break;
    case 16:
      handle_inner_block_elems(std::integral_constant<int, 16>{});
      break;
    case 32:
      handle_inner_block_elems(std::integral_constant<int, 32>{});
      break;
    default:
      LOG(FATAL) << "Invalid inner_block_elems_ " << inner_block_elems_;
  }
}

struct uint128 {
  uint64_t lo;
  uint64_t hi;
};
static_assert(sizeof(uint128) == 16, "uint128 should be 16 bytes in size");

void TransposePlan::ExecuteChunk(int chunk_id, const void* a, void* b,
                                 bool input_is_global,
                                 bool output_is_global) const {
  if (num_elems_ == 0) {
    return;
  }
  tsl::profiler::TraceMe traceme("Transpose::ExecuteChunk", /*level=*/2);

  absl::Span<Node const> nodes = nodes_[chunk_id];
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

  if (inner_kernel_is_memcpy_) {
    DCHECK(transformation_ == Transformation::kNone);
    // Memcpy-based plans all assume element size 1 (i.e., bytes).
    TransposeConstStride1(ac, bc, nodes.data());
    return;
  }

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
}

void TransposePlan::Execute(
    const void* a, void* b,
    std::optional<absl::FunctionRef<void(std::function<void(void)>)>>
        schedule_work) const {
  if (num_elems_ == 0) {
    return;
  }
  tsl::profiler::TraceMe traceme("Transpose::Execute", /*level=*/2);

  if (!schedule_work || Parallelism() <= 1) {
    for (int i = 0; i < Parallelism(); ++i) {
      ExecuteChunk(i, a, b, /*input_is_global=*/true,
                   /*output_is_global=*/true);
    }
  } else {
    absl::BlockingCounter counter(Parallelism() - 1);
    for (size_t i = 1; i < nodes_.size(); ++i) {
      (*schedule_work)([&, i]() {
        ExecuteChunk(i, a, b, /*input_is_global=*/true,
                     /*output_is_global=*/true);
        counter.DecrementCount();
      });
    }
    ExecuteChunk(0, a, b, /*input_is_global=*/true,
                 /*output_is_global=*/true);
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

// Helper function that builds a plan.
void TransposePlan::BuildPlanNodes(int chunk_id,
                                   std::vector<TransposePlan::Node>& nodes) {
  const int ndim = a_dims_.size();
  DCHECK_GT(ndim, 0);

  // Use the pre-computed chunk loops which have start/end bounds already set.
  absl::Span<const Loop> chunk_loops = chunk_loops_[chunk_id];

  // We build plans in a depth-first order, visiting loops from outermost to
  // innermost. We use a stack (depth-first) order to handle trailing partial
  // tiles, which we "come back to" after handling the non-trailing case.
  struct Agendum {
    // The ID of the loop to visit in chunk_loops.
    int loop_id;
    // The parent node ID whose trailing tile should be made to point to this
    // node.
    int parent_node_id;

    // For which dimensions of `a` are we to visit the partial trailing tile
    // a loop that visits that tile's interior?
    absl::InlinedVector<bool, 4> partial_tiles;
  };
  std::stack<Agendum> agenda;

  agenda.push(Agendum{/*loop_id=*/0, /*parent_node_id=*/-1,
                      absl::InlinedVector<bool, 4>(ndim, false)});

  auto loop_has_trivial_iteration_space = [](const Node& node) {
    return node.inc == node.end;
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

    if (agendum.loop_id == chunk_loops.size()) {
      // We've reached the end of the loop nest.
      // Transpose loops have a sentinel node, indicated by a negative `inc`
      // value, that describes the striding of the inner transpose kernel.
      if (!inner_kernel_is_memcpy_) {
        Node node;
        node.end = node.inc = -1;
        node.lda = sentinel_lda_;
        node.ldb = sentinel_ldb_;
        nodes.push_back(node);
      }
      DCHECK(!(inner_kernel_is_memcpy_ && agendum.parent_node_id >= 0));
      continue;
    }

    const Loop& loop = chunk_loops[agendum.loop_id];
    int a_dim = loop.dim_in_a;

    Node node;
    node.lda = loop.lda;
    node.ldb = loop.ldb;
    node.inc = loop.inc;
    node.is_inner_dim_in_a = loop.is_inner_dim_in_a;
    node.is_inner_dim_in_b = loop.is_inner_dim_in_b;

    if (loop.tile_interior) {
      // We are visiting the tile interior of a tiled dimension.
      bool partial = agendum.partial_tiles[a_dim];

      int64_t size = partial ? loop.dim_size % loop.tile_size : loop.tile_size;
      int64_t actual_start = std::max<int64_t>(0, loop.start);
      int64_t actual_end = std::min<int64_t>(size, loop.end);
      node.end = std::max<int64_t>(0, actual_end - actual_start);

      if (node.is_inner_dim_in_a && inner_kernel_is_memcpy_) {
        node.end *= elem_size_in_bytes_;
      }

      if (!loop_has_trivial_iteration_space(node) ||
          (inner_kernel_is_memcpy_ && node.is_inner_dim_in_a)) {
        nodes.push_back(node);
      }
      Agendum new_agendum;
      new_agendum.loop_id = agendum.loop_id + 1;
      new_agendum.parent_node_id = -1;
      new_agendum.partial_tiles = agendum.partial_tiles;
      agenda.push(std::move(new_agendum));
    } else {
      // We are either visiting an untiled dimension, or the loop that iterates
      // over tile exteriors.
      int64_t num_complete_tiles = loop.dim_size / loop.tile_size;
      bool has_partial_tile = (loop.dim_size % loop.tile_size != 0);

      // If there is a trailing partial tile as well as complete tiles, handle
      // it as a trailer on the loop over complete tiles.
      // A chunk is responsible for the trailing tile if its loop.end covers
      // the full dimension.
      int64_t full_size = CeilOfRatio(loop.dim_size, loop.tile_size);
      bool handles_trailing =
          loop.end >= full_size && loop.start <= num_complete_tiles;
      bool has_trailing_plan_node = false;
      if (num_complete_tiles > 0 && has_partial_tile && handles_trailing) {
        Agendum new_agendum;
        new_agendum.loop_id = agendum.loop_id + 1;
        new_agendum.parent_node_id = node_id;
        new_agendum.partial_tiles = agendum.partial_tiles;
        new_agendum.partial_tiles[a_dim] = true;
        agenda.push(std::move(new_agendum));
        has_trailing_plan_node = true;
      }

      // If this tiled dimension consists only of a single partial tile, handle
      // it here; there's no point emitting a degenerate loop and a separate
      // path to handle the trailing tile.
      bool partial = num_complete_tiles == 0 && has_partial_tile;

      // loop.start and loop.end are in tile units.
      int64_t num_tiles = partial ? 1 : num_complete_tiles;
      node.end = std::max<int64_t>(
          0, std::min<int64_t>(num_tiles, loop.end) - loop.start);

      if (node.is_inner_dim_in_a && inner_kernel_is_memcpy_) {
        node.end *= elem_size_in_bytes_;
      }

      // If this loop has a trivial iteration space, drop it.
      if (!loop_has_trivial_iteration_space(node) ||
          (inner_kernel_is_memcpy_ && node.is_inner_dim_in_a) ||
          has_trailing_plan_node) {
        nodes.push_back(node);
      }
      Agendum new_agendum;
      new_agendum.loop_id = agendum.loop_id + 1;
      new_agendum.parent_node_id = -1;
      new_agendum.partial_tiles = agendum.partial_tiles;
      new_agendum.partial_tiles[a_dim] = partial;
      agenda.push(std::move(new_agendum));
    }
  }
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

  TF_RETURN_IF_ERROR(ParseTilingSpecification(
      ndim, o.output_tiling,
      /*nonstandard_layout=*/output_contiguity, plan->b_tiling_));

  // Temporary vectors to hold un-permuted attributes
  absl::InlinedVector<int64_t, 4> temp_lda, temp_lda_tile, temp_a_tiling;

  // Parse the tile and stride specifications.
  TF_RETURN_IF_ERROR(ParseTilingSpecification(
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

  plan->transformation_ = o.transformation;
  switch (o.transformation) {
    case Transformation::kNone:
      break;
    case Transformation::kF64ToEf57:
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
  }
  plan->chunk_contiguity_ = o.chunk_contiguity;

  plan->Initialize();
  return plan;
}

void TransposePlan::Initialize() {
  if (num_elems_ == 0) {
    return;
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

  const int pos_stride1b_in_a = permutation_[pos_stride1b_in_b];

  int64_t stride_pos1a =
      inner_stride(pos_stride1a_in_a, lda_, lda_tile_, a_tiling_);
  int64_t stride_pos1b =
      inner_stride(pos_stride1b_in_b, ldb_, ldb_tile_, b_tiling_);

  inner_kernel_is_memcpy_ = (pos_stride1b_in_a == pos_stride1a_in_a) &&
                            (stride_pos1a == elem_size_in_bytes_) &&
                            (stride_pos1b == elem_size_in_bytes_);

  // Calculate sentinel strides.
  if (!inner_kernel_is_memcpy_) {
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

  constexpr int kMaxOuterBlockElems = 16;
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
      case 2:
      case 4:
      case 8:
        min_inner_block_elems = 1;
        max_inner_block_elems = std::min<int>(
            kMaxOuterBlockElems, kMaxInnerBlockSizeBytes / elem_size_in_bytes_);
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
        std::min<int64_t>(kMaxOuterBlockElems, a_stride1_size),
        inner_block_elems_);
    outer_block_elems_a_ = std::max<int64_t>(outer_block_elems_a_, 1);
    outer_block_elems_b_ = FloorOfRatio<int64_t>(
        std::min<int64_t>(kMaxOuterBlockElems, b_stride1_size),
        inner_block_elems_);
    outer_block_elems_b_ = std::max<int64_t>(outer_block_elems_b_, 1);
  }

  // Identify contiguous loops for chunk scheduling.
  // Loops with smallest strides in the contiguous buffer form the contiguous
  // region. Accumulate loops until their combined iteration count reaches
  // the target chunk size.
  IdentifyContiguousLoops(loop_order);
  ChooseLoopOrder(loop_order);

  for (Loop& loop : loop_order) {
    if (!inner_kernel_is_memcpy_ &&
        (loop.tile_interior || loop.tile_size == 1)) {
      if (loop.is_inner_dim_in_a) {
        loop.inc = inner_block_elems_ * outer_block_elems_a_;
      } else if (loop.is_inner_dim_in_b) {
        loop.inc = inner_block_elems_ * outer_block_elems_b_;
      }
    }
  }

  // It is a required invariant of the loop order that tile interiors always
  // appear after the corresponding tile exterior. This is a consequence of the
  // heuristic above, because the tile interior must have smaller strides in
  // both input and output.

  // The stride-1 loop must be innermost for a memcpy loop.
  DCHECK(!inner_kernel_is_memcpy_ || loop_order.back().is_inner_dim_in_a)
      << ToString();

  int num_chunks = ChooseParallelizationStrategy(loop_order);
  VLOG(5) << "After ChooseParallelizationStrategy num_chunks=" << num_chunks
          << " loops: " << loops_to_string(loop_order);
  PartitionLoops(a_dims_.size(), num_chunks, loop_order, chunk_loops_,
                 input_chunk_iteration_offsets_,
                 output_chunk_iteration_offsets_);
  ComputeChunkSizes();
  VLOG(8) << "Before plan build: " << ToString();
  nodes_.resize(num_chunks);
  for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    BuildPlanNodes(chunk_id, nodes_[chunk_id]);
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

  VLOG(5) << "Final plan: " << ToString();
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

  auto soft_cost = [&](const Loop& l) {
    int64_t a_stride = std::abs(l.lda);
    if (!inner_kernel_is_memcpy_ && l.is_inner_dim_in_a) {
      a_stride *= inner_block_elems_ * outer_block_elems_a_;
    }
    int64_t b_stride = std::abs(l.ldb);
    if (!inner_kernel_is_memcpy_ && l.is_inner_dim_in_b) {
      b_stride *= inner_block_elems_ * outer_block_elems_b_;
    }

    double stride;
    switch (chunk_contiguity_) {
      case ChunkContiguity::kOutput:
        stride = b_stride;
        return std::make_tuple(0, -stride);
      case ChunkContiguity::kInput:
        stride = a_stride;
        return std::make_tuple(0, -stride);
      case ChunkContiguity::kNone:
        // Add a small penalty to the input strides: given the choice between
        // consecutive writes and consecutive reads, we would prefer consecutive
        // writes.
        constexpr double kPenalty = 1.01;
        stride = std::min<double>(a_stride * kPenalty, b_stride);
        return std::make_tuple(l.contiguity, -stride);
    }
  };

  while (!remaining.empty()) {
    int best_idx = -1;
    for (int i = 0; i < remaining.size(); ++i) {
      const Loop& l = remaining[i];

      // Hard constraint 1: memcpy kernel requirement.
      if (inner_kernel_is_memcpy_ && l.is_inner_dim_in_a &&
          remaining.size() > 1) {
        continue;
      }

      // Hard constraint 2: tile ordering.
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
                       ? std::min(loop.tile_size, loop.dim_size)
                       : (CeilOfRatio(loop.dim_size, loop.tile_size));
    return CeilOfRatio<int64_t>(size, loop.inc);
  };

  // Estimate the number of bytes each iteration of each loop processes.
  absl::InlinedVector<int64_t, 4> work_in_bytes(loop_order.size());
  int64_t acc = elem_size_in_bytes_;
  if (!inner_kernel_is_memcpy_) {
    acc *= inner_block_elems_ * inner_block_elems_ * outer_block_elems_a_ *
           outer_block_elems_b_;
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

    int kMinBytesPerThread = inner_kernel_is_memcpy_ ? (1 << 20) : (1 << 26);
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
  }
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
                      "Node(end=%d,inc=%d,lda=%"
                      "d,ldb=%d,next_trailing=%d,inner_a=%s,inner_b=%s)",
                      node.end, node.inc, node.lda, node.ldb,
                      node.trailing_tile_next_node_inc,
                      node.is_inner_dim_in_a ? "y" : "n",
                      node.is_inner_dim_in_b ? "y" : "n");
                }));
      });
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
  }
  return absl::StrFormat(
      "elem_size=%d a_dims=%s b_dims=%s permutation=%s a_tiling=%s b_tiling=%s "
      "lda=%s lda_tile=%s ldb=%s ldb_tile=%s "
      "outer_bs=[%d,%d] inner_bs=%d "
      "transformation=%s scratch_size=%d num_chunks_requested=%d\n"
      "chunk_loops:\n%s\n"
      "nodes:\n%s",
      elem_size_in_bytes_, absl::StrJoin(a_dims_, ","),
      absl::StrJoin(Permute(a_dims_, permutation_), ","),
      absl::StrJoin(permutation_, ","), absl::StrJoin(a_tiling_, ","),
      absl::StrJoin(b_tiling_, ","), absl::StrJoin(lda_, ","),
      absl::StrJoin(lda_tile_, ","), absl::StrJoin(ldb_, ","),
      absl::StrJoin(ldb_tile_, ","), outer_block_elems_a_, outer_block_elems_b_,
      inner_block_elems_, transformation_str, scratch_size_,
      num_chunks_requested_, chunk_loops_str, nodes_str);
}

bool TransposePlanCacheKey::operator==(
    const TransposePlanCacheKey& other) const {
  return elem_size_in_bytes == other.elem_size_in_bytes && dims == other.dims &&
         permutation == other.permutation &&
         input_tiling == other.input_tiling &&
         input_striding == other.input_striding &&
         output_tiling == other.output_tiling &&
         transformation == other.transformation &&
         num_threads == other.num_threads;
}

template <typename H>
H AbslHashValue(H h, const TransposePlanCacheKey& key) {
  return H::combine(std::move(h), key.elem_size_in_bytes, key.num_threads,
                    key.transformation, key.dims, key.permutation,
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
  key.transformation = o.transformation;
  key.num_threads = o.num_threads;
  return cache_.GetOrCreateIfAbsent(
      key,
      [&](const TransposePlanCacheKey& key)
          -> absl::StatusOr<std::shared_ptr<TransposePlan>> {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<TransposePlan> plan,
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
      if (stride == 0) {
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
