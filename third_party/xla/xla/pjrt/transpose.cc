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
// we repeatedly halve the kernel size for trailing elements. Trailing tiles
// smaller than the primary microkernel size are recursively halved into smaller
// power of two SIMD microkernels (down to at least 4 elements and 16 byte rows)
// before falling back to a scalar transpose for any remaining elements.
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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/permutation_util.h"
#include "xla/pjrt/transpose_kernels.h"
#include "xla/pjrt/transpose_plan.h"
#include "xla/util.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

namespace {

static_assert(static_cast<int>(TransposePlan::Transformation::kNone) ==
              static_cast<int>(transpose_internal::Transformation::kNone));
static_assert(static_cast<int>(TransposePlan::Transformation::kF64ToEf57) ==
              static_cast<int>(transpose_internal::Transformation::kF64ToEf57));
static_assert(
    static_cast<int>(TransposePlan::Transformation::kPackSubbyte) ==
    static_cast<int>(transpose_internal::Transformation::kPackSubbyte));

static_assert(static_cast<int>(TransposePlan::ChunkContiguity::kNone) ==
              static_cast<int>(transpose_internal::ChunkContiguity::kNone));
static_assert(static_cast<int>(TransposePlan::ChunkContiguity::kInput) ==
              static_cast<int>(transpose_internal::ChunkContiguity::kInput));
static_assert(static_cast<int>(TransposePlan::ChunkContiguity::kOutput) ==
              static_cast<int>(transpose_internal::ChunkContiguity::kOutput));

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

absl::StatusOr<std::unique_ptr<TransposePlan>> TransposePlan::Create(
    Options o) {
  if (o.input_layout.has_value()) {
    if (const auto* t = std::get_if<Tiling>(&*o.input_layout)) {
      o.input_tiling = *t;
    } else if (const auto* s = std::get_if<Striding>(&*o.input_layout)) {
      o.input_striding = *s;
    }
  }

  transpose_internal::PlanOptions plan_options;
  plan_options.elem_size_in_bytes = o.elem_size_in_bytes;
  plan_options.dims = o.dims;
  plan_options.permutation = o.permutation;
  if (o.input_tiling.has_value()) {
    plan_options.input_tiling = o.input_tiling->tiling;
  }
  if (o.input_striding.has_value()) {
    plan_options.input_striding = o.input_striding->strides_in_bytes;
  }
  if (o.output_tiling.has_value()) {
    plan_options.output_tiling = o.output_tiling->tiling;
  }
  plan_options.transformation =
      static_cast<transpose_internal::Transformation>(o.transformation);
  plan_options.dest_bits_per_element = o.dest_bits_per_element;
  plan_options.num_threads = o.num_threads;
  plan_options.chunk_contiguity =
      static_cast<transpose_internal::ChunkContiguity>(o.chunk_contiguity);
  plan_options.tile_padding = o.tile_padding;

  ABSL_ASSIGN_OR_RETURN(transpose_internal::FlattenedPlan flattened,
                   transpose_internal::BuildTransposePlan(plan_options));

  auto plan = std::unique_ptr<TransposePlan>(new TransposePlan());
  plan->elem_size_in_bytes_ = flattened.elem_size_in_bytes;
  plan->num_elems_ = flattened.num_elems;
  plan->input_num_elems_ = flattened.input_num_elems;
  plan->output_num_elems_ = flattened.output_num_elems;
  plan->original_a_dims_.assign(o.dims.begin(), o.dims.end());
  if (o.input_striding.has_value()) {
    plan->original_a_strides_.assign(o.input_striding->strides_in_bytes.begin(),
                                     o.input_striding->strides_in_bytes.end());
  }
  plan->original_b_dims_ = Permute(o.dims, o.permutation);
  plan->input_chunk_offset_bytes_ =
      std::move(flattened.input_chunk_offset_bytes);
  plan->output_chunk_offset_bytes_ =
      std::move(flattened.output_chunk_offset_bytes);
  plan->input_chunk_iteration_offsets_ =
      std::move(flattened.input_chunk_iteration_offsets);
  plan->output_chunk_iteration_offsets_ =
      std::move(flattened.output_chunk_iteration_offsets);
  plan->input_chunk_size_bytes_ = std::move(flattened.input_chunk_size_bytes);
  plan->output_chunk_size_bytes_ = std::move(flattened.output_chunk_size_bytes);
  plan->max_input_chunk_size_bytes_ = flattened.max_input_chunk_size_bytes;
  plan->max_output_chunk_size_bytes_ = flattened.max_output_chunk_size_bytes;
  plan->chunk_ops_ = std::move(flattened.chunk_ops);
  plan->scratch_size_ = flattened.scratch_size;
  plan->bits_per_element_ = flattened.bits_per_element;
  plan->use_fallback_pack_ = flattened.use_fallback_pack;
  plan->inner_kernel_is_memcpy_ = flattened.inner_kernel_is_memcpy;
  plan->tile_padding_ = flattened.tile_padding;
  return plan;
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

}  // namespace xla
