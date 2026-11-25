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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_PLANNER_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_PLANNER_H_

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/libraries/cub/scratch_bump_allocator.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Logical buffer role used by radix sort ping-pong passes.
// Inputs (kIn) are never overwritten; the final pass always targets kOut.
enum class CubSortBufferRole {
  kIn,
  kOut,
  kTmp,
};

// Byte range [offset, offset + size) inside the CUB sort scratch buffer.
using CubSortScratchRange = ScratchRange;

// Launch chunk for CubRadixSortSingleTile{Keys,Pairs}Kernel.
struct CubSortSingleTileLaunch {
  int64_t segment_offset = 0;
  int32_t grid_x = 0;
};

// Single launch step of CubRadixSortOnesweep{Keys,Pairs}Kernel.
struct CubSortOnesweepPassLaunch {
  int pass = 0;
  int64_t portion = 0;
  CubSortBufferRole src_role = CubSortBufferRole::kIn;
  CubSortBufferRole dst_role = CubSortBufferRole::kOut;
  CubSortScratchRange bins_in;
  // Empty on the final portion of each pass (kernel receives nullptr).
  std::optional<CubSortScratchRange> bins_out;
  // Exact slice of the lookback buffer to zero and bind for this portion.
  CubSortScratchRange lookback;
  // Disjoint 256-byte-aligned slot inside `ctrs_all`, zeroed once by
  // ExclusiveSum before the Onesweep passes.
  CubSortScratchRange ctr;
  int64_t portion_item_offset = 0;
  int32_t portion_num_items = 0;
  int32_t grid_x = 0;
  int32_t current_bit = 0;
  int32_t num_bits = 0;
};

// Single launch step of CubUniformSegmentedRadixSort{Keys,Pairs}Kernel.
struct CubSortSegmentedPassLaunch {
  int pass = 0;
  bool use_alt_policy = false;
  CubSortBufferRole src_role = CubSortBufferRole::kIn;
  CubSortBufferRole dst_role = CubSortBufferRole::kOut;
  int64_t segment_offset = 0;
  int32_t grid_x = 0;
  int32_t current_bit = 0;
  int32_t pass_bits = 0;
};

// Execution plan for single-tile radix sort (unbatched or batched sorts whose
// segment fits in a single thread-block tile; requires 0 scratch bytes).
struct CubSortSingleTilePlan {
  std::vector<CubSortSingleTileLaunch> launches;
};

// Execution and scratch-buffer plan for unbatched multi-tile Onesweep radix
// sort: Histogram -> ExclusiveSum -> multi-pass / multi-portion scatter.
struct CubSortOnesweepPlan {
  int num_passes = 0;
  int64_t num_portions = 0;
  int32_t portion_size = 0;

  // Scratch sub-allocations (256-byte aligned; `keys_tmp`/`vals_tmp` are empty
  // when `num_passes <= 1` or for keys-only sorts).
  CubSortScratchRange bins_raw;
  CubSortScratchRange bins_portion0;
  CubSortScratchRange lookback_max;
  CubSortScratchRange ctrs_all;
  CubSortScratchRange keys_tmp;
  CubSortScratchRange vals_tmp;

  std::vector<CubSortOnesweepPassLaunch> launches;
};

// Execution and scratch-buffer plan for batched multi-pass segmented radix
// sort (`batch_size > 1` and `segment_size > single_tile.tile_items()`).
struct CubSortSegmentedMultiPassPlan {
  int num_passes = 0;

  // Scratch sub-allocations (256-byte aligned; `vals_tmp` is empty for
  // keys-only sorts).
  CubSortScratchRange keys_tmp;
  CubSortScratchRange vals_tmp;

  std::vector<CubSortSegmentedPassLaunch> launches;
};

// Complete pure-C++ execution and scratch-buffer plan for a CUDA CUB radix
// sort. Shared by both compile-time scratch estimation and thunk emission so
// buffer sizing and kernel launches can never drift.
struct CubSortExecutionPlan {
  CubRadixSortPolicySpec policy{};
  PrimitiveType key_type = PRIMITIVE_TYPE_INVALID;
  std::optional<PrimitiveType> value_type;
  int64_t num_items = 0;
  int64_t batch_size = 1;
  int64_t segment_size = 0;

  // Total scratch bytes required (0 for `CubSortSingleTilePlan`).
  int64_t scratch_bytes = 0;

  std::variant<CubSortSingleTilePlan, CubSortOnesweepPlan,
               CubSortSegmentedMultiPassPlan>
      mode_plan;
};

// Optional overrides intended for unit tests exercising multi-portion Onesweep
// chaining or multi-launch grid-X chunking on small inputs:
// - `portion_size_override`, if provided, is validated for all modes and must
//   be in `[onesweep.tile_items(), kCubSortMaxPortionLimit]` and an exact
//   multiple of `onesweep.tile_items()`.
// - `max_grid_x_override`, if provided, must be in `[1, INT32_MAX]` (used by
//   `CubSortSingleTilePlan` and `CubSortSegmentedMultiPassPlan`; ignored by
//   `CubSortOnesweepPlan`).
struct CubSortPlanOptions {
  std::optional<int32_t> portion_size_override;
  std::optional<int32_t> max_grid_x_override;
};

// Plans mode selection, scratch-buffer layout, and kernel launches for a CUDA
// CUB radix sort without requiring a live GPU or CUDA toolkit headers.
//
// Requires `num_items > 0`, `batch_size > 0`, `num_items % batch_size == 0`,
// and `num_items / batch_size >= 2` (empty and single-element sorts are
// rejected by `SortRewriter` and folded away by HLO simplification passes).
absl::StatusOr<CubSortExecutionPlan> PlanCubSortExecution(
    const stream_executor::CudaComputeCapability& cc, PrimitiveType key_type,
    std::optional<PrimitiveType> value_type, int64_t num_items,
    int64_t batch_size, CubSortPlanOptions options = {});

// Computes the exact scratch-buffer size in bytes needed for a CUDA CUB radix
// sort. Returns 0 when no scratch memory is needed (single-tile sorts); callers
// constructing an HLO `u8[N]` tuple leaf may clamp to
// `std::max<int64_t>(1, scratch_bytes)` if a non-zero buffer is desired.
absl::StatusOr<int64_t> ComputeCudaCubSortScratchSize(
    const stream_executor::CudaComputeCapability& cc, PrimitiveType key_type,
    std::optional<PrimitiveType> value_type, int64_t num_items,
    int64_t batch_size);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_PLANNER_H_
