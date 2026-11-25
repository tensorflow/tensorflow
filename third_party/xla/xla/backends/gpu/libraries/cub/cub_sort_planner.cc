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

#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_types.h"
#include "xla/backends/gpu/libraries/cub/scratch_bump_allocator.h"
#include "xla/overflow_util.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int64_t kOnesweepBinsPerPassBytes =
    static_cast<int64_t>(kCubSortOnesweepRadixDigits) * sizeof(CubGlobalOffset);

absl::Status ValidateDimensions(int64_t num_items, int64_t batch_size,
                                int32_t max_grid_x) {
  if (num_items <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_items must be positive, got ", num_items));
  }
  if (batch_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be positive, got ", batch_size));
  }
  if (num_items % batch_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_items (", num_items,
                     ") must be divisible by batch_size (", batch_size, ")"));
  }
  const int64_t segment_size = num_items / batch_size;
  if (segment_size <= 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("segment_size (num_items / batch_size) must be >= 2, got ",
                     segment_size));
  }
  if (max_grid_x <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_grid_x_override must be positive, got ", max_grid_x));
  }
  return absl::OkStatus();
}

absl::Status ValidatePortionSizeOverride(
    std::optional<int32_t> portion_size_override, int32_t tile_items) {
  if (!portion_size_override.has_value()) {
    return absl::OkStatus();
  }
  const int32_t portion_size = *portion_size_override;
  if (portion_size < tile_items || portion_size > kCubSortMaxPortionLimit ||
      portion_size % tile_items != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("portion_size_override (", portion_size,
                     ") must be a positive multiple of onesweep.tile_items() (",
                     tile_items, ") and <= ", kCubSortMaxPortionLimit));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> MultiplyNonNegative(int64_t a, int64_t b) {
  const auto [product, overflow] = OverflowSafeMultiply(a, b);
  if (overflow) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Integer overflow or negative operand multiplying ", a, " * ", b));
  }
  return product;
}

// Multi-pass radix sort alternates ("ping-pongs") each pass between two
// working buffers (`kOut` and `kTmp`) so a pass never overwrites its own input.
// Pass 0 always reads `kIn`, and the first destination is chosen from the
// parity of `num_passes` so the final pass always writes directly to `kOut`
// without a trailing copy.
//
// Returns (src_role, dst_role) for `pass` in [0, num_passes) such that:
// - kIn is read only on pass 0 and never overwritten.
// - The final pass (pass == num_passes - 1) always writes kOut.
// - Every pass has src_role != dst_role.
std::pair<CubSortBufferRole, CubSortBufferRole> PingPongRoles(int pass,
                                                              int num_passes) {
  const int remaining = num_passes - 1 - pass;
  const CubSortBufferRole dst =
      (remaining % 2 == 0) ? CubSortBufferRole::kOut : CubSortBufferRole::kTmp;
  const CubSortBufferRole src =
      (pass == 0) ? CubSortBufferRole::kIn
                  : ((remaining % 2 == 0) ? CubSortBufferRole::kTmp
                                          : CubSortBufferRole::kOut);
  return {src, dst};
}

struct TempBuffers {
  CubSortScratchRange keys_tmp;
  CubSortScratchRange vals_tmp;
};

// Allocates ping-pong temporary key (and optional value) buffers in `alloc`
// when a sort requires more than one radix pass.
absl::StatusOr<TempBuffers> AllocateTempBuffers(ScratchBumpAllocator& alloc,
                                                int num_passes,
                                                int64_t num_items,
                                                int key_byte_size,
                                                int value_byte_size) {
  TempBuffers buffers;
  if (num_passes <= 1) {
    return buffers;
  }
  ABSL_ASSIGN_OR_RETURN(const int64_t keys_tmp_bytes,
                   MultiplyNonNegative(num_items, key_byte_size));
  ABSL_ASSIGN_OR_RETURN(buffers.keys_tmp, alloc.Allocate(keys_tmp_bytes));
  if (value_byte_size > 0) {
    ABSL_ASSIGN_OR_RETURN(const int64_t vals_tmp_bytes,
                     MultiplyNonNegative(num_items, value_byte_size));
    ABSL_ASSIGN_OR_RETURN(buffers.vals_tmp, alloc.Allocate(vals_tmp_bytes));
  }
  return buffers;
}

CubSortSingleTilePlan PlanSingleTileSort(int64_t batch_size,
                                         int32_t max_grid_x) {
  CubSortSingleTilePlan plan;
  for (int64_t offset = 0; offset < batch_size; offset += max_grid_x) {
    const int32_t chunk = static_cast<int32_t>(
        std::min<int64_t>(batch_size - offset, max_grid_x));
    plan.launches.push_back(CubSortSingleTileLaunch{offset, chunk});
  }
  return plan;
}

// Builds the ordered sequence of Onesweep scatter kernel launches across all
// radix passes and input portions.
std::vector<CubSortOnesweepPassLaunch> BuildOnesweepLaunches(
    int total_bits, int64_t num_items, int32_t tile_items,
    const CubSortOnesweepPlan& onesweep,
    absl::Span<const CubSortScratchRange> bins_portions) {
  constexpr int kRadixBits = kCubSortOnesweepRadixBits;
  std::vector<CubSortOnesweepPassLaunch> launches;
  launches.reserve(static_cast<size_t>(onesweep.num_passes) *
                   static_cast<size_t>(onesweep.num_portions));
  for (int pass = 0; pass < onesweep.num_passes; ++pass) {
    const auto [src_role, dst_role] = PingPongRoles(pass, onesweep.num_passes);
    const int32_t current_bit = pass * kRadixBits;
    const int32_t num_bits = std::min(kRadixBits, total_bits - current_bit);
    const int64_t pass_bin_offset =
        static_cast<int64_t>(pass) * kOnesweepBinsPerPassBytes;

    for (int64_t portion = 0; portion < onesweep.num_portions; ++portion) {
      const int64_t item_offset = portion * onesweep.portion_size;
      const int32_t portion_items = static_cast<int32_t>(
          std::min<int64_t>(num_items - item_offset, onesweep.portion_size));
      const int32_t num_blocks =
          static_cast<int32_t>(CeilOfRatio<int64_t>(portion_items, tile_items));

      CubSortOnesweepPassLaunch step;
      step.pass = pass;
      step.portion = portion;
      step.src_role = src_role;
      step.dst_role = dst_role;
      step.bins_in =
          CubSortScratchRange{bins_portions[portion].offset + pass_bin_offset,
                              kOnesweepBinsPerPassBytes};
      if (portion + 1 < onesweep.num_portions) {
        step.bins_out = CubSortScratchRange{
            bins_portions[portion + 1].offset + pass_bin_offset,
            kOnesweepBinsPerPassBytes};
      }
      step.lookback = CubSortScratchRange{
          onesweep.lookback_max.offset,
          static_cast<int64_t>(num_blocks) * kCubSortOnesweepRadixDigits *
              static_cast<int64_t>(sizeof(int32_t))};
      const int64_t ctr_slot = portion * onesweep.num_passes + pass;
      step.ctr = CubSortScratchRange{
          onesweep.ctrs_all.offset + ctr_slot * kCubSortScratchAlignmentBytes,
          sizeof(int32_t)};
      step.portion_item_offset = item_offset;
      step.portion_num_items = portion_items;
      step.grid_x = num_blocks;
      step.current_bit = current_bit;
      step.num_bits = num_bits;
      launches.push_back(step);
    }
  }
  return launches;
}

struct OnesweepPlanResult {
  CubSortOnesweepPlan plan;
  int64_t scratch_bytes = 0;
};

// Plans an unbatched multi-tile Onesweep radix sort.
//
// Onesweep is CUB's least-significant-digit (LSD) radix sort for large 1D
// inputs that exceed a single thread block. Classic GPU radix sort needs two
// full memory sweeps per digit pass (upsweep histogram + downsweep scatter);
// Onesweep reduces this to one initial global histogram plus one scatter sweep
// per 8-bit digit pass (`num_passes = ceil(key_bits / 8)`):
//   1. Histogram: scans the input once and counts all 256 digit buckets for
//      every pass simultaneously (`bins_raw`).
//   2. ExclusiveSum: converts each pass's raw digit counts into starting
//      output offsets (`bins_portion0`) and zeroes the per-launch dynamic tile
//      counters (`ctrs_all`) in a single launch.
//   3. Onesweep passes: each pass assigns contiguous tiles dynamically via its
//      slot in `ctrs_all`, resolves per-digit prefix sums across blocks via
//      decoupled lookback (`lookback_max`), and scatters keys (and optional
//      values) directly into the ping-pong destination buffer.
//
// Because Onesweep packs intra-portion offsets and lookback status flags into
// 32-bit words, inputs larger than `portion_size` (~2^28 items, rounded down to
// a multiple of `tile_items`) are split into `num_portions` sequential chunks
// per pass. Each portion reads its starting bin offsets from `bins_in` and
// writes updated offsets to `bins_out` for the next portion.
absl::StatusOr<OnesweepPlanResult> PlanOnesweepSort(
    const CubRadixSortPolicySpec& policy, int64_t num_items, int key_byte_size,
    int value_byte_size, std::optional<int32_t> portion_size_override) {
  CubSortOnesweepPlan plan;
  const int total_bits = key_byte_size * 8;
  plan.num_passes = CeilOfRatio(total_bits, kCubSortOnesweepRadixBits);

  const int32_t tile_items = policy.onesweep.tile_items();
  const int32_t default_portion_size =
      (kCubSortMaxPortionLimit / tile_items) * tile_items;
  plan.portion_size = portion_size_override.value_or(default_portion_size);

  plan.num_portions = CeilOfRatio(num_items, int64_t{plan.portion_size});
  if (plan.num_portions > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Onesweep num_portions (", plan.num_portions,
        ") exceeds int32_t limit (", std::numeric_limits<int32_t>::max(), ")"));
  }

  const int32_t max_blocks_per_portion = static_cast<int32_t>(CeilOfRatio(
      std::min<int64_t>(num_items, plan.portion_size), int64_t{tile_items}));
  const int64_t bins_all_passes_bytes =
      static_cast<int64_t>(plan.num_passes) * kOnesweepBinsPerPassBytes;
  const int64_t max_lookback_bytes =
      static_cast<int64_t>(max_blocks_per_portion) *
      kCubSortOnesweepRadixDigits * sizeof(int32_t);

  ScratchBumpAllocator alloc(kCubSortScratchAlignmentBytes);
  ABSL_ASSIGN_OR_RETURN(plan.bins_raw, alloc.Allocate(bins_all_passes_bytes));

  std::vector<CubSortScratchRange> bins_portions;
  bins_portions.reserve(plan.num_portions);
  for (int64_t portion = 0; portion < plan.num_portions; ++portion) {
    ABSL_ASSIGN_OR_RETURN(CubSortScratchRange range,
                     alloc.Allocate(bins_all_passes_bytes));
    bins_portions.push_back(range);
  }
  plan.bins_portion0 = bins_portions.front();
  ABSL_ASSIGN_OR_RETURN(plan.lookback_max, alloc.Allocate(max_lookback_bytes));

  // One 256-byte-strided counter slot per (portion, pass) so ExclusiveSum can
  // zero all dynamic tile counters in a single launch without per-pass MemZero
  // nodes, while each Onesweep launch owns a disjoint slice.
  ABSL_ASSIGN_OR_RETURN(const int64_t total_ctr_slots,
                   MultiplyNonNegative(plan.num_passes, plan.num_portions));
  ABSL_ASSIGN_OR_RETURN(
      const int64_t ctrs_bytes,
      MultiplyNonNegative(total_ctr_slots, kCubSortScratchAlignmentBytes));
  ABSL_ASSIGN_OR_RETURN(plan.ctrs_all, alloc.Allocate(ctrs_bytes));

  ABSL_ASSIGN_OR_RETURN(TempBuffers temp_buffers,
                   AllocateTempBuffers(alloc, plan.num_passes, num_items,
                                       key_byte_size, value_byte_size));
  plan.keys_tmp = temp_buffers.keys_tmp;
  plan.vals_tmp = temp_buffers.vals_tmp;
  ABSL_ASSIGN_OR_RETURN(const int64_t scratch_bytes, alloc.TotalBytes());

  plan.launches = BuildOnesweepLaunches(total_bits, num_items, tile_items, plan,
                                        bins_portions);
  return OnesweepPlanResult{std::move(plan), scratch_bytes};
}

// Builds the ordered sequence of uniform-segment multi-pass launches across
// radix passes (with optional alternate-width passes) and grid-X batch chunks.
std::vector<CubSortSegmentedPassLaunch> BuildSegmentedLaunches(
    const CubRadixSortPolicySpec& policy, int total_bits, int num_passes,
    int64_t batch_size, int32_t max_grid_x) {
  const int primary_bits = policy.segmented.radix_bits;
  const int alt_bits = policy.alt_segmented.radix_bits;
  const int max_alt_passes = num_passes * primary_bits - total_bits;
  const int alt_end_bit = std::min(total_bits, max_alt_passes * alt_bits);

  std::vector<CubSortSegmentedPassLaunch> launches;
  int current_bit = 0;
  for (int pass = 0; pass < num_passes; ++pass) {
    const bool use_alt = current_bit < alt_end_bit;
    const int radix_bits = use_alt ? alt_bits : primary_bits;
    const int pass_bits = std::min(radix_bits, total_bits - current_bit);
    const auto [src_role, dst_role] = PingPongRoles(pass, num_passes);

    for (int64_t offset = 0; offset < batch_size; offset += max_grid_x) {
      const int32_t chunk = static_cast<int32_t>(
          std::min<int64_t>(batch_size - offset, max_grid_x));
      CubSortSegmentedPassLaunch step;
      step.pass = pass;
      step.use_alt_policy = use_alt;
      step.src_role = src_role;
      step.dst_role = dst_role;
      step.segment_offset = offset;
      step.grid_x = chunk;
      step.current_bit = current_bit;
      step.pass_bits = pass_bits;
      launches.push_back(step);
    }
    current_bit += pass_bits;
  }
  return launches;
}

struct SegmentedPlanResult {
  CubSortSegmentedMultiPassPlan plan;
  int64_t scratch_bytes = 0;
};

// Plans a batched multi-pass segmented radix sort (batch_size > 1 and
// segment_size > single_tile.tile_items()).
//
// Multi-pass segmented radix sort sorts `batch_size` independent, fixed-size
// segments of length `segment_size` when each segment is too large to fit in
// one thread block's registers/shared memory for `CubSortSingleTilePlan`.
// Unlike Onesweep, there is no cross-block coordination or global histogram:
//   - Each thread block owns one entire segment (`grid_x == batch_size`,
//     chunked into multiple launches of size `max_grid_x` if needed) and
//     computes its slice `[segment_idx * segment_size, ...)` arithmetically
//     with no host-to-device offset table copy.
//   - Each pass processes `pass_bits` remaining least-significant bits in two
//     in-block sweeps over the segment: an upsweep to histogram the segment's
//     digits and scan the bin counts in shared memory, followed by a downsweep
//     that stably scatters elements into the ping-pong destination buffer.
//   - Scratch memory holds only the ping-pong temporary key/value buffers when
//     `num_passes > 1` (no global bins, counters, or lookback state).
//   - When `total_bits` is not an exact multiple of `segmented.radix_bits`,
//     the initial passes use the 1-bit-narrower `alt_segmented` policy to
//     lower register and shared-memory usage without adding extra passes.
absl::StatusOr<SegmentedPlanResult> PlanSegmentedSort(
    const CubRadixSortPolicySpec& policy, int64_t num_items, int64_t batch_size,
    int64_t segment_size, int key_byte_size, int value_byte_size,
    int32_t max_grid_x) {
  // CUB's `DeviceSegmentedRadixSortKernel` asserts `INT32_MAX > segment_size`,
  // and `AgentRadixSortDownsweep::ProcessTile<false>` with `int32_t` offsets
  // advances `bin_offset` by up to `TILE_ITEMS` on a final partial tile.
  // Keeping `segment_size <= INT32_MAX - max_tile_items` satisfies CUB's
  // invariant and prevents signed `int32_t` overflow UB.
  const int64_t max_tile_items = std::max(policy.segmented.tile_items(),
                                          policy.alt_segmented.tile_items());
  const int64_t max_segment_size =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) -
      max_tile_items;
  if (segment_size > max_segment_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Batched CUDA CUB sort segment_size (", segment_size,
                     ") exceeds int32_t safe limit (", max_segment_size, ")"));
  }

  CubSortSegmentedMultiPassPlan plan;
  const int total_bits = key_byte_size * 8;
  plan.num_passes = CeilOfRatio(total_bits, policy.segmented.radix_bits);

  ScratchBumpAllocator alloc(kCubSortScratchAlignmentBytes);
  ABSL_ASSIGN_OR_RETURN(TempBuffers temp_buffers,
                   AllocateTempBuffers(alloc, plan.num_passes, num_items,
                                       key_byte_size, value_byte_size));
  plan.keys_tmp = temp_buffers.keys_tmp;
  plan.vals_tmp = temp_buffers.vals_tmp;
  ABSL_ASSIGN_OR_RETURN(const int64_t scratch_bytes, alloc.TotalBytes());

  plan.launches = BuildSegmentedLaunches(policy, total_bits, plan.num_passes,
                                         batch_size, max_grid_x);
  return SegmentedPlanResult{std::move(plan), scratch_bytes};
}

}  // namespace

absl::StatusOr<CubSortExecutionPlan> PlanCubSortExecution(
    const stream_executor::CudaComputeCapability& cc, PrimitiveType key_type,
    std::optional<PrimitiveType> value_type, int64_t num_items,
    int64_t batch_size, CubSortPlanOptions options) {
  ABSL_RETURN_IF_ERROR(ValidateCubSortTypes(key_type, value_type));
  const int32_t max_grid_x =
      options.max_grid_x_override.value_or(std::numeric_limits<int32_t>::max());
  ABSL_RETURN_IF_ERROR(ValidateDimensions(num_items, batch_size, max_grid_x));

  const int key_byte_size = primitive_util::ByteWidth(key_type);
  const int value_byte_size =
      value_type.has_value() ? primitive_util::ByteWidth(*value_type) : 0;
  const int sm_version = CanonicalCubSortPolicySmVersion(cc.major);

  CubSortExecutionPlan plan;
  plan.policy =
      GetCubRadixSortPolicySpec(sm_version, key_byte_size, value_byte_size,
                                /*is_float32_key=*/key_type == F32);
  ABSL_RETURN_IF_ERROR(ValidatePortionSizeOverride(
      options.portion_size_override, plan.policy.onesweep.tile_items()));
  plan.key_type = key_type;
  plan.value_type = value_type;
  plan.num_items = num_items;
  plan.batch_size = batch_size;
  plan.segment_size = num_items / batch_size;

  if (plan.segment_size <= plan.policy.single_tile.tile_items()) {
    plan.mode_plan = PlanSingleTileSort(batch_size, max_grid_x);
    return plan;
  }
  if (batch_size == 1) {
    ABSL_ASSIGN_OR_RETURN(
        OnesweepPlanResult result,
        PlanOnesweepSort(plan.policy, num_items, key_byte_size, value_byte_size,
                         options.portion_size_override));
    plan.scratch_bytes = result.scratch_bytes;
    plan.mode_plan = std::move(result.plan);
    return plan;
  }

  ABSL_ASSIGN_OR_RETURN(
      SegmentedPlanResult result,
      PlanSegmentedSort(plan.policy, num_items, batch_size, plan.segment_size,
                        key_byte_size, value_byte_size, max_grid_x));
  plan.scratch_bytes = result.scratch_bytes;
  plan.mode_plan = std::move(result.plan);
  return plan;
}

absl::StatusOr<int64_t> ComputeCudaCubSortScratchSize(
    const stream_executor::CudaComputeCapability& cc, PrimitiveType key_type,
    std::optional<PrimitiveType> value_type, int64_t num_items,
    int64_t batch_size) {
  ABSL_ASSIGN_OR_RETURN(
      CubSortExecutionPlan plan,
      PlanCubSortExecution(cc, key_type, value_type, num_items, batch_size));
  return plan.scratch_bytes;
}

}  // namespace xla::gpu
