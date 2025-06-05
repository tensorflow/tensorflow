/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/dot_search_space.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/bits.h"
#include "xla/util.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

// Returns the size (in number of elements) of the subshape of `shape` defined
// by `dimensions`.
int64_t GetSizeInDimensions(
    const Shape& shape,
    const tsl::protobuf::RepeatedField<int64_t>& dimensions) {
  int64_t size = 1;
  for (int64_t dim : dimensions) {
    size *= shape.dimensions(dim);
  }
  return size;
}

// Finds the next power of two larger than or equal to x.
//
// Unlike tsl::NextPowerOfTwo, doesn't crash for 0.
int64_t NextPowerOfTwo(int64_t x) {
  if (x == 0) {
    return 1;
  }
  return tsl::NextPowerOfTwoS64(x);
}

// Finds the previous power of two, smaller or equal to x.
//
// Returns 1 for an edge case of x = 0 (which we can get as a result of integer
// division). This might feel a bit weird, but it does the right thing when
// calculating tile sizes, since we need a strictly positive size.
int64_t PreviousPowerOfTwo(int64_t x) {
  if (x == 0) {
    return 1;
  }
  return tsl::NextPowerOfTwoS64(x + 1) / 2;
}

}  // namespace

TritonDotFusionSearchSpace::TritonDotFusionSearchSpace(
    const se::DeviceDescription& device_description,
    const HloDotInstruction* dot)
    :  // Set up basic information about the hardware and the problem.
      device_description_(device_description),
      contracting_size_(GetSizeInDimensions(
          dot->operand(0)->shape(),
          dot->dot_dimension_numbers().lhs_contracting_dimensions())),
      batch_size_(GetSizeInDimensions(
          dot->operand(0)->shape(),
          dot->dot_dimension_numbers().lhs_batch_dimensions())),
      lhs_parallel_size_(ShapeUtil::ElementsIn(dot->operand(0)->shape()) /
                         (contracting_size_ * batch_size_)),
      rhs_parallel_size_(ShapeUtil::ElementsIn(dot->operand(1)->shape()) /
                         (contracting_size_ * batch_size_)),
      operand_bitwidth_(  // The bitwitdth of both operands is the same.
          primitive_util::BitWidth(dot->operand(0)->shape().element_type())),
      compute_bitwidth_(primitive_util::BitWidth(dot->shape().element_type())),
      // Figure out some basic limitations on tiling based on the above.
      lhs_has_expensive_op_(HasExpensiveTransitiveParent(dot->operand(0))),
      rhs_has_expensive_op_(HasExpensiveTransitiveParent(dot->operand(1))),
      desired_total_warps_(GetDesiredTotalWarps()),
      max_out_tile_(GetMaxOutputTile()),
      should_optimize_for_occupancy_(ShouldOptimizeForOccupancy()),
      min_out_tile_(GetMinOutputTile()),
      min_warps_per_cta_(GetMinWarpsPerCta()),
      min_contracting_tile_size_(GetMinContractingTileSize()),
      max_contracting_split_(GetMaxContractingSplit(max_out_tile_)) {
  // Make sure that the range of output tile sizes is not empty
  // (min_output_tile_ is a hard limit, while max_output_tile_ is a soft one).
  max_out_tile_.lhs_dim =
      std::max(min_out_tile_.lhs_dim, max_out_tile_.lhs_dim);
  max_out_tile_.rhs_dim =
      std::max(min_out_tile_.rhs_dim, max_out_tile_.rhs_dim);
}

std::vector<TritonGemmConfig> TritonDotFusionSearchSpace::GenerateConfigs(
    std::optional<int64_t> force_contracting_split) const {
  std::vector<ConfigWithNotes> configs;
  if (force_contracting_split.has_value()) {
    ConfigWithNotes config;
    const int split = force_contracting_split.value();
    config.config.split_k = split;
    // It is possible that the user manually forced a huge contracting split
    // that is outside of the search space. In that case, we would end up
    // discarding all configs, and use the smallest possible tile size further
    // down, which is likely not what the user had in mind.
    config.keep_large_split = GetMaxContractingSplit(max_out_tile_) < split;
    VLOG(5) << "Forcing split_k, config = " << config.ToString();
    if (config.keep_large_split) {
      LOG(WARNING)
          << "split_k is larger than what we would have found automatically. "
             "Skipping split and output tile compatibility checks. Should we "
             "expand the split_k search space?";
    }
    configs.push_back(config);
  } else {
    configs = GenerateContractingSplitFactors();
  }

  ExtendConfigs(configs, &TritonDotFusionSearchSpace::AddOutputTilings);
  EliminateLowOccupancyConfigs(configs);
  ExtendConfigs(configs, &TritonDotFusionSearchSpace::AddCtaSizeParameter);
  ExtendConfigs(configs, &TritonDotFusionSearchSpace::AddContractingTiling);
  ExtendConfigs(configs, &TritonDotFusionSearchSpace::AddPipeliningParameter);

  std::vector<TritonGemmConfig> result;
  result.reserve(configs.size());
  for (ConfigWithNotes& config_with_notes : configs) {
    TritonGemmConfig& config = config_with_notes.config;
    // TODO: b/408386169 - Implement CTA cluster support.
    config.num_ctas = 1;
    result.push_back(config);
  }
  return result;
}

std::vector<TritonGemmConfig> TritonDotFusionSearchSpace::OptimizeConfigSet(
    const std::vector<TritonGemmConfig>& configs,
    const std::vector<TritonGemmConfig>& hints) const {
  if (hints.empty() || configs.empty()) {
    return configs;
  }

  auto split_limits = std::minmax_element(
      configs.begin(), configs.end(),
      [](const auto& a, const auto& b) { return a.split_k < b.split_k; });
  absl::flat_hash_set<TritonGemmConfig> filter;
  for (TritonGemmConfig config : hints) {
    // Our default config set does not take problem size into account, so we
    // might not even have some of them in the "exhaustive set", since they
    // might be outside of the efficient config range. Hence, we limit the tile
    // to what can appear in the exhaustive set.
    config.block_m = std::clamp(config.block_m, min_out_tile_.lhs_dim,
                                max_out_tile_.lhs_dim);
    config.block_n = std::clamp(config.block_n, min_out_tile_.rhs_dim,
                                max_out_tile_.rhs_dim);
    config.block_k =
        std::clamp(config.block_k, min_contracting_tile_size_,
                   GetMaxContractingTileSize({config.block_m, config.block_n},
                                             /*contracting_split=*/1));
    config.split_k = std::clamp(config.split_k, split_limits.first->split_k,
                                split_limits.second->split_k);
    VLOG(10) << "Adding config to hint filter: " << config.ToString();
    filter.insert(config);
  }

  std::vector<TritonGemmConfig> result_configs;
  for (const TritonGemmConfig& config : configs) {
    if (!filter.contains(config)) {
      continue;
    }
    VLOG(10) << "Filtering out configs based on hints: surviving config = "
             << config.ToString();
    result_configs.push_back(config);
  };

  if (result_configs.empty()) {
    LOG(WARNING) << "All configs were filtered out because none of them "
                    "sufficiently match the hints. Maybe the hints set does "
                    "not contain a good representative set of valid configs?"
                    "Working around this by using the full hints set instead.";
    return hints;
  }
  return result_configs;
}

std::string TritonDotFusionSearchSpace::ToString() const {
  return absl::StrFormat(
      "problem_size_BxMxNxKxE: %dx%dx%dx%dx(%d->%d) "
      "tile_range_SxMxNxK: [1-%d]x[%d-%d]x[%d-%d]x[%d-?] "
      "desired_total_warps: %d occupancy_optimization: %d "
      "warps_per_cta: [%d-?]",
      batch_size_, lhs_parallel_size_, rhs_parallel_size_, contracting_size_,
      operand_bitwidth_, compute_bitwidth_, max_contracting_split_,
      min_out_tile_.lhs_dim, max_out_tile_.lhs_dim, min_out_tile_.rhs_dim,
      max_out_tile_.rhs_dim, min_contracting_tile_size_, desired_total_warps_,
      should_optimize_for_occupancy_, min_warps_per_cta_);
}

bool TritonDotFusionSearchSpace::HasExpensiveTransitiveParent(
    const HloInstruction* operand) const {
  return HloBfsAnyOf({operand}, [](const HloInstruction* instr) {
    // XLA uses old absl that doesn't have absl:NoDestructor, so have to use
    // new instead to prevent the destructor from being called.
    static const auto kExpensiveOps = new absl::flat_hash_set<HloOpcode>{
        HloOpcode::kAtan2,    HloOpcode::kCos,   HloOpcode::kExp,
        HloOpcode::kExpm1,    HloOpcode::kLog,   HloOpcode::kLog1p,
        HloOpcode::kLogistic, HloOpcode::kPower, HloOpcode::kRsqrt,
        HloOpcode::kSin,      HloOpcode::kSqrt,  HloOpcode::kTan,
        HloOpcode::kTanh,
    };
    return kExpensiveOps->contains(instr->opcode());
  });
}

int TritonDotFusionSearchSpace::GetDesiredTotalWarps() const {
  constexpr int kSchedulersPerCore = 4;
  constexpr int kDesiredWarpsPerCore =
      kMaxWarpsPerScheduler * kSchedulersPerCore;
  return kDesiredWarpsPerCore * device_description_.core_count();
}

TritonDotFusionSearchSpace::OutputTile
TritonDotFusionSearchSpace::GetMaxOutputTile() const {
  constexpr int kRegisterSizeInBits = 32;
  const int64_t max_elements_per_cta =
      device_description_.registers_per_block_limit() * kRegisterSizeInBits /
      compute_bitwidth_;
  auto limit_other_size_to_fit = [max_elements_per_cta](int64_t this_size) {
    return PreviousPowerOfTwo(max_elements_per_cta / this_size);
  };
  // We generally want to have square-ish tiles if possible to get maximal
  // reuse. For wgmma the optimal instruction shape is 64x256, so optimizing for
  // larger RHS given the choice.
  OutputTile max_tile;
  max_tile.lhs_dim = PreviousPowerOfTwo(std::sqrt(max_elements_per_cta));
  max_tile.rhs_dim = limit_other_size_to_fit(max_tile.lhs_dim);
  VLOG(5) << "Computing max_output_tile: Based on available registers, "
             "max_output_tile = "
          << max_tile.lhs_dim << "x" << max_tile.rhs_dim;

  const int64_t lhs_parallel_limit = NextPowerOfTwo(lhs_parallel_size_);
  const int64_t rhs_parallel_limit = NextPowerOfTwo(rhs_parallel_size_);
  if (lhs_parallel_limit < max_tile.lhs_dim) {
    max_tile.lhs_dim = lhs_parallel_limit;
    max_tile.rhs_dim = std::min(limit_other_size_to_fit(lhs_parallel_limit),
                                rhs_parallel_limit);
    VLOG(5) << "Computing max_tile: However, due to small LHS parallel size,"
               "max_output_tile = "
            << max_tile.lhs_dim << "x" << max_tile.rhs_dim;
  }
  if (rhs_parallel_limit < max_tile.rhs_dim) {
    max_tile.lhs_dim = std::min(limit_other_size_to_fit(rhs_parallel_limit),
                                lhs_parallel_limit);
    max_tile.rhs_dim = rhs_parallel_limit;
    VLOG(5) << "Computing max_tile: However, due to small RHS parallel "
               "size, max_output_tile = "
            << max_tile.lhs_dim << "x" << max_tile.rhs_dim;
  }
  return max_tile;
}

bool TritonDotFusionSearchSpace::ShouldOptimizeForOccupancy() const {
  const int64_t desired_num_ctas =
      desired_total_warps_ / kMinWarpsPerCtaForWgmma;
  const int64_t min_result_tiles = GetNumResultTiles(max_out_tile_);
  if (desired_num_ctas > min_result_tiles) {
    VLOG(5) << "Occupancy optimization: Might have as few as "
            << min_result_tiles << " tiles, but want at least "
            << desired_num_ctas
            << " CTAs. Will consider trading off compute performance for "
               "occupancy.";
    return true;
  }
  return false;
}

TritonDotFusionSearchSpace::OutputTile
TritonDotFusionSearchSpace::GetMinOutputTile() const {
  // Triton currently doesn't support tiles smaller than 16x16.
  // TODO: b/395572776 - Lift this restriction, and calculate a smaller tile
  // based on the requested algorithm (e.g., if we want to use wgmma vs mma
  // vs fma, the minimal reasonable tile size is different).
  constexpr OutputTile kMinSupportedTile = {16, 16};
  constexpr OutputTile kMinWgmmaTile = {64, 16};
  if (device_description_.cuda_compute_capability().IsAtLeastHopper() &&
      !should_optimize_for_occupancy_) {
    VLOG(5) << "Computing output_tile: Want to use wgmma, so output_tile >= "
            << kMinWgmmaTile.lhs_dim << "x" << kMinWgmmaTile.rhs_dim;
    return kMinWgmmaTile;
  }
  VLOG(5)
      << "Computing output_tile: Might want to target mma, so output_tile >= "
      << kMinSupportedTile.lhs_dim << "x" << kMinSupportedTile.rhs_dim;
  return kMinSupportedTile;
}

int TritonDotFusionSearchSpace::GetMinWarpsPerCta() const {
  if (operand_bitwidth_ >= 32) {
    // Triton is generating quite suboptimal code for 32-bit dots, especially
    // when we use wgmma, or larger blocks.
    // TODO: b/422419331 - Remove this once Triton properly handles 32-bit dots.
    return kMinWarpsPerCtaForOccupancy;
  }
  if (device_description_.cuda_compute_capability().IsAtLeastHopper() &&
      !should_optimize_for_occupancy_) {
    VLOG(5) << "Computing num_warps: Want to use wgmma, so num_warps >= "
            << kMinWarpsPerCtaForWgmma;
    return kMinWarpsPerCtaForWgmma;
  }
  VLOG(5) << "Computing num_warps: Considering occupancy, so num_warps >= "
          << kMinWarpsPerCtaForOccupancy;
  return kMinWarpsPerCtaForOccupancy;
}

int64_t TritonDotFusionSearchSpace::GetNumResultTiles(
    OutputTile output_tile) const {
  return batch_size_ *
         CeilOfRatio<int64_t>(lhs_parallel_size_, output_tile.lhs_dim) *
         CeilOfRatio<int64_t>(rhs_parallel_size_, output_tile.rhs_dim);
}

int TritonDotFusionSearchSpace::GetMaxWarpsPerCta(OutputTile tile) const {
  // A single mma instruction is of output shape at least 16x8 (the same
  // also holds for wgmma: the warp-group level instruction is at least
  // 64x8, and split 4-ways across the 4 warps in the group).
  constexpr OutputTile kMmaSubTile = {16, 8};
  const int max_warps =
      device_description_.threads_per_block_limit() /
      std::max<int>(device_description_.threads_per_warp(), 1);
  const int lhs_warps = CeilOfRatio(tile.lhs_dim, kMmaSubTile.lhs_dim);
  const int rhs_warps = CeilOfRatio(tile.rhs_dim, kMmaSubTile.rhs_dim);
  return std::max(min_warps_per_cta_,
                  std::min(max_warps, lhs_warps * rhs_warps));
}

int TritonDotFusionSearchSpace::GetMinContractingTileSize() const {
  // The number of bits that both MMA and WGMMA instructions expect to have in
  // the contracting dimension. See
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shape
  constexpr int kMmaContractingBitwidth = 128;
  /// TODO: b/395572776 - Triton currently requires at least 16 elements, but we
  // should be able to relax this and remove this limit here.
  constexpr int kTritonLowerLimit = 16;
  const int min_contracting_tile_size =
      std::max(kMmaContractingBitwidth / operand_bitwidth_, kTritonLowerLimit);
  VLOG(5) << "Computing min_contracting_tile_size: Based on bitwidth of "
          << operand_bitwidth_
          << ", min_contracting_tile_size = " << min_contracting_tile_size;
  return min_contracting_tile_size;
}

int TritonDotFusionSearchSpace::GetMaxContractingSplit(
    OutputTile output_tile) const {
  const int64_t desired_num_ctas = desired_total_warps_ / min_warps_per_cta_;
  VLOG(5) << "Computing split_k: Considering output tile "
          << output_tile.lhs_dim << "x" << output_tile.rhs_dim;
  VLOG(5) << "Computing split_k: Want up to " << desired_num_ctas
          << " CTAs to occupy all cores.";

  const int64_t min_result_tiles = GetNumResultTiles(output_tile);
  VLOG(5) << "Computing split_k: Without split_k have " << min_result_tiles
          << " tiles.";

  const int64_t split_for_occupancy =
      NextPowerOfTwo(CeilOfRatio(desired_num_ctas, min_result_tiles));
  VLOG(5) << "Computing split_k: Want split_k of up to " << split_for_occupancy
          << " for sufficient occupancy.";

  const int64_t split_for_contracting_size =
      NextPowerOfTwo(contracting_size_ / min_contracting_tile_size_);
  VLOG(5) << "Computing split_k: Can't have split_k more than "
          << split_for_contracting_size
          << " to have sufficiently large contracting dimension.";

  const int64_t split =
      std::min(split_for_occupancy, split_for_contracting_size);
  VLOG(5) << "Computing split_k: max_split_k = " << split;
  return split;
}

int TritonDotFusionSearchSpace::GetContractingSizeLimitToFitSharedMemory(
    OutputTile output_tile) const {
  const int64_t shared_memory_budget =
      device_description_.shared_memory_per_block_optin();
  // Need to satisfy:
  //   (lhs_dim  + rhs_dim) * contracting_dim * bitwidth <= budget_in_bits
  return 8 * shared_memory_budget / compute_bitwidth_ /
         (output_tile.lhs_dim + output_tile.rhs_dim);
}

int TritonDotFusionSearchSpace::GetMaxContractingTileSize(
    OutputTile output_tile, int contracting_split) const {
  const int64_t available_size = contracting_size_ / contracting_split;
  const int size_limit = GetContractingSizeLimitToFitSharedMemory(output_tile);
  const int max_size =
      std::min(NextPowerOfTwo(available_size), PreviousPowerOfTwo(size_limit));
  VLOG(5) << "Computing max_contracting_tile_size for tiling BxMxN = "
          << contracting_split << "x" << output_tile.lhs_dim << "x"
          << output_tile.rhs_dim << ": limit based on problem is "
          << available_size << ", limit based on available shared memory is "
          << size_limit << ", max_contracting_tile_size = " << max_size;
  return std::max(min_contracting_tile_size_, max_size);
}

int TritonDotFusionSearchSpace::GetMaxNumStages(OutputTile output_tile,
                                                int contracting_tile_size,
                                                int contracting_split) const {
  const int64_t available_stages = CeilOfRatio<int64_t>(
      contracting_size_, contracting_split * contracting_tile_size);
  const int64_t stage_limit = std::max(
      1, CeilOfRatio(GetContractingSizeLimitToFitSharedMemory(output_tile),
                     contracting_tile_size));
  // Number of stages is basically a replacement for oversubscription, so
  // the maximum number we want is also limited by kMaxWarpsPerScheduler.
  const int stages = std::min({available_stages, stage_limit,
                               static_cast<int64_t>(kMaxWarpsPerScheduler)});
  VLOG(5) << "Computing max_num_stages for tiling BxMxNxK = "
          << contracting_split << "x" << output_tile.lhs_dim << "x"
          << output_tile.rhs_dim << "x" << contracting_tile_size
          << ": limit based on problem is " << available_stages
          << ", limit based on available shared memory is " << stage_limit
          << ", max_num_stages = " << stages;
  return stages;
}

std::vector<TritonDotFusionSearchSpace::ConfigWithNotes>
TritonDotFusionSearchSpace::GenerateContractingSplitFactors() const {
  CHECK_GE(max_contracting_split_, 1);
  std::vector<ConfigWithNotes> configs;
  ConfigWithNotes config;
  for (int split = 1; split <= max_contracting_split_; split *= 2) {
    config.config.split_k = split;
    VLOG(10) << "Generating contracting split factors: config = "
             << config.ToString();
    configs.push_back(config);
  }
  return configs;
}

void TritonDotFusionSearchSpace::ExtendConfigs(
    std::vector<ConfigWithNotes>& configs,
    ExtendConfigCallback extend_config) const {
  CHECK(!configs.empty());
  std::vector<ConfigWithNotes> updated_configs;
  for (ConfigWithNotes& config : configs) {
    (this->*extend_config)(config, updated_configs);
  }
  CHECK(!updated_configs.empty());
  configs = std::move(updated_configs);
}

void TritonDotFusionSearchSpace::AddOutputTilings(
    const ConfigWithNotes& config,
    std::vector<ConfigWithNotes>& updated_configs) const {
  CHECK_GT(config.config.split_k, 0)
      << "Need config with contracting split already set.";
  const int split = config.config.split_k;
  ConfigWithNotes new_config = config;
  for (int m = min_out_tile_.lhs_dim; m <= max_out_tile_.lhs_dim; m *= 2) {
    int min_n = min_out_tile_.rhs_dim;
    int max_n = max_out_tile_.rhs_dim;
    // If there are square-ish tiles contained within the search space, it is
    // extremely unlikely that a non-square-ish tile will perform better, since
    // it does not optimize data reuse. The one exception to this is the
    // edge-case where one of the dimensions is small: m >= LHS dim, or max_n >=
    // RHS dim.
    //
    // Thus, as soon as there are square-ish tiles in the search space, and
    // we're not in the edge case (i.e., m < LHS dim; the requirement on max_n
    // is satisfied by construction as soon as [m/2, m*2] and [min_n, max_n]
    // overlap), we can restrict the n-space to only these tiles.
    auto overlaps = [](std::pair<int, int> a, std::pair<int, int> b) {
      return !(a.second < b.first || b.second < a.first);
    };
    if (m < lhs_parallel_size_ && overlaps({m / 2, m * 2}, {min_n, max_n})) {
      // If one of the sides has an expensive op fused in, then we should allow
      // the tile of the other side to be larger, as that reduce the amount of
      // recomputation of the expensive op.
      if (!rhs_has_expensive_op_) {
        min_n = std::max(m / 2, min_n);
      }
      if (!lhs_has_expensive_op_) {
        max_n = std::min(m * 2, max_n);
      }
      VLOG(5) << "Computing output tile: For m = " << m
              << ", restricting n-space to [" << min_n << "," << max_n
              << "] to have square-ish tiles.";
    }
    for (int n = min_n; n <= max_n; n *= 2) {
      OutputTile tile = {m, n};
      // We could make the tile size limits depend on split_k, but then we
      // need to implement the "inverse" of `GetMaxContractingSplit`.
      // Simpler is to just verify that the given combination of tiling and
      // split_k is compatible.
      if (!config.keep_large_split && GetMaxContractingSplit(tile) < split) {
        VLOG(10) << "Skipping due to too large split_k, config = "
                 << new_config.ToString();
        continue;
      }
      new_config.not_enough_tiles =
          GetNumResultTiles(tile) * split < device_description_.core_count();
      new_config.config.block_m = m;
      new_config.config.block_n = n;
      VLOG(10) << "Adding output tiling: config = " << new_config.ToString();
      updated_configs.push_back(new_config);
    }
  }
}

void TritonDotFusionSearchSpace::AddCtaSizeParameter(
    const ConfigWithNotes& config,
    std::vector<ConfigWithNotes>& updated_configs) const {
  ConfigWithNotes new_config = config;
  const int tile_rows = config.config.block_m;
  const int tile_cols = config.config.block_n;
  CHECK_GT(tile_rows * tile_cols, 0)
      << "Need configs with output tilings determined.";
  const int max_warps = GetMaxWarpsPerCta({tile_rows, tile_cols});
  VLOG(5) << "Computing max_warps: For output_tile = " << tile_rows << "x"
          << tile_cols
          << " and (wg)mma instruction shape, max_warps = " << max_warps;
  for (int warps = min_warps_per_cta_; warps <= max_warps; warps *= 2) {
    new_config.config.num_warps = warps;
    VLOG(10) << "Adding CTA size parameter: config = " << new_config.ToString();
    updated_configs.push_back(new_config);
  }
}

void TritonDotFusionSearchSpace::AddContractingTiling(
    const ConfigWithNotes& config,
    std::vector<ConfigWithNotes>& updated_configs) const {
  const int tile_rows = config.config.block_m;
  const int tile_cols = config.config.block_n;
  const int split = config.config.split_k;
  CHECK_GT(tile_rows * tile_cols, 0)
      << "Need configs with output tilings determined.";
  CHECK_GT(split, 0) << "Need config with contracting split determined.";
  int max_tile_size =
      std::max(GetMaxContractingTileSize({tile_rows, tile_cols}, split),
               min_contracting_tile_size_);
  ConfigWithNotes new_config = config;
  for (int k = min_contracting_tile_size_; k <= max_tile_size; k *= 2) {
    new_config.config.block_k = k;
    VLOG(10) << "Adding contracting tiling: config = " << new_config.ToString();
    updated_configs.push_back(new_config);
  }
}

void TritonDotFusionSearchSpace::AddPipeliningParameter(
    const ConfigWithNotes& config,
    std::vector<ConfigWithNotes>& updated_configs) const {
  const int tile_rows = config.config.block_m;
  const int tile_cols = config.config.block_n;
  const int tile_contracting = config.config.block_k;
  const int split = config.config.split_k;
  CHECK_GT(tile_rows * tile_cols, 0)
      << "Need config with output tilings determined.";
  CHECK_GT(tile_contracting, 0)
      << "Need config with contracting tiling determined.";
  CHECK_GT(split, 0) << "Need config with contracting split determined.";
  int max_stages =
      GetMaxNumStages({tile_rows, tile_cols}, tile_contracting, split);
  ConfigWithNotes new_config = config;
  for (int num_stages = 1; num_stages <= max_stages; ++num_stages) {
    new_config.config.num_stages = num_stages;
    VLOG(10) << "Adding pipelining parameter: config = "
             << new_config.ToString();
    updated_configs.push_back(new_config);
  }
}

void TritonDotFusionSearchSpace::EliminateLowOccupancyConfigs(
    std::vector<ConfigWithNotes>& configs) const {
  CHECK(!configs.empty());
  ConfigWithNotes last_config = configs.back();  // Largest split.
  auto has_too_few_tiles = [](const ConfigWithNotes& config) {
    if (config.not_enough_tiles) {
      VLOG(10) << "Skipping due to fewer tiles than cores, config = "
               << config.ToString();
    }
    return config.not_enough_tiles;
  };
  configs.erase(llvm::remove_if(configs, has_too_few_tiles), configs.end());
  if (configs.empty()) {
    // We can get no configs if the problem is small enough to not even occupy
    // all cores. In that case, we just use the largest split and smallest
    // tiling.
    last_config.config.block_m = min_out_tile_.lhs_dim;
    last_config.config.block_n = min_out_tile_.rhs_dim;
    VLOG(10) << "No configs with sufficient occupancy, using config = "
             << last_config.ToString();
    configs.push_back(last_config);
  }
}

}  // namespace xla::gpu
