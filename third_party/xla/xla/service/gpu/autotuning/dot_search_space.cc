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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
      compute_bitwidth_(primitive_util::BitWidth(dot->shape().element_type())),
      // Figure out some basic limitations on tiling based on the above.
      desired_total_warps_(GetDesiredTotalWarps()),
      max_out_tile_(GetMaxOutputTile()),
      // TODO: b/404470821 - Compute these from the problem properties instead
      // of hardcoding.
      min_out_tile_{16, 16},
      min_warps_per_cta_(4),
      min_contracting_tile_size_(16),
      max_contracting_split_(GetMaxContractingSplit(max_out_tile_)) {
  // Make sure that the range of output tile sizes is not empty
  // (min_output_tile_ is a hard limit, while max_output_tile_ is a soft one).
  max_out_tile_.lhs_dim =
      std::max(min_out_tile_.lhs_dim, max_out_tile_.lhs_dim);
  max_out_tile_.rhs_dim =
      std::max(min_out_tile_.rhs_dim, max_out_tile_.rhs_dim);
}

std::vector<TritonGemmConfig> TritonDotFusionSearchSpace::GenerateConfigs(
    std::optional<int64_t> force_contracting_split) {
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

  std::vector<TritonGemmConfig> result;
  result.reserve(configs.size());
  for (ConfigWithNotes& config_with_notes : configs) {
    // TODO: b/404470821 - Implement this properly rather than hardcoding the
    // config parameters.
    TritonGemmConfig& config = config_with_notes.config;
    config.block_k = 64;
    config.num_stages = 3;
    config.num_ctas = 1;
    result.push_back(config);
  }
  return result;
}

std::string TritonDotFusionSearchSpace::Serialize() {
  return absl::StrFormat(
      "problem_size_BxMxNxKxE: %dx%dx%dx%dx%d "
      "tile_range_SxMxNxK: [1-%d]x[%d-%d]x[%d-%d]x[%d-?] "
      "desired_total_warps: %d warps_per_cta: [%d-?]",
      batch_size_, lhs_parallel_size_, rhs_parallel_size_, contracting_size_,
      compute_bitwidth_, max_contracting_split_, min_out_tile_.lhs_dim,
      max_out_tile_.lhs_dim, min_out_tile_.rhs_dim, max_out_tile_.rhs_dim,
      min_contracting_tile_size_, desired_total_warps_, min_warps_per_cta_);
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
  const int max_warps = device_description_.threads_per_block_limit() /
                        device_description_.threads_per_warp();
  const int lhs_warps = CeilOfRatio(tile.lhs_dim, kMmaSubTile.lhs_dim);
  const int rhs_warps = CeilOfRatio(tile.rhs_dim, kMmaSubTile.rhs_dim);
  return std::max(min_warps_per_cta_,
                  std::min(max_warps, lhs_warps * rhs_warps));
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

std::vector<TritonDotFusionSearchSpace::ConfigWithNotes>
TritonDotFusionSearchSpace::GenerateContractingSplitFactors() {
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
    std::vector<ConfigWithNotes>& configs, ExtendConfigCallback extend_config) {
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
    std::vector<ConfigWithNotes>& updated_configs) {
  CHECK_GT(config.config.split_k, 0)
      << "Need config with contracting split already set.";
  const int split = config.config.split_k;
  ConfigWithNotes new_config = config;
  int& m = new_config.config.block_m;
  int& n = new_config.config.block_n;
  for (m = min_out_tile_.lhs_dim; m <= max_out_tile_.lhs_dim; m *= 2) {
    for (n = min_out_tile_.rhs_dim; n <= max_out_tile_.rhs_dim; n *= 2) {
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
      VLOG(10) << "Adding output tiling: config = " << new_config.ToString();
      updated_configs.push_back(new_config);
    }
  }
}

void TritonDotFusionSearchSpace::AddCtaSizeParameter(
    const ConfigWithNotes& config,
    std::vector<ConfigWithNotes>& updated_configs) {
  ConfigWithNotes new_config = config;
  int tile_rows = config.config.block_m;
  int tile_cols = config.config.block_n;
  int& warps = new_config.config.num_warps;
  CHECK_GT(tile_rows * tile_cols, 0)
      << "Need configs with output tilings determined.";
  int max_warps = GetMaxWarpsPerCta({tile_rows, tile_cols});
  VLOG(5) << "Computing max_warps: For output_tile = " << tile_rows << "x"
          << tile_cols
          << " and (wg)mma instruction shape, max_warps = " << max_warps;
  for (warps = min_warps_per_cta_; warps <= max_warps; warps *= 2) {
    VLOG(10) << "Adding CTA size parameter: config = " << new_config.ToString();
    updated_configs.push_back(new_config);
  }
}

void TritonDotFusionSearchSpace::EliminateLowOccupancyConfigs(
    std::vector<ConfigWithNotes>& configs) {
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
