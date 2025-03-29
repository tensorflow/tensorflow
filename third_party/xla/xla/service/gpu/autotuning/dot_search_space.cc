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
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
    const Shape& shape, const ::proto2::RepeatedField<int64_t>& dimensions) {
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
      // TODO: b/404470821 - Compute these from the problem properties instead
      // of hardcoding.
      desired_total_warps_(2160),
      max_out_tile_{64, 128},
      min_warps_per_block_(4),
      min_contracting_tile_size_(16),
      max_contracting_split_(GetMaxContractingSplit(max_out_tile_)) {}

std::vector<TritonGemmConfig> TritonDotFusionSearchSpace::GenerateConfigs(
    std::optional<int64_t> force_contracting_split) {
  std::vector<TritonGemmConfig> configs;
  if (force_contracting_split.has_value()) {
    TritonGemmConfig config;
    config.split_k = force_contracting_split.value();
    configs.push_back(config);
  } else {
    configs = GenerateContractingSplitFactors();
  }
  // TODO: b/404470821 - Implement this properly rather than hardcoding the
  // config parameters.
  for (auto& config : configs) {
    config.block_m = 64;
    config.block_n = 128;
    config.block_k = 64;
    config.num_stages = 3;
    config.num_warps = 4;
    config.num_ctas = 1;
  }
  return configs;
}

std::string TritonDotFusionSearchSpace::Serialize() {
  return absl::StrFormat(
      "problem_size_BxMxNxK: %dx%dx%dx%d "
      "tile_range_SxMxNxK: [1-%d]x[1-%d]x[1-%d]x[%d-?] "
      "desired_total_warps: %d warps_per_block: [%d-?]",
      batch_size_, lhs_parallel_size_, rhs_parallel_size_, contracting_size_,
      max_contracting_split_, max_out_tile_.lhs_dim, max_out_tile_.rhs_dim,
      min_contracting_tile_size_, desired_total_warps_, min_warps_per_block_);
}

int64_t TritonDotFusionSearchSpace::GetNumResultTiles(
    TritonDotFusionSearchSpace::OutputTile output_tile) const {
  return batch_size_ *
         CeilOfRatio<int64_t>(lhs_parallel_size_, output_tile.lhs_dim) *
         CeilOfRatio<int64_t>(rhs_parallel_size_, output_tile.rhs_dim);
}

int TritonDotFusionSearchSpace::GetMaxContractingSplit(
    TritonDotFusionSearchSpace::OutputTile output_tile) const {
  const int64_t desired_num_blocks =
      desired_total_warps_ / min_warps_per_block_;
  VLOG(5) << "Computing split_k: Considering output tile "
          << output_tile.lhs_dim << "x" << output_tile.rhs_dim;
  VLOG(5) << "Computing split_k: Want up to " << desired_num_blocks
          << " blocks to occupy all cores.";

  const int64_t min_result_tiles = GetNumResultTiles(output_tile);
  VLOG(5) << "Computing split_k: Without split_k have " << min_result_tiles
          << " tiles.";

  const int64_t split_for_occupancy =
      NextPowerOfTwo(CeilOfRatio(desired_num_blocks, min_result_tiles));
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

std::vector<TritonGemmConfig>
TritonDotFusionSearchSpace::GenerateContractingSplitFactors() {
  std::vector<TritonGemmConfig> configs;
  TritonGemmConfig config;
  for (int split = 1; split <= max_contracting_split_; split *= 2) {
    config.split_k = split;
    VLOG(10) << "Generating contracting split factors: config = "
             << config.ToString();
    configs.push_back(config);
  }
  return configs;
}

}  // namespace xla::gpu
