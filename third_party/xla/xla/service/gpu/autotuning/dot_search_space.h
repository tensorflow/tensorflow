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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_DOT_SEARCH_SPACE_H_
#define XLA_SERVICE_GPU_AUTOTUNING_DOT_SEARCH_SPACE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Generates the space of promising Triton configs for a given dot fusion
// and hardware.
//
// Takes into account the properties of the problem (e.g., operand and result
// shapes, fused instructions), and the hardware (e.g., number of cores,
// available registers and memory per core).
//
// Internal doc with rationale: go/xla-gpu-dot-search
class TritonDotFusionSearchSpace {
 public:
  explicit TritonDotFusionSearchSpace(
      const se::DeviceDescription& device_description,
      const HloDotInstruction* dot);

  // Generates the list of promising configs in the search space for the
  // autotuner to try. If `force_contracting_split` is set, the search space
  // will be restricted to only include configs with the given split_k factor.
  std::vector<TritonGemmConfig> GenerateConfigs(
      std::optional<int64_t> force_contracting_split = std::nullopt);

  // Serializes the search space to a human-readable string.
  std::string Serialize();

 private:
  // Groups together the tiling of the dot's output dimensions: the parallel
  // dimensions of the left and right hand sides. We assume that any batch
  // dimensions are tiled by a factor of 1.
  struct OutputTile {
    int lhs_dim;  // LHS tiling (aka. block_m).
    int rhs_dim;  // RHS tiling (aka. block_n).
  };

  // Computes the number of result tiles we would have without
  // splitting the contracting dimension for a given output tile.
  int64_t GetNumResultTiles(OutputTile output_tile) const;

  // Computes the maximum sensible split in the contracting dimension
  // (split_k) to sufficiently occupy all available cores when using the given
  // output tile.
  int GetMaxContractingSplit(OutputTile output_tile) const;

  // Finds all promising values for splitting the contracting dimension to
  // achieve sufficient occupancy (split_k).
  std::vector<TritonGemmConfig> GenerateContractingSplitFactors();

  se::DeviceDescription device_description_;
  int64_t contracting_size_;
  int64_t batch_size_;
  int64_t lhs_parallel_size_;
  int64_t rhs_parallel_size_;
  int desired_total_warps_;
  OutputTile max_out_tile_;
  int min_warps_per_block_;
  int min_contracting_tile_size_;
  int max_contracting_split_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_AUTOTUNING_DOT_SEARCH_SPACE_H_
