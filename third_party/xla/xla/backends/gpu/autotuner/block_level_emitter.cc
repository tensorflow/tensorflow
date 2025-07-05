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

#include "xla/backends/gpu/autotuner/block_level_emitter.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {
// Computes the floor of the base-2 logarithm of a positive integer.
// This function returns the largest integer `n` such that 2^n <= x.
// It is equivalent to computing floor(log2(x)).
// Parameters:
// - x: a positive integer (must be >= 1)
// Note:
// - Behavior is undefined if `x` < 1.
constexpr int64_t Log2i(int64_t x) {
  int64_t log = 0;
  while (x >>= 1) {
    ++log;
  }
  return log;
}

// Computes a tile size for a given dimension `dim` such that:
// - It is a power of two,
// - It does not exceed `dim`,
// - It does not exceed `max_tile_size`.
//
// The returned tile size is the largest power of two less than or equal to
// `dim`, unless `dim` is less than or equal to 1, or greater than or equal to
// `max_tile_size`.
//
// Parameters:
// - dim: the size of the dimension to tile (must be ≥ 0).
// - max_tile_size: the maximum allowed tile size (must be ≥ 1).
//
// Returns:
// - If dim <= 1: returns dim.
// - If dim >= max_tile_size: returns max_tile_size.
// - Otherwise: returns the largest power of two less than or equal to `dim`.
//
// Examples:
//   GetTileSize(1, 16)   => 1
//   GetTileSize(7, 16)   => 4
//   GetTileSize(16, 16)  => 16
//   GetTileSize(20, 16)  => 16
//   GetTileSize(20, 8)   => 8
constexpr int64_t GetTileSize(int64_t dim, int max_tile_size) {
  if (dim <= 1) {
    return dim;
  }
  if (dim >= max_tile_size) {
    return max_tile_size;
  }
  return 1 << Log2i(dim);
}

// Generates all multi-dimensional integer combinations for a given shape.
//
// For each dimension `i` in `input`:
// - If input[i] >= 0: that index ranges from 0 to input[i], inclusive.
// - If input[i] < 0: the index is fixed to input[i] in all combinations.
//
// For example, given input = {2, MIN_INT, 3}, the function returns:
// {
//   {0, MIN_INT, 0}, {0, MIN_INT, 1}, {0, MIN_INT, 2}, {0, MIN_INT, 3},
//   {1, MIN_INT, 0}, {1, MIN_INT, 1}, {1, MIN_INT, 2}, {1, MIN_INT, 3},
//   {2, MIN_INT, 0}, {2, MIN_INT, 1}, {2, MIN_INT, 2}, {2, MIN_INT, 3}
// }
//
// Parameters:
// - input: a vector of integers representing upper bounds (inclusive) for each
//          dimension. A negative value indicates that the dimension is fixed to
//          that value.
//
// Returns:
// - A vector of integer vectors, where each inner vector is a unique
// combination.
//
// Notes:
// - The number of combinations is the product of all (input[i] + 1) where
// input[i] >= 0.
// - Each combination has the same length as `input`.
// - For dimensions with input[i] < 0, that value is used directly in all
//   outputs.
std::vector<std::vector<int64_t>> GenerateCombinations(
    const std::vector<int64_t>& input) {
  std::vector<std::vector<int64_t>> result;
  if (input.empty()) {
    return result;
  }
  int64_t dims = input.size();
  std::vector<int64_t> current(dims);

  // Initialize each dimension: 0 for variable, input[i] if fixed
  for (int64_t i = 0; i < dims; ++i) {
    current[i] = (input[i] < 0) ? input[i] : 0;
  }
  while (true) {
    result.push_back(current);
    // Increment from the last dimension backward
    int64_t i = dims - 1;
    while (i >= 0) {
      if (input[i] <= 0) {
        --i;
        continue;
      }
      current[i]++;
      if (current[i] <= input[i]) {
        break;
      }
      current[i] = 0;
      --i;
    }
    if (i < 0) {
      break;  // Done when we've looped through all dimensions
    }
  }
  return result;
}

// Recursively traverses a Shape object in depth-first order,
// collecting the dimensions of all array shapes encountered.
//
// Parameters:
// - shape: The Shape object to traverse. Can be a tuple (nested) or an array.
// - result: A vector where dimensions of each encountered array shape are
// appended.
//
// Behavior:
// - If `shape` is an array, its dimensions are added to `result`.
// - If `shape` is a tuple, each element is recursively traversed.
//
// This helper flattens a potentially nested shape into a flat list of array
// dimension spans.
void DfsShapes(const Shape& shape,
               std::vector<absl::Span<const int64_t>>& result) {
  if (shape.IsArray()) {
    result.push_back(shape.dimensions());
  } else if (shape.IsTuple()) {
    for (const Shape& element_shape : shape.tuple_shapes()) {
      DfsShapes(element_shape, result);
    }
  }
}

// Returns a flattened list of all array shapes (their dimension spans)
// contained within the shape of the given HLO Instruction.
//
// Parameters:
// - instr: The HLO Instruction whose shape is to be flattened.
//
// Returns:
// - A vector of spans, each representing dimensions of an array shape
//   found in the instruction’s (possibly nested) shape.
//
// Internally uses `DfsShapes()` to perform depth-first traversal.
std::vector<absl::Span<const int64_t>> FlatListOfShapes(
    const HloInstruction& instr) {
  std::vector<absl::Span<const int64_t>> result;
  DfsShapes(instr.shape(), result);
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBlockLevelFusionEmitterBackend::GetSupportedConfigs(
    const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  // This backend only supports array shapes (not tuples, etc.)
  if (!instr.shape().IsArray()) {
    return absl::InvalidArgumentError(
        "Only array shapes are supported in block-level emitter "
        "GetSupportedConfigs.");
  }
  // Compute the base-2 logarithm (rounded down) of each dimension size.
  // This determines the range of tile sizes to explore in log2 space.
  std::vector<int64_t> log2_dims;
  for (const int64_t dim : instr.shape().dimensions()) {
    // Exclude zero-sized dimensions from tiling configuration.
    if (dim == 0) {
      // Use INT64_MIN as a sentinel to mark zero-sized dimensions.
      // These will be handled specially later.
      log2_dims.push_back(INT64_MIN);
    } else {
      log2_dims.push_back(Log2i(dim));  // floor(log2(dim))
    }
  }
  // Generate all possible combinations of tile sizes across dimensions,
  // by iterating over the space of log2(tile size) values.
  //
  // For example, if one dimension has log2 = 2 (i.e., dim=4),
  // this will generate tile sizes of 1, 2, and 4 for that dim.
  std::vector<std::vector<int64_t>> tile_log2_combinations =
      GenerateCombinations(log2_dims);
  // For each valid tile size combination, construct a corresponding config.
  for (const std::vector<int64_t>& tile_log2_dims : tile_log2_combinations) {
    BlockLevelFusionConfig config;
    Tile* output_tile = config.add_output_tiles();
    for (const int64_t log2_dim : tile_log2_dims) {
      if (log2_dim == INT64_MIN) {
        // Preserve 0-sized dimensions in the tile configuration.
        output_tile->add_sizes(0);
      } else {
        // Convert log2 size back to actual tile size (1 << log2).
        output_tile->add_sizes(1 << log2_dim);
      }
    }
    // Set default kernel execution parameters.
    config.set_num_warps(1);   // Number of warps per block.
    config.set_num_ctas(1);    // Number of thread blocks (CTAs).
    config.set_num_stages(1);  // Number of pipeline stages.
    // Store the config (as a polymorphic BackendConfig).
    configs.push_back(
        std::make_unique<BlockLevelFusionConfig>(std::move(config)));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
TritonBlockLevelFusionEmitterBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  // Attempt to extract an existing BlockLevelFusionConfig from the instruction.
  if (instr.has_backend_config()) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                        instr.backend_config<GpuBackendConfig>());
    // Check if a FusionBackendConfig exists inside the GPU backend config.
    if (gpu_backend_config.has_fusion_backend_config()) {
      const FusionBackendConfig& fusion_backend_config =
          gpu_backend_config.fusion_backend_config();
      // If a BlockLevelFusionConfig is already present, return it directly.
      if (fusion_backend_config.has_block_level_fusion_config()) {
        return std::make_unique<BlockLevelFusionConfig>(
            fusion_backend_config.block_level_fusion_config());
      }
    }
  }
  // No explicit config found - construct a default one.
  BlockLevelFusionConfig config;
  // Flatten the output shape(s) of the instruction.
  const auto shapes = FlatListOfShapes(instr);
  for (const absl::Span<const int64_t> shape : shapes) {
    Tile* output_tile = config.add_output_tiles();
    for (const int64_t dim : shape) {
      // Choose a tile size as the nearest power-of-two <= `dim`, capped at 16.
      output_tile->add_sizes(GetTileSize(dim, /*max_tile_size=*/16));
    }
  }
  // Set default kernel execution parameters.
  config.set_num_warps(1);   // Number of warps per block.
  config.set_num_ctas(1);    // Number of thread blocks (CTAs).
  config.set_num_stages(1);  // Number of pipeline stages.
  return std::make_unique<BlockLevelFusionConfig>(std::move(config));
}

absl::Status TritonBlockLevelFusionEmitterBackend::ApplyConfig(
    HloInstruction& instr, const BackendConfig& config) {
  // Object nesting structure:
  // HloInstruction
  // └── GpuBackendConfig
  //     └── FusionBackendConfig
  //         └── BlockLevelFusionConfig
  // Ensure the provided config is of type BlockLevelFusionConfig.
  if (config.GetDescriptor() != BlockLevelFusionConfig::GetDescriptor()) {
    return absl::InvalidArgumentError(
        "Invalid backend config type for BlockLevelFusionConfig.");
  }
  // Safe to cast now since we've checked the descriptor above.
  const BlockLevelFusionConfig& block_level_fusion_config =
      static_cast<const BlockLevelFusionConfig&>(config);
  // Extract the current GPU backend config from the instruction.
  // This contains the nested FusionBackendConfig we want to modify.
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  // Get a mutable reference to the nested FusionBackendConfig.
  FusionBackendConfig& backend_config =
      *gpu_backend_config.mutable_fusion_backend_config();
  // Overwrite the block-level fusion config with the new one provided.
  *backend_config.mutable_block_level_fusion_config() =
      block_level_fusion_config;
  // Re-attach the modified GPU config back to the instruction.
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_backend_config)));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
