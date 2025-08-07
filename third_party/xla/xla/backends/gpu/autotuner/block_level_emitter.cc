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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/codegen/triton/tma_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {
// Computes a tile size for a given dimension `dim` such that:
// - It is a power of two,
// - It is at least `dim` (i.e., ≥ dim),
// - It does not exceed `max_tile_size`.
//
// - Special cases:
//     - If dim is a power of two ≤ max_tile_size, it returns dim.
//     - If dim is not a power of two, it returns the next power of two ≥ dim,
//       capped at max_tile_size.
//     - If dim <= 1, it returns dim.
//     - If dim >= max_tile_size, it returns max_tile_size.
//
// Parameters:
// - dim: the size of the dimension to tile (must be ≥ 0).
// - max_tile_size: the maximum allowed tile size (must be ≥ 1).
//
// Returns:
// - If dim <= 1: returns dim.
// - If dim >= max_tile_size: returns max_tile_size.
// - Otherwise: returns the smallest power of two ≥ dim, but ≤ max_tile_size.
//
// Examples:
//   GetTileSize(1, 16)   => 1
//   GetTileSize(7, 16)   => 8
//   GetTileSize(8, 16)   => 8
//   GetTileSize(16, 16)  => 16
//   GetTileSize(20, 16)  => 16
constexpr int64_t GetTileSize(int64_t dim, int max_tile_size) {
  if (dim <= 1) {
    return dim;
  }
  if (dim >= max_tile_size) {
    return max_tile_size;
  }
  return 1LL << static_cast<int64_t>(std::ceil(std::log2(dim)));
}

// Helper: resets all variable dimensions after 'index' to zero
void ResetTrailingDimensions(const std::vector<int64_t>& input,
                             std::vector<int64_t>& current, int64_t index) {
  int64_t dims = input.size();
  // Iterate over all dimensions after 'index'
  // Only reset dimensions that are variable (input[j] >= 0)
  for (int64_t j = index + 1; j < dims; ++j) {
    if (input[j] >= 0) {
      current[j] = 0;
    }
  }
}

// Helper: tries to advance to the next valid combination.
//
// Returns:
// - true: successfully advanced to the next combination (more combinations
//   available)
// - false: no more combinations (all combinations have been generated)
bool AdvanceToNextCombination(const std::vector<int64_t>& input,
                              std::vector<int64_t>& current) {
  int64_t dims = input.size();
  // Iterate dimensions from right to left
  for (int64_t i = dims - 1; i >= 0; --i) {
    // Skip fixed dimensions (negative values in input)
    if (input[i] < 0) {
      continue;
    }

    // If the current dimension can still be incremented
    if (current[i] < input[i]) {
      current[i]++;                                // Increment this dimension
      ResetTrailingDimensions(input, current, i);  // Reset all after it
      return true;  // Not done yet, next combination ready
    }
  }
  // If we reach here, all dimensions are at max and no increment possible
  return false;  // All combinations generated, done
}

// Generates all multi-dimensional integer combinations for a given shape.
//
// For each dimension `i` in `input`:
// - If input[i] >= 0 (variable dimension): the element at index `i` will
//   range from 0 up to `input[i]`, inclusive.
// - If input[i] < 0 (fixed dimension): the element at index `i` will be
//   fixed to the value of `input[i]`.
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
    current[i] = std::min(input[i], int64_t{0});
  }

  // Loop until all combinations are generated
  do {
    // Add a copy of the current combination to the result
    result.push_back(current);
    // Attempt to increment to the next combination
  } while (AdvanceToNextCombination(input, current));

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

void ExtendConfigsWithTma(
    std::vector<std::unique_ptr<BackendConfig>>& configs) {
  int64_t original_size = configs.size();
  for (int64_t i = 0; i < original_size; ++i) {
    BlockLevelFusionConfig original_config;
    if (!configs[i]->UnpackTo(&original_config)) {
      // This should not happen based on how configs are created.
      LOG(ERROR) << "Failed to unpack BlockLevelFusionConfig";
      continue;
    }
    BlockLevelFusionConfig new_config = original_config;
    new_config.set_is_tma_allowed(true);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(new_config);
    configs.push_back(std::move(any));
  }
}
}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
BlockLevelEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

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
      // ceil(log2(dim))
      log2_dims.push_back(static_cast<int64_t>(std::ceil(std::log2(dim))));
    }
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;

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
        output_tile->add_sizes(1LL << log2_dim);
      }
    }

    // Set default kernel execution parameters.
    config.set_num_warps(1);           // Number of warps per block.
    config.set_num_ctas(1);            // Number of thread blocks (CTAs).
    config.set_num_stages(1);          // Number of pipeline stages.
    config.set_is_tma_allowed(false);  // Can codegen attempt to use TMA?

    // Store the config (as a polymorphic BackendConfig).
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config);
    configs.push_back(std::move(any));
  }

  // Allow TMA tuning for Hopper+ devices when TMA flag is passed.
  bool autotune_tma =
      debug_options().xla_gpu_experimental_enable_triton_tma() &&
      IsTmaEnabledForDevice(target_config().device_description);
  if (autotune_tma) {
    ExtendConfigsWithTma(configs);
  }

  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
BlockLevelEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        absl::StrCat("BlockLevelEmitterBackend: unsupported instruction: ",
                     instr.ToString()));
  }
  // Attempt to extract an existing BlockLevelFusionConfig from the instruction.
  // Object nesting structure:
  // HloInstruction
  // └── GpuBackendConfig
  //     └── FusionBackendConfig
  //         └── BlockLevelFusionConfig
  if (instr.has_backend_config()) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                        instr.backend_config<GpuBackendConfig>());
    if (gpu_backend_config.has_fusion_backend_config()) {
      const FusionBackendConfig& fusion_backend_config =
          gpu_backend_config.fusion_backend_config();
      // If a BlockLevelFusionConfig is already present, return it directly.
      if (fusion_backend_config.has_block_level_fusion_config()) {
        auto any = std::make_unique<google::protobuf::Any>();
        any->PackFrom(fusion_backend_config.block_level_fusion_config());
        return any;
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
  config.set_num_warps(1);           // Number of warps per block.
  config.set_num_ctas(1);            // Number of thread blocks (CTAs).
  config.set_num_stages(1);          // Number of pipeline stages.
  config.set_is_tma_allowed(false);  // Can codegen attempt to use TMA?
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(config);
  return any;
}

absl::Status BlockLevelEmitterBackend::ApplyConfig(
    HloInstruction& instr, const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "BlockLevelEmitterBackend does not support this instruction.");
  }
  // Object nesting structure:
  // HloInstruction
  // └── GpuBackendConfig
  //     └── FusionBackendConfig
  //         └── BlockLevelFusionConfig
  // Ensure the provided config is of type BlockLevelFusionConfig.
  BlockLevelFusionConfig block_level_fusion_config;
  if (!config.UnpackTo(&block_level_fusion_config)) {
    return absl::InvalidArgumentError(
        "Invalid backend config type for BlockLevelFusionConfig.");
  }
  // Extract the current GPU backend config from the instruction.
  // This contains the nested FusionBackendConfig we want to modify.
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_backend_config.mutable_fusion_backend_config();
  // Overwrite the block-level fusion config with the new one provided.
  *backend_config.mutable_block_level_fusion_config() =
      block_level_fusion_config;
  // Re-attach the modified GPU config back to the instruction.
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_backend_config)));
  return absl::OkStatus();
}

bool BlockLevelEmitterBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();
  return backend_config.kind() == kTritonFusionKind;
}

}  // namespace gpu
}  // namespace xla
