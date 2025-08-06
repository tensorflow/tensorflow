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

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
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
BlockLevelEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  return absl::UnimplementedError("GetSupportedConfigs is not implemented yet");
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

absl::Status BlockLevelEmitterBackend::ApplyConfig(
    HloInstruction& instr, const BackendConfig& config) {
  return absl::UnimplementedError("ApplyConfig is not implemented yet");
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
