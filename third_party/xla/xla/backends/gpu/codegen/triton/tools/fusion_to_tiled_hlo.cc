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

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/codegen/xtile/tiling_from_block_parameters.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"

namespace xla::gpu {
namespace {

using ::xla::gpu::experimental::TiledHloComputation;
using ::xla::gpu::experimental::TilingSpace;
using ::xla::xtile::BlockLevelParameters;

absl::StatusOr<llvm::SmallVector<int64_t>> ParseTileSizes(
    absl::string_view tile_sizes_str) {
  llvm::SmallVector<int64_t> tile_sizes;
  for (absl::string_view part :
       absl::StrSplit(tile_sizes_str, ',', absl::SkipEmpty())) {
    int64_t size = 0;
    if (!absl::SimpleAtoi(part, &size) || size <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid tile size: '", part,
                       "'. Tile sizes must be positive integers."));
    }
    tile_sizes.push_back(size);
  }
  return tile_sizes;
}

absl::Status AssignTileSizesIfAvailable(
    const HloInstruction& instr,
    std::optional<llvm::SmallVector<int64_t>> explicit_tile_sizes,
    std::unique_ptr<TilingSpace>& tiling_space) {
  if (explicit_tile_sizes.has_value()) {
    if (explicit_tile_sizes->size() != tiling_space->num_dimensions()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of explicit tile sizes (", explicit_tile_sizes->size(),
          ") does not match number of dimensions in TilingSpace (",
          tiling_space->num_dimensions(), ")."));
    }
    return tiling_space->AssignTileSizes(*explicit_tile_sizes);
  }

  if (instr.opcode() != HloOpcode::kFusion || !instr.has_backend_config()) {
    return absl::OkStatus();
  }

  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok() || !gpu_config->has_fusion_backend_config() ||
      !gpu_config->fusion_backend_config().has_block_level_fusion_config()) {
    return absl::OkStatus();
  }

  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          gpu_config->fusion_backend_config().block_level_fusion_config());
  const DebugOptions& debug_options =
      instr.GetModule()->config().debug_options();
  bool enable_same_shape_mof =
      debug_options
          .xla_gpu_experimental_enable_same_shape_multi_output_fusion();
  if (block_level_parameters.output_tile_sizes.size() > 1 &&
      !enable_same_shape_mof &&
      debug_options.xla_gpu_unsupported_enable_triton_multi_output_fusion()) {
    std::vector<int64_t> flattened;
    for (const auto& sizes : block_level_parameters.output_tile_sizes) {
      flattened.insert(flattened.end(), sizes.begin(), sizes.end());
    }
    block_level_parameters.output_tile_sizes = {std::move(flattened)};
  }
  ABSL_ASSIGN_OR_RETURN(
      llvm::SmallVector<int64_t> tile_sizes,
      xtile::GetTilingSpaceConcreteSizes(*tiling_space, block_level_parameters,
                                         enable_same_shape_mof));
  for (int64_t& size : tile_sizes) {
    size = size == 0 ? 1 : llvm::PowerOf2Ceil(size);
  }
  return tiling_space->AssignTileSizes(tile_sizes);
}

absl::StatusOr<TiledHloComputation> TileInstruction(
    const HloInstruction& instr,
    std::optional<llvm::SmallVector<int64_t>> explicit_tile_sizes,
    bool symbolic, mlir::MLIRContext& mlir_context) {
  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(&instr);
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<TilingSpace> tiling_space,
                   TilingSpace::Create(*fusion_adaptor, &mlir_context));
  if (!symbolic) {
    ABSL_RETURN_IF_ERROR(AssignTileSizesIfAvailable(
        instr, std::move(explicit_tile_sizes), tiling_space));
  }
  ABSL_ASSIGN_OR_RETURN(
      TiledHloComputation tiled_computation,
      TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space)));
  tiled_computation.Simplify();
  tiled_computation.SortInstructionsPostOrder();
  return tiled_computation;
}

const HloInstruction* GetOrCreateEntryTarget(HloComputation* entry) {
  HloInstruction* root = entry->root_instruction();
  bool needs_fusion_wrapper = root->opcode() == HloOpcode::kTuple;
  for (const HloInstruction* operand : root->operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      needs_fusion_wrapper = true;
      break;
    }
  }
  if (!needs_fusion_wrapper) {
    return root;
  }
  std::vector<const HloInstruction*> reachable =
      HloBfsFindAll({root}, [](const HloInstruction*) { return true; });
  std::vector<HloInstruction*> post_order = entry->MakeInstructionPostOrder();
  std::vector<HloInstruction*> to_fuse;
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    HloInstruction* instr = *it;
    if (instr->opcode() != HloOpcode::kParameter &&
        absl::c_linear_search(reachable, instr)) {
      to_fuse.push_back(instr);
    }
  }
  return entry->CreateFusionInstruction(to_fuse,
                                        HloInstruction::FusionKind::kCustom);
}

absl::Status RealMain(absl::string_view input_file,
                      absl::string_view tile_sizes_str,
                      absl::string_view fusion_name, bool symbolic) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   xla::LoadModuleFromFile(std::string(input_file)));

  std::optional<llvm::SmallVector<int64_t>> explicit_tile_sizes;
  if (!tile_sizes_str.empty()) {
    ABSL_ASSIGN_OR_RETURN(explicit_tile_sizes, ParseTileSizes(tile_sizes_str));
  }

  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);

  std::vector<const HloInstruction*> targets;
  for (const HloInstruction* instr :
       hlo_module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      if (fusion_name.empty() || instr->name() == fusion_name ||
          instr->fused_instructions_computation()->name() == fusion_name) {
        targets.push_back(instr);
      }
    }
  }

  if (targets.empty()) {
    if (!fusion_name.empty()) {
      return absl::NotFoundError(
          absl::StrCat("Fusion '", fusion_name, "' not found in module."));
    }
    targets.push_back(GetOrCreateEntryTarget(hlo_module->entry_computation()));
  }

  for (const HloInstruction* target : targets) {
    if (targets.size() > 1) {
      llvm::outs() << "Fusion: " << target->name() << "\n";
    }
    ABSL_ASSIGN_OR_RETURN(
        TiledHloComputation tiled_computation,
        TileInstruction(*target, explicit_tile_sizes, symbolic, mlir_context));
    llvm::outs() << tiled_computation.ToString();
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  std::string tile_sizes;
  std::string fusion_name;
  bool symbolic = false;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("tile_sizes", &tile_sizes,
                "Comma-separated concrete tile sizes for TilingSpace "
                "dimensions. Overrides block_level_fusion_config if set."),
      tsl::Flag("fusion_name", &fusion_name,
                "Optional name of the fusion instruction or fused computation "
                "to tile."),
      tsl::Flag("symbolic", &symbolic,
                "If true, keep tile sizes symbolic even if "
                "block_level_fusion_config is present."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);
  if (!parse_ok) {
    // Print the usage using cerr to avoid truncation by LOG.
    std::cerr << kUsageString;
    return 1;
  }
  CHECK_GT(argc, 1) << "Must specify an input file";
  absl::Status status =
      xla::gpu::RealMain(argv[1], tile_sizes, fusion_name, symbolic);
  if (!status.ok()) {
    // We don't return non-zero codes as some of the tests check the status.
    std::cerr << status << "\n";
  }
  return 0;
}
