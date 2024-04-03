/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/symbolic_tile_analysis.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::MLIRContext;
using ::mlir::SmallVector;

}  // namespace

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeComputation(
    const HloComputation& computation, MLIRContext* ctx) {
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions;
  absl::flat_hash_map<std::pair<const HloInstruction*, IndexingMap>,
                      SymbolicTiledHloInstruction*>
      tiled_hlo_instructions_map;

  absl::flat_hash_map<SymbolicTiledHloInstruction*, int64_t> topological_order;

  std::function<std::variant<SymbolicTiledHloInstruction*, FusionDecision>(
      const HloInstruction*, IndexingMap)>
      get_tiled_hlo_instruction;

  // Create a new tiled hlo instruction or return existing instruction from
  // cache for the given hlo and indexing map.
  get_tiled_hlo_instruction = [&](const HloInstruction* hlo,
                                  IndexingMap indexing_map)
      -> std::variant<SymbolicTiledHloInstruction*, FusionDecision> {
    auto key = std::make_pair(hlo, indexing_map);

    auto it = tiled_hlo_instructions_map.find(key);
    if (it != tiled_hlo_instructions_map.end()) {
      return it->second;
    }

    // Bail out on instructions that are known to cause problems down the
    // line. This is not an inherent limitation of the approach, but simply
    // issues to be resolved in the current implementation.
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kReshape ||
        hlo->opcode() == HloOpcode::kBitcast ||
        hlo->opcode() == HloOpcode::kConcatenate) {
      return FusionDecision{} << "Bailing out on " << hlo->ToString();
    }

    // Bail out on instructions that do not output a single array.
    if (!hlo->shape().IsArray()) {
      return FusionDecision{} << hlo->ToString()
                              << " outputs more than a single array";
    }

    auto symbolic_tile = SymbolicTile::FromIndexingMap(indexing_map);
    if (!symbolic_tile.has_value()) {
      return FusionDecision{} << "Failed to compute symbolic tile for "
                              << indexing_map.ToString() << " for HLO "
                              << hlo->ToString();
    }

    tiled_hlo_instructions.push_back(
        std::make_unique<SymbolicTiledHloInstruction>(
            hlo, std::move(indexing_map), std::move(*symbolic_tile)));

    auto tiled_hlo_instruction = tiled_hlo_instructions.back().get();

    std::optional<HloInstructionIndexing> operands_indexing =
        ComputeOutputToInputIndexing(tiled_hlo_instruction->hlo(),
                                     /*output_id=*/0, ctx);

    if (!operands_indexing.has_value()) {
      return FusionDecision{} << "Failed to compute operands indexing for "
                              << tiled_hlo_instruction->hlo()->ToString();
    }

    for (auto [operand, operand_indexing_map_set] :
         llvm::zip(tiled_hlo_instruction->hlo()->operands(),
                   operands_indexing->indexing_maps)) {
      CHECK_EQ(operand_indexing_map_set.size(), 1);

      IndexingMap operand_indexing_map =
          ComposeIndexingMaps(tiled_hlo_instruction->indexing_map(),
                              *operand_indexing_map_set.begin());

      auto tiled_operand_or =
          get_tiled_hlo_instruction(operand, std::move(operand_indexing_map));

      if (auto fusion_decison =
              std::get_if<FusionDecision>(&tiled_operand_or)) {
        return *fusion_decison;
      }

      tiled_hlo_instruction->AppendOperand(
          std::get<SymbolicTiledHloInstruction*>(tiled_operand_or));
    }

    topological_order[tiled_hlo_instruction] = topological_order.size();
    tiled_hlo_instructions_map.emplace(key, tiled_hlo_instruction);
    return tiled_hlo_instruction;
  };

  const HloInstruction* root = computation.root_instruction();
  auto tiled_root =
      get_tiled_hlo_instruction(root, CreateIdentityMap(root->shape(), ctx));
  if (auto* fusion_decision = std::get_if<FusionDecision>(&tiled_root)) {
    return *fusion_decision;
  }

  // Order instructions in def-before-use order.
  absl::c_sort(tiled_hlo_instructions, [&](const auto& i1, const auto& i2) {
    return topological_order.at(i1.get()) < topological_order.at(i2.get());
  });

  return SymbolicTileAnalysis(std::move(tiled_hlo_instructions), ctx);
}

std::vector<int64_t> SymbolicTileAnalysis::TileOffsets(
    const SymbolicTiledHloInstruction& tiled_hlo) const {
  CHECK(tile_parameters_.has_value());
  return tiled_hlo.TileOffsets(*tile_parameters_);
}

// TODO(bchetioui): remove dependency on stride and offset parameters.
std::vector<int64_t> SymbolicTileAnalysis::TileSizes(
    const SymbolicTiledHloInstruction& tiled_hlo) const {
  CHECK(tile_parameters_.has_value());
  return tiled_hlo.TileSizes(*tile_parameters_);
}

std::vector<int64_t> SymbolicTileAnalysis::TileStrides(
    const SymbolicTiledHloInstruction& tiled_hlo) const {
  CHECK(tile_parameters_.has_value());
  return tiled_hlo.TileStrides(*tile_parameters_);
}

void SymbolicTileAnalysis::SetTileSizes(std::vector<int64_t> sizes) {
  // TODO(bchetioui): CHECK num parameters somehow?
  tile_parameters_ = std::vector(std::move(sizes));
}

}  // namespace gpu
}  // namespace xla
