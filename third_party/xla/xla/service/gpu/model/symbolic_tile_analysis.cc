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
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::MLIRContext;

// Computes indexing map from program id into the tile offset for the given
// shape and tile sizes.
IndexingMap ComputeBlockIdToOutputTileIndexing(
    absl::Span<const int64_t> dimensions, absl::Span<const int64_t> tile_sizes,
    mlir::MLIRContext* mlir_context) {
  CHECK_EQ(dimensions.size(), tile_sizes.size());  // Crash OK

  int num_tiles = 1;
  std::vector<int64_t> outer_loop_bounds;
  outer_loop_bounds.reserve(dimensions.size());
  for (auto [dim_size, tile_size] : llvm::zip(dimensions, tile_sizes)) {
    int num_tiles_per_dim = (dim_size + tile_size - 1) / tile_size;

    num_tiles *= num_tiles_per_dim;
    outer_loop_bounds.push_back(num_tiles_per_dim);
  }

  mlir::AffineExpr program_id = mlir::getAffineDimExpr(0, mlir_context);

  // Delinearize the block id.
  auto tile_exprs =
      DelinearizeIndex(outer_loop_bounds, program_id, mlir_context);

  // Scale each index by the tile size to produce tile offset.
  for (auto [tile_expr, tile_size] : llvm::zip(tile_exprs, tile_sizes)) {
    tile_expr = tile_expr * tile_size;
  }

  return IndexingMap::FromTensorSizes(
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0, tile_exprs, mlir_context),
      /*dim_upper_bounds=*/{num_tiles}, /*symbol_upper_bounds=*/{});
}

absl::StatusOr<IndexingMap> ComputeBlockIdToTileOffsetIndexing(
    const SymbolicTiledHloInstruction& tiled_hlo,
    const IndexingMap& block_id_to_root_tile_offset,
    mlir::MLIRContext* mlir_context) {
  IndexingMap block_id_to_tile_offset_indexing = ComposeIndexingMaps(
      block_id_to_root_tile_offset, tiled_hlo.indexing_map());

  // A symbol in an indexing map means that to produce on element of output, we
  // need to read all elements of input in the symbol range. Since this function
  // computes start of the tile, we need to substitute each symbol with its
  // lower bound value. We assume here the iteration order is normalized.
  // TODO(b/330906085): Support cases when tile offsets are not 0.
  if (absl::c_any_of(block_id_to_tile_offset_indexing.GetSymbolBounds(),
                     [](const Interval& symbol_bound) {
                       return symbol_bound.lower != 0;
                     })) {
    return absl::FailedPreconditionError(
        absl::StrCat("Symbol lower bound is not zero. ",
                     block_id_to_tile_offset_indexing.ToString()));
  }

  std::vector<AffineExpr> symbol_lower_bounds(
      block_id_to_tile_offset_indexing.GetSymbolCount(),
      mlir::getAffineConstantExpr(0, mlir_context));

  mlir::AffineMap simplified_affine_map =
      block_id_to_tile_offset_indexing.GetAffineMap().replaceDimsAndSymbols(
          /*dimReplacements=*/{}, symbol_lower_bounds,
          block_id_to_tile_offset_indexing.GetDimVarsCount(),
          /*numResultSyms=*/
          block_id_to_tile_offset_indexing.GetRangeVarsCount());

  IndexingMap simplified_indexing_map = IndexingMap{
      simplified_affine_map, block_id_to_tile_offset_indexing.GetDimVars(),
      block_id_to_tile_offset_indexing.GetRangeVars(),
      block_id_to_tile_offset_indexing.GetRTVars()};

  simplified_indexing_map.Simplify();
  simplified_indexing_map.RescaleSymbols();
  simplified_indexing_map.RemoveUnusedSymbols();

  return simplified_indexing_map;
}

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
      CHECK_EQ(operand_indexing_map_set.size(), 1);  // Crash OK

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

absl::StatusOr<TiledHloComputation>
SymbolicTileAnalysis::ComputeTiledHloInstructions(
    const std::vector<int64_t>& tile_parameters) const {
  IndexingMap block_id_to_root_tile_offset = ComputeBlockIdToOutputTileIndexing(
      GetRoot()->hlo()->shape().dimensions(), tile_parameters, context_);

  std::vector<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions;
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, TiledHloInstruction*>
      symbolic_to_tiled_hlo_map;
  absl::flat_hash_set<TiledHloInstruction*, TiledHloInstruction::PtrHash,
                      TiledHloInstruction::PtrEqual>
      tiled_hlo_instructions_set;

  absl::flat_hash_map<TiledHloInstruction*, int64_t> topological_order;

  std::function<absl::StatusOr<TiledHloInstruction*>(
      const SymbolicTiledHloInstruction*)>
      get_tiled_hlo_instruction;

  get_tiled_hlo_instruction =
      [&](const SymbolicTiledHloInstruction* symbolic_tiled_hlo)
      -> absl::StatusOr<TiledHloInstruction*> {
    auto it1 = symbolic_to_tiled_hlo_map.find(symbolic_tiled_hlo);
    if (it1 != symbolic_to_tiled_hlo_map.end()) {
      return it1->second;
    }

    std::vector<int64_t> tile_sizes =
        symbolic_tiled_hlo->TileSizes(tile_parameters);
    std::vector<int64_t> tile_strides =
        symbolic_tiled_hlo->TileStrides(tile_parameters);

    TF_ASSIGN_OR_RETURN(
        IndexingMap block_id_to_block_offset_indexing,
        ComputeBlockIdToTileOffsetIndexing(
            *symbolic_tiled_hlo, block_id_to_root_tile_offset, context_));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<TiledHloInstruction> tiled_hlo_holder,
                        TiledHloInstruction::Create(
                            symbolic_tiled_hlo->hlo(), std::move(tile_sizes),
                            std::move(tile_strides),
                            std::move(block_id_to_block_offset_indexing)));

    auto it2 = tiled_hlo_instructions_set.find(tiled_hlo_holder.get());
    if (it2 != tiled_hlo_instructions_set.end()) {
      return *it2;
    }

    tiled_hlo_instructions.push_back(std::move(tiled_hlo_holder));
    TiledHloInstruction* tiled_hlo = tiled_hlo_instructions.back().get();
    tiled_hlo_instructions_set.insert(tiled_hlo);
    symbolic_to_tiled_hlo_map[symbolic_tiled_hlo] = tiled_hlo;

    for (SymbolicTiledHloInstruction* operand :
         symbolic_tiled_hlo->operands()) {
      TF_ASSIGN_OR_RETURN(TiledHloInstruction * tiled_operand,
                          get_tiled_hlo_instruction(operand));
      tiled_hlo->AppendOperand(tiled_operand);
    }

    topological_order[tiled_hlo] = topological_order.size();
    return tiled_hlo;
  };

  TF_CHECK_OK(get_tiled_hlo_instruction(GetRoot()).status());

  // Order instructions in def-before-use order.
  absl::c_sort(tiled_hlo_instructions, [&](const auto& i1, const auto& i2) {
    return topological_order.at(i1.get()) < topological_order.at(i2.get());
  });

  return TiledHloComputation::FromSortedTiledHloInstructions(
      std::move(tiled_hlo_instructions));
}

std::string SymbolicTileAnalysis::ToString(
    const AffineMapPrinter& printer) const {
  std::stringstream ss;
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<SymbolicTiledHloInstruction*, std::string> tile_names;

  for (const auto& tiled_hlo : symbolic_tiled_hlo_instructions_) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo.get()] = tile_name;

    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }

    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ")\n";

    ss << tiled_hlo->ToString();
  }
  return ss.str();
}

}  // namespace gpu
}  // namespace xla
