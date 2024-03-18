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
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_context.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/tile_analysis.h"
#include "xla/shape.h"
#include "xla/status.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::SmallVector;

struct HloAndPath {
  const HloInstruction* hlo;
  SymbolicTileAnalysis::InstructionPathFromRoot path;
};

}  // namespace

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeComputation(
    const HloComputation& computation, IndexingContext* ctx) {
  absl::flat_hash_map<InstructionPathFromRoot, SymbolicTile>
      symbolic_tile_from_path;
  ConstHloInstructionMap<absl::flat_hash_set<InstructionPathFromRoot>>
      paths_from_root_to_instruction;
  absl::flat_hash_map<const InstructionPathFromRoot, IndexingMap>
      indexing_map_from_path;
  std::queue<HloAndPath> to_process;

  const HloInstruction* root = computation.root_instruction();
  paths_from_root_to_instruction.insert({root, {{}}});

  to_process.push(HloAndPath{root, /*path=*/{}});
  indexing_map_from_path.insert({{}, CreateIdentityMap(root->shape(), ctx)});

  while (!to_process.empty()) {
    const HloAndPath hlo_and_path = to_process.front();
    to_process.pop();

    const HloInstruction* hlo = hlo_and_path.hlo;

    // Bail out on instructions that are known to cause problems down the line.
    // This is not an inherent limitation of the approach, but simply issues
    // to be resolved in the current implementation.
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kReshape ||
        hlo->opcode() == HloOpcode::kBitcast ||
        hlo->opcode() == HloOpcode::kConcatenate) {
      return absl::StrCat("Bailing out on ", hlo->ToString()).c_str();
    }

    // Bail out on instructions that do not output a single array.
    if (!hlo->shape().IsArray()) {
      return absl::StrCat(hlo->ToString(), " outputs more than a single array")
          .c_str();
    }

    const IndexingMap& hlo_indexing_map =
        indexing_map_from_path.at(hlo_and_path.path);

    std::optional<SymbolicTile> symbolic_tile =
        SymbolicTile::FromIndexingMap(hlo_indexing_map);
    if (!symbolic_tile.has_value()) {
      return absl::StrCat("Failed to compute symbolic tile for ",
                          hlo_indexing_map.ToString(), " for HLO ",
                          hlo->ToString())
          .c_str();
    }
    symbolic_tile_from_path.insert({hlo_and_path.path, symbolic_tile.value()});

    std::optional<HloInstructionIndexing> operands_indexing =
        ComputeOutputToInputIndexing(hlo, /*output_id=*/0, ctx);

    if (!operands_indexing.has_value()) {
      return absl::StrCat("Failed to compute operands indexing for ",
                          hlo->ToString())
          .c_str();
    }

    int operand_id = 0;
    for (auto [operand, operand_indexing_map_set] :
         llvm::zip(hlo->operands(), operands_indexing->indexing_maps)) {
      // Assign hlo_indexing_map again, since the reference may have been
      // invalidated by the insertion below.
      const IndexingMap& hlo_indexing_map =
          indexing_map_from_path.at(hlo_and_path.path);
      CHECK_EQ(operand_indexing_map_set.size(), 1);

      IndexingMap operand_indexing_map = ComposeIndexingMaps(
          hlo_indexing_map, *operand_indexing_map_set.begin());

      InstructionPathFromRoot operand_path = InstructionPathFromRoot(
          hlo_and_path.path.begin(), hlo_and_path.path.end());
      operand_path.push_back(operand_id);

      indexing_map_from_path.insert({operand_path, operand_indexing_map});
      to_process.push(HloAndPath{operand, operand_path});

      // TODO(bchetioui): replace instances of 'count' with 'contains' once OSS
      // builds use C++20.
      if (paths_from_root_to_instruction.count(operand) == 0) {
        paths_from_root_to_instruction.insert({operand, {operand_path}});
      } else {
        paths_from_root_to_instruction.at(operand).insert(operand_path);
      }

      ++operand_id;
    }
  }

  return SymbolicTileAnalysis(symbolic_tile_from_path,
                              paths_from_root_to_instruction, ctx);
}

namespace {

std::vector<int64_t> EvaluateTileMap(AffineMap affine_map,
                                     absl::Span<int64_t const> parameters) {
  CHECK_EQ(affine_map.getNumSymbols(), parameters.size());
  CHECK_EQ(affine_map.getNumDims(), 0);

  SmallVector<AffineExpr> symbol_replacements = llvm::to_vector(
      llvm::map_range(parameters, [affine_map](const int64_t v) -> AffineExpr {
        return mlir::getAffineConstantExpr(v, affine_map.getContext());
      }));

  mlir::AffineMap simplified_affine_map =
      mlir::simplifyAffineMap(affine_map.replaceDimsAndSymbols(
          /*dimReplacements=*/{}, symbol_replacements, /*numResultDims=*/0,
          /*numResultSyms=*/0));

  SmallVector<int64_t> results = llvm::to_vector(llvm::map_range(
      simplified_affine_map.getResults(), [](AffineExpr result) -> int64_t {
        return llvm::cast<mlir::AffineConstantExpr>(result).getValue();
      }));

  return std::vector<int64_t>(results.begin(), results.end());
}

}  // namespace

std::vector<int64_t> SymbolicTileAnalysis::TileOffsets(
    const HloInstruction* hlo, const InstructionPathFromRoot& path) const {
  CHECK(tile_parameters_.has_value());
  // TODO(bchetioui): replace instances of 'count' with 'contains' once OSS
  // builds use C++20.
  CHECK_EQ(paths_from_root_to_instruction_.count(hlo), 1);
  CHECK_EQ(paths_from_root_to_instruction_.at(hlo).count(path), 1);
  return EvaluateTileMap(symbolic_tile_from_path_.at(path).offset_map(),
                         *tile_parameters_);
}

// TODO(bchetioui): remove dependency on stride and offset parameters.
std::vector<int64_t> SymbolicTileAnalysis::TileSizes(
    const HloInstruction* hlo, const InstructionPathFromRoot& path) const {
  CHECK(tile_parameters_.has_value());
  // TODO(bchetioui): replace instances of 'count' with 'contains' once OSS
  // builds use C++20.
  CHECK_EQ(paths_from_root_to_instruction_.count(hlo), 1);
  CHECK_EQ(paths_from_root_to_instruction_.at(hlo).count(path), 1);
  return EvaluateTileMap(symbolic_tile_from_path_.at(path).size_map(),
                         *tile_parameters_);
}

std::vector<int64_t> SymbolicTileAnalysis::TileStrides(
    const HloInstruction* hlo, const InstructionPathFromRoot& path) const {
  CHECK(tile_parameters_.has_value());
  // TODO(bchetioui): replace instances of 'count' with 'contains' once OSS
  // builds use C++20.
  CHECK_EQ(paths_from_root_to_instruction_.count(hlo), 1);
  CHECK_EQ(paths_from_root_to_instruction_.at(hlo).count(path), 1);
  return EvaluateTileMap(symbolic_tile_from_path_.at(path).stride_map(),
                         *tile_parameters_);
}

void SymbolicTileAnalysis::SetTileParameters(
    absl::Span<int64_t const> parameters) {
  // TODO(bchetioui): CHECK num parameters somehow?
  tile_parameters_ = std::vector(parameters.begin(), parameters.end());
}

}  // namespace gpu
}  // namespace xla
