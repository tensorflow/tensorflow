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

#ifndef XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"

namespace xla {
namespace gpu {

class SymbolicTileAnalysis;
using SymbolicTileAnalysisOrError =
    std::variant<SymbolicTileAnalysis, FusionDecision>;

// Constructs and holds symbolic tiles for all the instructions within a
// computation. We may hold several different symbolic tiles for the same
// instruction if the instruction is indexed in several different ways in order
// to produce a single chunk of the output. In order to handle this properly,
// we store a symbolic tile for each possible path starting from the root
// instruction of the computation to the relevant instruction.
class SymbolicTileAnalysis {
 public:
  // Tries to construct a symbolic tile analysis from a computation. Returns
  // a diagnostic if the construction fails for any reason.
  static SymbolicTileAnalysisOrError AnalyzeComputation(
      const HloComputation& computation, mlir::MLIRContext* ctx);

  // Evaluates the tile offsets of an instruction from the analyzed computation
  // following the provided path from the root. Tile parameters must have been
  // set before calling this method.
  std::vector<int64_t> TileOffsets(
      const SymbolicTiledHloInstruction& tiled_hlo) const;
  // Evaluates the tile sizes of an instruction from the analyzed computation
  // following the provided path from the root. Tile parameters must have been
  // set before calling this method.
  std::vector<int64_t> TileSizes(
      const SymbolicTiledHloInstruction& tiled_hlo) const;
  // Evaluates the tile strides of an instruction from the analyzed computation
  // following the provided path from the root. Tile parameters must have been
  // set before calling this method.
  std::vector<int64_t> TileStrides(
      const SymbolicTiledHloInstruction& tiled_hlo) const;

  // Computes the indexing map from block id to tile offset of the tiled HLO
  // instruction. The indexing map has the following form:
  //
  // (block_id) -> (tile_offset0, tile_offset1, ...)
  absl::StatusOr<IndexingMap> ComputeBlockIdToTileOffsetIndexing(
      const SymbolicTiledHloInstruction& tiled_hlo) const;

  // Populates input tile sizes. This is a prerequisite in order to extract
  // concrete values using `TileOffsets`, `TileSizes`, and `TileStrides`.
  void SetTileSizes(std::vector<int64_t> sizes);

  // Returns the tiled root instruction.
  const SymbolicTiledHloInstruction* GetRoot() const {
    return tiled_hlo_instructions_.back().get();
  }

  // Returns the tiled HLO instructions in def-before-use order.
  const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
  GetTiledHloInstructions() const {
    return tiled_hlo_instructions_;
  }

  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const { return context_; };

 private:
  SymbolicTileAnalysis(std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
                           tiled_hlo_instructions,
                       mlir::MLIRContext* context)
      : tiled_hlo_instructions_(std::move(tiled_hlo_instructions)),
        context_(context) {}

  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions_;

  mlir::MLIRContext* context_;
  // Optionally set tile parameters. These parameters can be set by calling
  // `SetTileParameters`, and correspond to the output tile for the analyzed
  // computation. The order and type of parameters are as explained in the
  // documentation of `SymbolicTile`.
  std::optional<std::vector<int64_t>> tile_parameters_;

  // Indexing map from block id to root tile offset. Computed from the tile
  // parameters.
  std::optional<IndexingMap> block_id_to_root_tile_offset_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_
