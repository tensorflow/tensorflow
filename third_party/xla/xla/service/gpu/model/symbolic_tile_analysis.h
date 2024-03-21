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
#include <optional>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_context.h"
#include "xla/service/gpu/model/tile_analysis.h"
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
  // `InstructionPathFromRoot` allows representing a graph path from the root
  // instruction of a computation up to one of its consumers. Each integer
  // in the path represents the index of the operand edge to follow to reach
  // the instruction, starting from the root instruction.
  using InstructionPathFromRoot = std::vector<int>;

  // Tries to construct a symbolic tile analysis from a computation. Returns
  // a diagnostic if the construction fails for any reason.
  static SymbolicTileAnalysisOrError AnalyzeComputation(
      const HloComputation& computation, IndexingContext* ctx);

  // Evaluates the tile offsets of an instruction from the analyzed computation
  // following the provided path from the root. Tile parameters must have been
  // set before calling this method.
  std::vector<int64_t> TileOffsets(const HloInstruction* hlo,
                                   const InstructionPathFromRoot& path) const;
  // Evaluates the tile sizes of an instruction from the analyzed computation
  // following the provided path from the root. Tile parameters must have been
  // set before calling this method.
  std::vector<int64_t> TileSizes(const HloInstruction* hlo,
                                 const InstructionPathFromRoot& path) const;
  // Evaluates the tile strides of an instruction from the analyzed computation
  // following the provided path from the root. Tile parameters must have been
  // set before calling this method.
  std::vector<int64_t> TileStrides(const HloInstruction* hlo,
                                   const InstructionPathFromRoot& path) const;

  // Populate tile parameters. This is a prerequisite in order to extract
  // concrete values using `TileOffsets`, `TileSizes`, and `TileStrides`.
  void SetTileParameters(absl::Span<int64_t const> parameters);

  // Populate tile parameters with given sizes. All offsets are 0 and strides
  // are 1.
  void SetTileParametersWithDefaultOffsetsAndStrides(
      absl::Span<int64_t const> sizes);

  // Return the underlying IndexingContext.
  IndexingContext* GetIndexingContext() const { return context_; };

 private:
  SymbolicTileAnalysis(
      absl::flat_hash_map<InstructionPathFromRoot, SymbolicTile>
          symbolic_tile_from_path,
      ConstHloInstructionMap<absl::flat_hash_set<InstructionPathFromRoot>>
          paths_from_root_to_instruction,
      IndexingContext* context)
      : symbolic_tile_from_path_(symbolic_tile_from_path),
        paths_from_root_to_instruction_(paths_from_root_to_instruction),
        context_(context) {}

  absl::flat_hash_map<InstructionPathFromRoot, SymbolicTile>
      symbolic_tile_from_path_;
  // Maps each instruction in the analyzed computation to a set containing all
  // the possible paths from the root instruction to the key instruction.
  ConstHloInstructionMap<absl::flat_hash_set<InstructionPathFromRoot>>
      paths_from_root_to_instruction_;
  IndexingContext* context_;
  // Optionally set tile parameters. These parameters can be set by calling
  // `SetTileParameters`, and correspond to the output tile for the analyzed
  // computation. The order and type of parameters are as explained in the
  // documentation of `SymbolicTile`.
  std::optional<std::vector<int64_t>> tile_parameters_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_
