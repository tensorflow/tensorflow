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
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
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

  // Returns a graph of HLO instructions tiled with the given tile parameters.
  // Result vector has instructions in def-before-use order.
  absl::StatusOr<std::vector<std::unique_ptr<TiledHloInstruction>>>
  ComputeTiledHloInstructions(
      const std::vector<int64_t>& tile_parameters) const;

  // Returns the tiled root instruction.
  const SymbolicTiledHloInstruction* GetRoot() const {
    return symbolic_tiled_hlo_instructions_.back().get();
  }

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
  GetSymbolicTiledHloComputation() const {
    return symbolic_tiled_hlo_instructions_;
  }

  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const { return context_; };

 private:
  SymbolicTileAnalysis(std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
                           symbolic_tiled_hlo_instructions,
                       mlir::MLIRContext* context)
      : symbolic_tiled_hlo_instructions_(
            std::move(symbolic_tiled_hlo_instructions)),
        context_(context) {}

  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      symbolic_tiled_hlo_instructions_;

  mlir::MLIRContext* context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_
