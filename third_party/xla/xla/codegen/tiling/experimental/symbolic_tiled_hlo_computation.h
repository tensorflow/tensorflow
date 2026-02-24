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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILED_HLO_COMPUTATION_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILED_HLO_COMPUTATION_H_

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/symbolic_tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/instruction_fusion.h"

namespace xla::gpu::experimental {

class SymbolicTiledComputation;
using SymbolicTileAnalysisOrError =
    std::variant<SymbolicTiledComputation, FusionDecision>;
using SymbolicTiledHloRegionOrError =
    std::variant<SymbolicTiledHloInstruction::Region, FusionDecision>;

// Constructs and holds symbolic tiles for all the instructions within a fusion.
class SymbolicTiledComputation {
 public:
  static SymbolicTileAnalysisOrError Tile(const HloFusionAdaptor& fusion,
                                          mlir::MLIRContext* ctx);

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  llvm::ArrayRef<std::unique_ptr<SymbolicTiledHloInstruction>>
  tiled_hlo_instructions() const {
    return tiled_hlo_instructions_;
  }
  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const {
    return tiling_space_->mlir_context();
  };

  // Returns a string representation of the analysis.
  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const SymbolicTiledComputation& tiled_computation) {
    sink.Append(tiled_computation.ToString());
  }

 private:
  SymbolicTiledComputation(
      std::unique_ptr<TilingSpace> tiling_space,
      std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
          tiled_hlo_instructions)
      : tiling_space_(std::move(tiling_space)),
        tiled_hlo_instructions_(std::move(tiled_hlo_instructions)) {}

  static SymbolicTiledHloRegionOrError CreateRegion(
      std::unique_ptr<SymbolicTiledHloInstruction> tiled_root,
      const HloFusionAdaptor& fusion, const TilingSpace& tiling_space);

  std::unique_ptr<TilingSpace> tiling_space_;
  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions_;
};

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILED_HLO_COMPUTATION_H_
