/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_TILED_HLO_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_TILED_HLO_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/instruction_fusion.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

// A node in the symbolic tiled representation of an HLO computation. During
// tiling and codegen an HLO instruction may need to be emitted multiple times
// with different tiling parameters.
class TiledHloInstruction {
 public:
  TiledHloInstruction(const HloInstruction* hlo, Tile tile)
      : hlo_(hlo), tile_(std::move(tile)) {}

  const HloInstruction* hlo() const { return hlo_; }

  const Tile& tile() const { return tile_; }
  void set_tile(Tile tile) { tile_ = std::move(tile); }

  const TiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }
  llvm::ArrayRef<const TiledHloInstruction*> operands() const {
    return operands_;
  }

  // Appends an operand to the end of the operand list.
  void AppendOperand(TiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

  // Returns a string representation of the instruction. Used only for error
  // messages and debugging.
  std::string ToString(absl::string_view field_separator = "\n\t") const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TiledHloInstruction& tiled_hlo) {
    sink.Append(tiled_hlo.ToString());
  }

  // A region is a collection of instructions grouped to represent a nested
  // control flow (e.g., loops) or a distinct computation branch.
  using Region = std::vector<std::unique_ptr<TiledHloInstruction>>;

  // Returns the regions of the instruction.
  absl::Span<const Region> regions() const { return regions_; }

  // Adds a region to the instruction. The region is owned by the instruction.
  void AddRegion(Region region) { regions_.push_back(std::move(region)); }

 private:
  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Symbolic tile.
  Tile tile_;

  // Operands of the instruction in the tiled computation graph.
  llvm::SmallVector<const TiledHloInstruction*, 2> operands_;

  // Regions of the instruction.
  llvm::SmallVector<Region> regions_;
};

inline bool operator==(const TiledHloInstruction& lhs,
                       const TiledHloInstruction& rhs) {
  return lhs.hlo() == rhs.hlo() && lhs.tile() == rhs.tile() &&
         lhs.operands() == rhs.operands();
}

inline bool operator!=(const TiledHloInstruction& lhs,
                       const TiledHloInstruction& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const TiledHloInstruction& tiled_hlo_instruction) {
  h = H::combine(std::move(h), *tiled_hlo_instruction.hlo(),
                 tiled_hlo_instruction.tile());
  for (const TiledHloInstruction* operand : tiled_hlo_instruction.operands()) {
    h = H::combine(std::move(h), operand);
  }
  return h;
}

class TiledHloComputation;
using TileAnalysisOrError = std::variant<TiledHloComputation, FusionDecision>;
using TiledHloRegionOrError =
    std::variant<TiledHloInstruction::Region, FusionDecision>;

// Constructs and holds symbolic tiles for all the instructions within a fusion.
class TiledHloComputation {
 public:
  static TileAnalysisOrError Tile(const HloFusionAdaptor& fusion,
                                  std::unique_ptr<TilingSpace> tiling_space);

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  llvm::ArrayRef<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions()
      const {
    return tiled_hlo_instructions_;
  }
  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const {
    return tiling_space_->mlir_context();
  };

  // Returns the tiling space.
  const TilingSpace& tiling_space() const { return *tiling_space_; }

  // Returns the root instructions.
  absl::Span<const TiledHloInstruction* const> roots() const { return roots_; }

  // Returns a string representation of the analysis.
  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const TiledHloComputation& tiled_computation) {
    sink.Append(tiled_computation.ToString());
  }

 private:
  TiledHloComputation(
      std::unique_ptr<TilingSpace> tiling_space,
      std::vector<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions,
      llvm::SmallVector<const TiledHloInstruction*> roots)
      : tiling_space_(std::move(tiling_space)),
        tiled_hlo_instructions_(std::move(tiled_hlo_instructions)),
        roots_(std::move(roots)) {}

  static TiledHloRegionOrError CreateRegion(
      std::unique_ptr<TiledHloInstruction> tiled_root,
      const HloFusionAdaptor& fusion, const TilingSpace& tiling_space);

  std::unique_ptr<TilingSpace> tiling_space_;
  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions_;

  // Stores pointers to the root instructions. Note that they do not necessarily
  // appear all at the end of `instructions_`.
  llvm::SmallVector<const TiledHloInstruction*> roots_;
};

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TILED_HLO_H_
