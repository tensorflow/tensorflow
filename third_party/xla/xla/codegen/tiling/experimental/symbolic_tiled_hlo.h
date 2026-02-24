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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILED_HLO_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILED_HLO_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/symbolic_tile.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu::experimental {

// A node in the symbolic tiled representation of an HLO computation. During
// tiling and codegen an HLO instruction may need to be emitted multiple times
// with different tiling parameters.
class SymbolicTiledHloInstruction {
 public:
  SymbolicTiledHloInstruction(const HloInstruction* hlo,
                              SymbolicTile symbolic_tile)
      : hlo_(hlo), symbolic_tile_(std::move(symbolic_tile)) {}

  const HloInstruction* hlo() const { return hlo_; }

  const SymbolicTile& symbolic_tile() const { return symbolic_tile_; }
  void set_symbolic_tile(SymbolicTile symbolic_tile) {
    symbolic_tile_ = std::move(symbolic_tile);
  }

  const SymbolicTiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }
  llvm::ArrayRef<const SymbolicTiledHloInstruction*> operands() const {
    return operands_;
  }

  // Appends an operand to the end of the operand list.
  void AppendOperand(SymbolicTiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

  // Returns a string representation of the instruction. Used only for error
  // messages and debugging.
  std::string ToString(absl::string_view field_separator = "\n\t") const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const SymbolicTiledHloInstruction& tiled_hlo) {
    sink.Append(tiled_hlo.ToString());
  }

  // A region is a collection of instructions grouped to represent a nested
  // control flow (e.g., loops) or a distinct computation branch.
  using Region = std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>;

  // Returns the regions of the instruction.
  absl::Span<const Region> regions() const { return regions_; }

  // Adds a region to the instruction. The region is owned by the instruction.
  void AddRegion(Region region) { regions_.push_back(std::move(region)); }

 private:
  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Symbolic tile.
  SymbolicTile symbolic_tile_;

  // Operands of the instruction in the tiled computation graph.
  llvm::SmallVector<const SymbolicTiledHloInstruction*, 2> operands_;

  // Regions of the instruction.
  llvm::SmallVector<Region> regions_;
};

inline bool operator==(const SymbolicTiledHloInstruction& lhs,
                       const SymbolicTiledHloInstruction& rhs) {
  return lhs.hlo() == rhs.hlo() && lhs.symbolic_tile() == rhs.symbolic_tile() &&
         lhs.operands() == rhs.operands();
}

inline bool operator!=(const SymbolicTiledHloInstruction& lhs,
                       const SymbolicTiledHloInstruction& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const SymbolicTiledHloInstruction& tiled_hlo_instruction) {
  h = H::combine(std::move(h), *tiled_hlo_instruction.hlo(),
                 tiled_hlo_instruction.symbolic_tile());
  for (const SymbolicTiledHloInstruction* operand :
       tiled_hlo_instruction.operands()) {
    h = H::combine(std::move(h), operand);
  }
  return h;
}

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILED_HLO_H_
