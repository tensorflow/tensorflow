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

#ifndef XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILED_HLO_INSTRUCTION_H_
#define XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILED_HLO_INSTRUCTION_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tile.h"

namespace xla {
namespace gpu {

// A node in the symbolic tiled representation of an HLO computation. During
// tiling and codegen an HLO instruction may need to be emitted multiple times
// with different tiling parameters.
class SymbolicTiledHloInstruction {
 public:
  SymbolicTiledHloInstruction(const HloInstruction* hlo,
                              IndexingMap indexing_map,
                              SymbolicTile symbolic_tile)
      : hlo_(hlo),
        indexing_map_(std::move(indexing_map)),
        symbolic_tile_(std::move(symbolic_tile)) {}

  // Evaluates the tile offsets of an instruction with given tile parameters.
  std::vector<int64_t> TileOffsets(
      absl::Span<int64_t const> tile_parameters) const;
  // Evaluates the tile sizes of an instruction with given tile parameters.
  std::vector<int64_t> TileSizes(
      absl::Span<int64_t const> tile_parameters) const;
  // Evaluates the tile strides of an instruction with given tile parameters.
  std::vector<int64_t> TileStrides(
      absl::Span<int64_t const> tile_parameters) const;

  const HloInstruction* hlo() const { return hlo_; }
  const IndexingMap& indexing_map() const { return indexing_map_; }
  const SymbolicTile& symbolic_tile() const { return symbolic_tile_; }

  const SymbolicTiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }
  SymbolicTiledHloInstruction* operand(int64_t operand_id) {
    return operands_[operand_id];
  }
  const std::vector<SymbolicTiledHloInstruction*>& operands() const {
    return operands_;
  }

  // Appends an operand to the end of the operand list.
  void AppendOperand(SymbolicTiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

 private:
  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Indexing map from the computation root to this instruction output.
  IndexingMap indexing_map_;

  // Symbolic tile derived from the indexing map.
  SymbolicTile symbolic_tile_;

  // Operands of the instruction in the tiled computation graph.
  std::vector<SymbolicTiledHloInstruction*> operands_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILED_HLO_INSTRUCTION_H_
