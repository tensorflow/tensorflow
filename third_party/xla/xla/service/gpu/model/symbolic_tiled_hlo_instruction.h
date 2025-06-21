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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/symbolic_tile.h"

namespace xla {
namespace gpu {

// A node in the symbolic tiled representation of an HLO computation. During
// tiling and codegen an HLO instruction may need to be emitted multiple times
// with different tiling parameters.
class SymbolicTiledHloInstruction {
 public:
  SymbolicTiledHloInstruction(
      const HloInstruction* hlo, IndexingMap indexing_map,
      std::vector<SymbolicTiledHloInstruction*> runtime_variables = {})
      : hlo_(hlo),
        indexing_map_(std::move(indexing_map)),
        runtime_variables_(std::move(runtime_variables)) {
    CHECK_EQ(indexing_map_.GetRTVars().size(), runtime_variables_.size());
  }

  virtual ~SymbolicTiledHloInstruction() = default;

  const HloInstruction* hlo() const { return hlo_; }
  const IndexingMap& indexing_map() const { return indexing_map_; }
  void set_symbolic_tile(SymbolicTile symbolic_tile) {
    symbolic_tile_ = std::move(symbolic_tile);
  }
  const SymbolicTile& symbolic_tile() const {
    CHECK(symbolic_tile_.has_value()) << "Symbolic tile was not computed";
    return *symbolic_tile_;
  }

  const SymbolicTiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }
  SymbolicTiledHloInstruction* operand(int64_t operand_id) {
    return operands_[operand_id];
  }
  const std::vector<SymbolicTiledHloInstruction*>& operands() const {
    return operands_;
  }

  // Tiling of runtime variables of the indexing map of the instruction.
  const std::vector<SymbolicTiledHloInstruction*>& runtime_variables() const {
    return runtime_variables_;
  }

  // Appends an operand to the end of the operand list.
  void AppendOperand(SymbolicTiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

  // Returns a string representation of the instruction. Used only for error
  // messages and debugging.
  std::string ToString(absl::string_view field_separator = "\n\t") const;

 private:
  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Indexing map from the computation root to this instruction output.
  IndexingMap indexing_map_;

  // Symbolic tile derived from the indexing map. Should be computed outside of
  // this class and set before usage. Wrapped in an optional, because
  // SymbolicTile does not have a default constructor.
  std::optional<SymbolicTile> symbolic_tile_;

  // Operands of the instruction in the tiled computation graph.
  std::vector<SymbolicTiledHloInstruction*> operands_;

  // Tiling of runtime variables of `indexing_map_`.
  std::vector<SymbolicTiledHloInstruction*> runtime_variables_;
};

inline bool operator==(const SymbolicTiledHloInstruction& lhs,
                       const SymbolicTiledHloInstruction& rhs) {
  return lhs.hlo() == rhs.hlo() && lhs.indexing_map() == rhs.indexing_map() &&
         lhs.runtime_variables() == rhs.runtime_variables() &&
         lhs.operands() == rhs.operands();
}

inline bool operator!=(const SymbolicTiledHloInstruction& lhs,
                       const SymbolicTiledHloInstruction& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const SymbolicTiledHloInstruction& tiled_hlo_instruction) {
  h = H::combine(std::move(h), tiled_hlo_instruction.hlo(),
                 tiled_hlo_instruction.indexing_map());
  for (const auto& runtime_variable :
       tiled_hlo_instruction.runtime_variables()) {
    h = H::combine(std::move(h), runtime_variable);
  }
  for (const auto& operand : tiled_hlo_instruction.operands()) {
    h = H::combine(std::move(h), operand);
  }
  return h;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILED_HLO_INSTRUCTION_H_
