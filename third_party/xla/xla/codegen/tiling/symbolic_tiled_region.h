/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_TILING_SYMBOLIC_TILED_REGION_H_
#define XLA_CODEGEN_TILING_SYMBOLIC_TILED_REGION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// A symbolic representation of a region of instructions that are tiled
// together. This is the symbolic counterpart to `TiledHloRegionInstruction`.
//
// Unlike `SymbolicTiledHloFusionInstruction`, which wraps a nested HLO fusion,
// `SymbolicTiledRegion` wraps a subgraph of instructions within the same
// computation that should be emitted within a single loop structure (e.g. dot
// prologue).
class SymbolicTiledRegion : public SymbolicTiledHloInstruction {
 public:
  SymbolicTiledRegion(
      const HloInstruction* hlo, IndexingMap indexing_map,
      std::vector<std::unique_ptr<SymbolicTiledHloInstruction>> instructions,
      std::vector<SymbolicTiledHloInstruction*> runtime_variables)
      : SymbolicTiledHloInstruction(hlo, std::move(indexing_map),
                                    std::move(runtime_variables)),
        instructions_(std::move(instructions)) {}

  // Returns the instructions belonging to this region, in def-before-use order.
  const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
  instructions() const {
    return instructions_;
  }

  std::string ToString(absl::string_view field_separator = "\n\t") const;

 private:
  // Instructions belonging to this region, in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>> instructions_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_SYMBOLIC_TILED_REGION_H_
