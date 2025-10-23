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

#ifndef XLA_CODEGEN_TILING_TILED_HLO_FUSION_INSTRUCTION_H_
#define XLA_CODEGEN_TILING_TILED_HLO_FUSION_INSTRUCTION_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// `TiledHloFusionInstruction` is to `TiledHloInstruction` what
// `HloFusionInstruction` is to `HloInstruction`.
//
// The main use case for `TiledHloFusionInstruction`s is to support nested
// fusions in block-level codegen.
//
// Similarly to `HloFusionInstruction`, this subclass holds a nested
// `TiledHloComputation` accessible through the `called_computation()` method.
class TiledHloFusionInstruction : public TiledHloInstruction {
 public:
  static absl::StatusOr<std::unique_ptr<TiledHloFusionInstruction>> Create(
      const HloInstruction* hlo,
      llvm::SmallVector<const TiledHloInstruction*> operands,
      llvm::SmallVector<const TiledHloInstruction*> runtime_variables,
      std::unique_ptr<TiledHloComputation> called_computation,
      llvm::SmallVector<int64_t> tile_sizes,
      llvm::SmallVector<int64_t> tile_strides,
      std::optional<IndexingMap> tile_offsets_indexing);

  // The `TiledHloComputation` called by this instruction.
  const TiledHloComputation* called_computation() const {
    return called_computation_.get();
  }

 private:
  TiledHloFusionInstruction(
      TiledHloInstruction tiled_hlo_instruction,
      std::unique_ptr<TiledHloComputation> called_computation);

  // See comment for `called_computation()`.
  std::unique_ptr<TiledHloComputation> called_computation_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_TILED_HLO_FUSION_INSTRUCTION_H_
