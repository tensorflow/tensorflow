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

#include "xla/codegen/tiling/tiled_hlo_fusion_instruction.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::unique_ptr<TiledHloFusionInstruction>>
TiledHloFusionInstruction::Create(
    const HloInstruction* hlo,
    llvm::SmallVector<const TiledHloInstruction*> operands,
    llvm::SmallVector<const TiledHloInstruction*> runtime_variables,
    std::unique_ptr<TiledHloComputation> called_computation,
    llvm::SmallVector<int64_t> tile_sizes,
    llvm::SmallVector<int64_t> tile_strides,
    std::optional<IndexingMap> tile_offsets_indexing) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TiledHloInstruction> tiled_hlo,
      TiledHloInstruction::Create(
          hlo, std::move(operands), std::move(runtime_variables),
          std::move(tile_sizes), std::move(tile_strides),
          std::move(tile_offsets_indexing)));

  return absl::WrapUnique(new TiledHloFusionInstruction(
      std::move(*tiled_hlo), std::move(called_computation)));
}

TiledHloFusionInstruction::TiledHloFusionInstruction(
    TiledHloInstruction tiled_hlo_instruction,
    std::unique_ptr<TiledHloComputation> called_computation)
    : TiledHloInstruction(std::move(tiled_hlo_instruction)),
      called_computation_(std::move(called_computation)) {}

}  // namespace xla
