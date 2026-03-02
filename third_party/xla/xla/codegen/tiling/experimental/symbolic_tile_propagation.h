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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILE_PROPAGATION_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILE_PROPAGATION_H_

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/symbolic_tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu::experimental {

using SymbolicTiles = llvm::SmallVector<SymbolicTile, 2>;

std::string ToString(const SymbolicTiles& tiles);

std::optional<SymbolicTiles> PropagateSymbolicTileToInput(
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const SymbolicTile& output_tile, int64_t output_index);

std::optional<SymbolicTiles> PropagateSymbolicTileToOutput(
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const SymbolicTile& input_tile, int64_t input_index);

std::optional<SymbolicTiles> PropagateTileToInput(
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const SymbolicTile& output_tile, int64_t output_index);

std::optional<SymbolicTiles> PropagateTileToOutput(
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const SymbolicTile& input_tile, int64_t input_index);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_SYMBOLIC_TILE_PROPAGATION_H_
