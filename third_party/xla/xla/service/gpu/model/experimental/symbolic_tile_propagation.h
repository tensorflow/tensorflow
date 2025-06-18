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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_PROPAGATION_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_PROPAGATION_H_

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"

namespace xla::gpu {

using SymbolicTiles = llvm::SmallVector<ExperimentalSymbolicTile, 2>;

struct TiledOperands {
  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TiledOperands& tiled_operands) {
    sink.Append(tiled_operands.ToString());
  }

  // Symbolic tiles per operand.
  SymbolicTiles tiles;
  ConstraintExpression constraint;
};

std::optional<TiledOperands> PropagateTileToInput(
    const HloInstruction& hlo, const ExperimentalSymbolicTile& result_tile,
    int64_t result_index);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_PROPAGATION_H_
