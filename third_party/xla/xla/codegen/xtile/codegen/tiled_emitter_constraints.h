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

#ifndef XLA_CODEGEN_XTILE_CODEGEN_TILED_EMITTER_CONSTRAINTS_H_
#define XLA_CODEGEN_XTILE_CODEGEN_TILED_EMITTER_CONSTRAINTS_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/AffineMap.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla {

// Constraints that are intrinsic to the tiled emitter itself that would
// otherwise result in tiling that would not be possible to emit.
class TiledEmitterConstraints : public EmitterSpecificConstraints {
 public:
  static std::unique_ptr<TiledEmitterConstraints> Create(
      const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&,
      const HloFusionAdaptor&);

  static EmitterSpecificConstraintsBuilder GetBuilder();

  absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const override;

 private:
  // Holds a constraint expression over derived parameters (d'0, ..., d'm) where
  //   (d'0, ..., d'm) = tile_parameters_transform(tile_parameters).
  struct CustomConstraints {
    mlir::AffineMap tile_parameters_transform;
    ConstraintExpression constraints;
  };

  explicit TiledEmitterConstraints(
      std::vector<CustomConstraints> custom_constraints)
      : custom_constraints_(std::move(custom_constraints)) {}

  // Derives a vector of `CustomConstraints` to be checked within
  // `ParametersSatisfyConstraints` from a vector of
  // `SymbolicTiledHloInstruction`s representing a symbolically tiled HLO
  // computation. The fusion adaptor is used to figure out which instructions
  // within the computation are operands of the fusion.
  //
  // Currently, this is used to work around an issue with reshapes/bitcasts when
  // instructions are tiled with non-power-of-2 shapes. The resulting custom
  // constraints contain
  //   * the reshape/bitcast's tile size map; this to allow deriving the
  //     output tile sizes for the reshape/bitcast instruction;
  //   * the constraint expression corresponding to the SymbolicTile derived
  //     from the reshape/bitcast instruction's output-to-input indexing map
  //     "in a vacuum" (i.e., without composing with any other indexing map).
  //
  // TODO(b/365727080): move tile derivation to support power of 2 tiles
  // everywhere, and deprecate this.
  static std::vector<CustomConstraints> DeriveCustomConstraints(
      const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
          instructions,
      const HloFusionAdaptor& fusion_adaptor);

  // Custom emitter-specific constraints to check in
  // `ParametersSatisfyConstraints`.
  std::vector<CustomConstraints> custom_constraints_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_XTILE_CODEGEN_TILED_EMITTER_CONSTRAINTS_H_
