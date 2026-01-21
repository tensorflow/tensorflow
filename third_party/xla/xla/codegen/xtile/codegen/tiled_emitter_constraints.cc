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

#include "xla/codegen/xtile/codegen/tiled_emitter_constraints.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/affine_map_evaluator.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/symbolic_tile.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/util.h"

namespace xla {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::MLIRContext;

std::vector<TiledEmitterConstraints::CustomConstraints>
TiledEmitterConstraints::DeriveCustomConstraints(
    const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        instructions,
    const HloFusionAdaptor& fusion_adaptor) {
  std::vector<CustomConstraints> result;

  for (const auto& instruction : instructions) {
    const HloInstruction* hlo = instruction->hlo();
    // Don't consider operands to the fusion computation for constraints.
    if (!fusion_adaptor.ContainsInstruction(hlo)) {
      continue;
    }

    // Construct custom constraints for parameters of bitcasts and reshapes
    // within `instructions`.
    if (hlo->opcode() == HloOpcode::kReshape ||
        hlo->opcode() == HloOpcode::kBitcast) {
      AffineMap size_map = instruction->symbolic_tile().size_map();
      MLIRContext* ctx = size_map.getContext();

      IndexingMap reshape_indexing_map =
          ComputeOutputToInputIndexing(hlo, /*output_id=*/0, ctx)
              .indexing_maps[0]
              .begin()
              ->map();

      std::optional<SymbolicTile> reshape_symbolic_tile =
          SymbolicTile::FromIndexingMap(reshape_indexing_map);

      // Since we managed to create a `SymbolicTiledHloInstruction` for this
      // instruction, it should never be the case that we fail to derive a
      // `SymbolicTile`, so we `CHECK`. This is enforced by checks in
      // `SymbolicTileAnalysis`'s internal function
      // `ShouldProceedWithSymbolicTileDerivation`.
      CHECK(reshape_symbolic_tile.has_value());

      ConstraintExpression reshape_constraints =
          reshape_symbolic_tile->constraints();
      result.push_back(
          CustomConstraints{size_map, std::move(reshape_constraints)});
      continue;
    }

    // Construct emitter-specific constraints for concatenates. This allows
    // filtering for tile sizes that divide the concatenated dimension for all
    // the operands exactly.
    if (hlo->opcode() == HloOpcode::kConcatenate) {
      AffineMap size_map = instruction->symbolic_tile().size_map();
      MLIRContext* ctx = size_map.getContext();
      int concatenate_dimension_index = hlo->concatenate_dimension();
      AffineExpr concatenate_dimension_map_parameter =
          mlir::getAffineDimExpr(concatenate_dimension_index, ctx);

      // Check that each operand's concatenation dimension is divisible by the
      // tile size along this dimension.
      ConstraintExpression divisibility_constraints =
          ConstraintExpression::GetAlwaysSatisfied();

      // The last operand of the concat does not require the divisibility
      // constraint.
      for (int operand_id = 0; operand_id < hlo->operand_count() - 1;
           ++operand_id) {
        const HloInstruction* operand = hlo->operand(operand_id);
        AffineExpr operand_concat_dimension = mlir::getAffineConstantExpr(
            operand->shape().dimensions(concatenate_dimension_index), ctx);
        ConstraintExpression::Constraint divisibility_constraint{
            operand_concat_dimension % concatenate_dimension_map_parameter,
            Interval{0, 0}};
        divisibility_constraints =
            divisibility_constraints && divisibility_constraint;
      }

      result.push_back(
          CustomConstraints{size_map, std::move(divisibility_constraints)});

      AffineMap identity_map =
          AffineMap::getMultiDimIdentityMap(size_map.getNumDims(), ctx);

      // Check that the offset along the contracting dimension is 0.
      ConstraintExpression::Constraint offset_constraint{
          instruction->symbolic_tile().offset_map().getResult(
              concatenate_dimension_index),
          Interval{0, 0}};
      result.push_back(CustomConstraints{
          identity_map, ConstraintExpression(offset_constraint)});

      // Check that the stride along the contracting dimension is 1.
      ConstraintExpression::Constraint stride_constraint{
          instruction->symbolic_tile().stride_map().getResult(
              concatenate_dimension_index),
          Interval{1, 1}};
      result.push_back(CustomConstraints{
          identity_map, ConstraintExpression(stride_constraint)});
      continue;
    }
  }

  return result;
}

std::unique_ptr<TiledEmitterConstraints> TiledEmitterConstraints::Create(
    const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        instructions,
    const HloFusionAdaptor& fusion_adaptor) {
  std::vector<CustomConstraints> custom_constraints =
      DeriveCustomConstraints(instructions, fusion_adaptor);

  return std::unique_ptr<TiledEmitterConstraints>(absl::WrapUnique(
      new TiledEmitterConstraints(std::move(custom_constraints))));
}

EmitterSpecificConstraintsBuilder TiledEmitterConstraints::GetBuilder() {
  return [=](const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
                 instructions,
             const HloFusionAdaptor& fusion_adaptor) {
    return Create(instructions, fusion_adaptor);
  };
}

absl::StatusOr<bool> TiledEmitterConstraints::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  // Ensure that we satisfy the custom constraints we derived when padding tile
  // sizes to a power of 2. This is a workaround while nested fusions are not
  // landed.
  //
  // TODO(b/365727080): get rid of this once tiling is using power of twos
  // everywhere, including when propagating into the prologue of reductions.
  VLOG(5) << "Checking custom constraints for tile parameters: "
          << absl::StrJoin(tile_parameters, ", ");
  for (const auto& custom_constraint : custom_constraints_) {
    VLOG(5) << "Checking custom constraint: transform  "
            << xla::ToString(custom_constraint.tile_parameters_transform)
            << " constraints " << custom_constraint.constraints.ToString();
    llvm::SmallVector<int64_t> transformed_tile_parameters =
        EvaluateAffineMap(custom_constraint.tile_parameters_transform,
                          /*dim_values=*/tile_parameters);
    if (!custom_constraint.constraints.IsSatisfiedBy(
            xtile::GetPaddedTileSizes(transformed_tile_parameters))) {
      return false;
    }
  }

  return true;
}

}  // namespace xla
