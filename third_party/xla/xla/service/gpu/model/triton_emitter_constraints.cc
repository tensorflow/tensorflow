/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/triton_emitter_constraints.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::MLIRContext;

// Triton enforces that all tensors in the program have less than 1048576
// elements, otherwise it will fail to compile.
constexpr int64_t kMaxTensorNumElements = 1048576;

llvm::SmallVector<int64_t> GetPaddedTileSizes(
    llvm::SmallVector<int64_t> tile_sizes) {
  llvm::SmallVector<int64_t> result;
  result.reserve(tile_sizes.size());
  for (int64_t value : tile_sizes) {
    result.push_back(llvm::PowerOf2Ceil(value));
  }
  return result;
}

}  // namespace

/*static*/ std::vector<TritonEmitterConstraints::CustomConstraints>
TritonEmitterConstraints::DeriveCustomConstraints(
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
      MLIRContext* ctx = instruction->symbolic_tile().size_map().getContext();

      IndexingMap reshape_indexing_map =
          *ComputeOutputToInputIndexing(hlo, /*output_id=*/0, ctx)
               .indexing_maps[0]
               .begin();

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
          CustomConstraints{instruction->symbolic_tile().size_map(),
                            std::move(reshape_constraints)});
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

      for (const HloInstruction* operand : hlo->operands()) {
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
    }
  }

  return result;
}

/*static*/ EmitterSpecificConstraintsBuilder
TritonEmitterConstraints::GetBuilder(
    const se::DeviceDescription& device_description) {
  return [=](const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
                 instructions,
             const HloFusionAdaptor& fusion_adaptor) {
    llvm::DenseSet<AffineMap> unique_tile_size_maps;
    llvm::SmallVector<RootTileInfo, 2> root_infos;
    auto roots = fusion_adaptor.GetRoots();
    for (const auto& tiled_hlo_instruction : instructions) {
      unique_tile_size_maps.insert(
          tiled_hlo_instruction->symbolic_tile().size_map());
      if (absl::c_any_of(roots, [&tiled_hlo_instruction](
                                    const HloInstructionAdaptor& instr) {
            return &instr.instruction() == tiled_hlo_instruction->hlo();
          })) {
        const auto& shape = tiled_hlo_instruction->hlo()->shape();
        root_infos.push_back(
            RootTileInfo{tiled_hlo_instruction->symbolic_tile().size_map(),
                         shape.IsArray() ? SpanToVector(shape.dimensions())
                                         : std::vector<int64_t>()});
      }
    }

    std::vector<CustomConstraints> custom_constraints =
        DeriveCustomConstraints(instructions, fusion_adaptor);

    llvm::SmallVector<AffineMap, 4> tile_size_maps(
        unique_tile_size_maps.begin(), unique_tile_size_maps.end());

    return std::unique_ptr<TritonEmitterConstraints>(
        absl::WrapUnique(new TritonEmitterConstraints(
            std::move(tile_size_maps), std::move(root_infos),
            std::move(custom_constraints),
            /*root_shape=*/instructions.back()->hlo()->shape(),
            device_description)));
  };
}

absl::StatusOr<bool> TritonEmitterConstraints::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  // Verify that the tile sizes are not too big.
  for (const auto& tile_size_map : tile_size_maps_) {
    int64_t tile_size = 1;
    for (auto expr : tile_size_map.getResults()) {
      tile_size *= llvm::PowerOf2Ceil(
          EvaluateAffineExpr(expr, /*dim_values=*/tile_parameters));
    }

    if (tile_size > kMaxTensorNumElements) {
      return false;
    }
  }

  int64_t num_tiles = 1;
  if (root_shape_.IsArray()) {
    for (auto [dim_size, tile_size] :
         llvm::zip(root_shape_.dimensions(), tile_parameters)) {
      num_tiles *= (dim_size + tile_size - 1) / tile_size;
    }
  }

  // Number of blocks will exceed the hardware limit. This limitation comes from
  // the fact that one tile is mapped to one block. This constraint can be
  // potentially hoisted to more generic "gpu-specific constraint".
  if (num_tiles >= device_info_.block_dim_limit().x) {
    return false;
  }

  // Ensure that we satisfy the custom constraints we derived when padding tile
  // sizes to a power of 2. This is a workaround while nested fusions are not
  // landed.
  //
  // TODO(b/365727080): get rid of this once tiling is using power of twos
  // everywhere, including when propagating into the prologue of reductions.
  for (const auto& custom_constraint : custom_constraints_) {
    llvm::SmallVector<int64_t> transformed_tile_parameters =
        EvaluateAffineMap(custom_constraint.tile_parameters_transform,
                          /*dim_values=*/tile_parameters);
    if (!custom_constraint.constraints.IsSatisfiedBy(
            GetPaddedTileSizes(transformed_tile_parameters))) {
      return false;
    }
  }
  for (const auto& root : roots_) {
    llvm::SmallVector<int64_t> transformed_tile_parameters =
        EvaluateAffineMap(root.size_map,
                          /*dim_values=*/tile_parameters);
    // We require that the propagated tile sizes for potential root tiles are
    // either powers of 2 or are equal to the dimension size.
    // TODO(b/365727080): Technically the tile size should always be a power of
    // 2, but currently if we capture a dimension fully, we use the dimension
    // size as tile size.
    for (auto [tile_size, dim_size] :
         llvm::zip(transformed_tile_parameters, root.dim_sizes)) {
      CHECK_GT(tile_size, 0);
      // If the tile size is neither a power of 2, nor equal to dim size, it is
      // invalid. Otherwise we would for example compute the launch config
      // incorrectly.
      if ((tile_size & (tile_size - 1)) && tile_size != dim_size) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace gpu
}  // namespace xla
