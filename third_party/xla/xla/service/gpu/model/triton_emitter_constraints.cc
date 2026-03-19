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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "xla/codegen/tiling/affine_map_evaluator.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/symbolic_tile.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/codegen/xtile/codegen/tiled_emitter_constraints.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;

// Triton enforces that all tensors in the program have less than 1048576
// elements, otherwise it will fail to compile. (See `TRITON_MAX_TENSOR_NUMEL`
// in the Triton codebase.)
constexpr int64_t kMaxTensorNumElements = 1048576;
// For dot operations we don't want to tile the contracting dimension to a size
// larger than this as that leads to a large number of registers being used.
// Also for MMA instruction we don't want tiles greater than 256.
constexpr int64_t kMaxMMADimSize = 256;

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

    if (hlo->opcode() == HloOpcode::kDot) {
      auto ctx = instruction->symbolic_tile().size_map().getContext();
      AffineMap identity_map = AffineMap::getMultiDimIdentityMap(
          instruction->symbolic_tile().size_map().getNumDims(), ctx);
      for (const auto& operand : instruction->operands()) {
        for (AffineExpr tile_size :
             operand->symbolic_tile().size_map().getResults()) {
          // TODO(393299275): There is also a lower bound limit for Triton
          // on what is accepted for dimension size (both contracting and free).
          ConstraintExpression dim_constraint(ConstraintExpression::Constraint{
              tile_size, Interval{1, kMaxMMADimSize}});
          result.push_back(CustomConstraints{identity_map, dim_constraint});
        }
      }
      continue;
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

    std::unique_ptr<TiledEmitterConstraints> tiled_emitter_constraints =
        TiledEmitterConstraints::Create(instructions, fusion_adaptor);

    return std::unique_ptr<TritonEmitterConstraints>(
        absl::WrapUnique(new TritonEmitterConstraints(
            std::move(tile_size_maps), std::move(root_infos),
            std::move(custom_constraints),
            /*root_shape=*/instructions.back()->hlo()->shape(),
            device_description, std::move(tiled_emitter_constraints))));
  };
}

namespace {

// Returns the number of elements in the (padded) tile described by
// tile_size_map` and `tiling_parameters`.
int64_t NumberOfElementsInPaddedTile(
    const AffineMap& tile_size_map,
    absl::Span<const int64_t> tiling_parameters) {
  int64_t num_elements = 1;
  for (auto expr : tile_size_map.getResults()) {
    num_elements *= llvm::PowerOf2Ceil(
        EvaluateAffineExpr(expr, /*dim_values=*/tiling_parameters));
  }

  return num_elements;
}

// Checks whether the number of programs to launch on the grid is under the
// limit enforced by the device.
//
// This currently returns `true` unconditionally if the output shape is not an
// array.
//
// TODO(b/418965008): persistent kernels will make this check obsolete.
bool NumberOfBlocksFitsOnDeviceGrid(const Shape& output_shape,
                                    absl::Span<const int64_t> tiling_parameters,
                                    const se::DeviceDescription& device_info) {
  int64_t num_blocks = 1;
  if (output_shape.IsArray()) {
    for (auto [dim_size, tile_size] :
         llvm::zip(output_shape.dimensions(), tiling_parameters)) {
      // Currently, each output tile is mapped to one block.
      num_blocks *= (dim_size + tile_size - 1) / tile_size;
    }
  }

  return num_blocks < device_info.block_dim_limit().x;
}

}  // namespace

absl::StatusOr<bool> TritonEmitterConstraints::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  // Verify that the tile sizes are not too big.
  if (absl::c_any_of(tile_size_maps_, [&](const auto& tile_size_map) {
        return NumberOfElementsInPaddedTile(tile_size_map, tile_parameters) >
               kMaxTensorNumElements;
      })) {
    VLOG(2) << "Found a tile with more than " << kMaxTensorNumElements
            << " elements. Bailing out.";
    return false;
  }

  // Verify that the number of blocks to launch on the device grid is not too
  // big.
  if (!NumberOfBlocksFitsOnDeviceGrid(root_shape_, tile_parameters,
                                      device_info_)) {
    VLOG(2) << "Number of blocks exceeds the device grid limit. Bailing out.";
    return false;
  }

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
        VLOG(5)
            << "Found a tile size that is not a power of 2 and is not equal "
               "to the dimension size. Bailing out."
            << tile_size << " " << dim_size;
        return false;
      }
    }
  }

  return tiled_emitter_constraints_->ParametersSatisfyConstraints(
      tile_parameters);
}

}  // namespace gpu
}  // namespace xla
