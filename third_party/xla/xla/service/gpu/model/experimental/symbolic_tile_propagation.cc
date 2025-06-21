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

#include "xla/service/gpu/model/experimental/symbolic_tile_propagation.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::getAffineDimExpr;
using ::mlir::MLIRContext;

TiledOperands PropagateTileToInputForCwiseOp(
    const HloInstruction& hlo, const ExperimentalSymbolicTile& result_tile) {
  return TiledOperands{SymbolicTiles(hlo.operand_count(), result_tile),
                       ConstraintExpression::GetAlwaysSatisfied()};
}

std::optional<TiledOperands> PropagateTileToInputForPadOp(
    const HloPadInstruction& pad, const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.tile_map.getContext();
  const PaddingConfig& padding_config = pad.padding_config();

  // We don't handle interior padding for now.
  for (const auto& dimension : padding_config.dimensions()) {
    if (dimension.interior_padding() != 0) {
      VLOG(2)
          << "Can't propagate tile to input of pad op with interior padding.";
      return std::nullopt;
    }
  }

  int64_t num_result_dims = result_tile.tile_map.getNumResults();
  std::vector<AffineExpr> transformation_exprs;
  transformation_exprs.reserve(num_result_dims);

  // For each dimension, the low padding is subtracted from the offsets.
  for (const auto [id, padding_dimension] :
       llvm::enumerate(padding_config.dimensions())) {
    AffineExpr offset_id = getAffineDimExpr(id, ctx);
    transformation_exprs.push_back(offset_id -
                                   padding_dimension.edge_padding_low());
  }

  // The tile sizes as well as the strides are unchanged in the undilated case.
  for (int64_t id = transformation_exprs.size(); id < num_result_dims; ++id) {
    transformation_exprs.push_back(getAffineDimExpr(id, ctx));
  }

  AffineMap transformation_map = AffineMap::get(
      num_result_dims, /*symbolCount=*/0, transformation_exprs, ctx);

  ExperimentalSymbolicTile operand_tile{
      transformation_map.compose(result_tile.tile_map), result_tile.rt_vars};

  // Pad also has a padding value, but it is a scalar, therefore we only need
  // to propagate the inputs.
  ExperimentalSymbolicTile padding_value_tile{
      AffineMap::get(result_tile.tile_map.getNumDims(),
                     result_tile.tile_map.getNumSymbols(), ctx),
      result_tile.rt_vars};

  return TiledOperands{SymbolicTiles{operand_tile, padding_value_tile},
                       ConstraintExpression::GetAlwaysSatisfied()};
}

}  // namespace

std::string TiledOperands::ToString() const {
  std::stringstream ss;
  for (const auto& [index, tile] : llvm::enumerate(tiles)) {
    ss << index << ") " << tile.ToString() << "\n";
  }
  if (!constraint.IsAlwaysSatisfied()) {
    ss << "constraint: " << constraint.ToString() << "\n";
  }
  return ss.str();
}

std::optional<TiledOperands> PropagateTileToInput(
    const HloInstruction& hlo, const ExperimentalSymbolicTile& result_tile,
    int64_t result_index) {
  if (HloInstruction::IsOpElementwise(hlo.opcode()) ||
      hlo.opcode() == HloOpcode::kMap) {
    return PropagateTileToInputForCwiseOp(hlo, result_tile);
  }

  if (hlo.opcode() == HloOpcode::kPad) {
    const HloPadInstruction& pad = *Cast<HloPadInstruction>(&hlo);
    return PropagateTileToInputForPadOp(pad, result_tile);
  }

  return std::nullopt;
}

}  // namespace xla::gpu
