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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"
#include "xla/service/gpu/model/experimental/tiling_space.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::MLIRContext;

TiledOperands PropagateTileToInputForCwiseOp(
    const HloInstruction& hlo, const ExperimentalSymbolicTile& result_tile) {
  return TiledOperands{SymbolicTiles(hlo.operand_count(), result_tile),
                       ConstraintExpression::GetAlwaysSatisfied()};
}

TiledOperands PropagateTileToInputForBroadcastOp(
    const HloBroadcastInstruction& broadcast,
    const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();

  int64_t num_operand_dims = broadcast.operand(0)->shape().dimensions().size();

  SmallVector<AffineExpr, 3> new_offsets, new_sizes, new_strides, new_bounds;
  new_offsets.reserve(num_operand_dims);
  new_sizes.reserve(num_operand_dims);
  new_strides.reserve(num_operand_dims);
  new_bounds.reserve(num_operand_dims);

  for (auto broadcast_dim : broadcast.dimensions()) {
    new_offsets.push_back(result_tile.offsets()[broadcast_dim]);
    new_sizes.push_back(result_tile.sizes()[broadcast_dim]);
    new_strides.push_back(result_tile.strides()[broadcast_dim]);
    new_bounds.push_back(result_tile.upper_bounds()[broadcast_dim]);
  }

  ExperimentalSymbolicTile operand_tile{ctx,
                                        result_tile.num_tile_ids(),
                                        result_tile.num_rt_vars(),
                                        new_offsets,
                                        new_sizes,
                                        new_strides,
                                        new_bounds};

  return TiledOperands{SymbolicTiles{operand_tile},
                       ConstraintExpression::GetAlwaysSatisfied()};
}

std::optional<TiledOperands> PropagateTileToInputForConcatenateOp(
    const HloConcatenateInstruction& concatenate,
    const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();

  int64_t num_operands = concatenate.operand_count();

  SymbolicTiles symbolic_tiles;
  symbolic_tiles.reserve(num_operands);

  // For concatenate, we need to adjust the offsets and the bounds in the
  // concatenate dimension.
  int64_t concat_dim = concatenate.concatenate_dimension();
  auto upper_bound = llvm::dyn_cast<mlir::AffineConstantExpr>(
      result_tile.upper_bounds()[concat_dim]);
  if (!upper_bound) {
    // TODO(b/422677091): Also support non-constant affine expressions for upper
    // bound.
    VLOG(2) << "Can't propagate tile to input of concatenate op with "
               "non-constant upper bound.";
    return std::nullopt;
  }
  int64_t offset = 0;
  for (const HloInstruction* operand : concatenate.operands()) {
    SmallVector<AffineExpr, 3> new_offsets(result_tile.offsets().begin(),
                                           result_tile.offsets().end());
    new_offsets[concat_dim] = new_offsets[concat_dim] - offset;
    int64_t operand_dim_size = operand->shape().dimensions(concat_dim);
    SmallVector<AffineExpr, 3> new_bounds(result_tile.upper_bounds().begin(),
                                          result_tile.upper_bounds().end());
    new_bounds[concat_dim] = mlir::getAffineConstantExpr(
        std::max(int64_t{0},
                 std::min(upper_bound.getValue() - offset, operand_dim_size)),
        ctx);
    ExperimentalSymbolicTile operand_tile{
        ctx,         result_tile.num_tile_ids(), result_tile.num_rt_vars(),
        new_offsets, result_tile.sizes(),        result_tile.strides(),
        new_bounds};
    symbolic_tiles.push_back(operand_tile);
    offset += operand_dim_size;
  }

  return TiledOperands{symbolic_tiles,
                       ConstraintExpression::GetAlwaysSatisfied()};
}

template <typename T>
SmallVector<T> Concat(ArrayRef<T> c1, ArrayRef<T> c2) {
  SmallVector<T> result;
  result.append(c1.begin(), c1.end());
  result.append(c2.begin(), c2.end());
  return result;
}

ExperimentalSymbolicTile PropagateTileToInputForSliceImpl(
    ArrayRef<AffineExpr> slice_offsets, ArrayRef<int64_t> slice_strides,
    const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();
  int64_t num_result_dims = result_tile.num_result_dims();

  SmallVector<AffineExpr, 3> new_offsets(num_result_dims),
      new_sizes(num_result_dims), new_strides(num_result_dims),
      new_bounds(num_result_dims);

  for (int64_t dim = 0; dim < num_result_dims; ++dim) {
    // To compute element r of the result we access input
    //   in = r * slice_strides[i] + slice_starts[i].
    // Replacing r with (t * strides[i] + offsets[i]) we get
    //   t * (strides[i] * slice_strides[i]) +
    //   (offsets[i] * slice_strides[i] + slice_starts[i]).
    new_strides[dim] = result_tile.strides()[dim] * slice_strides[dim];
    new_offsets[dim] =
        slice_offsets[dim] + result_tile.offsets()[dim] * slice_strides[dim];
    new_sizes[dim] = result_tile.sizes()[dim];
    // Upper bound condition is `r < upper_bounds[i](t)`.
    // By replacing r with `(in - slice_offsets[i]) / slice_strides[i]` we get
    // in < upper_bounds[i](t) * slice_strides[i] + slice_offsets[i].
    new_bounds[dim] = result_tile.upper_bounds()[dim] * slice_strides[dim] +
                      slice_offsets[dim];
  }
  return ExperimentalSymbolicTile{ctx,
                                  result_tile.num_tile_ids(),
                                  result_tile.num_rt_vars(),
                                  new_offsets,
                                  new_sizes,
                                  new_strides,
                                  new_bounds};
}

TiledOperands PropagateTileToInputForSliceOp(
    const HloInstruction& slice, const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();

  SmallVector<AffineExpr, 3> slice_offset_exprs;
  slice_offset_exprs.reserve(slice.shape().dimensions().size());
  for (int64_t slice_offset : slice.slice_starts()) {
    slice_offset_exprs.push_back(
        mlir::getAffineConstantExpr(slice_offset, ctx));
  }
  auto operand_tile = SymbolicTiles{PropagateTileToInputForSliceImpl(
      slice_offset_exprs, slice.slice_strides(), result_tile)};

  return TiledOperands{{operand_tile},
                       ConstraintExpression::GetAlwaysSatisfied()};
}

std::optional<int64_t> GetInt64FromConstant(const HloInstruction& hlo) {
  if (hlo.IsConstant() && hlo.shape().dimensions().empty()) {
    return LiteralUtil::LiteralAsScalarInt64(
        Cast<HloConstantInstruction>(&hlo)->literal());
  }
  return std::nullopt;
}

TiledOperands PropagateTileToInputForDynamicSliceOp(
    const TilingSpace& tiling_space,
    const HloDynamicSliceInstruction& dynamic_slice,
    const ExperimentalSymbolicTile& result_tile) {
  const int64_t first_index_operand_number =
      dynamic_slice.first_index_operand_number();
  CHECK(dynamic_slice.operand(first_index_operand_number)
            ->shape()
            .dimensions()
            .empty())
      << "b/118437727: Old form, not supported.";
  MLIRContext* ctx = result_tile.mlir_context();
  int64_t num_result_dims = result_tile.num_result_dims();

  SmallVector<AffineExpr, 3> slice_offset_exprs(num_result_dims);
  for (auto [dim, slice_size] :
       llvm::enumerate(dynamic_slice.dynamic_slice_sizes())) {
    auto slice_offset = dynamic_slice.operand(dim + first_index_operand_number);
    std::optional<int64_t> offset_const = GetInt64FromConstant(*slice_offset);
    const TilingSpace::RTVarInfo& rt_var_info = tiling_space.GetRTVarInfo(
        dynamic_slice, dim + first_index_operand_number);
    if (offset_const.has_value()) {
      int64_t clamped_offset = std::clamp(
          *offset_const, rt_var_info.bounds.lower, rt_var_info.bounds.upper);
      slice_offset_exprs[dim] =
          mlir::getAffineConstantExpr(clamped_offset, ctx);
      continue;
    }
    slice_offset_exprs[dim] = mlir::getAffineSymbolExpr(
        rt_var_info.id + result_tile.num_tile_ids(), ctx);
  }

  SymbolicTiles operand_tiles{PropagateTileToInputForSliceImpl(
      slice_offset_exprs, SmallVector<int64_t>(num_result_dims, 1),
      result_tile)};
  ExperimentalSymbolicTile scalar_tensor_tile{
      ctx, result_tile.num_tile_ids(), /*num_rt_vars=*/0, {}, {}, {}, {}};
  for (int i = 0; i < num_result_dims; ++i) {
    operand_tiles.push_back(scalar_tensor_tile);
  }
  return TiledOperands{std::move(operand_tiles),
                       ConstraintExpression::GetAlwaysSatisfied()};
}

std::optional<TiledOperands> PropagateTileToInputForPadOp(
    const HloPadInstruction& pad, const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();
  const PaddingConfig& padding_config = pad.padding_config();

  int64_t num_result_dims = result_tile.num_result_dims();
  SmallVector<AffineExpr, 3> new_offsets, new_sizes, new_bounds;
  new_offsets.reserve(num_result_dims);
  new_sizes.reserve(num_result_dims);
  new_bounds.reserve(num_result_dims);

  // For each dimension, the low padding is subtracted from the offsets.
  for (const auto [current_offset, current_size, padding_dim, operand_dim] :
       llvm::zip(result_tile.offsets(), result_tile.sizes(),
                 padding_config.dimensions(),
                 pad.operand(0)->shape().dimensions())) {
    if (padding_dim.interior_padding() != 0) {
      VLOG(2)
          << "Can't propagate tile to input of pad op with interior padding.";
      return std::nullopt;
    }
    new_offsets.push_back(current_offset - padding_dim.edge_padding_low());
    new_sizes.push_back(current_size);
    new_bounds.push_back(mlir::getAffineConstantExpr(operand_dim, ctx));
  }

  ExperimentalSymbolicTile operand_tile{ctx,
                                        result_tile.num_tile_ids(),
                                        result_tile.num_rt_vars(),
                                        new_offsets,
                                        new_sizes,
                                        result_tile.strides(),
                                        new_bounds};

  // Pad also has a padding value, but it is a scalar, therefore we only need
  // to propagate the inputs.
  ExperimentalSymbolicTile padding_value_tile{
      ctx, result_tile.num_tile_ids(), {}, {}, {}, {}, {}};

  return TiledOperands{SymbolicTiles{operand_tile, padding_value_tile},
                       ConstraintExpression::GetAlwaysSatisfied()};
}

TiledOperands PropagateTileToInputForTransposeOp(
    const HloInstruction& transpose,
    const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();
  int64_t num_result_dims = result_tile.num_result_dims();

  SmallVector<AffineExpr, 3> new_offsets(num_result_dims),
      new_sizes(num_result_dims), new_strides(num_result_dims),
      new_bounds(num_result_dims);

  for (int64_t dim = 0; dim < num_result_dims; ++dim) {
    int64_t operand_dim = transpose.dimensions()[dim];
    new_offsets[operand_dim] = result_tile.offsets()[dim];
    new_sizes[operand_dim] = result_tile.sizes()[dim];
    new_strides[operand_dim] = result_tile.strides()[dim];
    new_bounds[operand_dim] = result_tile.upper_bounds()[dim];
  }

  ExperimentalSymbolicTile operand_tile{ctx,
                                        result_tile.num_tile_ids(),
                                        result_tile.num_rt_vars(),
                                        new_offsets,
                                        new_sizes,
                                        new_strides,
                                        new_bounds};

  return TiledOperands{SymbolicTiles{operand_tile},
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
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const ExperimentalSymbolicTile& result_tile, int64_t result_index) {
  if (HloInstruction::IsOpElementwise(hlo.opcode()) ||
      hlo.opcode() == HloOpcode::kMap) {
    return PropagateTileToInputForCwiseOp(hlo, result_tile);
  }

  if (hlo.opcode() == HloOpcode::kBroadcast) {
    return PropagateTileToInputForBroadcastOp(
        *Cast<HloBroadcastInstruction>(&hlo), result_tile);
  }

  if (hlo.opcode() == HloOpcode::kConcatenate) {
    return PropagateTileToInputForConcatenateOp(
        *Cast<HloConcatenateInstruction>(&hlo), result_tile);
  }

  if (hlo.opcode() == HloOpcode::kDynamicSlice) {
    return PropagateTileToInputForDynamicSliceOp(
        tiling_space, *Cast<HloDynamicSliceInstruction>(&hlo), result_tile);
  }

  if (hlo.opcode() == HloOpcode::kPad) {
    const HloPadInstruction& pad = *Cast<HloPadInstruction>(&hlo);
    return PropagateTileToInputForPadOp(pad, result_tile);
  }

  if (hlo.opcode() == HloOpcode::kTranspose) {
    return PropagateTileToInputForTransposeOp(hlo, result_tile);
  }

  if (hlo.opcode() == HloOpcode::kSlice) {
    return PropagateTileToInputForSliceOp(hlo, result_tile);
  }

  LOG(INFO) << "Tile propagation not implemented for " << hlo.opcode();
  return std::nullopt;
}

}  // namespace xla::gpu
