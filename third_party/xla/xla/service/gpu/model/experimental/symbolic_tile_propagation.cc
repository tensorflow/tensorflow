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

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
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
#include "xla/service/gpu/matmul_indexing_utils.h"
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
  SmallVector<DimTile> one_dim_tiles;
  one_dim_tiles.reserve(broadcast.operand(0)->shape().dimensions().size());
  for (auto broadcast_dim : broadcast.dimensions()) {
    one_dim_tiles.push_back(result_tile.one_dim_tiles()[broadcast_dim]);
  }
  ExperimentalSymbolicTile operand_tile{ctx, result_tile.num_tile_ids(),
                                        result_tile.num_rt_vars(),
                                        std::move(one_dim_tiles)};

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
    SmallVector<DimTile> one_dim_tiles(result_tile.one_dim_tiles());
    one_dim_tiles[concat_dim].offset =
        one_dim_tiles[concat_dim].offset - offset;
    int64_t operand_dim_size = operand->shape().dimensions(concat_dim);

    one_dim_tiles[concat_dim].upper_bound = mlir::getAffineConstantExpr(
        std::max(int64_t{0},
                 std::min(upper_bound.getValue() - offset, operand_dim_size)),
        ctx);
    ExperimentalSymbolicTile operand_tile{ctx, result_tile.num_tile_ids(),
                                          result_tile.num_rt_vars(),
                                          std::move(one_dim_tiles)};
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

  SmallVector<DimTile> one_dim_tiles;
  one_dim_tiles.reserve(num_result_dims);

  for (const auto& [dim, result_one_dim_tile] :
       llvm::enumerate(result_tile.one_dim_tiles())) {
    // To compute element r of the result we access input
    //   in = r * slice_strides[i] + slice_starts[i].
    // Replacing r with (t * strides[i] + offsets[i]) we get
    //   t * (strides[i] * slice_strides[i]) +
    //   (offsets[i] * slice_strides[i] + slice_starts[i]).
    DimTile one_dim_tile;
    one_dim_tile.offset =
        slice_offsets[dim] + result_one_dim_tile.offset * slice_strides[dim];
    one_dim_tile.stride = result_one_dim_tile.stride * slice_strides[dim];
    one_dim_tile.size = result_one_dim_tile.size;
    // Upper bound condition is `r < upper_bounds[i](t)`.
    // By replacing r with `(in - slice_offsets[i]) / slice_strides[i]` we get
    // in < upper_bounds[i](t) * slice_strides[i] + slice_offsets[i].
    one_dim_tile.upper_bound =
        result_one_dim_tile.upper_bound * slice_strides[dim] +
        slice_offsets[dim];
    one_dim_tiles.push_back(std::move(one_dim_tile));
  }
  return ExperimentalSymbolicTile{ctx, result_tile.num_tile_ids(),
                                  result_tile.num_rt_vars(),
                                  std::move(one_dim_tiles)};
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

  // For each dimension, the low padding is subtracted from the offsets.
  SmallVector<DimTile> one_dim_tiles;
  one_dim_tiles.reserve(result_tile.num_result_dims());
  for (const auto [result_one_dim_tile, padding_dim, operand_dim] :
       llvm::zip(result_tile.one_dim_tiles(), padding_config.dimensions(),
                 pad.operand(0)->shape().dimensions())) {
    if (padding_dim.interior_padding() != 0) {
      VLOG(2)
          << "Can't propagate tile to input of pad op with interior padding.";
      return std::nullopt;
    }
    one_dim_tiles.push_back(
        DimTile{result_one_dim_tile.offset - padding_dim.edge_padding_low(),
                result_one_dim_tile.size, result_one_dim_tile.stride,
                mlir::getAffineConstantExpr(operand_dim, ctx)});
  }
  ExperimentalSymbolicTile operand_tile{ctx, result_tile.num_tile_ids(),
                                        result_tile.num_rt_vars(),
                                        std::move(one_dim_tiles)};

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
  SmallVector<DimTile> one_dim_tiles(num_result_dims);
  for (int64_t dim = 0; dim < num_result_dims; ++dim) {
    int64_t operand_dim = transpose.dimensions()[dim];
    one_dim_tiles[operand_dim] = result_tile.one_dim_tiles()[dim];
  }
  ExperimentalSymbolicTile operand_tile{ctx, result_tile.num_tile_ids(),
                                        result_tile.num_rt_vars(),
                                        std::move(one_dim_tiles)};
  return TiledOperands{SymbolicTiles{operand_tile},
                       ConstraintExpression::GetAlwaysSatisfied()};
}

TiledOperands PropagateTileToInputForDotOp(
    const TilingSpace& tiling_space, const HloDotInstruction& dot,
    const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  absl::Span<const int64_t> lhs_contracting_dims(
      dim_numbers.lhs_contracting_dimensions());
  absl::Span<const int64_t> rhs_contracting_dims =
      dim_numbers.rhs_contracting_dimensions();

  absl::Span<const int64_t> lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
  absl::Span<const int64_t> rhs_batch_dims = dim_numbers.rhs_batch_dimensions();

  const Shape& lhs_shape = dot.operand(0)->shape();
  const int64_t lhs_rank = lhs_shape.dimensions().size();
  SmallVector<DimTile> lhs_one_dim_tiles(lhs_rank);

  const Shape& rhs_shape = dot.operand(1)->shape();
  const int64_t rhs_rank = rhs_shape.dimensions().size();
  SmallVector<DimTile> rhs_one_dim_tiles(rhs_rank);

  // According to the StableHLO specification, the dimensions of the output
  // shape are ordered as follows:
  //   lhs_batch_dims | lhs_non_contracting_dims | rhs_non_contracting_dims

  // Populate lhs and rhs batch dimensions.
  for (auto [output_dim_id, batch_dims] :
       llvm::enumerate(llvm::zip(lhs_batch_dims, rhs_batch_dims))) {
    auto [lhs_batch_dim, rhs_batch_dim] = batch_dims;
    rhs_one_dim_tiles[rhs_batch_dim] = lhs_one_dim_tiles[lhs_batch_dim] =
        result_tile.one_dim_tiles()[output_dim_id];
  }

  // lhs_non_contracting_dims
  int64_t output_dim_id = lhs_batch_dims.size();
  auto lhs_non_contracting_dims =
      GetNonContractingDims(lhs_shape, lhs_batch_dims, lhs_contracting_dims);
  CHECK_OK(lhs_non_contracting_dims);
  for (int64_t lhs_non_contracting_dim : lhs_non_contracting_dims.value()) {
    lhs_one_dim_tiles[lhs_non_contracting_dim] =
        result_tile.one_dim_tiles()[output_dim_id];
    ++output_dim_id;
  }

  // rhs_non_contracting_dims
  auto rhs_non_contracting_dims =
      GetNonContractingDims(rhs_shape, rhs_batch_dims, rhs_contracting_dims);
  CHECK_OK(rhs_non_contracting_dims);
  for (int64_t rhs_non_contracting_dim : rhs_non_contracting_dims.value()) {
    rhs_one_dim_tiles[rhs_non_contracting_dim] =
        result_tile.one_dim_tiles()[output_dim_id];
    ++output_dim_id;
  }

  // lhs and rhs contracting dims
  for (auto [contracting_dim_id, contracting_dims] :
       llvm::enumerate(llvm::zip(lhs_contracting_dims, rhs_contracting_dims))) {
    auto [lhs_contracting_dim, rhs_contracting_dim] = contracting_dims;
    const TilingSpace::DimensionInfo& contracting_dim_info =
        tiling_space.GetDimensionInfo(dot, output_dim_id++);
    CHECK(contracting_dim_info.type ==
          TilingSpace::DimensionSemantics::kSequential)
        << "Expected a sequential dimension info for contracting dimension "
        << lhs_contracting_dim << " in dot op " << dot.ToString();
    AffineExpr tile_id = getAffineDimExpr(contracting_dim_info.id, ctx);
    AffineExpr tile_size = getAffineSymbolExpr(contracting_dim_info.id, ctx);
    AffineExpr c1 = getAffineConstantExpr(1, ctx);

    lhs_one_dim_tiles[lhs_contracting_dim] =
        rhs_one_dim_tiles[rhs_contracting_dim] =
            DimTile{tile_id * tile_size, tile_size, c1,
                    getAffineConstantExpr(
                        rhs_shape.dimensions(rhs_contracting_dim), ctx)};
  }
  return TiledOperands{
      SymbolicTiles{ExperimentalSymbolicTile{ctx, result_tile.num_tile_ids(),
                                             result_tile.num_rt_vars(),
                                             std::move(lhs_one_dim_tiles)},
                    ExperimentalSymbolicTile{ctx, result_tile.num_tile_ids(),
                                             result_tile.num_rt_vars(),
                                             std::move(rhs_one_dim_tiles)}},
      ConstraintExpression::GetAlwaysSatisfied()};
}

TiledOperands PropagateTileToInputForReduceOp(
    const TilingSpace& tiling_space, const HloReduceInstruction& reduce,
    const ExperimentalSymbolicTile& result_tile) {
  MLIRContext* ctx = result_tile.mlir_context();
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce.dimensions().begin(),
                                               reduce.dimensions().end());

  const Shape& input_shape = reduce.operand(0)->shape();
  const int64_t output_rank = GetFirstShape(&reduce).dimensions().size();

  SmallVector<DimTile> input_one_dim_tiles(input_shape.dimensions().size());
  int64_t output_dim_id = 0;
  int64_t reduction_dim_count = 0;
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (reduce_dims_ids.contains(input_dim_id)) {
      const TilingSpace::DimensionInfo& reduction_dim_info =
          tiling_space.GetDimensionInfo(reduce,
                                        output_rank + reduction_dim_count++);
      CHECK(reduction_dim_info.type ==
            TilingSpace::DimensionSemantics::kSequential)
          << "Expected a sequential dimension info for contracting dimension "
          << input_dim_id << " in reduce op " << reduce.ToString();

      AffineExpr tile_id = getAffineDimExpr(reduction_dim_info.id, ctx);
      AffineExpr tile_size = getAffineSymbolExpr(reduction_dim_info.id, ctx);
      AffineExpr c1 = getAffineConstantExpr(1, ctx);

      input_one_dim_tiles[input_dim_id] = DimTile{
          tile_id * tile_size, tile_size, c1,
          getAffineConstantExpr(input_shape.dimensions(input_dim_id), ctx)};
      continue;
    }
    input_one_dim_tiles[input_dim_id] =
        result_tile.one_dim_tiles()[output_dim_id++];
  }
  ExperimentalSymbolicTile init_value_tile{
      ctx, result_tile.num_tile_ids(), {}, {}, {}, {}, {}};

  SymbolicTiles operand_tiles(
      reduce.input_count(),
      ExperimentalSymbolicTile{ctx, result_tile.num_tile_ids(),
                               result_tile.num_rt_vars(),
                               std::move(input_one_dim_tiles)});
  operand_tiles.append(SymbolicTiles(reduce.input_count(), init_value_tile));
  return TiledOperands{std::move(operand_tiles),
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
  if (hlo.opcode() == HloOpcode::kDot) {
    return PropagateTileToInputForDotOp(
        tiling_space, *Cast<HloDotInstruction>(&hlo), result_tile);
  }
  if (hlo.opcode() == HloOpcode::kPad) {
    const HloPadInstruction& pad = *Cast<HloPadInstruction>(&hlo);
    return PropagateTileToInputForPadOp(pad, result_tile);
  }
  if (hlo.opcode() == HloOpcode::kReduce) {
    const HloReduceInstruction& reduce = *Cast<HloReduceInstruction>(&hlo);
    return PropagateTileToInputForReduceOp(tiling_space, reduce, result_tile);
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
