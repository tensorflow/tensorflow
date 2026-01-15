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
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
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
#include "xla/permutation_util.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"
#include "xla/service/gpu/model/experimental/tiling_space.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::MLIRContext;

SymbolicTiles PropagateTileToInputForCwiseOp(const HloInstruction& hlo,
                                             const SymbolicTile& input_tile) {
  return SymbolicTiles(hlo.operand_count(), input_tile);
};

SymbolicTiles PropagateTileToOutputForCwiseOp(const HloInstruction& hlo,
                                              const SymbolicTile& output_tile) {
  return {output_tile};
}

SymbolicTiles PropagateTileToInputForBroadcastOp(
    const HloBroadcastInstruction& bcast, const SymbolicTile& output_tile) {
  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(bcast.operand(0)->shape().dimensions().size());
  for (auto broadcast_dim : bcast.dimensions()) {
    dim_tiles.push_back(output_tile.dim_tiles()[broadcast_dim]);
  };

  return {SymbolicTile{output_tile.tiling_space(), std::move(dim_tiles)}};
}

SymbolicTiles PropagateTileToOutputForBroadcastOp(
    const HloBroadcastInstruction& bcast, const SymbolicTile& input_tile) {
  absl::Span<const int64_t> bcast_dims = bcast.dimensions();
  const Shape& output_shape = bcast.shape();
  auto output_rank = bcast.shape().dimensions().size();

  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(output_rank);
  for (auto [output_dim_id, output_dim] :
       llvm::enumerate(output_shape.dimensions())) {
    auto bcast_dim = absl::c_find(bcast_dims, output_dim_id);
    // If the dimension is not a broadcast dimension, create a tile that covers
    // the entire dimension.
    if (bcast_dim == bcast_dims.end()) {
      dim_tiles.push_back(
          GetFullDimTile(output_dim, input_tile.mlir_context()));
      continue;
    }
    dim_tiles.push_back(
        input_tile.dim_tiles()[std::distance(bcast_dims.begin(), bcast_dim)]);
  }
  return {SymbolicTile{input_tile.tiling_space(), std::move(dim_tiles)}};
}

std::optional<SymbolicTiles> PropagateTileToInputForConcatenateOp(
    const HloConcatenateInstruction& concatenate,
    const SymbolicTile& output_tile) {
  int64_t num_operands = concatenate.operand_count();

  SymbolicTiles symbolic_tiles;
  symbolic_tiles.reserve(num_operands);

  // For concatenate, we need to adjust the offsets and the bounds in the
  // concatenate dimension.
  int64_t concat_dim = concatenate.concatenate_dimension();
  auto upper_bound = llvm::dyn_cast<mlir::AffineConstantExpr>(
      output_tile.upper_bounds()[concat_dim]);
  if (!upper_bound) {
    // TODO(b/422677091): Also support non-constant affine expressions for upper
    // bound.
    VLOG(2) << "Can't propagate tile to input of concatenate op with "
               "non-constant upper bound.";
    return std::nullopt;
  }
  int64_t offset = 0;
  for (const HloInstruction* operand : concatenate.operands()) {
    SmallVector<DimTile> dim_tiles(output_tile.dim_tiles());
    dim_tiles[concat_dim].offset = dim_tiles[concat_dim].offset - offset;
    int64_t operand_dim_size = operand->shape().dimensions(concat_dim);

    dim_tiles[concat_dim].upper_bound = mlir::getAffineConstantExpr(
        std::max(int64_t{0},
                 std::min(upper_bound.getValue() - offset, operand_dim_size)),
        output_tile.mlir_context());
    SymbolicTile operand_tile{output_tile.tiling_space(), std::move(dim_tiles)};
    symbolic_tiles.push_back(operand_tile);
    offset += operand_dim_size;
  }
  return symbolic_tiles;
}

template <typename T>
SmallVector<T> Concat(ArrayRef<T> c1, ArrayRef<T> c2) {
  SmallVector<T> result;
  result.append(c1.begin(), c1.end());
  result.append(c2.begin(), c2.end());
  return result;
}

SymbolicTile PropagateTileToInputForSliceImpl(
    ArrayRef<AffineExpr> slice_offsets, ArrayRef<int64_t> slice_strides,
    const SymbolicTile& output_tile) {
  int64_t num_dim_tiles = output_tile.num_dim_tiles();

  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(num_dim_tiles);

  for (const auto& [dim, result_dim_tile] :
       llvm::enumerate(output_tile.dim_tiles())) {
    // To compute element r of the result we access input
    //   in = r * slice_strides[i] + slice_starts[i].
    // Replacing r with (t * strides[i] + offsets[i]) we get
    //   t * (strides[i] * slice_strides[i]) +
    //   (offsets[i] * slice_strides[i] + slice_starts[i]).
    DimTile dim_tile;
    dim_tile.offset =
        slice_offsets[dim] + result_dim_tile.offset * slice_strides[dim];
    dim_tile.stride = result_dim_tile.stride * slice_strides[dim];
    dim_tile.size = result_dim_tile.size;
    // Upper bound condition is `r < upper_bounds[i](t)`.
    // By replacing r with `(in - slice_offsets[i]) / slice_strides[i]` we get
    // in < upper_bounds[i](t) * slice_strides[i] + slice_offsets[i].
    dim_tile.upper_bound =
        result_dim_tile.upper_bound * slice_strides[dim] + slice_offsets[dim];
    dim_tiles.push_back(std::move(dim_tile));
  }
  return SymbolicTile{output_tile.tiling_space(), std::move(dim_tiles)};
}

SymbolicTiles PropagateTileToInputForSliceOp(const HloInstruction& slice,
                                             const SymbolicTile& output_tile) {
  SmallVector<AffineExpr, 3> slice_offset_exprs;
  slice_offset_exprs.reserve(slice.shape().dimensions().size());
  for (int64_t slice_offset : slice.slice_starts()) {
    slice_offset_exprs.push_back(
        mlir::getAffineConstantExpr(slice_offset, output_tile.mlir_context()));
  }
  auto operand_tile = SymbolicTiles{PropagateTileToInputForSliceImpl(
      slice_offset_exprs, slice.slice_strides(), output_tile)};
  return {operand_tile};
}

std::optional<int64_t> GetInt64FromConstant(const HloInstruction& hlo) {
  if (hlo.IsConstant() && hlo.shape().dimensions().empty()) {
    return LiteralUtil::LiteralAsScalarInt64(
        Cast<HloConstantInstruction>(&hlo)->literal());
  }
  return std::nullopt;
}

SymbolicTiles PropagateTileToInputForDynamicSliceOp(
    const TilingSpace& tiling_space,
    const HloDynamicSliceInstruction& dynamic_slice,
    const SymbolicTile& output_tile) {
  const int64_t first_index_operand_number =
      dynamic_slice.first_index_operand_number();
  CHECK(dynamic_slice.operand(first_index_operand_number)
            ->shape()
            .dimensions()
            .empty())
      << "b/118437727: Old form, not supported.";
  MLIRContext* ctx = output_tile.mlir_context();
  int64_t num_dim_tiles = output_tile.num_dim_tiles();

  SmallVector<AffineExpr, 3> slice_offset_exprs(num_dim_tiles);
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
        rt_var_info.id + output_tile.tiling_space().num_dimensions(), ctx);
  }

  SymbolicTiles operand_tiles{PropagateTileToInputForSliceImpl(
      slice_offset_exprs, SmallVector<int64_t>(num_dim_tiles, 1), output_tile)};
  SymbolicTile scalar_tensor_tile{output_tile.tiling_space(), {}, {}, {}, {}};
  for (int i = 0; i < num_dim_tiles; ++i) {
    operand_tiles.push_back(scalar_tensor_tile);
  }
  return operand_tiles;
}

std::optional<SymbolicTiles> PropagateTileToInputForPadOp(
    const HloPadInstruction& pad, const SymbolicTile& output_tile) {
  MLIRContext* ctx = output_tile.mlir_context();
  const PaddingConfig& padding_config = pad.padding_config();

  // For each dimension, the low padding is subtracted from the offsets.
  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(output_tile.num_dim_tiles());
  for (const auto [result_dim_tile, padding_dim, operand_dim] :
       llvm::zip(output_tile.dim_tiles(), padding_config.dimensions(),
                 pad.operand(0)->shape().dimensions())) {
    if (padding_dim.interior_padding() != 0) {
      VLOG(2)
          << "Can't propagate tile to input of pad op with interior padding.";
      return std::nullopt;
    }
    dim_tiles.push_back(
        DimTile{result_dim_tile.offset - padding_dim.edge_padding_low(),
                result_dim_tile.size, result_dim_tile.stride,
                mlir::getAffineConstantExpr(operand_dim, ctx)});
  }
  SymbolicTile operand_tile{output_tile.tiling_space(), std::move(dim_tiles)};

  // Pad also has a padding value, but it is a scalar, therefore we only need
  // to propagate the inputs.
  SymbolicTile padding_value_tile{output_tile.tiling_space(), {}, {}, {}, {}};

  return SymbolicTiles{operand_tile, padding_value_tile};
}

SymbolicTile PropagateTileThroughTransposeOp(
    const SymbolicTile& tile, absl::Span<const int64_t> permutation) {
  SmallVector<DimTile> dim_tiles(tile.num_dim_tiles());
  for (const auto [dim, permutated_dim] : llvm::enumerate(permutation)) {
    dim_tiles[permutated_dim] = tile.dim_tiles()[dim];
  }
  return SymbolicTile{tile.tiling_space(), std::move(dim_tiles)};
}

SymbolicTiles PropagateTileToInputForTransposeOp(
    const HloInstruction& transpose, const SymbolicTile& output_tile) {
  auto operand_tile =
      PropagateTileThroughTransposeOp(output_tile, transpose.dimensions());
  return {operand_tile};
}

SymbolicTiles PropagateTileToOutputForTransposeOp(
    const HloInstruction& transpose, const SymbolicTile& input_tile) {
  auto output_tile = PropagateTileThroughTransposeOp(
      input_tile, InversePermutation(transpose.dimensions()));
  return {output_tile};
}

SymbolicTiles PropagateTileToInputForDotOp(const TilingSpace& tiling_space,
                                           const HloDotInstruction& dot,
                                           const SymbolicTile& output_tile) {
  MLIRContext* ctx = output_tile.mlir_context();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  absl::Span<const int64_t> lhs_contracting_dims(
      dim_numbers.lhs_contracting_dimensions());
  absl::Span<const int64_t> rhs_contracting_dims =
      dim_numbers.rhs_contracting_dimensions();

  absl::Span<const int64_t> lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
  absl::Span<const int64_t> rhs_batch_dims = dim_numbers.rhs_batch_dimensions();

  const Shape& lhs_shape = dot.operand(0)->shape();
  const int64_t lhs_rank = lhs_shape.dimensions().size();
  SmallVector<DimTile> lhs_dim_tiles(lhs_rank);

  const Shape& rhs_shape = dot.operand(1)->shape();
  const int64_t rhs_rank = rhs_shape.dimensions().size();
  SmallVector<DimTile> rhs_dim_tiles(rhs_rank);

  // According to the StableHLO specification, the dimensions of the output
  // shape are ordered as follows:
  //   lhs_batch_dims | lhs_non_contracting_dims | rhs_non_contracting_dims

  // Populate lhs and rhs batch dimensions.
  for (auto [output_dim_id, batch_dims] :
       llvm::enumerate(llvm::zip(lhs_batch_dims, rhs_batch_dims))) {
    auto [lhs_batch_dim, rhs_batch_dim] = batch_dims;
    rhs_dim_tiles[rhs_batch_dim] = lhs_dim_tiles[lhs_batch_dim] =
        output_tile.dim_tiles()[output_dim_id];
  }

  // lhs_non_contracting_dims
  int64_t output_dim_id = lhs_batch_dims.size();
  auto lhs_non_contracting_dims =
      GetNonContractingDims(lhs_shape, lhs_batch_dims, lhs_contracting_dims);
  CHECK_OK(lhs_non_contracting_dims);
  for (int64_t lhs_non_contracting_dim : lhs_non_contracting_dims.value()) {
    lhs_dim_tiles[lhs_non_contracting_dim] =
        output_tile.dim_tiles()[output_dim_id];
    ++output_dim_id;
  }

  // rhs_non_contracting_dims
  auto rhs_non_contracting_dims =
      GetNonContractingDims(rhs_shape, rhs_batch_dims, rhs_contracting_dims);
  CHECK_OK(rhs_non_contracting_dims);
  for (int64_t rhs_non_contracting_dim : rhs_non_contracting_dims.value()) {
    rhs_dim_tiles[rhs_non_contracting_dim] =
        output_tile.dim_tiles()[output_dim_id];
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
    lhs_dim_tiles[lhs_contracting_dim] = rhs_dim_tiles[rhs_contracting_dim] =
        GetDefaultDimTile(contracting_dim_info.id,
                          contracting_dim_info.dimension_size, ctx);
  }
  return SymbolicTiles{
      SymbolicTile{output_tile.tiling_space(), std::move(lhs_dim_tiles)},
      SymbolicTile{output_tile.tiling_space(), std::move(rhs_dim_tiles)}};
}

SymbolicTiles PropagateTileToInputForReduceOp(
    const TilingSpace& tiling_space, const HloReduceInstruction& reduce,
    const SymbolicTile& output_tile) {
  MLIRContext* ctx = output_tile.mlir_context();
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce.dimensions().begin(),
                                               reduce.dimensions().end());

  const Shape& input_shape = reduce.operand(0)->shape();
  const int64_t output_rank = GetFirstShape(&reduce).dimensions().size();

  SmallVector<DimTile> input_dim_tiles(input_shape.dimensions().size());
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

      input_dim_tiles[input_dim_id] =
          GetDefaultDimTile(reduction_dim_info.id, input_dim, ctx);
      continue;
    }
    input_dim_tiles[input_dim_id] = output_tile.dim_tiles()[output_dim_id++];
  }
  SymbolicTile init_value_tile{output_tile.tiling_space(), {}, {}, {}, {}};

  SymbolicTiles operand_tiles(
      reduce.input_count(),
      SymbolicTile{output_tile.tiling_space(), std::move(input_dim_tiles)});
  operand_tiles.append(SymbolicTiles(reduce.input_count(), init_value_tile));
  return SymbolicTiles{std::move(operand_tiles)};
}

SymbolicTiles PropagateTileToOutputForReduceOp(
    const HloReduceInstruction& reduce, const SymbolicTile& input_tile) {
  absl::flat_hash_set<int64_t> reduce_dims(reduce.dimensions().begin(),
                                           reduce.dimensions().end());

  SmallVector<DimTile> output_dim_tiles;
  output_dim_tiles.reserve(input_tile.num_dim_tiles() - reduce_dims.size());
  for (auto [idx, input_dim_tile] : llvm::enumerate(input_tile.dim_tiles())) {
    if (!reduce_dims.contains(idx)) {
      output_dim_tiles.push_back(input_dim_tile);
    }
  }

  SymbolicTile output_tile{input_tile.tiling_space(),
                           std::move(output_dim_tiles)};
  return {std::move(output_tile)};
}

}  // namespace

std::string ToString(const SymbolicTiles& tiles) {
  std::stringstream ss;
  for (const auto& [index, tile] : llvm::enumerate(tiles)) {
    ss << index << ") " << tile.ToString() << "\n";
  }
  return ss.str();
}

std::optional<SymbolicTiles> PropagateTileToInput(
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const SymbolicTile& output_tile, int64_t output_index) {
  if (HloInstruction::IsOpElementwise(hlo.opcode()) ||
      hlo.opcode() == HloOpcode::kMap) {
    return {PropagateTileToInputForCwiseOp(hlo, output_tile)};
  }
  if (hlo.opcode() == HloOpcode::kBroadcast) {
    return PropagateTileToInputForBroadcastOp(
        *Cast<HloBroadcastInstruction>(&hlo), output_tile);
  }
  if (hlo.opcode() == HloOpcode::kConcatenate) {
    return PropagateTileToInputForConcatenateOp(
        *Cast<HloConcatenateInstruction>(&hlo), output_tile);
  }
  if (hlo.opcode() == HloOpcode::kDynamicSlice) {
    return PropagateTileToInputForDynamicSliceOp(
        tiling_space, *Cast<HloDynamicSliceInstruction>(&hlo), output_tile);
  }
  if (hlo.opcode() == HloOpcode::kDot) {
    return PropagateTileToInputForDotOp(
        tiling_space, *Cast<HloDotInstruction>(&hlo), output_tile);
  }
  if (hlo.opcode() == HloOpcode::kPad) {
    const HloPadInstruction& pad = *Cast<HloPadInstruction>(&hlo);
    return PropagateTileToInputForPadOp(pad, output_tile);
  }
  if (hlo.opcode() == HloOpcode::kReduce) {
    const HloReduceInstruction& reduce = *Cast<HloReduceInstruction>(&hlo);
    return PropagateTileToInputForReduceOp(tiling_space, reduce, output_tile);
  }
  if (hlo.opcode() == HloOpcode::kTranspose) {
    return PropagateTileToInputForTransposeOp(hlo, output_tile);
  }
  if (hlo.opcode() == HloOpcode::kSlice) {
    return PropagateTileToInputForSliceOp(hlo, output_tile);
  }
  LOG(INFO) << "Output to input tile propagation not implemented for "
            << hlo.opcode();
  return std::nullopt;
}

std::optional<SymbolicTiles> PropagateTileToOutput(
    const TilingSpace& tiling_space, const HloInstruction& hlo,
    const SymbolicTile& input_tile, int64_t input_index) {
  if (HloInstruction::IsOpElementwise(hlo.opcode()) ||
      hlo.opcode() == HloOpcode::kMap) {
    return PropagateTileToOutputForCwiseOp(hlo, input_tile);
  }
  if (hlo.opcode() == HloOpcode::kBroadcast) {
    return PropagateTileToOutputForBroadcastOp(
        *Cast<HloBroadcastInstruction>(&hlo), input_tile);
  }
  if (hlo.opcode() == HloOpcode::kTranspose) {
    return PropagateTileToOutputForTransposeOp(hlo, input_tile);
  }
  if (hlo.opcode() == HloOpcode::kReduce) {
    const HloReduceInstruction& reduce = *Cast<HloReduceInstruction>(&hlo);
    if (input_index >= reduce.input_count()) {
      LOG(INFO) << "Input to output tile propagation not implemented "
                   "from reduction init operands.";
      return std::nullopt;
    }
    return PropagateTileToOutputForReduceOp(reduce, input_tile);
  }
  LOG(INFO) << "Input to output tile propagation not implemented for "
            << hlo.opcode();
  return std::nullopt;
}

}  // namespace xla::gpu::experimental
