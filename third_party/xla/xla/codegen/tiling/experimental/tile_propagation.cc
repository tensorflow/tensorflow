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

#include "xla/codegen/tiling/experimental/tile_propagation.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/reshape_analysis.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;

DimTile GetDimTile(const TilingSpace::DimensionInfo& dim_info, bool is_symbolic,
                   int64_t num_dimensions, MLIRContext* ctx) {
  CHECK(is_symbolic || dim_info.tile_size >= 0)
      << "Concrete tile size cannot be negative.";
  SymbolicExpr tile_size =
      is_symbolic ? CreateSymbolExpr(dim_info.id.value(), num_dimensions, ctx)
                  : CreateSymbolicConstant(dim_info.tile_size, ctx);
  return GetDefaultDimTile(dim_info.id.value(), tile_size,
                           dim_info.dimension_size);
}

Tiles PropagateTileToInputForCwiseOp(const HloInstruction& hlo,
                                     const Tile& input_tile) {
  return Tiles(hlo.operand_count(), input_tile);
}

Tiles PropagateTileToOutputForCwiseOp(const HloInstruction& hlo,
                                      const Tile& output_tile) {
  return {output_tile};
}

Tiles PropagateTileToInputForBroadcastOp(const HloBroadcastInstruction& bcast,
                                         const Tile& output_tile) {
  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(bcast.operand(0)->shape().dimensions().size());
  for (auto broadcast_dim : bcast.dimensions()) {
    dim_tiles.push_back(output_tile.dim_tiles()[broadcast_dim]);
  };

  return {Tile{output_tile.tiling_space(), std::move(dim_tiles)}};
}

Tiles PropagateTileToOutputForBroadcastOp(const HloBroadcastInstruction& bcast,
                                          const Tile& input_tile) {
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
  return {Tile{input_tile.tiling_space(), std::move(dim_tiles)}};
}

absl::StatusOr<Tiles> PropagateTileToInputForConcatenateOp(
    const HloConcatenateInstruction& concatenate, const Tile& output_tile) {
  int64_t num_operands = concatenate.operand_count();

  Tiles tiles;
  tiles.reserve(num_operands);

  // For concatenate, we need to adjust the offsets and the bounds in the
  // concatenate dimension.
  int64_t concat_dim = concatenate.concatenate_dimension();
  auto upper_bound = output_tile.upper_bounds()[concat_dim];
  if (upper_bound.GetType() != SymbolicExprType::kConstant) {
    // TODO(b/422677091): Also support non-constant symbolic expressions for
    // upper bound.
    return absl::UnimplementedError(
        "Can't propagate tile to input of concatenate op with "
        "non-constant upper bound.");
  }
  int64_t offset = 0;
  for (const HloInstruction* operand : concatenate.operands()) {
    SmallVector<DimTile> dim_tiles(output_tile.dim_tiles());
    dim_tiles[concat_dim].offset = dim_tiles[concat_dim].offset - offset;
    CHECK_LT(concat_dim, operand->shape().dimensions().size());
    int64_t operand_dim_size = operand->shape().dimensions(concat_dim);

    dim_tiles[concat_dim].upper_bound = CreateSymbolicConstant(
        std::max(int64_t{0},
                 std::min(upper_bound.GetValue() - offset, operand_dim_size)),
        output_tile.mlir_context());
    Tile operand_tile{output_tile.tiling_space(), std::move(dim_tiles)};
    tiles.push_back(operand_tile);
    offset += operand_dim_size;
  }
  return tiles;
}

Tiles PropagateTileToOutputForConcatenateOp(
    const HloConcatenateInstruction& concatenate, const Tile& input_tile,
    int64_t input_index) {
  // Offsets and upper bounds need to be adjusted in the concatenate dimension
  // for the output.
  int64_t concat_dim = concatenate.concatenate_dimension();

  int64_t output_offset = 0;
  for (int i = 0; i < input_index; ++i) {
    CHECK_LT(concat_dim, concatenate.operand(i)->shape().dimensions().size());
    output_offset += concatenate.operand(i)->shape().dimensions(concat_dim);
  }

  SmallVector<DimTile> output_dim_tiles(input_tile.dim_tiles().begin(),
                                        input_tile.dim_tiles().end());

  output_dim_tiles[concat_dim].offset =
      output_offset + input_tile.dim_tiles()[concat_dim].offset;

  output_dim_tiles[concat_dim].upper_bound =
      input_tile.dim_tiles()[concat_dim].upper_bound + output_offset;

  return {Tile{input_tile.tiling_space(), std::move(output_dim_tiles)}};
}

template <typename T>
SmallVector<T> Concat(ArrayRef<T> c1, ArrayRef<T> c2) {
  SmallVector<T> result;
  result.append(c1.begin(), c1.end());
  result.append(c2.begin(), c2.end());
  return result;
}

Tile PropagateTileToInputForSliceImpl(ArrayRef<SymbolicExpr> slice_offsets,
                                      ArrayRef<int64_t> slice_strides,
                                      const Tile& output_tile) {
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
  return Tile{output_tile.tiling_space(), std::move(dim_tiles)};
}

Tiles PropagateTileToInputForSliceOp(const HloInstruction& slice,
                                     const Tile& output_tile) {
  SmallVector<SymbolicExpr, 3> slice_offset_exprs;
  slice_offset_exprs.reserve(slice.shape().dimensions().size());
  for (int64_t slice_offset : slice.slice_starts()) {
    slice_offset_exprs.push_back(
        CreateSymbolicConstant(slice_offset, output_tile.mlir_context()));
  }
  auto operand_tile = Tiles{PropagateTileToInputForSliceImpl(
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

Tiles PropagateTileToInputForDynamicSliceOp(
    const TilingSpace& tiling_space,
    const HloDynamicSliceInstruction& dynamic_slice, const Tile& output_tile) {
  const int64_t first_index_operand_number =
      dynamic_slice.first_index_operand_number();
  CHECK(dynamic_slice.operand(first_index_operand_number)
            ->shape()
            .dimensions()
            .empty())
      << "b/118437727: Old form, not supported.";
  MLIRContext* ctx = output_tile.mlir_context();
  int64_t num_dim_tiles = output_tile.num_dim_tiles();

  SmallVector<SymbolicExpr, 3> slice_offset_exprs(num_dim_tiles);
  for (auto [dim, slice_size] :
       llvm::enumerate(dynamic_slice.dynamic_slice_sizes())) {
    auto slice_offset = dynamic_slice.operand(dim + first_index_operand_number);
    std::optional<int64_t> offset_const = GetInt64FromConstant(*slice_offset);

    int64_t operand_id = dim + first_index_operand_number;
    auto rt_var_info_or = tiling_space.GetRTVarInfo(dynamic_slice, operand_id);
    CHECK(rt_var_info_or.has_value())
        << "Runtime variable not found for " << dynamic_slice.ToString()
        << " operand " << operand_id;
    const TilingSpace::RTVarInfo& rt_var_info = *rt_var_info_or.value();
    if (offset_const.has_value()) {
      int64_t clamped_offset = std::clamp(
          *offset_const, rt_var_info.bounds.lower, rt_var_info.bounds.upper);
      slice_offset_exprs[dim] = CreateSymbolicConstant(clamped_offset, ctx);
      continue;
    }
    slice_offset_exprs[dim] = CreateSymbolExpr(
        output_tile.tiling_space().num_dimensions() + rt_var_info.id,
        output_tile.tiling_space().num_dimensions(), ctx);
  }

  Tiles operand_tiles{PropagateTileToInputForSliceImpl(
      slice_offset_exprs, SmallVector<int64_t>(num_dim_tiles, 1), output_tile)};
  Tile scalar_tensor_tile{output_tile.tiling_space(), {}, {}, {}, {}};
  for (int i = 0; i < num_dim_tiles; ++i) {
    operand_tiles.push_back(scalar_tensor_tile);
  }
  return operand_tiles;
}

absl::StatusOr<Tiles> PropagateTileToInputForPadOp(const HloPadInstruction& pad,
                                                   const Tile& output_tile) {
  MLIRContext* ctx = output_tile.mlir_context();
  const PaddingConfig& padding_config = pad.padding_config();

  // For each dimension, the low padding is subtracted from the offsets.
  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(output_tile.num_dim_tiles());
  for (const auto [result_dim_tile, padding_dim, operand_dim] :
       llvm::zip(output_tile.dim_tiles(), padding_config.dimensions(),
                 pad.operand(0)->shape().dimensions())) {
    if (padding_dim.interior_padding() != 0) {
      return absl::UnimplementedError(
          "Can't propagate tile to input of pad op with interior padding.");
    }
    dim_tiles.push_back(
        DimTile{result_dim_tile.offset - padding_dim.edge_padding_low(),
                result_dim_tile.size, result_dim_tile.stride,
                CreateSymbolicConstant(operand_dim, ctx)});
  }
  Tile operand_tile{output_tile.tiling_space(), std::move(dim_tiles)};

  // Pad also has a padding value, but it is a scalar, therefore we only need
  // to propagate the inputs.
  Tile padding_value_tile{output_tile.tiling_space(), {}, {}, {}, {}};

  return Tiles{operand_tile, padding_value_tile};
}

Tile PropagateTileThroughTransposeOp(const Tile& tile,
                                     absl::Span<const int64_t> permutation) {
  SmallVector<DimTile> dim_tiles(tile.num_dim_tiles());
  for (const auto [dim, permutated_dim] : llvm::enumerate(permutation)) {
    dim_tiles[permutated_dim] = tile.dim_tiles()[dim];
  }
  return Tile{tile.tiling_space(), std::move(dim_tiles)};
}

Tiles PropagateTileToInputForTransposeOp(const HloInstruction& transpose,
                                         const Tile& output_tile) {
  auto operand_tile =
      PropagateTileThroughTransposeOp(output_tile, transpose.dimensions());
  return {operand_tile};
}

Tiles PropagateTileToOutputForTransposeOp(const HloInstruction& transpose,
                                          const Tile& input_tile) {
  auto output_tile = PropagateTileThroughTransposeOp(
      input_tile, InversePermutation(transpose.dimensions()));
  return {output_tile};
}

Tiles PropagateTileToInputForDotOp(const TilingSpace& tiling_space,
                                   const HloInstruction& hlo,
                                   const Tile& output_tile) {
  MLIRContext* ctx = output_tile.mlir_context();
  const DotDimensionNumbers& dim_numbers = hlo.dot_dimension_numbers();
  absl::Span<const int64_t> lhs_contracting_dims(
      dim_numbers.lhs_contracting_dimensions());
  absl::Span<const int64_t> rhs_contracting_dims =
      dim_numbers.rhs_contracting_dimensions();

  absl::Span<const int64_t> lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
  absl::Span<const int64_t> rhs_batch_dims = dim_numbers.rhs_batch_dimensions();

  const Shape& lhs_shape = hlo.operand(0)->shape();
  const int64_t lhs_rank = lhs_shape.dimensions().size();
  SmallVector<DimTile> lhs_dim_tiles(lhs_rank);

  const Shape& rhs_shape = hlo.operand(1)->shape();
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
        tiling_space.GetDimensionInfo(hlo, output_dim_id++);
    CHECK(contracting_dim_info.type ==
          TilingSpace::DimensionSemantics::kSequential)
        << "Expected a sequential dimension info for contracting dimension "
        << lhs_contracting_dim << " in dot (like) op " << hlo.ToString();
    lhs_dim_tiles[lhs_contracting_dim] = rhs_dim_tiles[rhs_contracting_dim] =
        GetDimTile(contracting_dim_info, tiling_space.IsSymbolic(),
                   tiling_space.num_dimensions(), ctx);
  }
  return Tiles{Tile{output_tile.tiling_space(), std::move(lhs_dim_tiles)},
               Tile{output_tile.tiling_space(), std::move(rhs_dim_tiles)}};
}

// Helper function for PropagateTileToInputForScaledDotOp to compute the
// scale tile from the operand tile.
Tile ComputeTileForScale(const Shape& scale_shape, const Shape& operand_shape,
                         const Tile& operand_tile, MLIRContext* ctx) {
  SmallVector<DimTile> scale_dim_tiles;
  scale_dim_tiles.reserve(scale_shape.dimensions().size());
  for (auto [dim, operand_dim_tile] :
       llvm::enumerate(operand_tile.dim_tiles())) {
    CHECK_LT(dim, scale_shape.dimensions().size());
    const int64_t scale_dim_size = scale_shape.dimensions(dim);
    CHECK_LT(dim, operand_shape.dimensions().size());
    const int64_t operand_dim_size = operand_shape.dimensions(dim);
    if (scale_dim_size == operand_dim_size) {
      scale_dim_tiles.push_back(operand_dim_tile);
      continue;
    }

    const int64_t block_size = operand_dim_size / scale_dim_size;
    CHECK_GT(block_size, 1);
    auto max_index = (operand_dim_tile.offset +
                      (operand_dim_tile.size - 1) * operand_dim_tile.stride)
                         .floorDiv(block_size);
    auto min_index = operand_dim_tile.offset.floorDiv(block_size);
    scale_dim_tiles.push_back(
        DimTile{operand_dim_tile.offset.floorDiv(block_size),
                max_index - min_index + 1, CreateSymbolicConstant(1, ctx),
                operand_dim_tile.upper_bound.floorDiv(block_size)});
  }
  return Tile{operand_tile.tiling_space(), std::move(scale_dim_tiles)};
}

Tiles PropagateTileToInputForScaledDotOp(const TilingSpace& tiling_space,
                                         const HloInstruction& hlo,
                                         const Tile& output_tile) {
  Tiles operand_tiles =
      PropagateTileToInputForDotOp(tiling_space, hlo, output_tile);

  auto lhs_scale_tile =
      ComputeTileForScale(hlo.operand(2)->shape(), hlo.operand(0)->shape(),
                          operand_tiles[0], output_tile.mlir_context());
  auto rhs_scale_tile =
      ComputeTileForScale(hlo.operand(3)->shape(), hlo.operand(1)->shape(),
                          operand_tiles[1], output_tile.mlir_context());

  return {std::move(operand_tiles[0]), std::move(operand_tiles[1]),
          std::move(lhs_scale_tile), std::move(rhs_scale_tile)};
}

Tiles PropagateTileToInputForReduceOp(const TilingSpace& tiling_space,
                                      const HloReduceInstruction& reduce,
                                      const Tile& output_tile) {
  MLIRContext* ctx = output_tile.mlir_context();
  SmallVector<int64_t, 2> reduce_dims_ids(reduce.dimensions().begin(),
                                          reduce.dimensions().end());

  const Shape& input_shape = reduce.operand(0)->shape();
  const int64_t output_rank = GetFirstShape(&reduce).dimensions().size();

  SmallVector<DimTile> input_dim_tiles(input_shape.dimensions().size());
  int64_t output_dim_id = 0;
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (auto it = absl::c_find(reduce_dims_ids, input_dim_id);
        it != reduce_dims_ids.end()) {
      const TilingSpace::DimensionInfo& reduction_dim_info =
          tiling_space.GetDimensionInfo(
              reduce, output_rank + std::distance(reduce_dims_ids.begin(), it));
      CHECK(reduction_dim_info.type ==
            TilingSpace::DimensionSemantics::kSequential)
          << "Expected a sequential dimension info for contracting dimension "
          << input_dim_id << " in reduce op " << reduce.ToString();
      input_dim_tiles[input_dim_id] =
          GetDimTile(reduction_dim_info, tiling_space.IsSymbolic(),
                     tiling_space.num_dimensions(), ctx);
      continue;
    }
    input_dim_tiles[input_dim_id] = output_tile.dim_tiles()[output_dim_id++];
  }
  Tile init_value_tile{output_tile.tiling_space(), {}, {}, {}, {}};

  Tiles operand_tiles(reduce.input_count(), Tile{output_tile.tiling_space(),
                                                 std::move(input_dim_tiles)});
  operand_tiles.append(Tiles(reduce.input_count(), init_value_tile));
  return Tiles{std::move(operand_tiles)};
}

Tiles PropagateTileToOutputForReduceOp(const HloReduceInstruction& reduce,
                                       const Tile& input_tile) {
  absl::flat_hash_set<int64_t> reduce_dims(reduce.dimensions().begin(),
                                           reduce.dimensions().end());

  SmallVector<DimTile> output_dim_tiles;
  output_dim_tiles.reserve(input_tile.num_dim_tiles() - reduce_dims.size());
  for (auto [idx, input_dim_tile] : llvm::enumerate(input_tile.dim_tiles())) {
    if (!reduce_dims.contains(idx)) {
      output_dim_tiles.push_back(input_dim_tile);
    }
  }

  Tile output_tile{input_tile.tiling_space(), std::move(output_dim_tiles)};
  return {std::move(output_tile)};
}

absl::Status IsSupportedReshape(const std::vector<MinimalReshape>& reshapes) {
  for (const auto& minimal_reshape : reshapes) {
    if (minimal_reshape.category == MinimalReshapeCategory::kExpandShape) {
      return absl::UnimplementedError("Unsupported reshape (kExpandShape).");
    }
    if (minimal_reshape.category == MinimalReshapeCategory::kGeneric) {
      return absl::UnimplementedError("Unsupported reshape (kGeneric).");
    }
  }
  return absl::OkStatus();
}

// Returns info for all dimensions in the range with size > 1.
// If all dimensions are size 1, returns the first dimension as representative.
SmallVector<int64_t> GetNonTrivialDimIds(const Shape& shape,
                                         const DimensionRange& range) {
  SmallVector<int64_t> result;
  for (int64_t i = range.start; i <= range.end(); ++i) {
    CHECK_LT(i, shape.dimensions().size());
    if (shape.dimensions(i) > 1) {
      result.push_back(i);
    }
  }
  if (!result.empty()) {
    return result;
  }
  return range.count == 0 ? SmallVector<int64_t>{}
                          : SmallVector<int64_t>{range.start};
}

// Checks source tile to see if kCollapseShape is supported.
//
// We consider a kCollapseShape reshape is supported if there is at most one
// dimension that is partially tiled. Specifically:
// - Only one dimension is truly "tiled" (size > 1).
// - Any dimensions inner to the tiled dimension are fully covered (ts_i = d_i)
// - Any dimensions outer to the tiled dimension are tile size 1 (ts_i = 1)
// - All dimensions except the innermost have stride 1.
// Example: for [3, 4] -> [12] we support:
// - ts_0 = 1, or
// - ts_1 = 4 (i.e., we take full rows)
absl::Status IsSupportedCollapseShape(absl::Span<const DimTile> source_tiles,
                                      absl::Span<const int64_t> source_dims) {
  // All dimensions except the innermost must have stride 1.
  int num_dims = static_cast<int>(source_tiles.size());
  auto IsStride1 = [](const DimTile& dt) {
    return dt.stride.GetType() == SymbolicExprType::kConstant &&
           dt.stride.GetValue() == 1;
  };
  if (!absl::c_all_of(source_tiles.subspan(0, num_dims - 1), IsStride1)) {
    return absl::UnimplementedError(
        "Unsupported minimal reshape (kCollapseShape): "
        "stride > 1 in non-innermost significant dimension");
  }

  // Find the first dimension that is not size 1.
  int i = 0;
  while (i < num_dims &&
         (source_tiles[i].size.GetType() == SymbolicExprType::kConstant &&
          source_tiles[i].size.GetValue() == 1)) {
    ++i;
  }
  // Find the last dimension that is not fully covered.
  int j = num_dims - 1;
  while (j >= 0 &&
         (source_tiles[j].size.GetType() == SymbolicExprType::kConstant &&
          source_tiles[j].size.GetValue() == source_dims[j])) {
    --j;
  }
  // All dimensions before i are size 1 and all dimensions after j are full.
  // If i >= j, then only index k=i=j potentially partially tiled.
  if (i < j) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported minimal reshape (kCollapseShape): "
                     "multiple dimensions are partially tiled. "
                     "The first dimension with tile size > 1 is at index ",
                     i,
                     " and the last dimension that is not fully covered is "
                     "at index ",
                     j, ". Expected i >= j."));
  }

  return absl::OkStatus();
}

// TODO(b/477615292) - Change MinimalReshape to store dim ids instead of ranges.
absl::Status PropagateTileThroughMinimalReshape(
    mlir::MLIRContext* mlir_context, const MinimalReshape& minimal_reshape,
    const Shape& source_shape, const Shape& target_shape,
    const Tile& source_tile, llvm::SmallVector<DimTile>& target_dim_tiles) {
  const DimensionRange& source_range = minimal_reshape.input_dim_ids;
  const DimensionRange& target_range = minimal_reshape.output_dim_ids;
  auto source_ids = GetNonTrivialDimIds(source_shape, source_range);
  auto target_ids = GetNonTrivialDimIds(target_shape, target_range);

  switch (minimal_reshape.category) {
    // 1-to-1 mapping of the "significant" dimensions (size > 1).
    case MinimalReshapeCategory::kIdentity:
    case MinimalReshapeCategory::kIncreaseRank:
    case MinimalReshapeCategory::kDecreaseRank: {
      for (auto [source_id, target_id] : llvm::zip(source_ids, target_ids)) {
        CHECK_LT(source_id, source_tile.num_dim_tiles())
            << absl::StrCat("Source dimension index ", source_id,
                            " out of bounds for tile with ",
                            source_tile.num_dim_tiles(), " dimensions");
        target_dim_tiles[target_id] = source_tile.dim_tiles()[source_id];
      }
      return absl::OkStatus();
    }
    // n-to-1 mapping of the "significant" dimensions (size > 1).
    case MinimalReshapeCategory::kCollapseShape: {
      if (source_ids.empty()) {
        return absl::UnimplementedError(
            "Unsupported minimal reshape (kCollapseShape): "
            "Expected at least one significant source dimension but found "
            "none.");
      }
      if (target_ids.size() != 1) {
        return absl::UnimplementedError(absl::StrCat(
            "Unsupported minimal reshape (kCollapseShape): "
            "Expected exactly one significant target dimension but found ",
            target_ids.size()));
      }

      if (!source_ids.empty()) {
        int64_t max_source_id = *absl::c_max_element(source_ids);
        CHECK_LT(max_source_id, source_tile.num_dim_tiles())
            << absl::StrCat("Source dimension index ", max_source_id,
                            " out of bounds for tile with ",
                            source_tile.num_dim_tiles(), " dimensions");
        CHECK_LT(max_source_id, source_shape.dimensions().size())
            << absl::StrCat("Source dimension index ", max_source_id,
                            " out of bounds for source shape with ",
                            source_shape.dimensions().size(), " dimensions");
      }
      SmallVector<DimTile> source_tiles = llvm::to_vector(llvm::map_range(
          source_ids, [&](int64_t id) { return source_tile.dim_tiles()[id]; }));
      SmallVector<int64_t> source_dims = llvm::to_vector(llvm::map_range(
          source_ids, [&](int64_t id) { return source_shape.dimensions(id); }));
      RETURN_IF_ERROR(IsSupportedCollapseShape(source_tiles, source_dims));

      llvm::SmallVector<SymbolicExpr> offsets, upper_bounds_inclusive;
      SymbolicExpr total_tile_elements =
          CreateSymbolicConstant(1, mlir_context);
      for (const auto& dt : source_tiles) {
        offsets.push_back(dt.offset);
        total_tile_elements = total_tile_elements * dt.size;
        // Compute the last valid element that each dimension allows to touch.
        upper_bounds_inclusive.push_back(
            (dt.offset + (dt.size - 1) * dt.stride).min(dt.upper_bound - 1));
      }

      int64_t target_id = target_ids[0];
      target_dim_tiles[target_id].offset =
          LinearizeShape(source_dims, offsets, mlir_context);
      // Due to IsSupportedCollapseShape, the linear stride of the collapsed
      // dimension is simply the stride of the innermost dimension.
      target_dim_tiles[target_id].stride = source_tiles.back().stride;
      target_dim_tiles[target_id].size = total_tile_elements;
      target_dim_tiles[target_id].upper_bound =
          LinearizeShape(source_dims, upper_bounds_inclusive, mlir_context) + 1;

      return absl::OkStatus();
    }
    // 1-to-n mapping of the "significant" dimensions (size > 1).
    case MinimalReshapeCategory::kExpandShape:
    // m-to-n mapping of the "significant" dimensions (size > 1).
    case MinimalReshapeCategory::kGeneric:
      return absl::UnimplementedError(
          "Unsupported minimal reshape: should already be rejected by "
          "IsSupportedReshape");
  }
}

absl::StatusOr<Tile> PropagateTileThroughReshape(const Tile& tile,
                                                 const Shape& src,
                                                 const Shape& dst) {
  VLOG(2) << "PropagateTileThroughReshape:\n"
          << "  src: " << src.ToString() << "\n"
          << "  dst: " << dst.ToString() << "\n"
          << "  tile: " << tile.ToString();
  std::vector<MinimalReshape> reshapes = GetMinimalReshapes(src, dst);
  VLOG(2) << "reshapes: " << absl::StrJoin(reshapes, ", ");
  RETURN_IF_ERROR(IsSupportedReshape(reshapes));

  SmallVector<DimTile> target_dim_tiles;
  target_dim_tiles.reserve(dst.dimensions().size());
  const TilingSpace& tiling_space = tile.tiling_space();
  mlir::MLIRContext* mlir_context = tiling_space.mlir_context();
  for (int64_t dim_size : dst.dimensions()) {
    target_dim_tiles.push_back(GetFullDimTile(dim_size, mlir_context));
  }
  for (const auto& minimal_reshape : reshapes) {
    RETURN_IF_ERROR(PropagateTileThroughMinimalReshape(
        mlir_context, minimal_reshape, src, dst, tile, target_dim_tiles));
  }
  return {Tile(tiling_space, std::move(target_dim_tiles))};
}

absl::StatusOr<Tiles> PropagateTileToInputForReshapeOp(
    const HloInstruction& hlo, const Tile& output_tile) {
  const Shape& input_shape = hlo.operand(0)->shape();
  const Shape& output_shape = hlo.shape();
  ASSIGN_OR_RETURN(
      auto input_tile,
      PropagateTileThroughReshape(output_tile, output_shape, input_shape));
  return Tiles{std::move(input_tile)};
}

absl::StatusOr<Tiles> PropagateTileToOutputForReshapeOp(
    const HloInstruction& hlo, const Tile& input_tile) {
  const Shape& input_shape = hlo.operand(0)->shape();
  const Shape& output_shape = hlo.shape();
  ASSIGN_OR_RETURN(
      auto output_tile,
      PropagateTileThroughReshape(input_tile, input_shape, output_shape));
  return Tiles{std::move(output_tile)};
}

absl::StatusOr<Tile> PropagateTileForBitcastOp(const Tile& tile,
                                               const Shape& src,
                                               const Shape& dst) {
  if (!ShapeUtil::IsDecomposableBitcast(src, dst)) {
    return absl::InvalidArgumentError("Bitcast is not decomposable.");
  }
  // Bitcast is transpose.
  if (!src.dimensions().empty()) {
    if (std::optional<std::vector<int64_t>> transpose_dims =
            ShapeUtil::DeduceTransposeDimensionsForBitcast(src, dst)) {
      return PropagateTileThroughTransposeOp(tile, *transpose_dims);
    }
  }
  // Bitcast is reshape.
  if (ShapeUtil::ReshapeIsBitcast(src, dst, /*ignore_element_type=*/true)) {
    return PropagateTileThroughReshape(tile, src, dst);
  }
  // Bitcast is transpose-reshape-transpose.
  auto maybe_trt = ShapeUtil::DecomposeBitcastToTrt(src, dst);
  if (!maybe_trt.has_value()) {
    return absl::InvalidArgumentError("Bitcast is not decomposable to TRT.");
  }
  const ShapeUtil::BitcastDecompositionTrt& trt = maybe_trt.value();
  Tile transpose1_tile =
      PropagateTileThroughTransposeOp(tile, trt.transpose1_dims);
  ASSIGN_OR_RETURN(auto reshape_tile, PropagateTileThroughReshape(
                                          transpose1_tile, trt.transpose1_shape,
                                          trt.reshape_shape));
  return PropagateTileThroughTransposeOp(reshape_tile, trt.transpose2_dims);
}

absl::StatusOr<Tiles> PropagateTileToInputForBitcastOp(
    const HloInstruction& hlo, const Tile& output_tile) {
  const Shape& input_shape = hlo.operand(0)->shape();
  const Shape& output_shape = hlo.shape();
  ASSIGN_OR_RETURN(
      auto input_tile,
      PropagateTileForBitcastOp(output_tile, output_shape, input_shape));
  return Tiles{std::move(input_tile)};
}

absl::StatusOr<Tiles> PropagateTileToOutputForBitcastOp(
    const HloInstruction& hlo, const Tile& input_tile) {
  const Shape& input_shape = hlo.operand(0)->shape();
  const Shape& output_shape = hlo.shape();
  ASSIGN_OR_RETURN(
      auto output_tile,
      PropagateTileForBitcastOp(input_tile, input_shape, output_shape));
  return Tiles{std::move(output_tile)};
}

}  // namespace

std::string ToString(const Tiles& tiles) {
  std::stringstream ss;
  for (const auto& [index, tile] : llvm::enumerate(tiles)) {
    ss << index << ") " << tile.ToString() << "\n";
  }
  return ss.str();
}

absl::StatusOr<Tiles> PropagateTileToInput(const TilingSpace& tiling_space,
                                           const HloInstruction& hlo,
                                           const Tile& output_tile,
                                           int64_t output_index) {
  VLOG(1) << "PropagateTileToInput:\n"
          << "  hlo: " << hlo.ToString() << "\n"
          << "  output_tile: " << output_tile.ToString() << "\n"
          << "  output_index: " << output_index;
  VLOG(2) << "tiling_space: " << tiling_space.ToString();
  if (HloInstruction::IsOpElementwise(hlo.opcode()) ||
      // For a single device, all-reduce is an elementwise op.
      HloPredicateIsOp<HloOpcode::kAllReduceStart, HloOpcode::kAllReduceDone,
                       HloOpcode::kMap>(&hlo)) {
    return {PropagateTileToInputForCwiseOp(hlo, output_tile)};
  }
  if (hlo.opcode() == HloOpcode::kBitcast) {
    return PropagateTileToInputForBitcastOp(hlo, output_tile);
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
    return PropagateTileToInputForDotOp(tiling_space, hlo, output_tile);
  }
  if (hlo.opcode() == HloOpcode::kScaledDot) {
    return PropagateTileToInputForScaledDotOp(tiling_space, hlo, output_tile);
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
  if (hlo.opcode() == HloOpcode::kReshape) {
    return PropagateTileToInputForReshapeOp(hlo, output_tile);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Output to input tile propagation not implemented for ", hlo.opcode()));
}

absl::StatusOr<Tiles> PropagateTileToOutput(const TilingSpace& tiling_space,
                                            const HloInstruction& hlo,
                                            const Tile& input_tile,
                                            int64_t input_index) {
  VLOG(1) << "PropagateTileToOutput:\n"
          << "  hlo: " << hlo.ToString() << "\n"
          << "  input_tile: " << input_tile.ToString() << "\n"
          << "  input_index: " << input_index;
  VLOG(2) << "tiling_space: " << tiling_space.ToString();
  if (HloInstruction::IsOpElementwise(hlo.opcode()) ||
      hlo.opcode() == HloOpcode::kMap) {
    return PropagateTileToOutputForCwiseOp(hlo, input_tile);
  }
  if (hlo.opcode() == HloOpcode::kBitcast) {
    return PropagateTileToOutputForBitcastOp(hlo, input_tile);
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
      return absl::InvalidArgumentError(
          "Input to output tile propagation not implemented from reduction "
          "init operands.");
    }
    return PropagateTileToOutputForReduceOp(reduce, input_tile);
  }
  if (hlo.opcode() == HloOpcode::kConcatenate) {
    return PropagateTileToOutputForConcatenateOp(
        *Cast<HloConcatenateInstruction>(&hlo), input_tile, input_index);
  }
  if (hlo.opcode() == HloOpcode::kReshape) {
    return PropagateTileToOutputForReshapeOp(hlo, input_tile);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Input to output tile propagation not implemented for ", hlo.opcode()));
}

}  // namespace xla::gpu::experimental
