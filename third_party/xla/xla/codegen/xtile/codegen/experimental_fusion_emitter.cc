/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/codegen/xtile/codegen/experimental_fusion_emitter.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/tiling/experimental/scheduling.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/xtile/codegen/dot_algorithms.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::xtile {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::FunctionOpInterface;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::Type;
using ::mlir::Value;
using ::stream_executor::GpuComputeCapability;

namespace arith = ::mlir::arith;
namespace stablehlo = ::mlir::stablehlo;
namespace ge = ::xla::gpu::experimental;

absl::StatusOr<std::vector<TensorValue>> EmitTiledComputation(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction::Region& region,
    absl::Span<const ge::TiledHloInstruction* const> roots);

Value MakeIndex(mlir::ImplicitLocOpBuilder& b, int64_t value) {
  return arith::ConstantIndexOp::create(b, value);
}

TensorValue Iota(mlir::ImplicitLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return stablehlo::IotaOp::create(b, type, /*iota_dimension=*/0);
}

template <typename T>
ArrayRef<T> MakeArrayRef(const absl::Span<const T> span) {
  return ArrayRef(span.data(), span.size());
}

absl::StatusOr<TensorValue> EmitBroadcast(
    mlir::ImplicitLocOpBuilder& b,
    const ge::TiledHloInstruction& tiled_broadcast, TensorValue input) {
  ASSIGN_OR_RETURN(SmallVector<int64_t> input_tile_shape,
                   tiled_broadcast.operand(0)->tile().GetStaticTileSizes());
  ASSIGN_OR_RETURN(SmallVector<int64_t> output_tile_shape,
                   tiled_broadcast.tile().GetStaticTileSizes());
  if (input_tile_shape.empty() && output_tile_shape.empty()) {
    return input;
  }
  CHECK(!output_tile_shape.empty());

  return xtile::BroadcastInDims(
      b, input, output_tile_shape,
      MakeArrayRef(tiled_broadcast.hlo()->dimensions()));
}

absl::StatusOr<TensorValue> EmitConcatenate(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_concat) {
  auto& b = emitter_ctx.b();
  const HloConcatenateInstruction* hlo_concat =
      ::xla::Cast<HloConcatenateInstruction>(tiled_concat.hlo());
  const int64_t concatenate_dimension = hlo_concat->concatenate_dimension();

  TF_RET_CHECK(tiled_concat.operands().size() == tiled_concat.regions().size())
      << "Concatenate must have the same number of operands and regions";

  ASSIGN_OR_RETURN(SmallVector<int64_t> tile_sizes,
                   tiled_concat.tile().GetStaticTileSizes());
  int64_t concat_dim_tile_size = tile_sizes[concatenate_dimension];

  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(emitter_ctx, tiled_concat));
  TF_RETURN_IF_ERROR(
      CheckConcatenateOperands(*hlo_concat, concat_dim_tile_size));
  Type result_type =
      mlir::RankedTensorType::get(tile_sizes, tile_info.storage_type());

  // We will load and compute from a single operand, so we need to figure out
  // which one by looking at the offset within the concatenation dimension.
  Value concatenate_dimension_offset =
      tile_info.offsets()[concatenate_dimension];

  // It would have been nice to be able to use `scf::IndexSwitchOp`, but Triton
  // does not want to deal with the `Index` type, and does not support the op.
  // Instead, we generate a sequence of nested `scf::IfOp`s.
  SmallVector<mlir::scf::IfOp, 4> if_ops;
  int64_t limit = 0;
  for (const auto& [i, operand] : llvm::enumerate(tiled_concat.operands())) {
    // Write in the else branch of the previous if op if one exists.
    if (!if_ops.empty()) {
      b.setInsertionPointToStart(if_ops.back().elseBlock());
    }
    // Add an `if_op` if we have not reached the last operand. The last operand
    // directly populates the `else` block of the previous `if_op`.
    if (if_ops.size() < tiled_concat.operands().size() - 1) {
      limit += operand->hlo()->shape().dimensions()[concatenate_dimension];
      Value offset_limit = CreateConst(b, b.getIndexType(), limit);

      auto cond =
          arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                concatenate_dimension_offset, offset_limit);
      auto if_op =
          mlir::scf::IfOp::create(b, mlir::TypeRange(result_type), cond,
                                  /*withElseRegion=*/true);

      // Propagate the result from the nested `if_op` if we were already within
      // an `if_op`.
      if (!if_ops.empty()) {
        mlir::scf::YieldOp::create(b, if_op.getResult(0));
      }
      b.setInsertionPointToStart(if_op.thenBlock());
      if_ops.push_back(if_op);
    }
    const auto& region = tiled_concat.regions()[i];
    const ge::TiledHloInstruction* const region_root = region.back().get();
    ASSIGN_OR_RETURN(auto results,
                     EmitTiledComputation(emitter_ctx, region, {region_root}));
    mlir::scf::YieldOp::create(b, results.back());
  }
  b.setInsertionPointAfter(if_ops.front());
  return mlir::cast<TensorValue>(if_ops.front().getResult(0));
}

// Computes and applies a mask to the reduction dimension of the dot operand
// passed as a parameter.
//
// Note: we currently assume that contracting_dimension_tile_index is an i32
// scalar.
absl::StatusOr<TensorValue> MaskDotOperand(
    mlir::ImplicitLocOpBuilder& b, const ge::TiledHloInstruction& dot_operand,
    TensorValue dot_operand_value, Value contracting_dimension_tile_index,
    int contraction_dimension_index) {
  llvm::ArrayRef<int64_t> tile_shape = dot_operand_value.getType().getShape();

  int64_t contracting_dimension_size =
      dot_operand.hlo()->shape().dimensions(contraction_dimension_index);
  int64_t tile_size = tile_shape[contraction_dimension_index];

  if (contracting_dimension_size % tile_size == 0) {
    return dot_operand_value;
  }

  // Only mask out tiles that we know to go beyond boundaries of the
  // contracting dimension---i.e. tiles whose index exceeds the number of
  // full tiles (tiles without padding).
  Type result_type = dot_operand_value.getType();
  Value tile_size_value = CreateConst(b, b.getI32Type(), tile_size);
  Value num_full_tiles = arith::DivSIOp::create(
      b, CreateConst(b, b.getI32Type(), contracting_dimension_size),
      tile_size_value);
  // if tile_index >= num_full_tiles...
  auto cond =
      arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                            contracting_dimension_tile_index, num_full_tiles);
  auto if_op = mlir::scf::IfOp::create(b, mlir::TypeRange(result_type), cond,
                                       /*withElseRegion=*/true);
  // then ...
  {
    b.setInsertionPointToStart(if_op.thenBlock());
    // indices =
    //   contracting_dimension_tile_index * tile_size + range(0, tile_size)
    // mask = indices < contracting_dimension_size
    // operand = select(broadcast(mask, operand.shape), operand, 0)
    Value tile_offset = arith::MulIOp::create(
        b, contracting_dimension_tile_index, tile_size_value);
    TensorValue range = Iota(b, tile_size);
    TensorValue broadcasted_tile_offset =
        xtile::Splat(b, tile_offset, {tile_size});
    Value indices = arith::AddIOp::create(b, range, broadcasted_tile_offset);

    Value boundary =
        CreateConst(b, b.getI32Type(), contracting_dimension_size, {tile_size});

    Value mask =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, indices, boundary);

    mask = xtile::BroadcastInDims(b, mlir::cast<TensorValue>(mask), tile_shape,
                                  {contraction_dimension_index});
    ASSIGN_OR_RETURN(
        auto element_type,
        PrimitiveTypeToMlirType(b, dot_operand.hlo()->shape().element_type()));

    TensorValue zero = CreateConst(b, element_type, 0.0f, tile_shape);

    Value masked_dot_operand =
        arith::SelectOp::create(b, mask, dot_operand_value, zero);
    mlir::scf::YieldOp::create(b, masked_dot_operand);
  }
  // else ...
  {
    b.setInsertionPointToStart(if_op.elseBlock());
    mlir::scf::YieldOp::create(b, dot_operand_value);
  }
  b.setInsertionPointAfter(if_op);
  return mlir::cast<TensorValue>(if_op.getResult(0));
}

// Returns the number of sequential dimensions in the HLO.
int64_t GetNumSequentialDimIds(const HloInstruction& hlo) {
  if (HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kScaledDot>(&hlo)) {
    const DotDimensionNumbers& dim_numbers = hlo.dot_dimension_numbers();
    return dim_numbers.lhs_contracting_dimensions().size();
  }
  if (HloPredicateIsOp<HloOpcode::kReduce>(&hlo)) {
    const HloReduceInstruction& reduce =
        *::xla::Cast<HloReduceInstruction>(&hlo);
    return reduce.dimensions().size();
  }
  return 0;
}

SmallVector<int64_t> GetSequentialDimIds(const HloInstruction& hlo) {
  int64_t num_sequential_dims = GetNumSequentialDimIds(hlo);
  SmallVector<int64_t> sequential_dim_ids;
  int64_t output_rank = hlo.shape().dimensions().size();
  for (int64_t dim_id = output_rank, e = output_rank + num_sequential_dims;
       dim_id < e; ++dim_id) {
    sequential_dim_ids.push_back(dim_id);
  }
  return sequential_dim_ids;
}

// Returns the number of iterations of the loop over the contraction/reduction
// dimensions.
absl::StatusOr<SmallVector<int64_t>> GetSequentialLoopIterationCounts(
    const ge::TiledHloInstruction& tiled_hlo,
    ArrayRef<int64_t> sequential_dim_ids) {
  const HloInstruction& hlo = *tiled_hlo.hlo();
  if (sequential_dim_ids.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "No sequential dimensions found for the HLO", hlo.ToString()));
  }
  const ge::TilingSpace& tiling_space = tiled_hlo.tile().tiling_space();

  int64_t output_rank = hlo.shape().dimensions().size();
  SmallVector<int64_t> loop_iteration_counts;
  for (int64_t dim_id : sequential_dim_ids) {
    const ge::TilingSpace::DimensionInfo& dim_info =
        tiling_space.GetDimensionInfo(hlo, output_rank++);
    CHECK(dim_info.type == ge::TilingSpace::DimensionSemantics::kSequential)
        << "Expected a sequential dimension info for contracting dimension "
        << dim_id << " in op " << hlo.ToString();
    CHECK(dim_info.IsTileSizeSet())
        << "Tile size is not set for contracting dimension ";
    loop_iteration_counts.push_back(
        CeilOfRatio(dim_info.dimension_size, dim_info.tile_size));
  }
  return loop_iteration_counts;
}

// Emits dot instruction that has LHS and RHS as part of its region.
// Tiling analysis identifies instructions that belong to the dot and puts them
// inside of the dot's regions.
//
// To emit it we create a loop over the contracting dimension and emit the
// region of the dot inside:
//
// acc = [tile_m, tile_n] 0.0f
// for (k = 0 .. size_k / tile_k) {
//   <contents of the region, including lhs and rhs>
//   acc += dot(lhs, rhs)
// }
// c = acc
absl::StatusOr<TensorValue> EmitDot(EmitterContext& emitter_ctx,
                                    const ge::TiledHloInstruction& tiled_dot) {
  TF_RET_CHECK(tiled_dot.regions().size() == 1);
  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_dot.tile().GetStaticTileSizes());

  SmallVector<int64_t, 2> padded_tile_sizes_no_unit_dims =
      CollapseUnitDims(padded_tile_sizes, padded_tile_sizes).first;

  // Sanity check: Triton historically did not support non-2D dots (and still
  // doesn't support arbitrary nD dots), so we require that the dot is tiled
  // with exactly two non-unit tile sizes. This anyway matches the hardware's
  // expectations, so seems like a reasonable requirement.
  // TODO(b/393299275): this needs to be enforced in tiling.
  if (padded_tile_sizes_no_unit_dims.size() != 2) {
    return absl::FailedPreconditionError(
        "Expected dot to be tiled with exactly two non-unit tile sizes");
  }
  auto& b = emitter_ctx.b();
  const auto& dot = *::xla::Cast<HloDotInstruction>(tiled_dot.hlo());
  // The specific accumulator type to use may not correspond to the output type
  // of the dot. In particular, that is the case when an algorithm is specified
  // and the dot's output type does not match its expectations.
  ASSIGN_OR_RETURN(Type accumulator_type, xtile::GetDotAccumulatorType(b, dot));
  TensorValue accumulator =
      CreateConst(b, accumulator_type, 0.0f, padded_tile_sizes_no_unit_dims);

  SmallVector<int64_t> sequential_dim_ids =
      GetSequentialDimIds(*tiled_dot.hlo());
  ASSIGN_OR_RETURN(
      SmallVector<int64_t> loop_iteration_count,
      GetSequentialLoopIterationCounts(tiled_dot, sequential_dim_ids));
  CHECK(loop_iteration_count.size() == 1)
      << "Expected exactly one loop iteration count for dot";

  auto for_op = mlir::scf::ForOp::create(
      b,
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_count.front()),
      /*step=*/MakeIndex(b, 1), accumulator);

  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value iv = for_op.getInductionVar();
    Value iv_i32 = Cast(b, for_op.getInductionVar(), b.getI32Type());
    CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(
        sequential_dim_ids.front(), iv));

    // Emit the dot region.
    const ge::TiledHloInstruction* lhs_operand = tiled_dot.operand(0);
    const ge::TiledHloInstruction* rhs_operand = tiled_dot.operand(1);
    ASSIGN_OR_RETURN(auto results, EmitTiledComputation(
                                       emitter_ctx, tiled_dot.regions().front(),
                                       {lhs_operand, rhs_operand}));

    // Canonicalize LHS to match Triton's expectations.
    TensorValue lhs_tensor = results[0];
    int64_t lhs_contracting_dim_idx =
        dot.dot_dimension_numbers().lhs_contracting_dimensions(0);
    ASSIGN_OR_RETURN(lhs_tensor,
                     MaskDotOperand(b, *lhs_operand, lhs_tensor, iv_i32,
                                    lhs_contracting_dim_idx));
    ASSIGN_OR_RETURN(lhs_tensor, CanonicalizeDotOperand(b, lhs_tensor,
                                                        lhs_contracting_dim_idx,
                                                        DotOperandSide::kLhs));

    // Canonicalize RHS to match Triton's expectations.
    TensorValue rhs_tensor = results[1];
    int64_t rhs_contracting_dim_idx =
        dot.dot_dimension_numbers().rhs_contracting_dimensions(0);
    ASSIGN_OR_RETURN(rhs_tensor,
                     MaskDotOperand(b, *rhs_operand, rhs_tensor, iv_i32,
                                    rhs_contracting_dim_idx));
    ASSIGN_OR_RETURN(rhs_tensor, CanonicalizeDotOperand(b, rhs_tensor,
                                                        rhs_contracting_dim_idx,
                                                        DotOperandSide::kRhs));

    // Emit the partial dot.
    Value acc = for_op.getRegionIterArgs().front();
    ASSIGN_OR_RETURN(
        Value acc_next,
        xtile::EmitSingleTileDot(
            b, dot, xtile::DotOperands{lhs_tensor, rhs_tensor, acc}));
    mlir::scf::YieldOp::create(b, acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  ASSIGN_OR_RETURN(Type dot_output_type,
                   PrimitiveTypeToMlirType(b, dot.shape().element_type()));

  Value result = for_op.getResult(0);
  if (dot_output_type != accumulator_type) {
    result = Cast(b, result, dot_output_type);
  }
  auto tensor_result = mlir::cast<TensorValue>(result);
  if (padded_tile_sizes.size() != padded_tile_sizes_no_unit_dims.size()) {
    return EmitTiledReshape(b, padded_tile_sizes, tensor_result);
  }
  return tensor_result;
}

absl::StatusOr<TensorValue> EmitIota(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_iota) {
  auto& b = emitter_ctx.b();
  const HloIotaInstruction* hlo_iota =
      ::xla::Cast<HloIotaInstruction>(tiled_iota.hlo());
  int64_t iota_dim = hlo_iota->iota_dimension();

  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_iota.tile().GetStaticTileSizes());

  // We can treat iota more or less as a parameter load, except that we need to
  // generate the right values in the right place as opposed to loading them.
  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(emitter_ctx, tiled_iota));

  // First, stride as needed between the iota components.
  Value range = arith::MulIOp::create(
      b, Iota(b, padded_tile_sizes[iota_dim]),
      xtile::Splat(
          b, CreateConst(b, b.getI32Type(), tile_info.tile_strides()[iota_dim]),
          padded_tile_sizes[iota_dim]));

  // Cast the offset to the iota dimension to i32, because
  // stable_hlo.broadcast_in_dims does not support index type.
  auto iota_dim_offset = Cast(b, tile_info.offsets()[iota_dim], b.getI32Type());
  // Then, add the base offset to the iota components.
  range = arith::AddIOp::create(
      b, range, xtile::Splat(b, iota_dim_offset, padded_tile_sizes[iota_dim]));

  // Cast the result to the targeted type.
  range = Cast(b, range, tile_info.storage_type());

  // And finally, produce a broadcast along the non-iota dimensions in order to
  // produce the whole iota tile.
  return xtile::BroadcastInDims(b, mlir::cast<TensorValue>(range),
                                padded_tile_sizes,
                                /*dims=*/{iota_dim});
}

TensorValue EmitTranspose(mlir::ImplicitLocOpBuilder& b,
                          ArrayRef<int64_t> tile_sizes,
                          ArrayRef<int64_t> dimensions, TensorValue input) {
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  Type input_element_type = input.getType().getElementType();
  Type output_tensor_type =
      mlir::RankedTensorType::get(padded_tile_sizes, input_element_type);

  mlir::DenseI64ArrayAttr order = b.getDenseI64ArrayAttr(dimensions);
  return ::mlir::stablehlo::TransposeOp::create(b, output_tensor_type, input,
                                                order);
}

absl::StatusOr<TensorValue> EmitPad(EmitterContext& emitter_ctx,
                                    const ge::TiledHloInstruction& tiled_pad) {
  auto& b = emitter_ctx.b();
  ASSIGN_OR_RETURN(SmallVector<int64_t> tile_sizes,
                   tiled_pad.tile().GetStaticTileSizes());

  const ge::TiledHloInstruction* tiled_operand = tiled_pad.operand(0);
  const auto& pad_input_shape = tiled_operand->hlo()->shape().dimensions();

  // Compute tile offsets.
  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(emitter_ctx, tiled_pad));
  SmallVector<Value, 3> tile_offsets = tile_info.offsets();

  // Compute mask.
  Type i32_type = b.getI32Type();
  Value mask;
  for (auto [dim_index, sizes] : llvm::enumerate(
           llvm::zip(pad_input_shape, tile_sizes, tile_offsets,
                     tiled_pad.hlo()->padding_config().dimensions()))) {
    auto [pad_input_dim_size, pad_output_dim_size, tile_offset, dim_config] =
        sizes;
    if (dim_config.edge_padding_low() != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Low padding is not supported but got edge_padding_low: ",
          dim_config.edge_padding_low()));
    }
    if (dim_config.interior_padding() != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Interior padding is not supported but got interior_padding: ",
          dim_config.interior_padding()));
    }

    if (pad_input_dim_size == pad_output_dim_size) {
      continue;
    }

    // LHS for the compare is an iota broadcasted to the output shape.
    TensorValue range = Iota(b, pad_output_dim_size);
    TensorValue bcast = xtile::BroadcastInDims(
        b, range, tile_sizes, {static_cast<int64_t>(dim_index)});

    // RHS for the compare is splat(pad_input_dim_size - tile_offset).
    Value tile_offset_i32 = Cast(b, tile_offset, i32_type);
    Value threshold = arith::SubIOp::create(
        b, CreateConst(b, i32_type, pad_input_dim_size), tile_offset_i32);
    TensorValue threshold_splat = xtile::Splat(b, threshold, tile_sizes);
    Value cmp = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, bcast,
                                      threshold_splat);
    mask = mask ? stablehlo::AndOp::create(b, mask, cmp) : cmp;
  }
  if (!mask) {
    return emitter_ctx.TiledHloToTensorValue(*tiled_operand);
  }
  const ge::TiledHloInstruction* padding_value = tiled_pad.operand(1);

  TensorValue pad_value_splat = xtile::Splat(
      b, emitter_ctx.TiledHloToTensorValue(*padding_value), tile_sizes);
  return mlir::cast<TensorValue>(
      arith::SelectOp::create(b, mask,
                              emitter_ctx.TiledHloToTensorValue(*tiled_operand),
                              pad_value_splat)
          .getResult());
}

absl::StatusOr<TensorValue> EmitTiledHloInstruction(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_hlo) {
  auto& b = emitter_ctx.b();
  const HloInstruction* hlo = tiled_hlo.hlo();
  VLOG(4) << "EmitTiledHloInstruction: " << hlo->ToString();

  const HloFusionInstruction& fusion = emitter_ctx.fusion();
  if (hlo->opcode() == HloOpcode::kParameter && !fusion.IsUserOf(hlo)) {
    hlo = hlo->parent()->FusionInstruction()->operand(hlo->parameter_number());
  }

  if (fusion.IsUserOf(hlo)) {
    int64_t arg_index = fusion.operand_index(hlo);
    // Walk up the parameter chain to find the outermost operand index.
    while (auto* instr = hlo->parent()->FusionInstruction()) {
      arg_index = hlo->parameter_number();  // Nested operands are parameters.
      hlo = instr->operand(arg_index);
    }
    ASSIGN_OR_RETURN(TileInfo tile_info,
                     TileInfo::Construct(emitter_ctx, tiled_hlo));
    TensorValue parameter = EmitParameterExtract(
        b, tile_info, emitter_ctx.entry_func().getArgument(arg_index));

    // Workaround(i1_to_i8_workaround)
    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to type checking that we perform a conversion after
    // loading if the type of the loaded parameter does not match what is
    // expected.
    Type loaded_element_type = getElementTypeOrSelf(parameter.getType());
    ASSIGN_OR_RETURN(Type expected_element_type,
                     PrimitiveTypeToMlirType(b, hlo->shape().element_type()));

    if (expected_element_type != loaded_element_type) {
      // Ensure that we didn't mess up somewhere else by checking that we
      // indeed loaded the expected storage type for the expected element type.
      if (loaded_element_type != StorageType(expected_element_type)) {
        return absl::InternalError(absl::StrCat(
            "Parameters were loaded with an unexpected element type "
            "while lowering ",
            fusion.called_computation()->ToString()));
      }
      parameter =
          mlir::cast<TensorValue>(Cast(b, parameter, expected_element_type));
    }
    return parameter;
  }
  if (hlo->opcode() == HloOpcode::kDot) {
    return EmitDot(emitter_ctx, tiled_hlo);
  }
  if (hlo->opcode() == HloOpcode::kConcatenate) {
    return EmitConcatenate(emitter_ctx, tiled_hlo);
  }
  std::vector<Value> operands;
  operands.reserve(hlo->operands().size());
  for (const ge::TiledHloInstruction* operand : tiled_hlo.operands()) {
    operands.push_back(emitter_ctx.TiledHloToTensorValue(*operand));
  }
  switch (hlo->opcode()) {
    case HloOpcode::kTranspose: {
      ASSIGN_OR_RETURN(auto static_tile_sizes,
                       tiled_hlo.tile().GetStaticTileSizes());
      auto padded_tile_sizes = GetPaddedTileSizes(static_tile_sizes);
      return EmitTranspose(b, padded_tile_sizes, hlo->dimensions(),
                           mlir::cast<TensorValue>(operands[0]));
    }
    case HloOpcode::kBroadcast: {
      return EmitBroadcast(b, tiled_hlo, mlir::cast<TensorValue>(operands[0]));
    }
    case HloOpcode::kConstant: {
      if (ShapeUtil::IsEffectiveScalar(hlo->shape())) {
        return EmitConstant(b, *hlo);
      }
      return absl::UnimplementedError(
          absl::StrCat("Unsupported non-scalar constant ", hlo->ToString()));
    }
    case HloOpcode::kDynamicSlice: {
      return emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0));
    }
    case HloOpcode::kIota: {
      return EmitIota(emitter_ctx, tiled_hlo);
    }
    case HloOpcode::kSlice: {
      return emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0));
    }
    case HloOpcode::kPad: {
      return EmitPad(emitter_ctx, tiled_hlo);
    }
    default:
      break;
  }
  if (hlo->IsElementwise()) {
    ASSIGN_OR_RETURN(Value result, EmitElementwise(b, *hlo, operands));
    return mlir::cast<TensorValue>(result);
  }
  return absl::UnimplementedError(
      absl::StrCat("Unsupported operation ", hlo->ToString()));
}

absl::StatusOr<std::vector<TensorValue>> EmitTiledComputation(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction::Region& region,
    absl::Span<const ge::TiledHloInstruction* const> roots) {
  for (const auto& tiled_hlo : region) {
    const HloInstruction* hlo = tiled_hlo->hlo();
    VLOG(8) << "Emitting " << hlo->ToString(HloPrintOptions::ShortParsable());
    ASSIGN_OR_RETURN(TensorValue result,
                     EmitTiledHloInstruction(emitter_ctx, *tiled_hlo));
    TF_RET_CHECK(emitter_ctx.MapTiledHloToTensorValue(tiled_hlo.get(), result))
        << hlo->ToString();
  }
  std::vector<TensorValue> results;
  results.reserve(roots.size());
  for (const auto* root : roots) {
    results.push_back(emitter_ctx.TiledHloToTensorValue(*root));
  }
  VLOG(8) << "Emitted computation";
  return std::move(results);
}

absl::Status EmitGeneric(ImplicitLocOpBuilder& b,
                         const HloFusionInstruction* fusion,
                         const ge::TiledHloComputation& tiled_computation,
                         const ::xla::IndexingMap& schedule,
                         xtile::EntryFuncOp fn, MLIRContext* mlir_context) {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Emitting XTile IR for fusion\n"
            << ExtractInstructionIntoNewModule(*fusion)->ToString();
    VLOG(6) << "Tiled computation: \n" << tiled_computation.ToString();
  }
  Value tile_id = fn.getTileId();
  EmitterContext emitter_ctx{b,        fusion, tile_id,
                             schedule, fn,     tiled_computation};

  VLOG(2) << "EmitTiledComputation: " << tiled_computation.ToString();
  ASSIGN_OR_RETURN(auto results,
                   EmitTiledComputation(
                       emitter_ctx, tiled_computation.tiled_hlo_instructions(),
                       tiled_computation.roots()));
  const HloComputation* computation = fusion->fused_instructions_computation();
  for (const auto& [root, result, arg] :
       llvm::zip(tiled_computation.roots(), results,
                 fn.getArguments().drop_front(computation->num_parameters()))) {
    // Workaround(i1_to_i8_workaround)
    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to check converted types before storing if the type
    // of the result does not match the type of the output pointer.
    Type result_element_type = getElementTypeOrSelf(result.getType());
    Type result_storage_type = StorageType(result_element_type);

    if (result_element_type != result_storage_type) {
      result = mlir::cast<TensorValue>(Cast(b, result, result_storage_type));
    }

    ASSIGN_OR_RETURN(auto tile_info, TileInfo::Construct(emitter_ctx, *root));

    xtile::InsertTileOp::create(b, result, arg, tile_info.offsets(),
                                tile_info.padded_tile_sizes(),
                                tile_info.tile_strides());
  }

  return absl::OkStatus();
}

}  // namespace

// TODO(b/447133106): Contrary to the name, this function still does a lot of
// triton specific things. It should be migrated to use non-triton specific
// utilities.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitXTileModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const ::xla::gpu::experimental::TiledHloComputation& tiled_computation,
    MLIRContext& mlir_context, absl::Span<mlir::Type> opaque_args_types,
    const std::optional<GpuComputeCapability>& gpu_cc) {
  const HloComputation* hlo_computation =
      fusion->fused_instructions_computation();

  Location loc = mlir::NameLoc::get(
      mlir::StringAttr::get(&mlir_context, hlo_computation->name()));
  ImplicitLocOpBuilder b(loc, &mlir_context);

  mlir::OwningOpRef<mlir::ModuleOp> xtile_module =
      llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(xtile_module->getBody());

  // Compute function argument types.
  ASSIGN_OR_RETURN(SmallVector<Type> fn_arg_types,
                   GetFnArgTypes(b, fusion, opaque_args_types, gpu_cc));
  // Metadata arguments are opaque to the tiling infra.
  llvm::SmallVector<mlir::NamedAttribute> named_attributes{b.getNamedAttr(
      "num_opaque_args", b.getI32IntegerAttr(opaque_args_types.size()))};

  auto fn = xtile::EntryFuncOp::create(b, fn_name, fn_arg_types,
                                       named_attributes, {});
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  ASSIGN_OR_RETURN(auto schedule, Schedule(tiled_computation));
  TF_RETURN_IF_ERROR(
      EmitGeneric(b, fusion, tiled_computation, schedule, fn, &mlir_context));

  b.create<xtile::EntryFuncReturnOp>();

  // This should be enabled only in debug mode probably.
  {
    // Verify that the emitted module contains only ops from dialects that can
    // be shared between backends.
    mlir::PassManager pm(&mlir_context);
    pm.addPass(xtile::createVerifyLegalXTileOpsPass());
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    TF_RETURN_IF_ERROR(diagnostic_handler.consumeStatus(pm.run(*xtile_module)));
  }
  return xtile_module;
}

}  // namespace xla::xtile
