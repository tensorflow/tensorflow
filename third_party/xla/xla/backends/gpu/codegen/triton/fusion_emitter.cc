/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_fusion_instruction.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/tiling/tiled_hlo_schedule.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace arith = ::mlir::arith;
namespace stablehlo = ::mlir::stablehlo;

using ::llvm::SmallVector;
using ::mlir::AffineMap;
using ::mlir::ArrayRef;
using ::mlir::MLIRContext;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

using ::xla::xtile::Cast;
using ::xla::xtile::CreateConst;
using ::xla::xtile::EmitConstant;
using ::xla::xtile::EmitElementwise;
using ::xla::xtile::EmitScope;
using ::xla::xtile::GetPaddedTileSizes;
using ::xla::xtile::PrimitiveTypeToMlirType;
using ::xla::xtile::StorageType;
using ::xla::xtile::TensorValue;
using ::xla::xtile::TileInfo;

namespace {

Value MakeIndex(mlir::ImplicitLocOpBuilder& b, int64_t value) {
  return arith::ConstantIndexOp::create(b, value);
}

TensorValue Iota(mlir::ImplicitLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return stablehlo::IotaOp::create(b, type, /*iota_dimension=*/0);
}

absl::Status EmitReduceComputation(mlir::ImplicitLocOpBuilder& b,
                                   const HloInstruction* hlo_reduction,
                                   const HloComputation* reduction_computation,
                                   mlir::Operation* reduction) {
  TF_ASSIGN_OR_RETURN(
      Type result_ty,
      PrimitiveTypeToMlirType(b, hlo_reduction->shape().element_type()));
  result_ty = mlir::RankedTensorType::get({}, result_ty);

  mlir::Location loc = b.getLoc();
  mlir::Block* reducer = b.createBlock(&reduction->getRegion(0), {},
                                       {result_ty, result_ty}, {loc, loc});
  b.setInsertionPointToStart(reducer);

  std::vector<const HloInstruction*> to_emit;
  absl::flat_hash_map<const HloInstruction*, TensorValue> region_values;
  for (const HloInstruction* instr :
       reduction_computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      int parameter_number = instr->parameter_number();
      TF_RET_CHECK(parameter_number < 2);
      auto argument = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
          reducer->getArgument(parameter_number));

      if (!argument) {
        return Internal("Expected reducer argument to be a tensor.");
      }

      TF_RET_CHECK(region_values.insert({instr, argument}).second);
    } else {
      to_emit.push_back(instr);
    }
  }

  TF_RET_CHECK(!to_emit.empty());

  TF_ASSIGN_OR_RETURN(TensorValue result, EmitScope(b, to_emit, region_values));
  stablehlo::ReturnOp::create(b, SmallVector<Value>({result}));
  b.setInsertionPointAfter(reduction);
  return absl::OkStatus();
}

absl::StatusOr<TensorValue> EmitReduce(
    mlir::ImplicitLocOpBuilder& b, const TiledHloInstruction& tiled_hlo_reduce,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  // At the moment, we should only emit a full reduction over a single
  // dimension using a scalar as a neutral element.
  const HloReduceInstruction& hlo_reduce =
      *::xla::Cast<HloReduceInstruction>(tiled_hlo_reduce.hlo());
  TensorValue input = values[tiled_hlo_reduce.operand(0)];

  // Since every shape is padded to a power of 2 in Triton, the input tile may
  // be padded with arbitrary values. These values could affect the result of
  // the reduction, so we need to mask them away. Luckily, we have a monoid
  // structure (element_type, hlo_reduce.to_apply(), hlo_reduce.operand(1))---
  // up to floating-point inaccuracies. Masking the input using
  // hlo_reduce.operand(1) is thus always the right choice to ensure that the
  // reduction is computed correctly, since it is the neutral value with
  // regards to the reducer.

  absl::Span<const int64_t> unpadded_tile_sizes =
      tiled_hlo_reduce.operand(0)->tile_sizes();
  llvm::SmallVector<int64_t> mask_dim_bounds;
  mask_dim_bounds.reserve(unpadded_tile_sizes.size());
  for (auto [idx, dim_size] : llvm::enumerate(unpadded_tile_sizes)) {
    if (absl::c_contains(hlo_reduce.dimensions(), idx)) {
      // We only need to mask the reduction dimensions.
      mask_dim_bounds.push_back(dim_size);
    } else {
      mask_dim_bounds.push_back(input.getType().getDimSize(idx));
    }
  }
  mlir::Value neutral_value =
      mlir::tensor::ExtractOp::create(b, values[tiled_hlo_reduce.operand(1)]);
  // Use createOrFold as the mask may be be reduntant, in which case it will be
  // folded away.
  input = mlir::cast<TensorValue>(
      b.createOrFold<xtile::MaskOp>(input, mask_dim_bounds, neutral_value));

  Value init_value = values[tiled_hlo_reduce.operand(1)];

  stablehlo::ReduceOp reduction = stablehlo::ReduceOp::create(
      b, input, init_value, hlo_reduce.dimensions());
  TF_RETURN_IF_ERROR(
      EmitReduceComputation(b, &hlo_reduce, hlo_reduce.to_apply(), reduction));

  return mlir::cast<TensorValue>(reduction.getResult(0));
}

template <typename T>
ArrayRef<T> MakeArrayRef(const absl::Span<const T> span) {
  return ArrayRef(span.data(), span.size());
}

TensorValue EmitTiledBroadcast(
    mlir::ImplicitLocOpBuilder& b, const TiledHloInstruction& tiled_broadcast,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  const SmallVector<int64_t>& input_tile_shape =
      tiled_broadcast.operand(0)->tile_sizes();
  const SmallVector<int64_t>& output_tile_shape = tiled_broadcast.tile_sizes();

  if (input_tile_shape.empty() && output_tile_shape.empty()) {
    return values[tiled_broadcast.operand(0)];
  }
  CHECK(!output_tile_shape.empty());

  SmallVector<int64_t> padded_output_tile_shape =
      GetPaddedTileSizes(output_tile_shape);

  TensorValue input = values[tiled_broadcast.operand(0)];
  return xtile::BroadcastInDims(
      b, input, padded_output_tile_shape,
      MakeArrayRef(tiled_broadcast.hlo()->dimensions()));
}

absl::StatusOr<TensorValue> EmitTiledIota(
    mlir::ImplicitLocOpBuilder& b, Value pid,
    const TiledHloInstruction& tiled_iota) {
  const HloIotaInstruction* hlo_iota =
      ::xla::Cast<HloIotaInstruction>(tiled_iota.hlo());
  int64_t iota_dim = hlo_iota->iota_dimension();

  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_iota.tile_sizes());

  // We can treat iota more or less as a parameter load, except that we need to
  // generate the right values in the right place as opposed to loading them.
  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_iota.tile_offsets_indexing());

  auto iota_dim_offset =
      Cast(b,
           emitters::ApplyIndexing(tile_offsets_indexing, /*dims=*/pid,
                                   /*symbols=*/{}, b)[iota_dim],
           b.getI32Type());

  // First, stride as needed between the iota components.
  Value range = arith::MulIOp::create(
      b, Iota(b, padded_tile_sizes[iota_dim]),
      xtile::Splat(
          b,
          CreateConst(b, b.getI32Type(), tiled_iota.tile_strides()[iota_dim]),
          padded_tile_sizes[iota_dim]));

  // Then, add the base offset to the iota components.
  range = arith::AddIOp::create(
      b, range, xtile::Splat(b, iota_dim_offset, padded_tile_sizes[iota_dim]));

  // Cast the result to the targeted type.
  TF_ASSIGN_OR_RETURN(
      Type iota_element_type,
      PrimitiveTypeToMlirType(b, hlo_iota->shape().element_type()));

  range = Cast(b, range, iota_element_type);

  // And finally, produce a broadcast along the non-iota dimensions in order to
  // produce the whole iota tile.
  return xtile::BroadcastInDims(b, mlir::cast<TensorValue>(range),
                                padded_tile_sizes,
                                /*dims=*/{iota_dim});
}

SmallVector<Value> GetRuntimeValues(
    const TiledHloInstruction& tiled_hlo,
    const absl::flat_hash_map<const TiledHloInstruction*, TensorValue>&
        values) {
  SmallVector<Value> runtime_values;
  if (!tiled_hlo.runtime_variables().empty()) {
    for (const TiledHloInstruction* rt : tiled_hlo.runtime_variables()) {
      CHECK(values.contains(rt))
          << absl::StrCat(" runtime variable ", rt->ToString(), " not found");
      TensorValue value = values.at(rt);
      mlir::OpBuilder builder(value.getContext());
      builder.setInsertionPointAfterValue(value);
      runtime_values.push_back(
          mlir::tensor::ExtractOp::create(builder, value.getLoc(), value));
    }
  }
  return runtime_values;
}

absl::StatusOr<TensorValue> EmitTiledReshape(mlir::ImplicitLocOpBuilder& b,
                                             ArrayRef<int64_t> tile_sizes,
                                             TensorValue input) {
  mlir::RankedTensorType input_type = input.getType();
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  // At this point we know that neither the input nor the output are 0D tensors.
  auto output_tensor_type = mlir::RankedTensorType::get(
      padded_tile_sizes, input_type.getElementType());

  if (input_type.getNumElements() != output_tensor_type.getNumElements()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Reshape input and output shapes must be the same, got ",
                     absl::StrJoin(input_type.getShape(), "x"), " -> ",
                     absl::StrJoin(output_tensor_type.getShape(), "x")));
  }

  return stablehlo::ReshapeOp::create(b, output_tensor_type, input);
}

TensorValue EmitTiledTranspose(mlir::ImplicitLocOpBuilder& b,
                               ArrayRef<int64_t> tile_sizes,
                               SmallVector<int64_t> dimensions,
                               TensorValue input) {
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  Type input_element_type = input.getType().getElementType();
  Type output_tensor_type =
      mlir::RankedTensorType::get(padded_tile_sizes, input_element_type);

  mlir::DenseI64ArrayAttr order = b.getDenseI64ArrayAttr(dimensions);

  return stablehlo::TransposeOp::create(b, output_tensor_type, input, order);
}

absl::StatusOr<TensorValue> EmitTiledBitcast(
    mlir::ImplicitLocOpBuilder& b, const TiledHloInstruction& tiled_bitcast,
    TensorValue input) {
  Shape input_shape = tiled_bitcast.hlo()->operand(0)->shape();
  const Shape& output_shape = tiled_bitcast.hlo()->shape();
  // If the bitcast changes the element type to an element type of the same
  // bitwidth, we need to emit a ttir::BitcastOp.
  if (input_shape.element_type() != output_shape.element_type()) {
    if (primitive_util::BitWidth(input_shape.element_type()) !=
        primitive_util::BitWidth(output_shape.element_type())) {
      return absl::InvalidArgumentError(
          "Bitcast with different bitwidth for operand and output shape "
          "element type is not yet supported.");
    }
    TF_ASSIGN_OR_RETURN(
        Type output_element_type,
        PrimitiveTypeToMlirType(b, output_shape.element_type()));
    auto output_type = mlir::RankedTensorType::get(
        GetPaddedTileSizes(tiled_bitcast.operand(0)->tile_sizes()),
        output_element_type);
    input = mlir::cast<TensorValue>(
        mlir::tensor::BitcastOp::create(b, output_type, input).getResult());
    input_shape.set_element_type(output_shape.element_type());
  }

  // Any Bitcast is decomposable to a transpose+reshape+transpose.
  auto trt = ShapeUtil::DecomposeBitcastToTrt(input_shape, output_shape);
  TF_RET_CHECK(trt.has_value());

  // When replacing the `bitcast` with `transpose` + `reshape` + `transpose` we
  // need to provide the tile sizes at output of each op. We already have the
  // tiling of the `input` (before the first transpose) and the tiling of the
  // final output (after the second transpose), so what's missing are the two
  // tilings in between - after the first transpose and after the reshape. In
  // the case of arbitrary ops, we would need to run the tiling analysis to
  // compute this, but in the case of bitcast we can trivially compute the
  // needed tile sizes from the input and output.

  // The tiles sizes we need to use for the output of the first transpose
  // are the permuted tiles sizes of the input. Note that these are
  // different, even in rank, compared to the tile sizes of the final shape of
  // the bitcast, so it's not possible to easily propagate them from the output.
  std::vector<int64_t> transpose1_tile_sizes =
      Permute(tiled_bitcast.operand(0)->tile_sizes(), trt->transpose1_dims);
  TensorValue normalized_input =
      trt->IsTranspose1Identity()
          ? input
          : EmitTiledTranspose(b, transpose1_tile_sizes,
                               llvm::to_vector(trt->transpose1_dims), input);

  // Like the first transpose above, the tile sizes after the second transpose
  // are a permutation (according to transpose2_dims) of the tile sizes of
  // the reshape. Since we know the tile sizes of the final transpose and need
  // the tile sizes of the reshape, we compute the tile sizes backwards, taking
  // the inverse permutation.
  std::vector<int64_t> reshape_tile_sizes =
      PermuteInverse(tiled_bitcast.tile_sizes(), trt->transpose2_dims);
  TensorValue normalized_reshape;
  if (ShapeUtil::Equal(trt->transpose1_shape, trt->reshape_shape)) {
    normalized_reshape = normalized_input;
  } else {
    TF_ASSIGN_OR_RETURN(
        normalized_reshape,
        EmitTiledReshape(b, reshape_tile_sizes, normalized_input));
  }

  // The final transpose simply uses the tile sizes computed for the original
  // bitcast by the tiling analysis.
  return trt->IsTranspose2Identity()
             ? normalized_reshape
             : EmitTiledTranspose(b, tiled_bitcast.tile_sizes(),
                                  llvm::to_vector(trt->transpose2_dims),
                                  normalized_reshape);
}

absl::StatusOr<std::vector<TensorValue>> EmitTiledComputation(
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation, mlir::FunctionOpInterface fn,
    Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values);
// Returns the number of iterations of the loop over the contracting
// dimension of matrix multiplication.
absl::StatusOr<int64_t> GetDotLoopIterationCount(
    const TiledHloInstruction& tiled_dot) {
  // As LHS (and RHS) must point to the outline fusion computation that is
  // tiled with contracting dimension, we can get the
  // - size from the shape of the operand
  // - tile size from the tiling of the nested fusion root
  // using the contracting dimension from the dot instruction.
  const HloInstruction& dot = *tiled_dot.hlo();
  const auto& dims = dot.dot_dimension_numbers();
  if (dims.lhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only one contracting dimension is supported, got ",
                     dims.lhs_contracting_dimensions_size()));
  }
  auto dim_idx = dims.lhs_contracting_dimensions(0);
  int64_t k_size = tiled_dot.hlo()->operand(0)->shape().dimensions(dim_idx);
  int64_t k_tile = tiled_dot.operand(0)->tile_size(dim_idx);
  return CeilOfRatio(k_size, k_tile);
}

// TODO(b/393299275): unify with the logic in `EmitReduce`.
// Computes and applies a mask to the reduction dimension of the dot operand
// passed as a parameter.
//
// Note: we currently assume that contracting_dimension_tile_index is an i32
// scalar.
absl::StatusOr<TensorValue> MaskDotOperand(
    mlir::ImplicitLocOpBuilder& b, const TiledHloInstruction& dot_operand,
    TensorValue dot_operand_value, Value contracting_dimension_tile_index,
    int contraction_dimension_index) {
  if (contracting_dimension_tile_index.getType() != b.getI32Type()) {
    return absl::FailedPreconditionError(
        "contracting_dimension_tile_index must be an i32 scalar");
  }

  llvm::ArrayRef<int64_t> tile_shape = dot_operand_value.getType().getShape();

  int64_t contracting_dimension_size =
      dot_operand.hlo()->shape().dimensions(contraction_dimension_index);
  int64_t tile_size = tile_shape[contraction_dimension_index];

  if (contracting_dimension_size % tile_size != 0) {
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

      Value boundary = CreateConst(b, b.getI32Type(),
                                   contracting_dimension_size, {tile_size});

      Value mask = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, indices,
                                         boundary);

      mask = xtile::BroadcastInDims(b, mlir::cast<TensorValue>(mask),
                                    tile_shape, {contraction_dimension_index});
      TF_ASSIGN_OR_RETURN(auto element_type,
                          PrimitiveTypeToMlirType(
                              b, dot_operand.hlo()->shape().element_type()));

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

  return dot_operand_value;
}

// Returns `shape` without all its unit dimensions, as well as the index of the
// remaining dimensions in the original `shape`.
std::pair<SmallVector<int64_t>, SmallVector<int64_t>> CollapseUnitDims(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> counterpart_shape) {
  SmallVector<int64_t> shape_without_unit_dims;
  SmallVector<int64_t> non_unit_dims_indices;
  for (auto [i, size] : llvm::enumerate(shape)) {
    if (size != 1 || size != counterpart_shape[i]) {
      shape_without_unit_dims.push_back(size);
      non_unit_dims_indices.push_back(i);
    }
  }
  return {std::move(shape_without_unit_dims), std::move(non_unit_dims_indices)};
}

enum class DotOperandSide { kLhs, kRhs };

// Canonicalizes the given operand of a dot operation, i.e. make it a 2D tensor,
// and make sure that the contracting dimension is where we expect it to be for
// the given side (the second dimension for LHS, the first dimension for the
// RHS).
//
// If it is a scaled-dot scale operand then we drop the extra dims only
// when they equal to 1  and are matching with the corresponding operand.
// Example:
//   when lhs_scale operand with shape [1,128, 1] (passed as operand parameter)
//   and lhs operand with shape [1,128, 32] (passed as counterpart_operand
//   parameter)
//   the function will drop only the first dim and will keep the last one
//   because the last one of the lhs operand is not equal to 1.
//
// Returns an error if canonicalization is not possible.
absl::StatusOr<TensorValue> CanonicalizeDotOperand(
    mlir::ImplicitLocOpBuilder& b, TensorValue operand,
    int64_t contracting_dim_idx, DotOperandSide side,
    TensorValue counterpart_operand = nullptr) {
  llvm::ArrayRef<int64_t> shape = operand.getType().getShape();
  llvm::ArrayRef<int64_t> counterpart_shape =
      counterpart_operand == nullptr ? shape
                                     : counterpart_operand.getType().getShape();

  auto [shape_without_unit_dims, non_unit_dims_indices] =
      CollapseUnitDims(shape, counterpart_shape);

  if (shape_without_unit_dims.size() != 2) {
    return absl::FailedPreconditionError(
        "Expected dot operand tile to have exactly two non-unit tile sizes");
  }

  if (shape.size() != shape_without_unit_dims.size()) {
    TF_ASSIGN_OR_RETURN(operand,
                        EmitTiledReshape(b, shape_without_unit_dims, operand));
  }

  int expected_contracting_dim_position = side == DotOperandSide::kLhs ? 1 : 0;
  bool is_transposed =
      non_unit_dims_indices[expected_contracting_dim_position] !=
      contracting_dim_idx;

  if (is_transposed) {
    SmallVector<int64_t, 2> transposed_shape{shape_without_unit_dims[1],
                                             shape_without_unit_dims[0]};
    operand =
        EmitTiledTranspose(b, transposed_shape, /*dimensions=*/{1, 0}, operand);
  }

  return operand;
}

absl::StatusOr<TensorValue> EmitDot(
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo_dot, mlir::FunctionOpInterface fn,
    Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  // We expect to get a tiled HLO in form:
  //
  // left { ... }
  // right { ... }
  // kernel {
  //   p0 = parameter(0)
  //   p1 = parameter(1)
  //   ..
  //   a = fusion(p0, p1, ...), calls=left
  //   b = fusion(p0, p1, ...), calls=right
  //   ...
  //   c = f32[32,512]{1,0} dot(a, b),
  //     lhs_contracting_dims={1}, rhs_contracting_dims={0}
  //   ...
  // }
  //
  // Where `left` and `right` fusions already have been tiled to be emitted
  // as part of the loop over the contracting dimension. Their
  // parameters are literally the parameters of `kernel`, not the results of
  // other instructions in the `kernel`. From that we will emit:
  //
  // acc = [tile_m, tile_n] 0.0f
  // for (k = 0 .. size_k / tile_k) {
  //   a = "left" computation for left tiling at (pid)[k]
  //   b = "right" computation for right tiling at (pid)[k]
  //   acc = a x b
  // }
  // c = acc
  VLOG(2) << "EmitDot: " << tiled_hlo_dot.ToString();
  const HloDotInstruction& dot =
      *::xla::Cast<HloDotInstruction>(tiled_hlo_dot.hlo());
  if (!absl::c_all_of(tiled_hlo_dot.operands(),
                      [](const TiledHloInstruction* operand) {
                        return operand->hlo()->opcode() == HloOpcode::kFusion;
                      })) {
    return absl::FailedPreconditionError("Expected dot operands to be fusions");
  }

  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_hlo_dot.tile_sizes());

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

  // The specific accumulator type to use may not correspond to the output type
  // of the dot. In particular, that is the case when an algorithm is specified
  // and the dot's output type does not match its expectations.
  TF_ASSIGN_OR_RETURN(Type accumulator_type,
                      xtile::GetDotAccumulatorType(b, dot));
  TensorValue accumulator =
      CreateConst(b, accumulator_type, 0.0f, padded_tile_sizes_no_unit_dims);

  TF_ASSIGN_OR_RETURN(int64_t loop_iteration_count,
                      GetDotLoopIterationCount(tiled_hlo_dot));
  auto pid_dim = b.getAffineDimExpr(0);
  auto ki_symbol = b.getAffineSymbolExpr(0);
  // Nested fusions are tiled with indexing map 'pid * loop_iter_count + ki'
  IndexingMap computation_index_map{
      AffineMap::get(1, 1, pid_dim * loop_iteration_count + ki_symbol),
      {IndexingMap::Variable{
          tiled_hlo_dot.tile_offsets_indexing()->GetDimensionBound(0), "pid"}},
      {IndexingMap::Variable{{0, loop_iteration_count - 1}, "k"}},
      /*rt_vars=*/{}};

  auto for_op = mlir::scf::ForOp::create(
      b,
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_count),
      /*step=*/MakeIndex(b, 1), accumulator);

  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value ki = for_op.getInductionVar();
    Value computation_index = xla::ApplyIndexingOp::create(
                                  b, ValueRange{pid, ki}, computation_index_map)
                                  .getResult(0);
    SmallVector<TensorValue> dot_args;
    for (const TiledHloInstruction* operand : tiled_hlo_dot.operands()) {
      VLOG(3) << "Emitting dot operand: " << operand->ToString();
      const TiledHloFusionInstruction* tiled_fusion_operand =
          static_cast<const TiledHloFusionInstruction*>(operand);
      TF_ASSIGN_OR_RETURN(
          std::vector<TensorValue> result,
          EmitTiledComputation(
              b, ::xla::Cast<HloFusionInstruction>(tiled_fusion_operand->hlo()),
              *tiled_fusion_operand->called_computation(), fn,
              computation_index, values));
      if (result.size() != 1) {
        return absl::InternalError(absl::StrCat(
            "Expected nested fusion computation to emit a single value, got ",
            result.size()));
      }
      dot_args.push_back(result.front());
    }
    Value acc = for_op.getRegionIterArgs().front();
    int64_t lhs_contracting_dim_idx =
        dot.dot_dimension_numbers().lhs_contracting_dimensions(0);

    int64_t rhs_contracting_dim_idx =
        dot.dot_dimension_numbers().rhs_contracting_dimensions(0);

    Value ki_i32 = Cast(b, ki, b.getI32Type());
    TF_ASSIGN_OR_RETURN(
        TensorValue lhs,
        MaskDotOperand(b, *tiled_hlo_dot.operand(0), dot_args[0], ki_i32,
                       lhs_contracting_dim_idx));

    TF_ASSIGN_OR_RETURN(
        TensorValue rhs,
        MaskDotOperand(b, *tiled_hlo_dot.operand(1), dot_args[1], ki_i32,
                       rhs_contracting_dim_idx));

    // Canonicalize the dot operands to match Triton's/the hardware's
    // expectations.
    TF_ASSIGN_OR_RETURN(lhs,
                        CanonicalizeDotOperand(b, lhs, lhs_contracting_dim_idx,
                                               DotOperandSide::kLhs));
    TF_ASSIGN_OR_RETURN(rhs,
                        CanonicalizeDotOperand(b, rhs, rhs_contracting_dim_idx,
                                               DotOperandSide::kRhs));

    TF_ASSIGN_OR_RETURN(
        Value acc_next,
        xtile::EmitSingleTileDot(b, dot, xtile::DotOperands{lhs, rhs, acc}));
    mlir::scf::YieldOp::create(b, acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  TF_ASSIGN_OR_RETURN(Type dot_output_type,
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

absl::StatusOr<TensorValue> EmitScaledDot(
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo_dot, mlir::FunctionOpInterface fn,
    Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  VLOG(2) << "EmitScaledDot: " << tiled_hlo_dot.ToString();
  const HloScaledDotInstruction& scaled_dot =
      *::xla::Cast<HloScaledDotInstruction>(tiled_hlo_dot.hlo());
  if (!absl::c_all_of(tiled_hlo_dot.operands(),
                      [](const TiledHloInstruction* operand) {
                        return operand->hlo()->opcode() == HloOpcode::kFusion;
                      })) {
    return absl::FailedPreconditionError("Expected dot operands to be fusions");
  }

  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_hlo_dot.tile_sizes());

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

  Type accumulator_type = b.getF32Type();
  TensorValue accumulator =
      CreateConst(b, accumulator_type, 0.0f, padded_tile_sizes_no_unit_dims);

  TF_ASSIGN_OR_RETURN(int64_t loop_iteration_count,
                      GetDotLoopIterationCount(tiled_hlo_dot));
  auto pid_dim = b.getAffineDimExpr(0);
  auto ki_symbol = b.getAffineSymbolExpr(0);
  // Nested fusions are tiled with indexing map 'pid * loop_iter_count + ki'
  IndexingMap computation_index_map{
      AffineMap::get(1, 1, pid_dim * loop_iteration_count + ki_symbol),
      {IndexingMap::Variable{
          tiled_hlo_dot.tile_offsets_indexing()->GetDimensionBound(0), "pid"}},
      {IndexingMap::Variable{{0, loop_iteration_count - 1}, "k"}},
      /*rt_vars=*/{}};

  // TODO(b/449668102): Consider adding warp specialization support for scaled
  // dot. At the moment, there are no benchmarks that use scaled dot.
  auto for_op = mlir::scf::ForOp::create(
      b,
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_count),
      /*step=*/MakeIndex(b, 1), accumulator);
  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value ki = for_op.getInductionVar();
    Value computation_index = xla::ApplyIndexingOp::create(
                                  b, ValueRange{pid, ki}, computation_index_map)
                                  .getResult(0);
    SmallVector<TensorValue> dot_args;
    for (const TiledHloInstruction* operand : tiled_hlo_dot.operands()) {
      VLOG(3) << "Emitting scaled dot operand: " << operand->ToString();
      const TiledHloFusionInstruction* tiled_fusion_operand =
          static_cast<const TiledHloFusionInstruction*>(operand);
      TF_ASSIGN_OR_RETURN(
          std::vector<TensorValue> result,
          EmitTiledComputation(
              b, ::xla::Cast<HloFusionInstruction>(tiled_fusion_operand->hlo()),
              *tiled_fusion_operand->called_computation(), fn,
              computation_index, values));
      if (result.size() != 1) {
        return absl::InternalError(absl::StrCat(
            "Expected nested fusion computation to emit a single value, got ",
            result.size()));
      }
      dot_args.push_back(result.front());
    }
    Value acc = for_op.getRegionIterArgs().front();
    int64_t lhs_contracting_dim_idx =
        scaled_dot.dot_dimension_numbers().lhs_contracting_dimensions(0);

    int64_t rhs_contracting_dim_idx =
        scaled_dot.dot_dimension_numbers().rhs_contracting_dimensions(0);

    // TODO(b/393299275): masking is only necessary during the last iteration of
    // the loop. We should evaluate whether adding a conditional mask helps or
    // hinders performance for Triton.
    Value ki_i32 = Cast(b, ki, b.getI32Type());
    TF_ASSIGN_OR_RETURN(
        TensorValue lhs,
        MaskDotOperand(b, *tiled_hlo_dot.operand(0), dot_args[0], ki_i32,
                       lhs_contracting_dim_idx));
    TF_ASSIGN_OR_RETURN(
        TensorValue rhs,
        MaskDotOperand(b, *tiled_hlo_dot.operand(1), dot_args[1], ki_i32,
                       rhs_contracting_dim_idx));

    TF_ASSIGN_OR_RETURN(
        TensorValue lhs_scale,
        MaskDotOperand(b, *tiled_hlo_dot.operand(2), dot_args[2], ki_i32,
                       lhs_contracting_dim_idx));

    TF_ASSIGN_OR_RETURN(
        TensorValue rhs_scale,
        MaskDotOperand(b, *tiled_hlo_dot.operand(3), dot_args[3], ki_i32,
                       rhs_contracting_dim_idx));

    // Canonicalize the dot operands to match Triton's/the hardware's
    // expectations.

    TF_ASSIGN_OR_RETURN(
        lhs_scale, CanonicalizeDotOperand(b, lhs_scale, lhs_contracting_dim_idx,
                                          DotOperandSide::kLhs, lhs));
    TF_ASSIGN_OR_RETURN(
        rhs_scale, CanonicalizeDotOperand(b, rhs_scale, rhs_contracting_dim_idx,
                                          DotOperandSide::kRhs, rhs));
    TF_ASSIGN_OR_RETURN(lhs,
                        CanonicalizeDotOperand(b, lhs, lhs_contracting_dim_idx,
                                               DotOperandSide::kLhs));
    TF_ASSIGN_OR_RETURN(rhs,
                        CanonicalizeDotOperand(b, rhs, rhs_contracting_dim_idx,
                                               DotOperandSide::kRhs));

    TF_ASSIGN_OR_RETURN(
        Value acc_next,
        xtile::EmitSingleTileScaledDot(
            b, scaled_dot,
            xtile::ScaledDotOperands{lhs, rhs, lhs_scale, rhs_scale, acc}));
    mlir::scf::YieldOp::create(b, acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  TF_ASSIGN_OR_RETURN(
      Type dot_output_type,
      PrimitiveTypeToMlirType(b, scaled_dot.shape().element_type()));

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

absl::StatusOr<TensorValue> EmitConcatenate(
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_concatenate, mlir::FunctionOpInterface fn,
    Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  const int64_t concatenate_dimension =
      tiled_concatenate.hlo()->concatenate_dimension();

  // TODO(b/393299275): get rid of calls to `GetPaddedTileSizes` once tiling
  // is using power of twos everywhere, including when propagating into the
  // prologue of reductions.
  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_concatenate.tile_sizes());
  int64_t concat_dim_tile_size = padded_tile_sizes[concatenate_dimension];

  int64_t num_operands = tiled_concatenate.operands().size();
  for (const auto [index, operand] :
       llvm::enumerate(tiled_concatenate.operands())) {
    if (operand->hlo()->opcode() != HloOpcode::kFusion) {
      // Sanity check: all operands should be nested fusions.
      return absl::FailedPreconditionError(
          "Expected concatenate operands to be nested fusions.");
    }

    int64_t operand_concat_dim_size =
        operand->hlo()->shape().dimensions(concatenate_dimension);

    // The last operand does not have to be a multiple of the tile size, since
    // we can pad it.
    if (index != num_operands - 1 &&
        operand_concat_dim_size % concat_dim_tile_size != 0) {
      // Sanity check: concatenation dimension should be divisible by the tile
      // size for each operand. This is not a fundamental limitation, but this
      // lowering will emit incorrect code if this does not hold---so we gate
      // against it explicitly.
      return absl::FailedPreconditionError(absl::StrCat(
          "Expected the tile size of the concatenation dimension of operand ",
          operand->ToString(), "to divide the dimension size exactly, but got",
          operand_concat_dim_size, " % ", concat_dim_tile_size, " != 0"));
    }
  }
  TF_ASSIGN_OR_RETURN(Type element_type,
                      PrimitiveTypeToMlirType(
                          b, tiled_concatenate.hlo()->shape().element_type()));
  Type result_type =
      mlir::RankedTensorType::get(padded_tile_sizes, element_type);

  // We will load and compute from a single operand, so we need to figure out
  // which one by looking at the offset within the concatenation dimension.
  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_concatenate.tile_offsets_indexing());

  Value concatenate_dimension_offset =
      emitters::ApplyIndexing(tile_offsets_indexing, /*dims=*/pid,
                              /*symbols=*/{}, b)[concatenate_dimension];

  // It would have been nice to be able to use `scf::IndexSwitchOp`, but Triton
  // does not want to deal with the `Index` type, and does not support the op.
  // Instead, we generate a sequence of nested `scf::IfOp`s.
  SmallVector<mlir::scf::IfOp, 4> if_ops;
  int64_t limit = 0;
  for (auto [i, operand] : llvm::enumerate(tiled_concatenate.operands())) {
    // Write in the else branch of the previous if op if one exists.
    if (!if_ops.empty()) {
      b.setInsertionPointToStart(if_ops.back().elseBlock());
    }

    // Add an `if_op` if we have not reached the last operand. The last operand
    // directly populates the `else` block of the previous `if_op`.
    if (if_ops.size() < tiled_concatenate.operands().size() - 1) {
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

    const TiledHloFusionInstruction* tiled_fusion_operand =
        static_cast<const TiledHloFusionInstruction*>(
            tiled_concatenate.operand(i));
    TF_ASSIGN_OR_RETURN(
        std::vector<TensorValue> result,
        EmitTiledComputation(
            b, ::xla::Cast<HloFusionInstruction>(tiled_fusion_operand->hlo()),
            *tiled_fusion_operand->called_computation(), fn, pid, values));
    CHECK_EQ(result.size(), 1);
    mlir::scf::YieldOp::create(b, result.front());
  }

  b.setInsertionPointAfter(if_ops.front());

  return mlir::cast<TensorValue>(if_ops.front().getResult(0));
}

absl::StatusOr<TensorValue> EmitPad(
    mlir::ImplicitLocOpBuilder& b, const TiledHloInstruction& tiled_pad,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values,
    Value pid) {
  // TODO(b/393299275): get rid of calls to `GetPaddedTileSizes` once tiling
  // is using power of twos everywhere, including when propagating into the
  // prologue of reductions.
  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_pad.tile_sizes());

  const TiledHloInstruction* tiled_operand = tiled_pad.operand(0);
  const auto& pad_input_shape = tiled_operand->hlo()->shape().dimensions();

  // Compute tile offsets.
  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_pad.tile_offsets_indexing());
  SmallVector<Value, 3> tile_offsets =
      emitters::ApplyIndexing(tile_offsets_indexing, /*dims=*/pid,
                              /*symbols=*/{}, b);

  // Compute mask.
  Type i32_type = b.getI32Type();
  Value mask;
  for (auto [dim_index, sizes] : llvm::enumerate(
           llvm::zip(pad_input_shape, padded_tile_sizes, tile_offsets))) {
    auto [pad_input_dim_size, pad_output_dim_size, tile_offset] = sizes;
    if (pad_input_dim_size == pad_output_dim_size) {
      continue;
    }

    // LHS for the compare is an iota broadcasted to the output shape.
    TensorValue range = Iota(b, pad_output_dim_size);
    TensorValue bcast = xtile::BroadcastInDims(
        b, range, padded_tile_sizes, {static_cast<int64_t>(dim_index)});

    // RHS for the compare is splat(pad_input_dim_size - tile_offset).
    Value tile_offset_i32 = Cast(b, tile_offset, i32_type);
    Value threshold = arith::SubIOp::create(
        b, CreateConst(b, i32_type, pad_input_dim_size), tile_offset_i32);
    TensorValue threshold_splat = xtile::Splat(b, threshold, padded_tile_sizes);
    Value cmp = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, bcast,
                                      threshold_splat);
    mask = mask ? stablehlo::AndOp::create(b, mask, cmp) : cmp;
  }
  if (!mask) {
    return values[tiled_operand];
  }
  const TiledHloInstruction* padding_value = tiled_pad.operand(1);

  TensorValue pad_value_splat =
      xtile::Splat(b, values[padding_value], padded_tile_sizes);
  return mlir::cast<TensorValue>(
      arith::SelectOp::create(b, mask, values[tiled_operand], pad_value_splat)
          .getResult());
}

absl::StatusOr<TensorValue> EmitAllReduce(
    mlir::ImplicitLocOpBuilder& b, const HloComputation* computation,
    const HloAllReduceInstruction& all_reduce,
    const TiledHloInstruction& tiled_hlo_reduce,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  llvm::SmallVector<mlir::Value> operands;
  operands.reserve(tiled_hlo_reduce.operands().size());

  for (const auto operand : tiled_hlo_reduce.operands()) {
    if (!values.contains(operand)) {
      return Internal("Operand %s not found in the values map.",
                      operand->ToString());
    }
    operands.push_back(values[operand]);
  }

  if (all_reduce.device_list().replica_groups().empty()) {
    return Internal(
        "Triton emitting AllReduce without replica groups is not supported.");
  }

  llvm::SmallVector<int64_t> flattened_replica_group_ids;
  for (const auto& replica_group : all_reduce.replica_groups()) {
    for (const auto& replica_id : replica_group.replica_ids()) {
      flattened_replica_group_ids.push_back(replica_id);
    }
  }

  std::optional<int64_t> channel_handle = all_reduce.channel_id();
  bool use_global_device_ids = all_reduce.use_global_device_ids();

  TF_ASSIGN_OR_RETURN(
      auto output_element_type,
      xtile::PrimitiveTypeToMlirType(b, all_reduce.shape().element_type()));
  auto output_type = mlir::RankedTensorType::get(tiled_hlo_reduce.tile_sizes(),
                                                 output_element_type);

  auto replica_groups_type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(all_reduce.replica_groups().size()),
       static_cast<int64_t>(all_reduce.replica_groups()[0].replica_ids_size())},
      b.getI64Type());
  auto replica_groups_attr = mlir::DenseIntElementsAttr::get(
      replica_groups_type, flattened_replica_group_ids);
  auto channel_handle_attr =
      channel_handle ? mlir::stablehlo::ChannelHandleAttr::get(b.getContext(),
                                                               *channel_handle,
                                                               /*type=*/0)
                     : nullptr;

  auto all_reduce_op = mlir::stablehlo::AllReduceOp::create(
      b, b.getLoc(), output_type, mlir::ValueRange(operands),
      replica_groups_attr, channel_handle_attr, use_global_device_ids);

  TF_RETURN_IF_ERROR(EmitReduceComputation(
      b, &all_reduce, all_reduce.to_apply(), all_reduce_op));

  return mlir::cast<TensorValue>(all_reduce_op.getResult(0));
}

absl::StatusOr<TensorValue> EmitTiledHloInstruction(
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo, mlir::FunctionOpInterface fn,
    Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  const HloInstruction* hlo = tiled_hlo.hlo();
  VLOG(4) << "EmitTiledHloInstruction: " << hlo->ToString();

  if (hlo->opcode() == HloOpcode::kParameter && !fusion->IsUserOf(hlo)) {
    hlo = hlo->parent()->FusionInstruction()->operand(hlo->parameter_number());
  }

  if (fusion->IsUserOf(hlo)) {
    int64_t arg_index = fusion->operand_index(hlo);
    // Walk up the parameter chain to find the outermost operand index.
    while (auto* instr = hlo->parent()->FusionInstruction()) {
      arg_index = hlo->parameter_number();  // Nested operands are parameters.
      hlo = instr->operand(arg_index);
    }
    TF_ASSIGN_OR_RETURN(
        TileInfo tile_info,
        TileInfo::Construct(b, pid, GetRuntimeValues(tiled_hlo, values),
                            tiled_hlo));
    TensorValue parameter =
        EmitParameterExtract(b, tile_info, fn.getArgument(arg_index));

    // Workaround(i1_to_i8_workaround)
    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to type checking that we perform a conversion after
    // loading if the type of the loaded parameter does not match what is
    // expected.
    Type loaded_element_type = getElementTypeOrSelf(parameter.getType());
    TF_ASSIGN_OR_RETURN(
        Type expected_element_type,
        PrimitiveTypeToMlirType(b, hlo->shape().element_type()));

    if (expected_element_type != loaded_element_type) {
      // Ensure that we didn't mess up somewhere else by checking that we
      // indeed loaded the expected storage type for the expected element type.
      if (loaded_element_type != StorageType(expected_element_type)) {
        return absl::InternalError(absl::StrCat(
            "Parameters were loaded with an unexpected element type "
            "while lowering ",
            fusion->called_computation()->ToString()));
      }
      parameter =
          mlir::cast<TensorValue>(Cast(b, parameter, expected_element_type));
    }

    return parameter;
  }

  if (hlo->opcode() == HloOpcode::kConcatenate) {
    return EmitConcatenate(b, fusion, tiled_hlo, fn, pid, values);
  }

  if (hlo->opcode() == HloOpcode::kPad) {
    return EmitPad(b, tiled_hlo, values, pid);
  }

  if (hlo->opcode() == HloOpcode::kDot) {
    return EmitDot(b, fusion, tiled_hlo, fn, pid, values);
  }

  if (hlo->opcode() == HloOpcode::kScaledDot) {
    return EmitScaledDot(b, fusion, tiled_hlo, fn, pid, values);
  }

  if (hlo->opcode() == HloOpcode::kConstant) {
    if (ShapeUtil::IsEffectiveScalar(hlo->shape())) {
      return EmitConstant(b, *hlo);
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported non-scalar constant ", hlo->ToString()));
  }

  if (hlo->opcode() == HloOpcode::kIota) {
    return EmitTiledIota(b, pid, tiled_hlo);
  }

  if (hlo->opcode() == HloOpcode::kBroadcast) {
    return EmitTiledBroadcast(b, tiled_hlo, values);
  }

  if (hlo->opcode() == HloOpcode::kReduce) {
    return EmitReduce(b, tiled_hlo, values);
  }

  if (hlo->opcode() == HloOpcode::kAllReduceStart) {
    const HloComputation* computation =
        fusion->fused_instructions_computation();
    const HloInstruction* root_instruction = computation->root_instruction();
    if (root_instruction->opcode() == HloOpcode::kAllReduceDone) {
      root_instruction = root_instruction->operand(0);
    }
    return EmitAllReduce(b, computation,
                         *xla::Cast<HloAllReduceInstruction>(root_instruction),
                         tiled_hlo, values);
  }

  if (hlo->opcode() == HloOpcode::kAllReduceDone) {
    return values[tiled_hlo.operand(0)];
  }

  if (hlo->IsElementwise()) {
    std::vector<Value> operands;
    operands.reserve(hlo->operands().size());

    for (const TiledHloInstruction* operand : tiled_hlo.operands()) {
      operands.push_back(values[operand]);
    }
    TF_ASSIGN_OR_RETURN(Value result, EmitElementwise(b, *hlo, operands));
    return mlir::cast<TensorValue>(result);
  }

  if (hlo->opcode() == HloOpcode::kReshape) {
    return EmitTiledReshape(b, tiled_hlo.tile_sizes(),
                            values[tiled_hlo.operand(0)]);
  }

  if (hlo->opcode() == HloOpcode::kBitcast) {
    return EmitTiledBitcast(b, tiled_hlo, values[tiled_hlo.operand(0)]);
  }

  if (hlo->opcode() == HloOpcode::kTranspose) {
    auto transpose =
        ::xla::Cast<const HloTransposeInstruction>(tiled_hlo.hlo());
    return EmitTiledTranspose(b, tiled_hlo.tile_sizes(),
                              llvm::to_vector(transpose->dimensions()),
                              values[tiled_hlo.operand(0)]);
  }

  // Slice is currently supported only as an operation on indices
  // which is pushed to loads and stores. We don't generate any further code.
  if (hlo->opcode() == HloOpcode::kSlice) {
    return values[tiled_hlo.operand(0)];
  }

  if (hlo->opcode() == HloOpcode::kDynamicSlice) {
    // Dynamic slice is implemented as a load and does not require any further
    // processing.
    return values[tiled_hlo.operand(0)];
  }

  return absl::UnimplementedError(
      absl::StrCat("Unsupported operation ", hlo->ToString()));
}

absl::StatusOr<std::vector<TensorValue>> EmitTiledComputation(
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation, mlir::FunctionOpInterface fn,
    Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, TensorValue>& values) {
  VLOG(2) << "EmitTiledComputation: " << tiled_computation.ToString();
  for (const TiledHloInstruction* tiled_hlo :
       tiled_computation.instructions()) {
    const HloInstruction* hlo = tiled_hlo->hlo();
    // Skip generating nested fusions, they are emitted by their consumer.
    if (hlo->parent()->IsFusionComputation() &&
        hlo->opcode() == HloOpcode::kFusion) {
      VLOG(1) << "Skipping nested fusion: " << hlo->ToString();
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        TensorValue result,
        EmitTiledHloInstruction(b, fusion, *tiled_hlo, fn, pid, values));
    TF_RET_CHECK(values.insert({tiled_hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  std::vector<TensorValue> results;
  results.reserve(tiled_computation.GetRoots().size());
  for (const auto* root : tiled_computation.GetRoots()) {
    results.push_back(values[root]);
  }
  return std::move(results);
}

}  // namespace

namespace {

absl::Status EmitGeneric(mlir::OpBuilder builder,
                         const HloFusionInstruction* fusion,
                         const SymbolicTileAnalysis& symbolic_tile_analysis,
                         const Tiling& tiling, xtile::EntryFuncOp fn,
                         MLIRContext* mlir_context) {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Emitting XTile IR for fusion\n"
            << ExtractInstructionIntoNewModule(*fusion)->ToString();
  }

  // TODO(b/372454662): Decide which root to use. Currently, we only support
  // "simple" multi-output fusions that have just one root without users. This
  // root appears last in def-before-use order. We derive the tiling from this
  // root.
  const HloInstruction* root =
      symbolic_tile_analysis.GetSymbolicTiledHloComputation().back()->hlo();
  auto loc = mlir::NameLoc::get(builder.getStringAttr(root->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);
  absl::Span<const HloInstruction* const> roots =
      symbolic_tile_analysis.GetRoots();
  int64_t root_index = FindIndex(roots, root);
  TiledHloScheduleBuilder schedule_builder = CreateMajorToMinorTiledHloSchedule;

  // TODO(b/417977182): this is a hacky heuristic to avoid regressing cases
  // involving hardcoded grid tiling in the legacy emitter, as we enable the new
  // one for `dot` fusions.
  //
  // The idea here is that, if `lhs` can fully fit in L2 cache, and `rhs` does
  // not, we should start with iterating over the full `lhs` in order to have it
  // in cache for all subsequent iterations over `rhs`. That means we should
  // iterate over `lhs`'s non-contracting dimensions first.
  //
  // Whenever it is not true that one of the operands can fit fully in cache, it
  // is more beneficial to use a "planar snake" space-filling curve to optimize
  // L2 cache hits, but this is not implemented yet.
  if (roots.size() == 1 && root->opcode() == HloOpcode::kDot) {
    int64_t lhs_bytes_size =
        Product(root->operand(0)->shape().dimensions()) *
        primitive_util::ByteWidth(root->operand(0)->shape().element_type());
    int64_t rhs_bytes_size =
        Product(root->operand(1)->shape().dimensions()) *
        primitive_util::ByteWidth(root->operand(1)->shape().element_type());
    if (lhs_bytes_size < rhs_bytes_size) {
      // Validates whether the expected invariants are upheld by the analysis to
      // ensure we don't crash later.
      //
      // TODO(b/417977182): use a "conformance" API instead of a builder to
      // reuse what we build here directly.
      absl::StatusOr<std::unique_ptr<TransposedDotTiledHloSchedule>>
          transposed_schedule = TransposedDotTiledHloSchedule::Create(
              symbolic_tile_analysis.GetTilingSpecification());
      if (transposed_schedule.ok()) {
        schedule_builder = TransposedDotTiledHloSchedule::Create;
      }
    }
  }
  TF_RET_CHECK(root_index < symbolic_tile_analysis.GetRoots().size());
  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      symbolic_tile_analysis.ComputeTiledHloInstructions(
                          tiling, schedule_builder,
                          /*constraints_are_known_satisfied=*/false,
                          /*compute_all_tile_offset_indexing_maps=*/true));
  VLOG(3) << "EmitGeneric: tiled HLO computation:\n"
          << tiled_hlo_computation.ToString();

  Value tile_id = fn.getTileId();
  absl::flat_hash_map<const TiledHloInstruction*, TensorValue> values;
  TF_ASSIGN_OR_RETURN(auto results,
                      EmitTiledComputation(b, fusion, tiled_hlo_computation, fn,
                                           tile_id, values));

  const HloComputation* computation = fusion->fused_instructions_computation();
  for (auto [root, result, arg] :
       llvm::zip(tiled_hlo_computation.GetRoots(), results,
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

    TF_ASSIGN_OR_RETURN(
        auto tile_info,
        TileInfo::Construct(b, tile_id, /*runtime_values=*/{}, *root));

    xtile::InsertTileOp::create(b, result, arg, tile_info.offsets(),
                                tile_info.padded_tile_sizes(),
                                tile_info.tile_strides());
  }

  return absl::OkStatus();
}

}  // namespace

mlir::MemRefType GetMemRefType(const Shape& shape, mlir::Type element_type) {
  mlir::MLIRContext* context = element_type.getContext();
  mlir::Type storage_type = StorageType(element_type);

  // Don't add any attribute for default layouts as it adds a lot of noise to
  // the printed IR.
  if (LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return mlir::MemRefType::get(shape.dimensions(), storage_type);
  }

  auto minor_to_major_attr =
      mlir::DenseI64ArrayAttr::get(context, shape.layout().minor_to_major());
  auto layout = xtile::LayoutAttr::get(context, minor_to_major_attr);

  return mlir::MemRefType::get(shape.dimensions(), storage_type, layout);
}

// TODO(b/447133106): Contrary to the name, this function still does a lot of
// triton specific things. It should be migrated to use non-triton specific
// utilities.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitXTileModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const SymbolicTileAnalysis& symbolic_tile_analysis, const Tiling& tiling,
    MLIRContext& mlir_context, absl::Span<mlir::Type> opaque_args_types) {
  const auto debug_options = fusion->GetModule()->config().debug_options();

  const HloComputation* hlo_computation =
      fusion->fused_instructions_computation();

  auto loc = mlir::NameLoc::get(
      mlir::StringAttr::get(&mlir_context, hlo_computation->name()));
  mlir::ImplicitLocOpBuilder b(loc, &mlir_context);

  mlir::OwningOpRef<mlir::ModuleOp> xtile_module =
      llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(xtile_module->getBody());

  // Build Triton kernel.
  SmallVector<Type> fn_arg_types;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    PrimitiveType type = p->shape().element_type();
    Type ir_type;
    if (type == U16) {
      ir_type = b.getI16Type();
    } else if (type == S4) {
      ir_type = b.getI4Type();
    } else {
      TF_ASSIGN_OR_RETURN(ir_type, PrimitiveTypeToMlirType(b, type));
    }
    fn_arg_types.push_back(GetMemRefType(p->shape(), ir_type));
  }

  for (const auto& [index, shape] : ShapeUtil::GetLeafShapes(fusion->shape())) {
    TF_ASSIGN_OR_RETURN(Type triton_ty,
                        PrimitiveTypeToMlirType(b, shape.element_type()));
    fn_arg_types.push_back(GetMemRefType(shape, triton_ty));
  }

  // Add opaque arguments.
  fn_arg_types.reserve(fn_arg_types.size() + opaque_args_types.size());

  for (const auto& type : opaque_args_types) {
    fn_arg_types.push_back(type);
  }

  // Metadata arguments are opaque to the tiling infra.
  llvm::SmallVector<mlir::NamedAttribute> named_attributes{b.getNamedAttr(
      "num_opaque_args", b.getI32IntegerAttr(opaque_args_types.size()))};

  auto fn = xtile::EntryFuncOp::create(b, fn_name, fn_arg_types,
                                       named_attributes, {});

  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  TF_RETURN_IF_ERROR(EmitGeneric(b, fusion, symbolic_tile_analysis, tiling, fn,
                                 &mlir_context));

  b.create<xtile::EntryFuncReturnOp>();

  {
    // Verify that the emitted module contains only ops from dialects that can
    // be shared between backends.
    mlir::PassManager pm(&mlir_context);
    pm.addPass(xtile::createVerifyLegalXTileOpsPass());
    if (mlir::failed(pm.run(*xtile_module))) {
      return Internal("Failed to verify XTile module.");
    }
  }

  return xtile_module;
}

}  // namespace gpu
}  // namespace xla
