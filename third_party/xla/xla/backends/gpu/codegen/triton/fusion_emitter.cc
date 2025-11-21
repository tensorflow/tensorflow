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
#include <string>
#include <system_error>  // NOLINT
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
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_fusion_instruction.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/tiling/tiled_hlo_schedule.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
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
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla {
namespace gpu {

namespace arith = ::mlir::arith;
namespace ttir = ::mlir::triton;
namespace mtx = ::mlir::triton::xla;
namespace stablehlo = ::mlir::stablehlo;
namespace xgt = ::xla::gpu::triton;

using ::llvm::SmallVector;
using ::mlir::AffineMap;
using ::mlir::ArrayRef;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

using ::xla::gpu::triton::Cast;
using ::xla::gpu::triton::CreateConst;
using ::xla::gpu::triton::EmitConstant;
using ::xla::gpu::triton::EmitElementwise;
using ::xla::gpu::triton::EmitScope;
using ::xla::gpu::triton::GetPaddedTileSizes;
using ::xla::gpu::triton::StorageType;
using ::xla::gpu::triton::TensorValue;
using ::xla::gpu::triton::TileInfo;
using ::xla::gpu::triton::TritonType;

namespace {

Value MakeIndex(EmitterLocOpBuilder& b, int64_t value) {
  return b.create<arith::ConstantIndexOp>(value);
}

// Same as HLO BroadcastInDims. The sorted indices in `dims` specify the mapping
// of the input dimensions to the output dimensions.
TensorValue BroadcastInDims(EmitterLocOpBuilder b, TensorValue value,
                            ArrayRef<int64_t> output_shape,
                            ArrayRef<int64_t> dims) {
  CHECK(llvm::is_sorted(dims)) << "broadcast dims must be sorted";

  auto result_type = mlir::RankedTensorType::get(
      output_shape, value.getType().getElementType());

  return b.create<stablehlo::BroadcastInDimOp>(result_type, value, dims);
}

TensorValue Splat(EmitterLocOpBuilder b, Value value,
                  ArrayRef<int64_t> output_shape) {
  auto tensor_value = mlir::dyn_cast<TensorValue>(value);
  if (!tensor_value) {
    tensor_value = b.create<mlir::tensor::FromElementsOp>(
        mlir::RankedTensorType::get({}, value.getType()), value);
  }
  return BroadcastInDims(b, tensor_value, output_shape, /*dims=*/{});
}

TensorValue Iota(EmitterLocOpBuilder b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return b.create<stablehlo::IotaOp>(type, /*iota_dimension=*/0);
}

absl::StatusOr<TensorValue> EmitReduce(
    EmitterLocOpBuilder b, const TiledHloInstruction& tiled_hlo_reduce,
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

  stablehlo::ReduceOp reduction =
      b.create<stablehlo::ReduceOp>(input, init_value, hlo_reduce.dimensions());
  {
    TF_ASSIGN_OR_RETURN(Type result_ty,
                        TritonType(b, hlo_reduce.shape().element_type()));
    result_ty = mlir::RankedTensorType::get({}, result_ty);

    mlir::Location loc = b.getLoc();
    mlir::Block* reducer = b.createBlock(&reduction->getRegion(0), {},
                                         {result_ty, result_ty}, {loc, loc});
    b.setInsertionPointToStart(reducer);

    HloComputation* reduction_computation = hlo_reduce.to_apply();

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

    TF_ASSIGN_OR_RETURN(TensorValue result, EmitScope(b, /*analysis=*/nullptr,
                                                      to_emit, region_values));
    b.create<stablehlo::ReturnOp>(SmallVector<Value>({result}));
    b.setInsertionPointAfter(reduction);
  }

  return mlir::cast<TensorValue>(reduction.getResult(0));
}

template <typename T>
ArrayRef<T> MakeArrayRef(const absl::Span<const T> span) {
  return ArrayRef(span.data(), span.size());
}

TensorValue EmitTiledBroadcast(
    EmitterLocOpBuilder b, const TiledHloInstruction& tiled_broadcast,
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
  return BroadcastInDims(b, input, padded_output_tile_shape,
                         MakeArrayRef(tiled_broadcast.hlo()->dimensions()));
}

absl::StatusOr<TensorValue> EmitTiledIota(
    EmitterLocOpBuilder b, Value pid, const TiledHloInstruction& tiled_iota) {
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
  Value range = b.create<arith::MulIOp>(
      Iota(b, padded_tile_sizes[iota_dim]),
      Splat(b,
            CreateConst(b, b.getI32Type(), tiled_iota.tile_strides()[iota_dim]),
            padded_tile_sizes[iota_dim]));

  // Then, add the base offset to the iota components.
  range = b.create<arith::AddIOp>(
      range, Splat(b, iota_dim_offset, padded_tile_sizes[iota_dim]));

  // Cast the result to the targeted type.
  TF_ASSIGN_OR_RETURN(Type iota_element_type,
                      TritonType(b, hlo_iota->shape().element_type()));

  range = Cast(b, range, iota_element_type);

  // And finally, produce a broadcast along the non-iota dimensions in order to
  // produce the whole iota tile.
  return BroadcastInDims(b, mlir::cast<TensorValue>(range), padded_tile_sizes,
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

absl::StatusOr<TensorValue> EmitTiledReshape(EmitterLocOpBuilder b,
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

  return b.create<stablehlo::ReshapeOp>(output_tensor_type, input);
}

TensorValue EmitTiledTranspose(EmitterLocOpBuilder b,
                               ArrayRef<int64_t> tile_sizes,
                               SmallVector<int64_t> dimensions,
                               TensorValue input) {
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  Type input_element_type = input.getType().getElementType();
  Type output_tensor_type =
      mlir::RankedTensorType::get(padded_tile_sizes, input_element_type);

  mlir::DenseI64ArrayAttr order = b.getDenseI64ArrayAttr(dimensions);

  return b.create<stablehlo::TransposeOp>(output_tensor_type, input, order);
}

absl::StatusOr<TensorValue> EmitTiledBitcast(
    EmitterLocOpBuilder b, const TiledHloInstruction& tiled_bitcast,
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
    TF_ASSIGN_OR_RETURN(Type output_element_type,
                        TritonType(b, output_shape.element_type()));
    auto output_type = mlir::RankedTensorType::get(
        GetPaddedTileSizes(tiled_bitcast.operand(0)->tile_sizes()),
        output_element_type);
    input = mlir::cast<TensorValue>(
        b.create<mlir::tensor::BitcastOp>(output_type, input).getResult());
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
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, Value pid,
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
    EmitterLocOpBuilder b, const TiledHloInstruction& dot_operand,
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
    Value num_full_tiles = b.create<arith::DivSIOp>(
        CreateConst(b, b.getI32Type(), contracting_dimension_size),
        tile_size_value);
    // if tile_index >= num_full_tiles...
    auto cond = b.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                        contracting_dimension_tile_index,
                                        num_full_tiles);
    auto if_op = b.create<mlir::scf::IfOp>(mlir::TypeRange(result_type), cond,
                                           /*withElseRegion=*/true);
    // then ...
    {
      b.setInsertionPointToStart(if_op.thenBlock());
      // indices =
      //   contracting_dimension_tile_index * tile_size + range(0, tile_size)
      // mask = indices < contracting_dimension_size
      // operand = select(broadcast(mask, operand.shape), operand, 0)
      Value tile_offset = b.create<arith::MulIOp>(
          contracting_dimension_tile_index, tile_size_value);
      TensorValue range = Iota(b, tile_size);
      TensorValue broadcasted_tile_offset = Splat(b, tile_offset, {tile_size});
      Value indices = b.create<arith::AddIOp>(range, broadcasted_tile_offset);

      Value boundary = CreateConst(b, b.getI32Type(),
                                   contracting_dimension_size, {tile_size});

      Value mask =
          b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, indices, boundary);

      mask = BroadcastInDims(b, mlir::cast<TensorValue>(mask), tile_shape,
                             {contraction_dimension_index});
      TF_ASSIGN_OR_RETURN(
          auto element_type,
          TritonType(b, dot_operand.hlo()->shape().element_type()));

      TensorValue zero = CreateConst(b, element_type, 0.0f, tile_shape);

      Value masked_dot_operand =
          b.create<arith::SelectOp>(mask, dot_operand_value, zero);
      b.create<mlir::scf::YieldOp>(masked_dot_operand);
    }
    // else ...
    {
      b.setInsertionPointToStart(if_op.elseBlock());
      b.create<mlir::scf::YieldOp>(dot_operand_value);
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
    EmitterLocOpBuilder b, TensorValue operand, int64_t contracting_dim_idx,
    DotOperandSide side, TensorValue counterpart_operand = nullptr) {
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
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo_dot,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, Value pid,
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
                      triton::GetDotAccumulatorType(b, dot));
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

  auto for_op = b.create<mlir::scf::ForOp>(
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_count),
      /*step=*/MakeIndex(b, 1), accumulator);

  if (block_level_parameters.is_warp_specialization_allowed) {
    for_op->setAttr("tt.warp_specialize", b.getBoolAttr(true));
  }

  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value ki = for_op.getInductionVar();
    Value computation_index = b.create<xla::ApplyIndexingOp>(
                                   ValueRange{pid, ki}, computation_index_map)
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
              *tiled_fusion_operand->called_computation(),
              block_level_parameters, fn, computation_index, values));
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
        triton::EmitSingleTileDot(b, dot, triton::DotOperands{lhs, rhs, acc}));
    b.create<mlir::scf::YieldOp>(acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  TF_ASSIGN_OR_RETURN(Type dot_output_type,
                      TritonType(b, dot.shape().element_type()));

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
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo_dot,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, Value pid,
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
  auto for_op = b.create<mlir::scf::ForOp>(
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_count),
      /*step=*/MakeIndex(b, 1), accumulator);
  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value ki = for_op.getInductionVar();
    Value computation_index = b.create<xla::ApplyIndexingOp>(
                                   ValueRange{pid, ki}, computation_index_map)
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
              *tiled_fusion_operand->called_computation(),
              block_level_parameters, fn, computation_index, values));
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
        triton::EmitSingleTileScaledDot(
            b, scaled_dot,
            triton::ScaledDotOperands{lhs, rhs, lhs_scale, rhs_scale, acc}));
    b.create<mlir::scf::YieldOp>(acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  TF_ASSIGN_OR_RETURN(Type dot_output_type,
                      TritonType(b, scaled_dot.shape().element_type()));

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
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_concatenate,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, Value pid,
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
  TF_ASSIGN_OR_RETURN(
      Type element_type,
      TritonType(b, tiled_concatenate.hlo()->shape().element_type()));
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
          b.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                  concatenate_dimension_offset, offset_limit);
      auto if_op = b.create<mlir::scf::IfOp>(mlir::TypeRange(result_type), cond,
                                             /*withElseRegion=*/true);

      // Propagate the result from the nested `if_op` if we were already within
      // an `if_op`.
      if (!if_ops.empty()) {
        b.create<mlir::scf::YieldOp>(if_op.getResult(0));
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
            *tiled_fusion_operand->called_computation(), block_level_parameters,
            fn, pid, values));
    CHECK_EQ(result.size(), 1);
    b.create<mlir::scf::YieldOp>(result.front());
  }

  b.setInsertionPointAfter(if_ops.front());

  return mlir::cast<TensorValue>(if_ops.front().getResult(0));
}

absl::StatusOr<TensorValue> EmitPad(
    EmitterLocOpBuilder b, const TiledHloInstruction& tiled_pad,
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
    TensorValue bcast = BroadcastInDims(b, range, padded_tile_sizes,
                                        {static_cast<int64_t>(dim_index)});

    // RHS for the compare is splat(pad_input_dim_size - tile_offset).
    Value tile_offset_i32 = Cast(b, tile_offset, i32_type);
    Value threshold = b.create<arith::SubIOp>(
        CreateConst(b, i32_type, pad_input_dim_size), tile_offset_i32);
    TensorValue threshold_splat = Splat(b, threshold, padded_tile_sizes);
    Value cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, bcast,
                                        threshold_splat);
    mask = mask ? b.create<arith::AndIOp>(mask, cmp) : cmp;
  }
  if (!mask) {
    return values[tiled_operand];
  }
  const TiledHloInstruction* padding_value = tiled_pad.operand(1);

  TensorValue pad_value_splat =
      Splat(b, values[padding_value], padded_tile_sizes);
  return mlir::cast<TensorValue>(
      b.create<arith::SelectOp>(mask, values[tiled_operand], pad_value_splat)
          .getResult());
}

absl::StatusOr<TensorValue> EmitTiledHloInstruction(
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, Value pid,
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

    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to type checking that we perform a conversion after
    // loading if the type of the loaded parameter does not match what is
    // expected.
    Type loaded_element_type = getElementTypeOrSelf(parameter.getType());
    TF_ASSIGN_OR_RETURN(Type expected_element_type,
                        TritonType(b, hlo->shape().element_type()));

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
    return EmitConcatenate(b, fusion, tiled_hlo, block_level_parameters, fn,
                           pid, values);
  }

  if (hlo->opcode() == HloOpcode::kPad) {
    return EmitPad(b, tiled_hlo, values, pid);
  }

  if (hlo->opcode() == HloOpcode::kDot) {
    return EmitDot(b, fusion, tiled_hlo, block_level_parameters, fn, pid,
                   values);
  }

  if (hlo->opcode() == HloOpcode::kScaledDot) {
    return EmitScaledDot(b, fusion, tiled_hlo, block_level_parameters, fn, pid,
                         values);
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
    return EmitCollective(b, fusion, tiled_hlo, block_level_parameters, fn, pid,
                          values);
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
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, Value pid,
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
        EmitTiledHloInstruction(b, fusion, *tiled_hlo, block_level_parameters,
                                fn, pid, values));
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

namespace ir_emitter_triton_internal {

absl::StatusOr<Tiling> TilingFromAnnotatedFusion(
    const HloFusionInstruction* fusion,
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const BlockLevelParameters& block_level_parameters) {
  Tiling::TileMapping tile_mapping;
  int64_t real_root_index = symbolic_tile_analysis.real_root_index();
  const HloInstruction* real_root =
      symbolic_tile_analysis.GetRoots()[real_root_index];

  for (const auto& [hlo, num_tiling_parameters] :
       symbolic_tile_analysis.GetTilingSpecification().parameter_mapping()) {
    // TODO(b/419026602): handle reductions.
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kScaledDot) {
      const HloInstruction* lhs = hlo->operand(0);
      // When encountering a `dot`, we always expect its operands to be nests.
      auto backend_config = lhs->backend_config<GpuBackendConfig>();
      if (!backend_config.ok() || !backend_config->fusion_backend_config()
                                       .has_block_level_fusion_config()) {
        return absl::FailedPreconditionError(
            absl::StrCat("No block_level_fusion_config in ", lhs->ToString()));
      }
      std::vector<int64_t> lhs_output_tile_sizes =
          BlockLevelParameters::FromBlockLevelFusionConfig(
              backend_config->fusion_backend_config()
                  .block_level_fusion_config())
              .output_tile_sizes.front();

      absl::InlinedVector<int64_t, 4> dot_tiling_parameters;
      dot_tiling_parameters.reserve(num_tiling_parameters);
      for (int64_t contracting_dim_id :
           hlo->dot_dimension_numbers().lhs_contracting_dimensions()) {
        if (contracting_dim_id >= lhs_output_tile_sizes.size()) {
          return absl::FailedPreconditionError(
              absl::StrCat("Output tile sizes index ", contracting_dim_id,
                           " is out of bounds for ", lhs->ToString()));
        }
        dot_tiling_parameters.push_back(
            lhs_output_tile_sizes[contracting_dim_id]);
      }

      tile_mapping[hlo] = dot_tiling_parameters;
    }

    // TODO(b/390559452): this should change for generalized multi-output
    // fusions.
    if (hlo == real_root) {
      if (real_root_index >= block_level_parameters.output_tile_sizes.size()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Output tile sizes index ", real_root_index,
            " is out of bounds for block level fusion config: ",
            block_level_parameters.ToBlockLevelFusionConfig().DebugString()));
      }
      absl::Span<const int64_t> output_tile_sizes =
          block_level_parameters.output_tile_sizes[real_root_index];
      tile_mapping[hlo].insert(tile_mapping[hlo].end(),
                               output_tile_sizes.begin(),
                               output_tile_sizes.end());
    }
  }

  return Tiling(std::move(tile_mapping));
}

}  // namespace ir_emitter_triton_internal

namespace {
using ::xla::gpu::ir_emitter_triton_internal::DumpTritonIR;

absl::Status EmitGeneric(
    mlir::OpBuilder builder,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
    const HloFusionInstruction* fusion, xtile::EntryFuncOp fn,
    const BlockLevelParameters& block_level_parameters,
    SymbolicExprContext* symbolic_expr_context) {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Emitting Triton IR for fusion\n"
            << ExtractInstructionIntoNewModule(*fusion)->ToString();
  }
  const HloComputation* computation = fusion->fused_instructions_computation();
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          *computation, symbolic_expr_context,
          emitter_specific_constraints_builder);

  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in EmitGeneric: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  const auto& symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(symbolic_tile_analysis_or);

  // TODO(b/421837868): unify the logic to extract tiling parameters with
  // `BlockLevelParameters`.
  TF_ASSIGN_OR_RETURN(
      Tiling tiling,
      ir_emitter_triton_internal::TilingFromAnnotatedFusion(
          fusion, symbolic_tile_analysis, block_level_parameters));

  // TODO(b/372454662): Decide which root to use. Currently, we only support
  // "simple" multi-output fusions that have just one root without users. This
  // root appears last in def-before-use order. We derive the tiling from this
  // root.
  const HloInstruction* root =
      symbolic_tile_analysis.GetSymbolicTiledHloComputation().back()->hlo();
  auto loc = mlir::NameLoc::get(builder.getStringAttr(root->name()));
  EmitterLocOpBuilder b(loc, builder,
                        root->GetModule()
                            ->config()
                            .debug_options()
                            .xla_gpu_unsupported_annotate_with_emitter_loc());
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
  TF_ASSIGN_OR_RETURN(
      auto results,
      EmitTiledComputation(b, fusion, tiled_hlo_computation,
                           block_level_parameters, fn, tile_id, values));

  for (auto [root, result, arg] :
       llvm::zip(tiled_hlo_computation.GetRoots(), results,
                 fn.getArguments().drop_front(computation->num_parameters()))) {
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

    b.create<xtile::InsertTileOp>(result, arg, tile_info.offsets(),
                                  tile_info.padded_tile_sizes(),
                                  tile_info.tile_strides());
  }

  return absl::OkStatus();
}

}  // namespace

void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context) {
  mlir_context.loadDialect<
      ttir::TritonDialect, ttir::gpu::TritonGPUDialect,
      mlir::arith::ArithDialect, mlir::affine::AffineDialect,
      mlir::LLVM::LLVMDialect, xla::XlaDialect, xla::gpu::XlaGpuDialect,
      ttir::xla::XlaTritonDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, xla::xtile::XTileDialect,
      mlir::NVVM::NVVMDialect, stablehlo::StablehloDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir_context.appendDialectRegistry(registry);
}

// Simplified copy of translateLLVMToLLVMIR which in addition takes
// path to libdevice directly as an argument.
absl::StatusOr<std::unique_ptr<llvm::Module>> TranslateLLVMToLLVMIR(
    llvm::LLVMContext* llvmContext, mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    return Internal("Failed to emit LLVM IR.");
  }
  // TODO: b/363203060 - Upstream Triton sets specific flags for the LLVM
  // optimizer to get best performance. Figure out if we can gain any of it by
  // propagating these flags to
  // xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc.
  return llvmModule;
}

absl::Status CreateInternalError(absl::string_view message,
                                 const HloFusionInstruction* fusion,
                                 mlir::ModuleOp triton_module) {
  std::string err;
  llvm::raw_string_ostream os(err);
  os << message << "\n";
  os << "fusion instruction: " << fusion->ToString() << "\n";
  os << "HLO module to reproduce:\n"
     << ExtractInstructionIntoNewModule(*fusion)->ToString();
  os << "triton_module>>>\n";
  triton_module->print(os, mlir::OpPrintingFlags().enableDebugInfo(true, true));
  os << "<<<triton_module\n";
  return absl::InternalError(err);
}

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
  auto layout = mtx::LayoutAttr::get(context, minor_to_major_attr);

  return mlir::MemRefType::get(shape.dimensions(), storage_type, layout);
}

absl::Status IsTritonSupportedFusion(const HloFusionInstruction& fusion,
                                     const se::DeviceDescription& device_info) {
  const HloComputation* computation = fusion.fused_instructions_computation();
  for (const HloInstruction* hlo : computation->instructions()) {
    // Skip generating nested fusions, they are emitted by their consumer.
    if (hlo->parent()->IsFusionComputation() &&
        hlo->opcode() == HloOpcode::kFusion) {
      if (hlo->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_experimental_scaled_dot_with_triton()) {
        continue;
      }
      CodegenDecision decision = IsTritonSupportedInstruction(
          *hlo, device_info.gpu_compute_capability());
      if (!decision.CanFuse()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Fusion ", hlo->ToString(),
                         " is not supported: ", decision.Explain()));
      }
      VLOG(1) << "Skipping nested fusion: " << hlo->ToString();
      continue;
    }

    if (hlo->opcode() == HloOpcode::kPad) {
      if (!IsTritonSupportedInstruction(*hlo,
                                        device_info.gpu_compute_capability())) {
        return absl::FailedPreconditionError(
            absl::StrCat("Pad is not supported: ", hlo->ToString()));
      }
    }

    if (hlo->opcode() == HloOpcode::kReduce && hlo->dimensions().size() != 1) {
      return absl::FailedPreconditionError(
          absl::StrCat("Reduction with only a single dimension is supported: ",
                       hlo->ToString()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    SymbolicExprContext& symbolic_expr_context) {
  TF_RETURN_IF_ERROR(IsTritonSupportedFusion(*fusion, device_info));

  // TODO: b/451959933 - Use reference or check pointer.
  mlir::MLIRContext& mlir_context = *symbolic_expr_context.GetMLIRContext();

  TF_ASSIGN_OR_RETURN(
      auto triton_module,
      ir_emitter_triton_internal::EmitXTileModule(
          fn_name, TritonEmitterConstraints::GetBuilder(device_info), fusion,
          block_level_parameters, symbolic_expr_context,
          ir_emitter_triton_internal::LegacyMatmulEmitter(device_info)));

  const HloComputation* hlo_computation =
      fusion->fused_instructions_computation();

  const auto debug_options = fusion->GetModule()->config().debug_options();

  if (DumpingEnabledForHloModule(*hlo_computation->parent())) {
    auto suffix = absl::StrCat(fusion->name(), ".before_validation.ttir.txt");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", suffix,
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
    std::string fusion_suffix = absl::StrCat(fusion->name(), ".hlo");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", fusion_suffix,
        ExtractInstructionIntoNewModule(*fusion)->ToString());
  }

  TF_RETURN_IF_ERROR(ir_emitter_triton_internal::LowerXTileToTriton(
      triton_module.get(), mlir_context, *fusion, device_info));

  VLOG(6) << DumpTritonIR(triton_module.get(),
                          fusion->GetModule()
                              ->config()
                              .debug_options()
                              .xla_gpu_unsupported_annotate_with_emitter_loc());
  if (DumpingEnabledForHloModule(*hlo_computation->parent())) {
    std::string suffix = absl::StrCat(fusion->name(), ".ttir.txt");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", suffix,
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
  }

  return std::move(triton_module);
}

absl::Status CheckAtLeastAmpere(const se::GpuComputeCapability& gpu_cc) {
  if (auto* cuda_cc = gpu_cc.cuda_compute_capability();
      cuda_cc != nullptr && !cuda_cc->IsAtLeastAmpere()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Triton support is only enabled for Ampere GPUs (compute ",
                     "capability 8.0) and up, but got compute capability ",
                     cuda_cc->ToString(), "."));
  }
  return absl::OkStatus();
}

absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& gpu_cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    llvm::Module* llvm_module, SymbolicExprContext& symbolic_expr_context) {
  mlir::MLIRContext& mlir_context = *symbolic_expr_context.GetMLIRContext();
  TF_RETURN_IF_ERROR(CheckAtLeastAmpere(gpu_cc));

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> triton_module,
      CreateTritonModule(fn_name, fusion, device_info, block_level_parameters,
                         symbolic_expr_context));

  VLOG(3) << fusion->ToString(HloPrintOptions::ShortParsable());
  VLOG(3) << fusion->fused_instructions_computation()->ToString(
      HloPrintOptions::ShortParsable());

  // Compile Triton kernel to LLVM.
  const HloModule* hlo_module = fusion->GetModule();
  return CompileTritonToLLVM(fn_name, *hlo_module, device_info,
                             block_level_parameters, triton_module.get(),
                             llvm_module, mlir_context,
                             /*is_xla_fusion=*/true);
}

absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
    mlir::MLIRContext& mlir_context, bool is_xla_fusion, bool emit_kernel) {
  const auto& gpu_cc = device_info.gpu_compute_capability();
  TF_RETURN_IF_ERROR(CheckAtLeastAmpere(gpu_cc));
  std::string arch_name = gpu_cc.ToString();

  const HloModuleConfig& hlo_config = hlo_module.config();

  bool should_verify =
      (hlo_config.debug_options().xla_gpu_llvm_verification_level() >= 1);
#ifndef NDEBUG
  should_verify = true;
#endif

  bool should_dump_mlir_passes =
      hlo_config.debug_options().xla_enable_dumping() &&
      DumpingEnabledForHloModule(hlo_module) &&
      DumpingEnabledForHloPass("triton-fusion-emitter",
                               hlo_config.debug_options());

  mlir::PassManager pm(&mlir_context);
  pm.enableVerifier(should_verify);

  std::optional<llvm::raw_fd_ostream> log_stream;
  if (should_dump_mlir_passes) {
    std::string outputs_dir = hlo_config.debug_options().xla_dump_to();
    if (outputs_dir == "sponge") {
      if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
        LOG(ERROR) << "Failed to get test undeclared outputs dir. Lets skip "
                      "dumping triton passes.";
        outputs_dir = "";
      }
    }
    if (!outputs_dir.empty()) {
      const std::string basename =
          absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module.name())),
                       ".", kernel_name, ".triton-passes.log");
      std::string path = tsl::io::JoinPath(outputs_dir, basename);
      std::error_code err;
      log_stream.emplace(path, err, llvm::sys::fs::OF_None);
      if (err) {
        log_stream.reset();
        LOG(ERROR) << "Failed to dump triton passes to " << path << ": "
                   << err.message();
      } else {
        pm.getContext()->disableMultithreading();
        auto print_always = [](mlir::Pass*, mlir::Operation*) { return true; };
        pm.enableIRPrinting(/*shouldPrintBeforePass=*/print_always,
                            /*shouldPrintAfterPass=*/print_always,
                            /*printModuleScope=*/true,
                            /*printAfterOnlyOnChange=*/false,
                            /*printAfterOnlyOnFailure=*/true, *log_stream);
      }
    } else {
      LOG(ERROR)
          << "--xla_dump_hlo_pass_re=triton-fusion-emitter is set, but neither "
          << "the environment variable TEST_UNDECLARED_OUTPUTS_DIR nor the "
          << "flag --xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  CreateTritonXlaPipeline(&pm, gpu_cc, /*rewrite_int4=*/is_xla_fusion,
                          block_level_parameters.is_tma_allowed,
                          block_level_parameters.num_stages);

  int num_warps = block_level_parameters.num_warps;
  int num_ctas = block_level_parameters.num_ctas;
  int num_stages = block_level_parameters.num_stages;
  if (num_warps <= 0 || num_ctas <= 0 || num_stages <= 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "(num_warps, num_ctas, num_stages) must be positive, but got: (",
        num_warps, ", ", num_ctas, ", ", num_stages, ")"));
  }
  mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
  CreateTritonPipeline(&pm, gpu_cc, num_warps, num_ctas, num_stages,
                       cluster_info);

  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(mlir::triton::xla::CreateGeneralizeKernelSignaturePass());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  if (failed(pm.run(triton_module))) {
    return Internal("Failed to compile Triton kernel.");
  }

  const int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (shared_mem_bytes > device_info.shared_memory_per_block_optin()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Shared memory size limit exceeded: requested %d, available: %d",
        shared_mem_bytes, device_info.shared_memory_per_block_optin()));
  }

  if (auto* cuda_cc = gpu_cc.cuda_compute_capability();
      cuda_cc != nullptr && cuda_cc->IsBlackwell()) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory
    constexpr int kTensorMemoryColumns = 512;
    const int tensor_mem_columns =
        triton_module
            ->getAttrOfType<mlir::IntegerAttr>("ttg.tensor_memory_size")
            .getInt();
    if (tensor_mem_columns > 0) {
      VLOG(2) << "Tensor memory usage: " << tensor_mem_columns << " columns";
    }
    if (tensor_mem_columns > kTensorMemoryColumns) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Tensor memory size limit exceeded: requested %d, available: %d",
          tensor_mem_columns, kTensorMemoryColumns));
    }
  }

  std::vector<llvm::Metadata*> captured_nvvm_annotations;
  if (emit_kernel) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<llvm::Module> ll_triton_module,
        TranslateLLVMToLLVMIR(&llvm_module->getContext(), triton_module));

    XLA_VLOG_LINES(5, llvm_ir::DumpToString(ll_triton_module.get()));
    if (should_verify) {
      VerifyModule(*ll_triton_module);
    }

    // Integrate LLVM matmul kernel into XLA's LLVM module.
    captured_nvvm_annotations =
        xgt::ExtractNvvmAnnotations(ll_triton_module.get());
    ll_triton_module->setDataLayout(llvm_module->getDataLayout());
    ll_triton_module->setTargetTriple(llvm_module->getTargetTriple());
    // Use override flag because libdevice functions can be present in both.
    TF_RET_CHECK(
        !llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module),
                                   llvm::Linker::Flags::OverrideFromSrc));

    XLA_VLOG_LINES(5, llvm_ir::DumpToString(llvm_module));
    if (should_verify) {
      VerifyModule(*llvm_module);
    }
  }

  // `cluster_info` must be read after pm.run().
  std::optional<se::ClusterDim> cluster_dim;
  if (block_level_parameters.num_ctas > 1) {
    VLOG(3) << "num_ctas: " << block_level_parameters.num_ctas
            << ", cluster_info: " << cluster_info.clusterDimX << ","
            << cluster_info.clusterDimY << "," << cluster_info.clusterDimZ;
    if (cluster_info.clusterDimX > 1 || cluster_info.clusterDimY > 1 ||
        cluster_info.clusterDimZ > 1) {
      cluster_dim =
          se::ClusterDim(cluster_info.clusterDimX, cluster_info.clusterDimY,
                         cluster_info.clusterDimZ);
    }
  } else {
    TF_RET_CHECK(cluster_info.clusterDimX == 1 &&
                 cluster_info.clusterDimY == 1 &&
                 cluster_info.clusterDimZ == 1);
  }

  SmallVector<mlir::LLVM::LLVMFuncOp> func_ops;
  for (auto func : triton_module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    // Custom calls will also match to LLVMFuncOp, so we are only interested in
    // the entry function.
    if (func.getName().str() == kernel_name) {
      func_ops.push_back(func);
    }
  }
  CHECK_EQ(func_ops.size(), 1)
      << "Expected a single LLVMFuncOp in the module for the entry function.";
  mlir::LLVM::LLVMFuncOp func_op = func_ops[0];

  TF_ASSIGN_OR_RETURN(se::ThreadDim thread_dims,
                      xgt::ExtractThreadDims(triton_module, func_op));
  TF_ASSIGN_OR_RETURN(stream_executor::gpu::TmaMetadata tma_metadata,
                      xgt::ExtractTmaMetadata(func_op));

  // Propagate the following extracted information from the Triton module:
  // - TMA metadata.
  // - Total threads per block. Computed from module attributes.
  // - Captured NVVM annotations.
  TritonWrapperResult result = {
      shared_mem_bytes,          cluster_dim, tma_metadata, thread_dims,
      captured_nvvm_annotations,
  };
  return result;
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  if (device_info.gpu_compute_capability().IsCuda()) {
    return nvptx::LibDevicePath(
        hlo_config.debug_options().xla_gpu_cuda_data_dir());
  }
  return "";
}

namespace ir_emitter_triton_internal {

absl::Status LegacyMatmulEmitter::Emit(
    EmitterLocOpBuilder& b, const HloFusionInstruction* fusion,
    xtile::EntryFuncOp& fn,
    const BlockLevelParameters& block_level_parameters) {
  std::string libdevice_path =
      GetLibdevicePath(fusion->GetModule()->config(), device_info_);
  TF_RETURN_IF_ERROR(EmitMatMul(b, libdevice_path, device_info_, fusion, fn,
                                block_level_parameters));
  return absl::OkStatus();
}

// TODO(b/447133106): Contrary to the name, this function still does a lot of
// triton specific things. It should be migrated to use non-triton specific
// utilities.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitXTileModule(
    absl::string_view fn_name,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
    const HloFusionInstruction* fusion,
    const BlockLevelParameters& block_level_parameters,
    SymbolicExprContext& symbolic_expr_context,
    std::optional<LegacyMatmulEmitter> legacy_matmul_emitter) {
  mlir::MLIRContext& mlir_context = *symbolic_expr_context.GetMLIRContext();
  LoadMlirDialectsForTriton(mlir_context);
  const auto debug_options = fusion->GetModule()->config().debug_options();

  const HloComputation* hlo_computation =
      fusion->fused_instructions_computation();

  auto loc = mlir::NameLoc::get(
      mlir::StringAttr::get(&mlir_context, hlo_computation->name()));
  EmitterLocOpBuilder b(
      loc, &mlir_context,
      debug_options.xla_gpu_unsupported_annotate_with_emitter_loc());

  mlir::OwningOpRef<mlir::ModuleOp> triton_module =
      llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(triton_module->getBody());

  auto backend_config =
      fusion->backend_config<GpuBackendConfig>()->fusion_backend_config();
  absl::string_view fusion_kind = backend_config.kind();

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
      TF_ASSIGN_OR_RETURN(ir_type, TritonType(b, type));
    }
    fn_arg_types.push_back(GetMemRefType(p->shape(), ir_type));
  }

  for (const auto& [index, shape] : ShapeUtil::GetLeafShapes(fusion->shape())) {
    TF_ASSIGN_OR_RETURN(Type triton_ty, TritonType(b, shape.element_type()));
    fn_arg_types.push_back(GetMemRefType(shape, triton_ty));
  }

  // Add metadata arguments for collectives.
  // This is done after the input and output arguments but before the tile
  // index.
  int32_t num_metadata_arguments = 0;
  if (fusion_kind == kTritonCollectiveFusionKind) {
    TF_ASSIGN_OR_RETURN(
        num_metadata_arguments,
        AddCollectiveMetadataArguments(fn_arg_types, b, hlo_computation));
  }
  // Metadata arguments are opaque to the tiling infra.
  llvm::SmallVector<mlir::NamedAttribute> named_attributes{b.getNamedAttr(
      "num_opaque_args", b.getI32IntegerAttr(num_metadata_arguments))};

  auto fn =
      b.create<xtile::EntryFuncOp>(fn_name, fn_arg_types, named_attributes, {});

  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  if (fusion_kind == kTritonGemmFusionKind) {
    if (absl::c_contains(
            fusion->GetModule()
                ->config()
                .debug_options()
                .xla_gpu_unsupported_generic_triton_emitter_features(),
            DebugOptions::GENERIC_TRITON_EMITTER_DISABLE_LEGACY_GEMM)) {
      return Internal("Legacy GEMM emitter is disabled.");
    }
    CHECK(legacy_matmul_emitter.has_value())
        << "emit_legacy_matmul_fn is not set";
    TF_RETURN_IF_ERROR(
        legacy_matmul_emitter->Emit(b, fusion, fn, block_level_parameters));
  } else if (fusion_kind == kTritonFusionKind ||
             fusion_kind == kTritonNestedGemmFusionKind ||
             fusion_kind == kTritonScaledDotFusionKind ||
             fusion_kind == kTritonCollectiveFusionKind) {
    TF_RETURN_IF_ERROR(EmitGeneric(b, emitter_specific_constraints_builder,
                                   fusion, fn, block_level_parameters,
                                   &symbolic_expr_context));
  } else {
    return Internal("Unsupported fusion kind: %s", fusion_kind);
  }

  b.create<xtile::EntryFuncReturnOp>();

  return triton_module;
}

absl::Status LowerXTileToTriton(mlir::ModuleOp xtile_dialect_module,
                                mlir::MLIRContext& mlir_context,
                                const HloFusionInstruction& fusion,
                                const se::DeviceDescription& device_info) {
  {
    auto backend_config =
        fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
    absl::string_view fusion_kind = backend_config.kind();

    // Convert xTile ops to Triton ops.
    mlir::PassManager pm(&mlir_context);
    // Disable verifier because the Triton code may be invalid due to the
    // unsupported types.
    pm.enableVerifier(/*enabled=*/false);
    // The legacy emitter supports 0D tensors so we would get inconsistent
    // results if we try to rewrite them.
    if (fusion_kind != kTritonGemmFusionKind) {
      pm.addPass(xtile::createConvertElementwise0DTensorToScalarPass());
    }
    pm.addPass(mlir::triton::xla::CreateTensorLowerToTritonPass());
    pm.addPass(mlir::triton::xla::CreateStableHLOLowerToTritonPass());
    pm.addPass(mlir::triton::xla::CreateXTileLowerToTritonPass());

    std::string libdevice_path =
        GetLibdevicePath(fusion.GetModule()->config(), device_info);
    absl::string_view triple = device_info.gpu_compute_capability().IsRocm()
                                   ? "amdgcn-unknown-unknown"
                                   : "nvptx64-unknown-unknown";
    pm.addPass(mlir::triton::xla::CreateTritonXLAMathToLibdevicePass(
        libdevice_path, triple));

    tsl::StatusScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    if (absl::Status status =
            diagnostic_handler.consumeStatus(pm.run(xtile_dialect_module));
        !status.ok()) {
      return CreateInternalError(
          "Failed to lower from shared dialect to Triton.", &fusion,
          xtile_dialect_module);
    }
  }

  if (fusion.GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_scaled_dot_with_triton()) {
    // Convert unsupported types before verification.
    mlir::PassManager pm(&mlir_context);
    pm.addPass(mlir::triton::xla::CreateTritonXLAConvertUnsupportedTypesPass());
    if (mlir::failed(pm.run(xtile_dialect_module))) {
      return CreateInternalError(
          "Failed to fix unsupported types in Triton module for fusion:",
          &fusion, xtile_dialect_module);
    }
  }

  if (mlir::failed(mlir::verify(xtile_dialect_module))) {
    return CreateInternalError("Failed to verify Triton module for fusion:",
                               &fusion, xtile_dialect_module);
  }
  mlir::PassManager pm(&mlir_context);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(xtile_dialect_module))) {
    return CreateInternalError("Failed to create Triton module for fusion:",
                               &fusion, xtile_dialect_module);
  }

  return absl::OkStatus();
}

}  // namespace ir_emitter_triton_internal

}  // namespace gpu
}  // namespace xla
