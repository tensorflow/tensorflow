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

#include <cstddef>
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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/tma_utils.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/analysis/indexing_analysis.h"
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
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
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
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace arith = ::mlir::arith;
namespace ttir = ::mlir::triton;
namespace mtx = ::mlir::triton::xla;

using ::llvm::SmallVector;
using ::mlir::ArrayRef;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

using ::xla::gpu::triton::Cast;
using ::xla::gpu::triton::CreateConst;
using ::xla::gpu::triton::EmitConstant;
using ::xla::gpu::triton::EmitElementwise;
using ::xla::gpu::triton::GetPaddedTileSizes;
using ::xla::gpu::triton::ScalarOrTensor;
using ::xla::gpu::triton::StorageType;
using ::xla::gpu::triton::TritonType;

namespace {

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

ScalarOrTensor Broadcast(EmitterLocOpBuilder& b, TensorValue value,
                         ArrayRef<int64_t> shape) {
  return ScalarOrTensor(
      b.create<ttir::BroadcastOp>(value.getType().clone(shape), value));
}

ScalarOrTensor Range(EmitterLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return ScalarOrTensor(b.create<ttir::MakeRangeOp>(type, 0, limit));
}

ScalarOrTensor EmitParameterExtract(EmitterLocOpBuilder& b,
                                    mlir::triton::xla::TileOp tile_op) {
  // For a pointer to a scalar or a zero-dimensional tensor, load the base
  // pointer directly. This shortcut is necessary because Triton does not
  // support 0-D tensors. Looking for the defining make_tensor_ptr op is
  // sufficient because pointers to 0-D tensors are never modified by e.g.
  // `tt.advance`.

  auto tiled_tensor_type =
      mlir::dyn_cast<mtx::TiledTensorType>(tile_op.getResult().getType());
  CHECK(tiled_tensor_type) << "Expected a TiledTensorType\n";

  if (tiled_tensor_type.getTileShape().empty()) {
    return ScalarOrTensor(
        b.create<mlir::tensor::ExtractOp>(tile_op.getTensor(), {}));
  }

  SmallVector<Value> offsets(
      tile_op.getTiledTensor().getType().getTileShape().size(),
      CreateConst(b, b.getIndexType(), 0).UnwrapScalar());

  return ScalarOrTensor(b.create<mtx::ExtractOp>(
      mlir::RankedTensorType::get(
          tiled_tensor_type.getTileShape(),
          StorageType(b, tiled_tensor_type.getElementType())),
      tile_op.getResult(), offsets));
}

absl::StatusOr<ScalarOrTensor> EmitScope(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const TritonFusionAnalysis* analysis,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, ScalarOrTensor>& values);

absl::StatusOr<ScalarOrTensor> EmitReduce(
    EmitterLocOpBuilder& b, const TiledHloInstruction& tiled_hlo_reduce,
    absl::flat_hash_map<const TiledHloInstruction*, ScalarOrTensor>& values,
    absl::string_view libdevice_path,
    const se::DeviceDescription& device_info) {
  // At the moment, we should only emit a full reduction over a single
  // dimension using a scalar as a neutral element.
  const HloReduceInstruction& hlo_reduce =
      *::xla::Cast<HloReduceInstruction>(tiled_hlo_reduce.hlo());
  ScalarOrTensor input = values[tiled_hlo_reduce.operand(0)];
  llvm::ArrayRef<int64_t> input_shape =
      mlir::cast<ShapedType>(input.getType()).getShape();
  absl::Span<const int64_t> source_tensor_shape =
      hlo_reduce.operand(0)->shape().dimensions();

  int reduction_dimension = hlo_reduce.dimensions().front();

  // Since every shape is padded to a power of 2 in Triton, the input tile may
  // be padded with arbitrary values. These values could affect the result of
  // the reduction, so we need to mask them away. Luckily, we have a monoid
  // structure (element_type, hlo_reduce.to_apply(), hlo_reduce.operand(1))---
  // up to floating-point inaccuracies. Masking the input using
  // hlo_reduce.operand(1) is thus always the right choice to ensure that the
  // reduction is computed correctly, since it is the neutral value with
  // regards to the reducer.
  int64_t source_tensor_reduction_dimension_size =
      source_tensor_shape[reduction_dimension];
  int64_t input_reduction_dimension_size = input_shape[reduction_dimension];
  if (input_reduction_dimension_size !=
      source_tensor_reduction_dimension_size) {
    Value range = Range(b, input_reduction_dimension_size).UnwrapUnsafe();
    // Triton's broadcast requires that the rank of the source and broadcasted
    // result are equal.
    for (int i = 0; i < input_shape.size() - 1; i++) {
      if (i < reduction_dimension) {
        range = b.create<ttir::ExpandDimsOp>(range, /*axis*/ 0);
      } else {
        range = b.create<ttir::ExpandDimsOp>(range, /*axis*/ i + 1);
      }
    }
    Value mask = Broadcast(b, mlir::cast<TensorValue>(range), input_shape)
                     .UnwrapUnsafe();
    ScalarOrTensor constant = CreateConst(
        b, b.getI32Type(), source_tensor_reduction_dimension_size, input_shape);
    mask = b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, mask,
                                   constant.UnwrapUnsafe());

    ScalarOrTensor neutral = values[tiled_hlo_reduce.operand(1)];
    // Triton's broadcast requires that the rank of the source and broadcasted
    // result are equal.
    if (neutral.IsScalar()) {
      neutral = Splat(b, neutral, input_shape);
    } else {
      for (int i = 0; i < input_shape.size(); i++) {
        neutral = ScalarOrTensor(
            b.create<ttir::ExpandDimsOp>(neutral.UnwrapUnsafe(), /*axis*/ 0));
      }
      neutral = Broadcast(b, mlir::cast<TensorValue>(neutral.UnwrapUnsafe()),
                          input_shape);
    }
    input = ScalarOrTensor(b.create<arith::SelectOp>(mask, input.UnwrapUnsafe(),
                                                     neutral.UnwrapUnsafe()));
  }

  ttir::ReduceOp reduction =
      b.create<ttir::ReduceOp>(input.UnwrapUnsafe(), reduction_dimension);
  {
    TF_ASSIGN_OR_RETURN(Type result_ty,
                        TritonType(b, hlo_reduce.shape().element_type()));
    mlir::Location loc = b.getLoc();
    mlir::Block* reducer = b.createBlock(&reduction->getRegion(0), {},
                                         {result_ty, result_ty}, {loc, loc});

    HloComputation* reduction_computation = hlo_reduce.to_apply();

    std::vector<const HloInstruction*> to_emit;
    absl::flat_hash_map<const HloInstruction*, ScalarOrTensor> region_values;
    for (const HloInstruction* instr :
         reduction_computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kParameter) {
        int parameter_number = instr->parameter_number();
        TF_RET_CHECK(parameter_number < 2);
        TF_RET_CHECK(region_values
                         .insert({instr, ScalarOrTensor(reducer->getArgument(
                                             parameter_number))})
                         .second);
      } else {
        to_emit.push_back(instr);
      }
    }

    TF_RET_CHECK(!to_emit.empty());

    b.setInsertionPointToStart(reducer);
    TF_ASSIGN_OR_RETURN(
        ScalarOrTensor result,
        EmitScope(b, libdevice_path, device_info, /*analysis=*/nullptr, to_emit,
                  region_values));
    b.create<ttir::ReduceReturnOp>(SmallVector<Value>({result.UnwrapUnsafe()}));
    b.setInsertionPointAfter(reduction);
  }

  return ScalarOrTensor(reduction.getResult().front());
}

// Emit code corresponding to a fusion instruction somehow nested within the
// initial Triton fusion. This can happen when we carry around auxiliary
// computations, e.g. with reduces. Since we are emitting a single Triton
// fusion, we simply flatten the fusion inside the computation.
//
// TODO(b/331413981): get rid of this special handling once this is solved.
absl::StatusOr<ScalarOrTensor> EmitNestedFusion(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction& fusion_instruction,
    absl::flat_hash_map<const HloInstruction*, ScalarOrTensor>& values) {
  // TODO(b/331402498): revisit the order of scope once we completely
  // deprecate Triton fusion analysis.
  const HloComputation* fusion_computation =
      fusion_instruction.fused_instructions_computation();

  absl::flat_hash_map<const HloInstruction*, ScalarOrTensor> region_values;

  std::vector<const HloInstruction*> to_emit;
  for (const HloInstruction* instr :
       fusion_computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      int64_t parameter_number = instr->parameter_number();
      auto it = values.find(fusion_instruction.operand(parameter_number));
      TF_RET_CHECK(it != values.end());
      TF_RET_CHECK(region_values.insert({instr, it->second}).second);
    } else {
      to_emit.push_back(instr);
    }
  }

  TF_RET_CHECK(to_emit.back() == fusion_computation->root_instruction());

  return EmitScope(b, libdevice_path, device_info, /*analysis=*/nullptr,
                   to_emit, region_values);
}

ScalarOrTensor EmitTiledBroadcast(
    EmitterLocOpBuilder& b, const TiledHloInstruction& tiled_broadcast,
    absl::flat_hash_map<const TiledHloInstruction*, ScalarOrTensor>& values) {
  const SmallVector<int64_t>& input_tile_shape =
      tiled_broadcast.operand(0)->tile_sizes();
  const SmallVector<int64_t>& output_tile_shape = tiled_broadcast.tile_sizes();

  if (input_tile_shape.empty() && output_tile_shape.empty()) {
    return values[tiled_broadcast.operand(0)];
  }
  CHECK(!output_tile_shape.empty());

  SmallVector<int64_t> padded_output_tile_shape =
      GetPaddedTileSizes(output_tile_shape);

  ScalarOrTensor input = values[tiled_broadcast.operand(0)];
  // Handle the 0d special case.
  if (input.IsScalar()) {
    return Splat(b, input, padded_output_tile_shape);
  }

  Value expanded_input = input.UnwrapTensor();

  // Returns true if `dim_id` is broadcasted.
  auto is_broadcasted_dim = [&](int64_t dim_id) {
    return !llvm::is_contained(tiled_broadcast.hlo()->dimensions(), dim_id);
  };

  // The loop below iterates over output dimensions and tracks matching dims in
  // input_tile_shape and expended_input value.
  // `input_dim_id != expanded_input_dim_id`, because size-1 dims are present in
  // the input tile shape, but not in the MLIR value. Triton doesn't like size-1
  // dims, so they are inserted only for dimensions that will be broadcasted.
  int64_t input_dim_id = 0;
  int64_t expanded_input_dim_id = 0;
  for (size_t output_dim_id = 0; output_dim_id < output_tile_shape.size();
       ++output_dim_id) {
    if (is_broadcasted_dim(output_dim_id)) {
      // Expand dim for broadcast.
      expanded_input =
          b.create<ttir::ExpandDimsOp>(expanded_input, expanded_input_dim_id);
      ++expanded_input_dim_id;
    } else {
      // The dim is not broadcasted. Validate that it's equal in the input and
      // output tile.
      CHECK_EQ(input_tile_shape[input_dim_id],
               output_tile_shape[output_dim_id]);
      ++input_dim_id;
      ++expanded_input_dim_id;
    }
  }

  return Broadcast(b, mlir::cast<TensorValue>(expanded_input),
                   padded_output_tile_shape);
}

absl::StatusOr<ScalarOrTensor> EmitTiledIota(
    EmitterLocOpBuilder& b, ValueRange tile_multi_index,
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

  auto iota_dim_offset = b.create<arith::IndexCastUIOp>(
      b.getI32Type(),
      emitters::ApplyIndexing(tile_offsets_indexing, /*dims=*/tile_multi_index,
                              /*symbols=*/{}, b)[iota_dim]);

  // First, stride as needed between the iota components.
  Value range = b.create<arith::MulIOp>(
      Range(b, padded_tile_sizes[iota_dim]).UnwrapTensor(),
      Splat(b,
            CreateConst(b, b.getI32Type(), tiled_iota.tile_strides()[iota_dim]),
            padded_tile_sizes[iota_dim])
          .UnwrapTensor());

  // Then, add the base offset to the iota components.
  range = b.create<arith::AddIOp>(
      range,
      Splat(b, ScalarOrTensor(iota_dim_offset), padded_tile_sizes[iota_dim])
          .UnwrapTensor());

  // Cast the result to the targeted type.
  TF_ASSIGN_OR_RETURN(Type iota_element_type,
                      TritonType(b, hlo_iota->shape().element_type()));

  range = Cast(b, range, iota_element_type);

  // And finally, produce a broadcast along the non-iota dimensions in order to
  // produce the whole iota tile.
  for (int i = 0; i < padded_tile_sizes.size() - 1; i++) {
    if (i < iota_dim) {
      range = b.create<ttir::ExpandDimsOp>(range, /*axis*/ 0);
    } else {
      range = b.create<ttir::ExpandDimsOp>(range, /*axis*/ i + 1);
    }
  }

  return Broadcast(b, mlir::cast<TensorValue>(range), padded_tile_sizes);
}

// Reshapes a non-0D tensor of shape [1, 1, 1, ...] to a scalar.
ScalarOrTensor ReshapeTensorToScalar(EmitterLocOpBuilder& b, Value input) {
  auto element_type = mlir::cast<ShapedType>(input.getType()).getElementType();

  // First, reshape to a 1D tensor if not already the case. This is needed
  // because triton::ReduceOp can only reduce 1 dimension at a time.
  auto single_dim_tensor = input;
  if (mlir::cast<ShapedType>(input.getType()).getRank() > 1) {
    Type output_tensor_type = mlir::RankedTensorType::get({1}, element_type);
    single_dim_tensor = b.create<ttir::ReshapeOp>(output_tensor_type, input,
                                                  /*allow_reorder=*/true);
  }

  // Second, reduce to a scalar.
  ttir::ReduceOp reduction =
      b.create<ttir::ReduceOp>(single_dim_tensor, /*axis*/ 0);

  mlir::Location loc = b.getLoc();
  mlir::Block* reducer = b.createBlock(
      &reduction->getRegion(0), /*insertPt=*/{},
      /*argTypes=*/{element_type, element_type}, /*locs=*/{loc, loc});

  b.setInsertionPointToStart(reducer);
  Value result = mlir::isa<mlir::IntegerType>(element_type)
                     ? b.create<arith::AddIOp>(reducer->getArgument(0),
                                               reducer->getArgument(1))
                           .getResult()
                     : b.create<arith::AddFOp>(reducer->getArgument(0),
                                               reducer->getArgument(1))
                           .getResult();
  b.create<ttir::ReduceReturnOp>(SmallVector<Value>({result}));
  b.setInsertionPointAfter(reduction);

  return ScalarOrTensor(reduction.getResult().front());
}

absl::StatusOr<ScalarOrTensor> EmitTiledReshape(EmitterLocOpBuilder& b,
                                                ArrayRef<int64_t> tile_sizes,
                                                ScalarOrTensor input) {
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  if (input.IsScalar()) {
    if (tile_sizes.empty()) {
      // Nothing to do.
      return input;
    }
    // Convert the scalar to a tensor.
    return Splat(b, input, padded_tile_sizes);
  }

  // At this point we know that the input is a non-0D tensor.
  auto input_shaped_type = mlir::cast<ShapedType>(input.getType());

  // Handle the case of reshaping [1,1,1...] to a scalar.
  if (tile_sizes.empty()) {
    return ReshapeTensorToScalar(b, input.UnwrapTensor());
  }

  // At this point we know that neither the input nor the output are 0D tensors.
  Type output_tensor_type = mlir::RankedTensorType::get(
      padded_tile_sizes, input_shaped_type.getElementType());

  // Conservatively prevent Triton from reordering elements within the tile.
  // TODO(b/353637689): see if this restriction can be lifted.
  bool allow_reorder = false;
  auto reshape = b.create<ttir::ReshapeOp>(output_tensor_type,
                                           input.UnwrapUnsafe(), allow_reorder);
  return ScalarOrTensor(reshape.getResult());
}

Value EmitTiledTranspose(EmitterLocOpBuilder& b, ArrayRef<int64_t> tile_sizes,
                         SmallVector<int64_t> dimensions, Value input) {
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  Type input_element_type =
      mlir::cast<ShapedType>(input.getType()).getElementType();
  Type output_tensor_type =
      mlir::RankedTensorType::get(padded_tile_sizes, input_element_type);

  SmallVector<int32_t> order = llvm::to_vector_of<int32_t>(dimensions);

  return b.create<ttir::TransOp>(output_tensor_type, input, order);
}

absl::StatusOr<ScalarOrTensor> EmitTiledBitcast(
    EmitterLocOpBuilder& b, const TiledHloInstruction& tiled_bitcast,
    Value input) {
  // Any Bitcast is decomposable to a transpose+reshape+transpose.
  auto trt = ShapeUtil::DecomposeBitcastToTrt(
      tiled_bitcast.hlo()->operand(0)->shape(), tiled_bitcast.hlo()->shape());

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
      Permute(tiled_bitcast.operand(0)->tile_sizes(), trt.transpose1_dims);
  Value normalized_input =
      trt.IsTranspose1Identity()
          ? input
          : EmitTiledTranspose(b, transpose1_tile_sizes,
                               llvm::to_vector(trt.transpose1_dims), input);

  // Like the first transpose above, the tile sizes after the second transpose
  // are a permutation (according to transpose2_dims) of the tile sizes of
  // the reshape. Since we know the tile sizes of the final transpose and need
  // the tile sizes of the reshape, we compute the tile sizes backwards, taking
  // the inverse permutation.
  std::vector<int64_t> reshape_tile_sizes =
      PermuteInverse(tiled_bitcast.tile_sizes(), trt.transpose2_dims);
  Value normalized_reshape;
  if (ShapeUtil::Equal(trt.transpose1_shape, trt.reshape_shape)) {
    normalized_reshape = normalized_input;
  } else {
    TF_ASSIGN_OR_RETURN(auto reshape,
                        EmitTiledReshape(b, reshape_tile_sizes,
                                         ScalarOrTensor(normalized_input)));
    normalized_reshape = reshape.UnwrapUnsafe();
  }

  // The final transpose simply uses the tile sizes computed for the original
  // bitcast by the tiling analysis.
  return ScalarOrTensor{
      trt.IsTranspose2Identity()
          ? normalized_reshape
          : EmitTiledTranspose(b, tiled_bitcast.tile_sizes(),
                               llvm::to_vector(trt.transpose2_dims),
                               normalized_reshape)};
}

absl::StatusOr<std::vector<ScalarOrTensor>> EmitTiledComputation(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation, mlir::FunctionOpInterface fn,
    ValueRange tile_multi_index);

bool UseGenericTritonEmitterForGemms(const HloInstruction* hlo) {
  return hlo->GetModule()
      ->config()
      .debug_options()
      .xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms();
}

// Returns the number of iterations of the loop over the contracting
// dimension of matrix multiplication.
absl::StatusOr<int64_t> GetDotLoopIterationCount(
    const TiledHloInstruction& tiled_dot) {
  // As LHS (and RHS) must point to the outline fusion computation that is
  // tiled with contracting dimension, we can get the
  // - size from the shape of the operand
  // - tile size from the tiling of the nested fusion root
  // using the contracting dimension from the dot instruction.
  const HloDotInstruction& dot =
      *::xla::Cast<HloDotInstruction>(tiled_dot.hlo());
  const auto& dims = dot.dot_dimension_numbers();
  if (dims.lhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only one contracting dimension is supported, got ",
                     dims.lhs_contracting_dimensions_size()));
  }
  auto contracting_dim_idx = dims.lhs_contracting_dimensions(0);
  int64_t k = dot.operand(0)->shape().dimensions(contracting_dim_idx);

  const TiledHloFusionInstruction* tiled_hlo_fusion =
      static_cast<const TiledHloFusionInstruction*>(tiled_dot.operand(0));
  auto fusion_tile_sizes =
      tiled_hlo_fusion->called_computation()->GetRoots()[0]->tile_sizes();
  int64_t tile_k = fusion_tile_sizes[contracting_dim_idx];

  return CeilOfRatio(k, tile_k);
}

// TODO(b/393299275): unify with the logic in `EmitReduce`.
// Computes and applies a mask to the reduction dimension of the dot operand
// passed as a parameter.
//
// Note: we currently assume that contracting_dimension_tile_index is an i32
// scalar.
absl::StatusOr<Value> MaskDotOperand(EmitterLocOpBuilder& b,
                                     const TiledHloInstruction& dot_operand,
                                     Value dot_operand_value,
                                     Value contracting_dimension_tile_index,
                                     int contraction_dimension_index) {
  if (contracting_dimension_tile_index.getType() != b.getI32Type()) {
    return absl::FailedPreconditionError(
        "contracting_dimension_tile_index must be an i32 scalar");
  }

  llvm::ArrayRef<int64_t> tile_shape =
      mlir::cast<ShapedType>(dot_operand_value.getType()).getShape();

  int64_t rank = dot_operand.hlo()->shape().dimensions().size();
  int64_t contracting_dimension_size =
      dot_operand.hlo()->shape().dimensions(contraction_dimension_index);
  int64_t tile_size = tile_shape[contraction_dimension_index];

  if (contracting_dimension_size % tile_size != 0) {
    // When the contracting dimension is not divisible by the tile size, we
    // need to mask out the last tile. We do this with the following logic:
    //
    // indices =
    //   contracting_dimension_tile_index * tile_size + range(0, tile_size)
    // mask = indices < contracting_dimension_size
    // operand = select(broadcast(mask, operand.shape), operand, 0)
    Value range = Range(b, tile_size).UnwrapTensor();
    Value tile_size_value =
        CreateConst(b, b.getI32Type(), tile_size, {}).UnwrapScalar();
    Value tile_offset = b.create<arith::MulIOp>(
        contracting_dimension_tile_index, tile_size_value);
    Value broadcasted_tile_offset =
        Splat(b, ScalarOrTensor(tile_offset), {tile_size}).UnwrapTensor();
    Value indices = b.create<arith::AddIOp>(range, broadcasted_tile_offset);

    Value boundary =
        CreateConst(b, b.getI32Type(), contracting_dimension_size, {tile_size})
            .UnwrapTensor();

    Value mask =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, indices, boundary);

    // Triton's broadcast requires that the rank of the source and broadcasted
    // result are equal.
    for (int i = 0; i < rank - 1; i++) {
      int axis = (i < contraction_dimension_index) ? 0 : i + 1;
      mask = b.create<ttir::ExpandDimsOp>(mask, axis);
    }
    mask =
        Broadcast(b, mlir::cast<TensorValue>(mask), tile_shape).UnwrapTensor();

    TF_ASSIGN_OR_RETURN(
        auto element_type,
        TritonType(b, dot_operand.hlo()->shape().element_type()));

    ScalarOrTensor zero = CreateConst(b, element_type, 0.0f, tile_shape);

    return b.create<arith::SelectOp>(mask, dot_operand_value,
                                     zero.UnwrapTensor());
  }

  return dot_operand_value;
}

// Computes the base pointer offset for the given tile multi-index and hlo shape
// taking into account the physical layout of the hlo buffer.
absl::StatusOr<SmallVector<Value>> ComputeBasePtrOffset(
    EmitterLocOpBuilder& b, ValueRange tile_multi_index,
    const TiledHloInstruction& tiled_hlo) {
  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_hlo.tile_offsets_indexing());

  return emitters::ApplyIndexing(tile_offsets_indexing,
                                 /*dims=*/tile_multi_index,
                                 /*symbols=*/{}, b);
}

// Returns `shape` without all its unit dimensions, as well as the index of the
// remaining dimensions in the original `shape`.
std::pair<SmallVector<int64_t>, SmallVector<int64_t>> CollapseUnitDims(
    llvm::ArrayRef<int64_t> shape) {
  SmallVector<int64_t> shape_without_unit_dims;
  SmallVector<int64_t> non_unit_dims_indices;
  for (auto [i, size] : llvm::enumerate(shape)) {
    if (size != 1) {
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
// Returns an error if canonicalization is not possible.
absl::StatusOr<Value> CanonicalizeDotOperand(EmitterLocOpBuilder& b,
                                             Value operand,
                                             int64_t contracting_dim_idx,
                                             DotOperandSide side) {
  llvm::ArrayRef<int64_t> shape =
      mlir::cast<ShapedType>(operand.getType()).getShape();
  auto [shape_without_unit_dims, non_unit_dims_indices] =
      CollapseUnitDims(shape);

  if (shape_without_unit_dims.size() != 2) {
    return absl::FailedPreconditionError(
        "Expected dot operand tile to have exactly two non-unit tile sizes");
  }

  if (shape.size() != shape_without_unit_dims.size()) {
    TF_ASSIGN_OR_RETURN(
        ScalarOrTensor wrapped_operand,
        EmitTiledReshape(b, shape_without_unit_dims, ScalarOrTensor(operand)));
    operand = wrapped_operand.UnwrapTensor();
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

absl::StatusOr<ScalarOrTensor> EmitDot(EmitterLocOpBuilder& b,
                                       absl::string_view libdevice_path,
                                       const se::DeviceDescription& device_info,
                                       const HloFusionInstruction* fusion,
                                       const TiledHloInstruction& tiled_hlo_dot,
                                       mlir::FunctionOpInterface fn,
                                       ValueRange tile_multi_index) {
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
  if (dot.sparse_operands() > 0) {
    return absl::UnimplementedError("Sparse configuration is not supported");
  }
  if (!absl::c_all_of(tiled_hlo_dot.operands(),
                      [](const TiledHloInstruction* operand) {
                        return operand->hlo()->opcode() == HloOpcode::kFusion;
                      })) {
    return absl::FailedPreconditionError("Expected dot operands to be fusions");
  }

  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_hlo_dot.tile_sizes());

  SmallVector<int64_t, 2> padded_tile_sizes_no_unit_dims =
      CollapseUnitDims(padded_tile_sizes).first;

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
  Value accumulator =
      CreateConst(b, accumulator_type, 0.0f, padded_tile_sizes_no_unit_dims)
          .UnwrapTensor();

  auto ci64 = [&](int64_t value) -> Value {
    return b.create<arith::ConstantOp>(b.getIntegerAttr(b.getI64Type(), value));
  };
  TF_ASSIGN_OR_RETURN(int64_t loop_iteration_count,
                      GetDotLoopIterationCount(tiled_hlo_dot));
  auto for_op = b.create<mlir::scf::ForOp>(
      /*lowerBound=*/ci64(0), /*upperBound=*/ci64(loop_iteration_count),
      /*step=*/ci64(1), SmallVector<Value>{accumulator});
  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    SmallVector<TensorValue> dot_args;
    // Nested fusions are tiled with indexing map
    // (tile multi-index.., loop index) -> ....
    SmallVector<Value> computation_index(tile_multi_index);
    Value ki = for_op.getInductionVar();
    const Value ki_index = b.create<arith::IndexCastUIOp>(b.getIndexType(), ki);
    computation_index.push_back(ki_index);
    for (const TiledHloInstruction* operand : tiled_hlo_dot.operands()) {
      VLOG(3) << "Emitting dot operand: " << operand->ToString();
      const TiledHloFusionInstruction* tiled_fusion_operand =
          static_cast<const TiledHloFusionInstruction*>(operand);
      TF_ASSIGN_OR_RETURN(
          std::vector<ScalarOrTensor> result,
          EmitTiledComputation(
              b, libdevice_path, device_info,
              ::xla::Cast<HloFusionInstruction>(tiled_fusion_operand->hlo()),
              *tiled_fusion_operand->called_computation(), fn,
              computation_index));
      if (result.size() != 1) {
        return absl::InternalError(absl::StrCat(
            "Expected nested fusion computation to emit a single value, got ",
            result.size()));
      }
      dot_args.push_back(result.front().UnwrapTensor());
    }
    Value acc = for_op.getRegionIterArgs().front();
    int64_t lhs_contracting_dim_idx =
        dot.dot_dimension_numbers().lhs_contracting_dimensions(0);

    int64_t rhs_contracting_dim_idx =
        dot.dot_dimension_numbers().rhs_contracting_dimensions(0);

    // TODO(b/393299275): masking is only necessary during the last iteration of
    // the loop. We should evaluate whether adding a conditional mask helps or
    // hinders performance for Triton.
    Value ki_i32 = b.create<arith::TruncIOp>(b.getI32Type(), ki);
    TF_ASSIGN_OR_RETURN(
        Value lhs, MaskDotOperand(b, *tiled_hlo_dot.operand(0), dot_args[0],
                                  ki_i32, lhs_contracting_dim_idx));

    TF_ASSIGN_OR_RETURN(
        Value rhs, MaskDotOperand(b, *tiled_hlo_dot.operand(1), dot_args[1],
                                  ki_i32, rhs_contracting_dim_idx));

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

  if (padded_tile_sizes.size() != padded_tile_sizes_no_unit_dims.size()) {
    TF_ASSIGN_OR_RETURN(
        ScalarOrTensor wrapped_result,
        EmitTiledReshape(b, padded_tile_sizes, ScalarOrTensor(result)));
    result = wrapped_result.UnwrapTensor();
  }

  return ScalarOrTensor(result);
}

absl::StatusOr<ScalarOrTensor> EmitConcatenate(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_concatenate, mlir::FunctionOpInterface fn,
    ValueRange tile_multi_index) {
  const int64_t concatenate_dimension =
      tiled_concatenate.hlo()->concatenate_dimension();

  // TODO(b/393299275): get rid of calls to `GetPaddedTileSizes` once tiling
  // is using power of twos everywhere, including when propagating into the
  // prologue of reductions.
  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_concatenate.tile_sizes());
  int64_t concatenate_dimension_tile_size =
      padded_tile_sizes[concatenate_dimension];

  for (const TiledHloInstruction* operand : tiled_concatenate.operands()) {
    if (operand->hlo()->opcode() != HloOpcode::kFusion) {
      // Sanity check: all operands should be nested fusions.
      return absl::FailedPreconditionError(
          "Expected concatenate operands to be nested fusions.");
    }

    int64_t operand_concatenate_dimension_size =
        tiled_concatenate.hlo()->shape().dimensions(concatenate_dimension);

    if (operand_concatenate_dimension_size % concatenate_dimension_tile_size !=
        0) {
      // Sanity check: concatenation dimension should be divisible by the tile
      // size for each operand. This is not a fundamental limitation, but this
      // lowering will emit incorrect code if this does not hold---so we gate
      // against it explicitly.
      return absl::FailedPreconditionError(absl::StrCat(
          "Expected the tile size of the concatenation dimension of operand ",
          operand->ToString(), "to divide the dimension size exactly, but got",
          operand_concatenate_dimension_size, " % ",
          concatenate_dimension_tile_size, " != 0"));
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
      emitters::ApplyIndexing(tile_offsets_indexing, /*dims=*/tile_multi_index,
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
      Value offset_limit =
          CreateConst(b, b.getIndexType(), limit, {}).UnwrapScalar();

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
        std::vector<ScalarOrTensor> result,
        EmitTiledComputation(
            b, libdevice_path, device_info,
            ::xla::Cast<HloFusionInstruction>(tiled_fusion_operand->hlo()),
            *tiled_fusion_operand->called_computation(), fn, tile_multi_index));
    CHECK_EQ(result.size(), 1);
    b.create<mlir::scf::YieldOp>(result.front().UnwrapTensor());
  }

  b.setInsertionPointAfter(if_ops.front());

  return ScalarOrTensor(if_ops.front().getResult(0));
}

// Given an operand to a (potentially nested) fusion instruction, finds the
// index of the operand to the outermost fusion it corresponds to.
//
// Nested fusion parameter chains should always only traverse parameter nodes.
int64_t GetOutermostFusionOperandParameterIndex(
    const HloFusionInstruction* fusion, const HloInstruction* operand) {
  CHECK(fusion->IsUserOf(operand));

  // Simple case: `fusion` is the outermost fusion.
  if (!operand->parent()->IsFusionComputation()) {
    return fusion->operand_index(operand);
  }

  // While operand is in a nested fusion, walk up to the outermost fusion.
  while (
      operand->parent()->FusionInstruction()->parent()->IsFusionComputation()) {
    // Nests operands should always point to parameters.
    CHECK(operand->opcode() == HloOpcode::kParameter);
    int64_t param_number = operand->parameter_number();
    operand = operand->parent()->FusionInstruction()->operand(param_number);
  }

  CHECK(operand->parent()->IsFusionComputation());
  CHECK(operand->opcode() == HloOpcode::kParameter);
  return operand->parameter_number();
}

absl::StatusOr<ScalarOrTensor> EmitTiledHloInstruction(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion, const TiledHloInstruction& tiled_hlo,
    mlir::FunctionOpInterface fn, ValueRange tile_multi_index,
    absl::flat_hash_map<const TiledHloInstruction*, ScalarOrTensor>& values) {
  const HloInstruction* hlo = tiled_hlo.hlo();
  VLOG(4) << "EmitTiledHloInstruction: " << hlo->ToString();

  if (fusion->IsUserOf(hlo)) {
    // If the fusion instruction is a user of `hlo`, then `hlo` is an operand
    // to the fusion instruction.
    int64_t arg_index = GetOutermostFusionOperandParameterIndex(fusion, hlo);
    TF_ASSIGN_OR_RETURN(auto tile_op, ir_emitter_triton_internal::CreateTileOp(
                                          b, tile_multi_index, tiled_hlo,
                                          fn.getArgument(arg_index)));

    ScalarOrTensor parameter = EmitParameterExtract(b, tile_op);

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
      if (loaded_element_type != StorageType(b, expected_element_type)) {
        return absl::InternalError(absl::StrCat(
            "Parameters were loaded with an unexpected element type "
            "while lowering ",
            fusion->called_computation()->ToString()));
      }
      parameter = ScalarOrTensor(
          Cast(b, parameter.UnwrapUnsafe(), expected_element_type));
    }

    return parameter;
  }

  if (hlo->opcode() == HloOpcode::kConcatenate) {
    return EmitConcatenate(b, libdevice_path, device_info, fusion, tiled_hlo,
                           fn, tile_multi_index);
  }

  if (hlo->opcode() == HloOpcode::kDot) {
    return EmitDot(b, libdevice_path, device_info, fusion, tiled_hlo, fn,
                   tile_multi_index);
  }

  if (hlo->opcode() == HloOpcode::kConstant) {
    if (ShapeUtil::IsEffectiveScalar(hlo->shape())) {
      return EmitConstant(b, *hlo);
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported non-scalar constant ", hlo->ToString()));
  }

  if (hlo->opcode() == HloOpcode::kIota) {
    return EmitTiledIota(b, tile_multi_index, tiled_hlo);
  }

  if (hlo->opcode() == HloOpcode::kBroadcast) {
    return EmitTiledBroadcast(b, tiled_hlo, values);
  }

  if (hlo->opcode() == HloOpcode::kReduce) {
    return EmitReduce(b, tiled_hlo, values, libdevice_path, device_info);
  }

  if (hlo->IsElementwise()) {
    std::vector<Value> operands;
    operands.reserve(hlo->operands().size());

    for (const TiledHloInstruction* operand : tiled_hlo.operands()) {
      operands.push_back(values[operand].UnwrapUnsafe());
    }
    TF_ASSIGN_OR_RETURN(
        Value result,
        EmitElementwise(b, libdevice_path, device_info, *hlo, operands));
    return ScalarOrTensor(result);
  }

  if (hlo->opcode() == HloOpcode::kReshape) {
    return EmitTiledReshape(b, tiled_hlo.tile_sizes(),
                            values[tiled_hlo.operand(0)]);
  }

  if (hlo->opcode() == HloOpcode::kBitcast) {
    return EmitTiledBitcast(b, tiled_hlo,
                            values[tiled_hlo.operand(0)].UnwrapUnsafe());
  }

  if (hlo->opcode() == HloOpcode::kTranspose) {
    auto transpose =
        ::xla::Cast<const HloTransposeInstruction>(tiled_hlo.hlo());
    return ScalarOrTensor(EmitTiledTranspose(
        b, tiled_hlo.tile_sizes(), llvm::to_vector(transpose->dimensions()),
        values[tiled_hlo.operand(0)].UnwrapUnsafe()));
  }

  // Slice is currently supported only as an operation on indices
  // which is pushed to loads and stores. We don't generate any further code.
  if (hlo->opcode() == HloOpcode::kSlice) {
    return values[tiled_hlo.operand(0)];
  }

  return absl::UnimplementedError(
      absl::StrCat("Unsupported operation ", hlo->ToString()));
}

// Emit a sequence of instructions using compatible tiling with producers
// ordered before consumers in `tiled_computation`. Returns the results for the
// roots of `tiled_computation`.
absl::StatusOr<std::vector<ScalarOrTensor>> EmitTiledComputation(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation, mlir::FunctionOpInterface fn,
    ValueRange tile_multi_index) {
  VLOG(2) << "EmitTiledComputation: " << tiled_computation.ToString();
  absl::flat_hash_map<const TiledHloInstruction*, ScalarOrTensor> values;
  for (const TiledHloInstruction* tiled_hlo :
       tiled_computation.instructions()) {
    const HloInstruction* hlo = tiled_hlo->hlo();
    // Skip generating nested fusions, they are emitted by their consumer.
    if (hlo->parent()->IsFusionComputation() &&
        hlo->opcode() == HloOpcode::kFusion) {
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
    TF_ASSIGN_OR_RETURN(
        ScalarOrTensor result,
        EmitTiledHloInstruction(b, libdevice_path, device_info, fusion,
                                *tiled_hlo, fn, tile_multi_index, values));
    TF_RET_CHECK(values.insert({tiled_hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  std::vector<ScalarOrTensor> results;
  results.reserve(tiled_computation.GetRoots().size());
  for (const auto* root : tiled_computation.GetRoots()) {
    results.push_back(values[root]);
  }
  return std::move(results);
}

// Emit sequence of instructions using compatible tiling ordered producers
// before consumers.
absl::StatusOr<ScalarOrTensor> EmitScope(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const TritonFusionAnalysis* analysis,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, ScalarOrTensor>& values) {
  for (const HloInstruction* hlo : instructions) {
    ScalarOrTensor result;
    if (hlo->opcode() == HloOpcode::kConcatenate ||
        hlo->opcode() == HloOpcode::kDynamicSlice) {
      // Parameter loads and their concatenations are handled outside EmitScope.
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    } else if (hlo->opcode() == HloOpcode::kParameter) {
      if (hlo->users()[0]->opcode() == HloOpcode::kConcatenate ||
          hlo->users()[0]->opcode() == HloOpcode::kDynamicSlice) {
        continue;
      }
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    } else if (hlo->opcode() == HloOpcode::kConstant) {
      return EmitConstant(b, *hlo);
    } else if (hlo->opcode() == HloOpcode::kBroadcast) {
      return absl::InvalidArgumentError(
          "Broadcast is not yet supported in EmitScope().");
    } else if (HloInstruction::IsOpElementwise(hlo->opcode())) {
      std::vector<Value> operands;
      operands.reserve(hlo->operands().size());
      for (const HloInstruction* operand : hlo->operands()) {
        operands.push_back(values[operand].UnwrapUnsafe());
      }
      TF_ASSIGN_OR_RETURN(
          Value elementwise_result,
          EmitElementwise(b, libdevice_path, device_info, *hlo, operands));
      result = ScalarOrTensor(elementwise_result);
    } else if (hlo->opcode() == HloOpcode::kTuple) {
      TF_RET_CHECK(hlo->IsRoot()) << hlo->ToString();
    } else if (hlo->opcode() == HloOpcode::kBitcast ||
               hlo->opcode() == HloOpcode::kTranspose ||
               hlo->opcode() == HloOpcode::kSlice ||
               hlo->opcode() == HloOpcode::kReshape ||
               hlo->opcode() == HloOpcode::kPad) {
      // All these are currently supported only as operations on indices
      // which are pushed to loads and stores. No operations on tiles are
      // performed here.
      result = values[hlo->operand(0)];
    } else if (hlo->opcode() == HloOpcode::kFusion) {
      const auto* fusion_instruction = ::xla::Cast<HloFusionInstruction>(hlo);
      TF_ASSIGN_OR_RETURN(result,
                          EmitNestedFusion(b, libdevice_path, device_info,
                                           *fusion_instruction, values));
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported operation ", hlo->ToString()));
    }
    TF_RET_CHECK(values.insert({hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  return values[instructions.back()];
}
}  // namespace

namespace ir_emitter_triton_internal {

SmallVector<Value, 3> ComputeDelinearizedTileIndex(
    EmitterLocOpBuilder& b,
    absl::Span<const int64_t> num_output_tiles_per_dim) {
  // TODO(b/389955087): we can decide whether to sign extend by understanding if
  // we need 64 bits to encode indices or if 32 bits are enough. For now, just
  // use 64 bits to avoid issues.
  Value pid = b.create<arith::IndexCastUIOp>(
      b.getIndexType(),
      b.create<arith::ExtSIOp>(b.getI64Type(), b.create<ttir::GetProgramIdOp>(
                                                   ttir::ProgramIDDim::X)));

  // Delinearize the block id.
  mlir::AffineExpr program_id = mlir::getAffineDimExpr(0, b.getContext());
  auto tile_exprs =
      DelinearizeIndex(num_output_tiles_per_dim, program_id, b.getContext());

  IndexingMap program_id_to_root_tile_offset = IndexingMap::FromTensorSizes(
      mlir::AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, tile_exprs,
                           b.getContext()),
      /*dim_upper_bounds=*/{Product(num_output_tiles_per_dim)},
      /*symbol_upper_bounds=*/{});

  return emitters::ApplyIndexing(program_id_to_root_tile_offset,
                                 /*dims=*/pid,
                                 /*symbols=*/{}, b);
}

SmallVector<Value> CreateIndexValues(EmitterLocOpBuilder& builder,
                                     const ArrayRef<int64_t>& values) {
  SmallVector<Value> result;
  result.reserve(values.size());
  for (int64_t value : values) {
    result.push_back(
        CreateConst(builder, builder.getIndexType(), value).UnwrapScalar());
  }
  return result;
}

absl::StatusOr<mlir::triton::xla::TileOp> CreateTileOp(
    EmitterLocOpBuilder& b, ValueRange tile_multi_index,
    const TiledHloInstruction& tiled_hlo, Value parent_base_ptr) {
  TF_ASSIGN_OR_RETURN(SmallVector<Value> ptr_offsets,
                      ComputeBasePtrOffset(b, tile_multi_index, tiled_hlo));

  // Triton requires that all block dimensions are a power of 2.
  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_hlo.tile_sizes());

  const Shape& shape = tiled_hlo.hlo()->shape();
  TF_ASSIGN_OR_RETURN(Type expected_element_type,
                      TritonType(b, shape.element_type()));
  Type storage_type = StorageType(b, expected_element_type);
  auto result_type = mtx::TiledTensorType::get(
      b.getContext(), padded_tile_sizes,
      llvm::ArrayRef<int64_t>(shape.dimensions().data(),
                              shape.dimensions().size()),
      storage_type);

  return b.create<mtx::TileOp>(
      result_type, parent_base_ptr, ptr_offsets,
      CreateIndexValues(b, tiled_hlo.tile_strides()),
      llvm::to_vector(LayoutUtil::MinorToMajor(shape)));
}

}  // namespace ir_emitter_triton_internal

namespace {

using ::xla::gpu::ir_emitter_triton_internal::DumpTritonIR;

// Generate Triton IR inside 'fn', using the given block_level_parameters.
absl::StatusOr<SmallVector<Value>> EmitGeneric(
    mlir::OpBuilder builder, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion, mlir::FunctionOpInterface fn,
    const BlockLevelParameters& block_level_parameters) {
  const HloComputation* computation = fusion->fused_instructions_computation();
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          *computation, builder.getContext(),
          TritonEmitterConstraints::GetBuilder(device_info));
  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in EmitGeneric: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  const auto& symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(symbolic_tile_analysis_or);
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

  int64_t root_index = FindIndex(symbolic_tile_analysis.GetRoots(), root);
  TF_RET_CHECK(root_index < symbolic_tile_analysis.GetRoots().size());
  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      symbolic_tile_analysis.ComputeTiledHloInstructions(
                          block_level_parameters.output_tile_sizes[root_index],
                          /*constraints_are_known_satisfied=*/false,
                          /*compute_all_tile_offset_indexing_maps=*/true));
  VLOG(3) << "EmitGeneric: tiled HLO computation:\n"
          << tiled_hlo_computation.ToString();

  SmallVector<Value, 3> tile_multi_index =
      ir_emitter_triton_internal::ComputeDelinearizedTileIndex(
          b, tiled_hlo_computation.num_output_tiles_per_dim());

  TF_ASSIGN_OR_RETURN(
      auto results,
      EmitTiledComputation(b, libdevice_path, device_info, fusion,
                           tiled_hlo_computation, fn, tile_multi_index));

  SmallVector<Value> insert_results;
  for (auto [root, result, parent_base_ptr] :
       llvm::zip(tiled_hlo_computation.GetRoots(), results,
                 fn.getArguments().drop_front(computation->num_parameters()))) {
    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to check converted types before storing if the type
    // of the result does not match the type of the output pointer.
    Type result_element_type = getElementTypeOrSelf(result.getType());
    Type result_storage_type = StorageType(b, result_element_type);

    if (result_element_type != result_storage_type) {
      result =
          ScalarOrTensor(Cast(b, result.UnwrapUnsafe(), result_storage_type));
    }

    if (result.IsScalar()) {
      ValueRange indices = {};
      insert_results.push_back(
          b.create<mlir::tensor::InsertOp>(result.UnwrapScalar(),
                                           parent_base_ptr, indices)
              .getResult());
      continue;
    }

    CHECK(root->hlo()->shape().IsArray() &&
          !root->hlo()->shape().dimensions().empty());
    TF_ASSIGN_OR_RETURN(mlir::triton::xla::TileOp tile_op,
                        ir_emitter_triton_internal::CreateTileOp(
                            b, tile_multi_index, *root, parent_base_ptr));

    // Should not be scalar at this point.
    auto tiled_tensor_type =
        mlir::dyn_cast<mtx::TiledTensorType>(tile_op.getResult().getType());
    CHECK(tiled_tensor_type) << "Expected a tiled tensor type since scalars "
                                "should've been handled at this point.";

    SmallVector<Value> offsets(
        tiled_tensor_type.getTileShape().size(),
        CreateConst(b, b.getIndexType(), 0).UnwrapScalar());

    insert_results.push_back(
        b.create<mtx::InsertOp>(
             mlir::RankedTensorType::get(tiled_tensor_type.getOriginalShape(),
                                         result_storage_type),
             result.UnwrapTensor(), tile_op.getResult(), offsets)
            .getResult());
  }

  return insert_results;
}

}  // namespace

void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context) {
  mlir_context
      .loadDialect<ttir::TritonDialect, ttir::gpu::TritonGPUDialect,
                   mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                   mlir::LLVM::LLVMDialect, xla::XlaDialect,
                   xla::gpu::XlaGpuDialect, ttir::xla::XlaTritonDialect,
                   mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
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

// Legacy emitter works with tt.func. New emitter works with func.func.
// TODO(393299275): Remove legacy optionality once migration is complete.
void AppendFuncArgType(EmitterLocOpBuilder& b, absl::Span<const int64_t> dims,
                       absl::string_view fusion_kind, Type ir_type,
                       SmallVector<Type>& fn_arg_types) {
  if (fusion_kind == kTritonGemmFusionKind) {
    fn_arg_types.push_back(ttir::PointerType::get(
        StorageType(b, ir_type), mlir::NVVM::kGlobalMemorySpace));
  } else {
    fn_arg_types.push_back(mlir::RankedTensorType::get(
        llvm::ArrayRef<int64_t>(dims.data(), dims.size()),
        StorageType(b, ir_type)));
  }
}

// Only needed for the new emitter since we are using func.func instead of
// tt.func.
// TODO(393299275): Remove legacy optionality once migration is complete.
void AppendFuncResultType(EmitterLocOpBuilder& b, absl::string_view fusion_kind,
                          absl::Span<const int64_t> dims, Type ir_type,
                          SmallVector<Type>& fn_result_types) {
  if (fusion_kind != kTritonGemmFusionKind) {
    fn_result_types.push_back(mlir::RankedTensorType::get(
        llvm::ArrayRef<int64_t>(dims.data(), dims.size()),
        StorageType(b, ir_type)));
  }
}

// Legacy emitter works with tt.func. New emitter works with func.func.
// TODO(393299275): Remove legacy optionality once migration is complete.
mlir::FunctionOpInterface CreateFuncOp(EmitterLocOpBuilder& b,
                                       absl::string_view fn_name,
                                       absl::string_view fusion_kind,
                                       SmallVector<Type>& fn_arg_types,
                                       SmallVector<Type>& fn_result_types) {
  mlir::FunctionOpInterface fn;
  if (fusion_kind == kTritonGemmFusionKind) {
    fn = b.create<ttir::FuncOp>(fn_name,
                                b.getFunctionType(fn_arg_types, std::nullopt));
    for (int i = 0; i < fn.getNumArguments(); ++i) {
      fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
    }
  } else {
    fn = b.create<mlir::func::FuncOp>(
        fn_name, b.getFunctionType(fn_arg_types, fn_result_types));
  }
  return fn;
}

// Legacy emitter works with tt.return. New emitter works with func.return.
// TODO(393299275): Remove legacy optionality once migration is complete.
void EmitReturnOp(EmitterLocOpBuilder& b, absl::string_view fusion_kind,
                  SmallVector<Value> insert_results) {
  if (fusion_kind == kTritonGemmFusionKind) {
    b.create<ttir::ReturnOp>();
  } else {
    b.create<mlir::func::ReturnOp>(insert_results);
  }
}

absl::StatusOr<stream_executor::gpu::TmaMetadata> ExtractTmaMetadata(
    mlir::ModuleOp triton_module, absl::string_view kernel_name) {
  stream_executor::gpu::TmaMetadata tma_metadata;
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

  for (auto [idx, arg] : llvm::enumerate(func_ops[0].getArguments())) {
    if (auto attr = func_ops[0].getArgAttrOfType<mtx::TmaDescriptorAttr>(
            idx, "tt.tma_descriptor")) {
      TF_ASSIGN_OR_RETURN(
          auto tma_desc,
          Create2DTmaDescriptor(attr.getGlobalShape(), attr.getBlockShape(),
                                attr.getElementByteSize()));
      tma_metadata.arg_index_to_tma_info.insert({idx, tma_desc});
    }
  }
  return tma_metadata;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context) {
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

    AppendFuncArgType(b, p->shape().dimensions(), fusion_kind, ir_type,
                      fn_arg_types);
  }

  SmallVector<Type> fn_result_types;

  for (const ShapeUtil::IndexedShape& s :
       ShapeUtil::GetLeafShapes(fusion->shape())) {
    TF_ASSIGN_OR_RETURN(Type triton_ty, TritonType(b, s.shape.element_type()));
    AppendFuncArgType(b, s.shape.dimensions(), fusion_kind, triton_ty,
                      fn_arg_types);
    AppendFuncResultType(b, fusion_kind, s.shape.dimensions(), triton_ty,
                         fn_result_types);
  }

  mlir::FunctionOpInterface fn =
      CreateFuncOp(b, fn_name, fusion_kind, fn_arg_types, fn_result_types);

  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  std::string libdevice_path =
      GetLibdevicePath(fusion->GetModule()->config(), device_info);

  SmallVector<Value> insert_results;
  if (fusion_kind == kTritonGemmFusionKind) {
    // If the generic Triton emitter is enabled, we should never go through the
    // legacy MatMul emitter.
    if (UseGenericTritonEmitterForGemms(fusion)) {
      return absl::FailedPreconditionError(
          "The generic Triton emitter is enabled, but the legacy MatMul "
          "emitter is being used.");
    }
    TF_RETURN_IF_ERROR(EmitMatMul(b, libdevice_path, device_info, fusion, fn,
                                  block_level_parameters));
  } else if (fusion_kind == kTritonFusionKind ||
             fusion_kind == kTritonNestedGemmFusionKind) {
    TF_ASSIGN_OR_RETURN(insert_results,
                        EmitGeneric(b, libdevice_path, device_info, fusion, fn,
                                    block_level_parameters));
  } else {
    return Internal("Unsupported fusion kind: %s", fusion_kind);
  }

  EmitReturnOp(b, fusion_kind, insert_results);

  if (DumpingEnabledForHloModule(*hlo_computation->parent())) {
    auto suffix = absl::StrCat(fusion->name(), ".before_validation.ttir");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", suffix,
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
    std::string fusion_suffix = absl::StrCat(hlo_computation->name(), ".hlo");
    DumpToFileInDirOrStdout(*hlo_computation->parent(), "", fusion_suffix,
                            hlo_computation->ToString());
  }

  if (mlir::failed(mlir::verify(*triton_module))) {
    return CreateInternalError(
        "Failed to verify Triton module for fusion:", fusion, *triton_module);
  }

  mlir::PassManager pm(&mlir_context);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(triton_module.get()))) {
    return CreateInternalError(
        "Failed to create Triton module for fusion:", fusion, *triton_module);
  }

  VLOG(6) << DumpTritonIR(triton_module.get(),
                          fusion->GetModule()
                              ->config()
                              .debug_options()
                              .xla_gpu_unsupported_annotate_with_emitter_loc());
  // TODO(loislo): Remove this dump once we have the Triton IR dump in
  // CompileTritonToLLVM after the Triton optimization passes.
  if (DumpingEnabledForHloModule(*hlo_computation->parent())) {
    std::string suffix = absl::StrCat(fusion->name(), ".ttir");
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

absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    llvm::Module* llvm_module, mlir::MLIRContext& mlir_context) {
  if (std::holds_alternative<se::CudaComputeCapability>(cc)) {
    auto ccCuda = std::get<se::CudaComputeCapability>(cc);
    if (!ccCuda.IsAtLeastAmpere()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Triton support is only enabled for Ampere GPUs (compute ",
          "capability 8.0) and up, but got compute capability ", ccCuda.major,
          ".", ccCuda.minor, "."));
    }
  }

  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> triton_module,
                      CreateTritonModule(fn_name, fusion, device_info,
                                         block_level_parameters, mlir_context));

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
  const auto& cc = device_info.gpu_compute_capability();
  std::string arch_name =
      std::visit([](auto& cc) { return cc.ToString(); }, cc);
  if (std::holds_alternative<se::CudaComputeCapability>(cc)) {
    auto ccCuda = std::get<se::CudaComputeCapability>(cc);
    if (!ccCuda.IsAtLeastAmpere()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Triton support is only enabled for Ampere GPUs (compute ",
          "capability 8.0) and up, but got compute capability ", ccCuda.major,
          ".", ccCuda.minor, "."));
    }
  }

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
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = hlo_config.debug_options().xla_dump_to();
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
        LOG(ERROR) << err.message();
      } else {
        pm.getContext()->disableMultithreading();
        auto print_always = [](mlir::Pass*, mlir::Operation*) { return true; };
        pm.enableIRPrinting(/*shouldPrintBeforePass=*/print_always,
                            /*shouldPrintAfterPass=*/print_always,
                            /*printModuleScope=*/true,
                            /*printAfterOnlyOnChange=*/false,
                            /*printAfterOnlyOnFailure=*/true, *log_stream,
                            /*opPrintingFlags=*/{});
      }
    } else {
      LOG(ERROR)
          << "--xla_dump_hlo_pass_re=triton-fusion-emitter is set, but neither "
          << "the environment variable TEST_UNDECLARED_OUTPUTS_DIR nor the "
          << "flag --xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  // TODO(b/315957220): Propagate TMA flag once it's supported.
  pm.addPass(mlir::triton::xla::CreateTritonXLAExtractInsertToTritonPass(
      device_info, /*tma_enabled=*/false));

  // Lower affine expressions into arithmetic ops.
  pm.addPass(mlir::createLowerAffinePass());

  // Lower xla_gpu.apply_indexing into arithmetic ops.
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(CreateConvertIndexTypePass());

  int64_t num_warps = block_level_parameters.num_warps;
  int num_ctas = block_level_parameters.num_ctas;
  int num_stages = block_level_parameters.num_stages;

  if (num_warps <= 0 || num_ctas <= 0 || num_stages <= 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "(num_warps, num_ctas, num_stages) must be positive, but got: (",
        num_warps, ", ", num_ctas, ", ", num_stages, ")"));
  }

  mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
  if (!CreateTritonPipeline(&pm, arch_name, num_warps, num_ctas, num_stages,
                            cluster_info, is_xla_fusion)
           .ok()) {
    return Internal("Failed to create Triton pipeline.");
  }
  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(mlir::triton::xla::CreateGeneralizeKernelSignaturePass());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  bool succeeded = mlir::succeeded(pm.run(triton_module));

  if (!succeeded) {
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

  if (std::holds_alternative<se::CudaComputeCapability>(cc) &&
      std::get<se::CudaComputeCapability>(cc).IsBlackwell()) {
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

  if (emit_kernel) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<llvm::Module> ll_triton_module,
        TranslateLLVMToLLVMIR(&llvm_module->getContext(), triton_module));
    VLogModule(5, *ll_triton_module);
    if (should_verify) {
      VerifyModule(*ll_triton_module);
    }

    // Integrate LLVM matmul kernel into XLA's LLVM module.
    // TODO(goncharov): remove once we integrated past LLVM
    // 6c2e170d043d3a7d7b32635e887cfd255ef5c2ce that removes nvvm.annotations.
    auto* nvvm_annotations =
        ll_triton_module->getNamedMetadata("nvvm.annotations");
    if (nvvm_annotations) {
      ll_triton_module->eraseNamedMetadata(nvvm_annotations);
    }
    ll_triton_module->setDataLayout(llvm_module->getDataLayout());
    ll_triton_module->setTargetTriple(llvm_module->getTargetTriple());
    // Use override flag because libdevice functions can be present in both.
    TF_RET_CHECK(
        !llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module),
                                   llvm::Linker::Flags::OverrideFromSrc));

    VLogModule(5, *llvm_module);
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

  // It's okay for tma_metadata to be empty; it's only populated when used
  // explicitly.
  TF_ASSIGN_OR_RETURN(stream_executor::gpu::TmaMetadata tma_metadata,
                      ExtractTmaMetadata(triton_module, kernel_name));

  return {{shared_mem_bytes, cluster_dim, tma_metadata}};
}

}  // namespace gpu
}  // namespace xla
