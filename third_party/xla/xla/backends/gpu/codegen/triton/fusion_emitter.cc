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
#include <numeric>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11): required to interface with LLVM
#include <utility>
#include <variant>
#include <vector>

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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
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
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"
#include "xla/backends/gpu/codegen/triton/passes.h"
#include "xla/backends/gpu/codegen/triton/xla_triton_ops.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"
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
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
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

Value AddPtr(EmitterLocOpBuilder& b, Value ptr, Value offset) {
  return b.create<ttir::AddPtrOp>(ptr.getType(), ptr, offset);
}

ScalarOrTensor EmitParameterLoad(EmitterLocOpBuilder& b, Value pointer,
                                 ArrayRef<int32_t> boundary_checks) {
  // For a pointer to a scalar or a zero-dimensional tensor, load the base
  // pointer directly. This shortcut is necessary because Triton does not
  // support 0-D tensors. Looking for the defining make_tensor_ptr op is
  // sufficient because pointers to 0-D tensors are never modified by e.g.
  // `tt.advance`.
  if (auto make_tensor_ptr = pointer.getDefiningOp<ttir::MakeTensorPtrOp>();
      make_tensor_ptr && make_tensor_ptr.getShape().empty()) {
    pointer = make_tensor_ptr.getBase();
  }

  std::optional<ttir::PaddingOption> padding;
  if (!boundary_checks.empty()) {
    padding = ttir::PaddingOption::PAD_ZERO;
  }
  bool is_volatile = false;
  return ScalarOrTensor(b.create<ttir::LoadOp>(
      pointer, boundary_checks, padding, ttir::CacheModifier::NONE,
      ttir::EvictionPolicy::NORMAL, is_volatile));
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
  const llvm::SmallVector<int64_t>& input_tile_shape =
      tiled_broadcast.operand(0)->tile_sizes();
  const llvm::SmallVector<int64_t>& output_tile_shape =
      tiled_broadcast.tile_sizes();

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
                                                  /*allow_reorder*/ true);
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

absl::StatusOr<ScalarOrTensor> EmitTiledHloInstruction(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion, const TiledHloInstruction& tiled_hlo,
    mlir::triton::FuncOp fn, ValueRange tile_multi_index,
    absl::flat_hash_map<const TiledHloInstruction*, ScalarOrTensor>& values) {
  const HloInstruction* hlo = tiled_hlo.hlo();

  if (fusion->IsUserOf(hlo)) {
    TF_ASSIGN_OR_RETURN(auto make_tensor,
                        ir_emitter_triton_internal::CreateMakeTensorPtrOp(
                            b, tile_multi_index, tiled_hlo,
                            fn.getArgument(fusion->operand_index(hlo))));

    ScalarOrTensor parameter =
        EmitParameterLoad(b, make_tensor.op, make_tensor.boundary_checks);

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

// Emit sequence of instructions using compatible tiling ordered producers
// before consumers.
absl::StatusOr<ScalarOrTensor> EmitTiledComputation(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion,
    const TiledHloComputation& tiled_computation, mlir::triton::FuncOp fn,
    ValueRange tile_multi_index) {
  absl::flat_hash_map<const TiledHloInstruction*, ScalarOrTensor> values;
  for (const TiledHloInstruction* tiled_hlo :
       tiled_computation.instructions()) {
    TF_ASSIGN_OR_RETURN(
        ScalarOrTensor result,
        EmitTiledHloInstruction(b, libdevice_path, device_info, fusion,
                                *tiled_hlo, fn, tile_multi_index, values));
    TF_RET_CHECK(values.insert({tiled_hlo, result}).second)
        << tiled_hlo->hlo()->ToString();
    VLOG(8) << "Emitted "
            << tiled_hlo->hlo()->ToString(HloPrintOptions::ShortParsable());
  }
  return values[tiled_computation.GetRoot()];
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

// Computes the base pointer offset for the given tile multi-index and hlo shape
// taking into account the physical layout of the hlo buffer.
absl::StatusOr<Value> ComputeBasePtrOffset(
    EmitterLocOpBuilder& b, ValueRange tile_multi_index,
    const TiledHloInstruction& tiled_hlo) {
  const Shape& shape = tiled_hlo.hlo()->shape();
  Shape linear_shape = ShapeUtil::MakeShape(shape.element_type(),
                                            {ShapeUtil::ElementsIn(shape)});

  // Bitcast map gives an indexing map from the parameter shape (multi-index) to
  // a linear index respecting physical layout of the memory.
  auto bitcast_map = GetBitcastMap(shape, linear_shape, b.getContext());

  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_hlo.tile_offsets_indexing());

  auto compose_indexing_maps =
      ComposeIndexingMaps(tile_offsets_indexing, bitcast_map);
  compose_indexing_maps.Simplify();

  return b.create<arith::IndexCastUIOp>(
      b.getI64Type(), emitters::ApplyIndexing(compose_indexing_maps,
                                              /*dims=*/tile_multi_index,
                                              /*symbols=*/{}, b)[0]);
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

absl::StatusOr<MakeTensorPtrOpAndBoundaryChecks> CreateMakeTensorPtrOp(
    EmitterLocOpBuilder& b, ValueRange tile_multi_index,
    const TiledHloInstruction& tiled_hlo, Value parent_base_ptr) {
  const Shape& shape = tiled_hlo.hlo()->shape();

  // Compute physical strides of the tile. `tile_strides` contains strides for
  // individual dimensions. We need to convert them to strides in the buffer
  // taking into account physical layout.
  // TODO(b/331332678): Compute indexing maps to physical layout indexing in
  // SymbolicTileAnalysis.
  const llvm::SmallVector<int64_t>& tile_strides = tiled_hlo.tile_strides();
  llvm::SmallVector<Value> strides(tile_strides.size());
  int64_t current_stride = 1;
  for (int64_t cur_dim : LayoutUtil::MinorToMajor(shape)) {
    strides[cur_dim] =
        CreateConst(b, b.getI64Type(), tile_strides[cur_dim] * current_stride)
            .UnwrapScalar();
    current_stride *= shape.dimensions(cur_dim);
  }

  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_hlo.tile_offsets_indexing());
  auto tile_offsets_as_indices =
      emitters::ApplyIndexing(tile_offsets_indexing,
                              /*dims=*/tile_multi_index,
                              /*symbols=*/{}, b);

  // Triton requires that all block dimensions are a power of 2.
  SmallVector<int64_t> padded_tile_sizes =
      GetPaddedTileSizes(tiled_hlo.tile_sizes());

  // TensorPtr is intended to be a  base pointer of the TiledHloInstruction and
  // plus the necessary offsets so that Triton can compute the pointer to the
  // block specific to the given pid. This option would yield simpler code, but
  // cannot handle all combinations of strides and offsets, because Triton
  // always multiplies the offset by the stride. E.g., it's not possible to
  // slice [10] with [1:5:2] because the first element will always be at an even
  // offset.
  //
  // Instead, we output a TensorPtr that points directly to the tile specific
  // to the pid. All offset computation is done in advance. MakeTensorPtrOp
  // sees 0 offsets. This allows Triton to read any block regardless of strides
  // size or offsets. To make sure that masking is correct, we compute a
  // "residual shape" which is the original parent shape minus the offsets.

  llvm::SmallVector<Value> residual_shape;
  llvm::SmallVector<int32_t> boundary_checks;
  for (int dim_idx = 0; dim_idx < padded_tile_sizes.size(); ++dim_idx) {
    Value parent_size =
        CreateConst(b, b.getI64Type(), shape.dimensions(dim_idx))
            .UnwrapScalar();
    // Offsets are necessarily positive since they represent a distance between
    // 0 and the size of the tensor on the given axis. Therefore, it is safe to
    // use 'IndexCastUI' here. This allows index canonicalizations later on.
    Value offset = b.create<arith::IndexCastUIOp>(
        b.getI64Type(), tile_offsets_as_indices[dim_idx]);
    residual_shape.push_back(b.create<arith::SubIOp>(parent_size, offset));

    if (shape.dimensions(dim_idx) % padded_tile_sizes[dim_idx] != 0) {
      boundary_checks.push_back(dim_idx);
    }
  }

  TF_ASSIGN_OR_RETURN(Value ptr_offset,
                      ComputeBasePtrOffset(b, tile_multi_index, tiled_hlo));
  auto tile_ptr = AddPtr(b, parent_base_ptr, ptr_offset);

  llvm::SmallVector<Value> offsets(
      padded_tile_sizes.size(),
      CreateConst(b, b.getI32Type(), 0).UnwrapScalar());

  // TODO(b/342989850): Clarify and comment what `order` exactly is. It's not
  // entirely clear from the Triton docs.
  llvm::SmallVector<int32_t> order(padded_tile_sizes.size());
  std::iota(order.rbegin(), order.rend(), 0);

  auto make_tensor_ptr = b.create<ttir::MakeTensorPtrOp>(
      /*base*/ tile_ptr,
      /*shape*/ residual_shape,
      /*strides*/ strides,
      /*offsets*/ offsets,
      /*tensorShape*/ llvm::to_vector_of<int32_t>(padded_tile_sizes),
      /*order*/ order);

  return MakeTensorPtrOpAndBoundaryChecks{make_tensor_ptr, boundary_checks};
}

}  // namespace ir_emitter_triton_internal

namespace {
// Generate Triton IR inside 'fn', using the given block_level_parameters.
absl::Status EmitGeneric(mlir::OpBuilder builder,
                         absl::string_view libdevice_path,
                         const se::DeviceDescription& device_info,
                         const HloFusionInstruction* fusion,
                         mlir::triton::FuncOp fn,
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
  const HloInstruction* root = computation->root_instruction();
  auto loc = mlir::NameLoc::get(builder.getStringAttr(root->name()));
  EmitterLocOpBuilder b(loc, builder,
                        root->GetModule()
                            ->config()
                            .debug_options()
                            .xla_gpu_unsupported_annotate_with_emitter_loc());

  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      symbolic_tile_analysis.ComputeTiledHloInstructions(
                          block_level_parameters.output_tile_sizes,
                          /*constraints_are_known_satisfied=*/false,
                          /*compute_all_tile_offset_indexing_maps=*/true));

  SmallVector<Value, 3> tile_multi_index =
      ir_emitter_triton_internal::ComputeDelinearizedTileIndex(
          b, tiled_hlo_computation.num_output_tiles_per_dim());

  TF_ASSIGN_OR_RETURN(
      ScalarOrTensor result,
      EmitTiledComputation(b, libdevice_path, device_info, fusion,
                           tiled_hlo_computation, fn, tile_multi_index));

  // Some types are stored using different types, e.g. i1 is stored in memory
  // as i8. It's important to type checking that we perform a conversion before
  // storing if the type of the result does not match the type of the output
  // pointer.
  Type result_element_type = getElementTypeOrSelf(result.getType());
  Type result_storage_type = StorageType(b, result_element_type);

  if (result_element_type != result_storage_type) {
    result =
        ScalarOrTensor(Cast(b, result.UnwrapUnsafe(), result_storage_type));
  }

  const auto& tiled_hlo = *tiled_hlo_computation.GetRoot();

  Value parent_base_ptr = fn.getArgument(computation->num_parameters());

  if (result.IsScalar()) {
    b.create<ttir::StoreOp>(parent_base_ptr, result.UnwrapScalar(),
                            ttir::CacheModifier::NONE,
                            ttir::EvictionPolicy::NORMAL);
    return absl::OkStatus();
  }

  CHECK(tiled_hlo.hlo()->shape().IsArray() &&
        tiled_hlo.hlo()->shape().rank() > 0);
  TF_ASSIGN_OR_RETURN(auto make_tensor,
                      ir_emitter_triton_internal::CreateMakeTensorPtrOp(
                          b, tile_multi_index, tiled_hlo, parent_base_ptr));
  b.create<ttir::StoreOp>(
      make_tensor.op, result.UnwrapTensor(), make_tensor.boundary_checks,
      ttir::CacheModifier::NONE, ttir::EvictionPolicy::NORMAL);

  return absl::OkStatus();
}

}  // namespace

void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context) {
  mlir_context
      .loadDialect<ttir::TritonDialect, ttir::gpu::TritonGPUDialect,
                   mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                   mlir::LLVM::LLVMDialect, xla::XlaDialect,
                   xla::gpu::XlaGpuDialect, ttir::xla::XlaTritonDialect>();
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

std::string DumpTritonIR(mlir::ModuleOp triton_module, bool dump_annotations) {
  std::string triton_ir;
  llvm::raw_string_ostream os(triton_ir);
  triton_module.print(os, mlir::OpPrintingFlags().enableDebugInfo(
                              dump_annotations, dump_annotations));
  if (dump_annotations) {
    return EmitterLocOpBuilder::FormatTritonIrWithAnnotations(triton_ir);
  }
  return triton_ir;
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
  os << "triton_module: \n";
  triton_module->print(os, mlir::OpPrintingFlags().enableDebugInfo(true, true));
  return absl::InternalError(err);
}

absl::StatusOr<TritonModule> CreateTritonModule(
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
    fn_arg_types.push_back(ttir::PointerType::get(
        StorageType(b, ir_type), mlir::NVVM::kGlobalMemorySpace));
  }

  for (const ShapeUtil::IndexedShape& s :
       ShapeUtil::GetLeafShapes(fusion->shape())) {
    TF_ASSIGN_OR_RETURN(Type triton_ty, TritonType(b, s.shape.element_type()));
    fn_arg_types.push_back(ttir::PointerType::get(
        StorageType(b, triton_ty), mlir::NVVM::kGlobalMemorySpace));
  }

  auto fn = b.create<ttir::FuncOp>(
      fn_name, b.getFunctionType(fn_arg_types, std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }

  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  std::string libdevice_path =
      GetLibdevicePath(fusion->GetModule()->config(), device_info);

  auto backend_config =
      fusion->backend_config<GpuBackendConfig>()->fusion_backend_config();
  absl::string_view fusion_kind = backend_config.kind();

  // It's okay for tma_metadata to be empty; it's only populated when used
  // explicitly.
  std::optional<stream_executor::gpu::TmaMetadata> tma_metadata = std::nullopt;
  if (fusion_kind == kTritonGemmFusionKind) {
    TF_ASSIGN_OR_RETURN(tma_metadata,
                        EmitMatMul(b, libdevice_path, device_info, fusion, fn,
                                   block_level_parameters));
  } else if (fusion_kind == kTritonFusionKind) {
    TF_RETURN_IF_ERROR(EmitGeneric(b, libdevice_path, device_info, fusion, fn,
                                   block_level_parameters));
  } else {
    return Internal("Unsupported fusion kind: %s", fusion_kind);
  }

  b.create<ttir::ReturnOp>();

  if (DumpingEnabledForHloModule(*hlo_computation->parent())) {
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "triton_ir", "before_validation.ttir",
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
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
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "triton_ir", "ttir",
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
  }

  return TritonModule{std::move(triton_module), tma_metadata};
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

  TF_ASSIGN_OR_RETURN(auto triton_module,
                      CreateTritonModule(fn_name, fusion, device_info,
                                         block_level_parameters, mlir_context));

  VLOG(3) << fusion->ToString(HloPrintOptions::ShortParsable());
  VLOG(3) << fusion->fused_instructions_computation()->ToString(
      HloPrintOptions::ShortParsable());

  // Compile Triton kernel to LLVM.
  const HloModule* hlo_module = fusion->GetModule();
  TF_ASSIGN_OR_RETURN(
      TritonWrapperResult result,
      CompileTritonToLLVM(hlo_module->config(), hlo_module->name(), device_info,
                          block_level_parameters, triton_module.module.get(),
                          llvm_module, mlir_context,
                          /*is_xla_fusion=*/true));
  result.tma_metadata = triton_module.tma_metadata;
  return result;
}

absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    const HloModuleConfig& hlo_config, absl::string_view hlo_module_name,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
    mlir::MLIRContext& mlir_context, bool is_xla_fusion, bool emit_kernel) {
  const auto& cc = device_info.gpu_compute_capability();
  const std::string arch_name =
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

  bool should_verify =
      (hlo_config.debug_options().xla_gpu_llvm_verification_level() >= 1);
#ifndef NDEBUG
  should_verify = true;
#endif

  mlir::PassManager pm(&mlir_context);
  pm.enableVerifier(should_verify);

  std::optional<llvm::raw_fd_ostream> log_stream;
  if (hlo_config.debug_options().xla_gpu_dump_llvmir()) {
    const std::string basename =
        absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module_name)),
                     ".triton-passes.log");
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = hlo_config.debug_options().xla_dump_to();
    }
    if (!outputs_dir.empty()) {
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
      LOG(ERROR) << "--xla_gpu_dump_llvmir is set, but neither the environment "
                 << "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                 << "--xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  // Lower affine expressions into arithmetic ops.
  pm.addPass(mlir::createLowerAffinePass());

  // Lower xla_gpu.apply_indexing into arithmetic ops.
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(CreateConvertIndexTypePass());

  mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
  if (!CreateTritonPipeline(&pm, arch_name, block_level_parameters.num_warps,
                            block_level_parameters.num_ctas,
                            block_level_parameters.num_stages, cluster_info,
                            is_xla_fusion)
           .ok()) {
    return Internal("Failed to create Triton pipeline.");
  }
  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(CreateGeneralizeKernelSignaturePass());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  if (log_stream.has_value()) {
    pm.printAsTextualPipeline(log_stream.value());
    log_stream->write("\n\n", 2);
  }
  bool succeeded = mlir::succeeded(pm.run(triton_module));

  if (log_stream.has_value()) {
    log_stream->flush();
  }

  if (!succeeded) {
    return Internal("Failed to compile Triton kernel.");
  }

  const int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (std::holds_alternative<se::CudaComputeCapability>(cc) &&
      shared_mem_bytes > device_info.shared_memory_per_block_optin()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Shared memory size limit exceeded: requested %d, available: %d",
        shared_mem_bytes, device_info.shared_memory_per_block_optin()));
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
    ll_triton_module->eraseNamedMDNode(
        ll_triton_module->getNamedMetadata("nvvm.annotations"));
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
  return {{shared_mem_bytes, cluster_dim}};
}

}  // namespace gpu
}  // namespace xla
