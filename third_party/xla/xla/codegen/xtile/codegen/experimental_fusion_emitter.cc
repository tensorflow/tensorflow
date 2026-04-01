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

#include "absl/container/flat_hash_map.h"
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
    const ge::TiledHloInstruction& tiled_broadcast,
    absl::flat_hash_map<const ge::TiledHloInstruction*, TensorValue>& values) {
  ASSIGN_OR_RETURN(SmallVector<int64_t> input_tile_shape,
                   tiled_broadcast.operand(0)->tile().GetStaticTileSizes());
  ASSIGN_OR_RETURN(SmallVector<int64_t> output_tile_shape,
                   tiled_broadcast.tile().GetStaticTileSizes());
  if (input_tile_shape.empty() && output_tile_shape.empty()) {
    return values[tiled_broadcast.operand(0)];
  }
  CHECK(!output_tile_shape.empty());

  TensorValue input = values[tiled_broadcast.operand(0)];
  return xtile::BroadcastInDims(
      b, input, output_tile_shape,
      MakeArrayRef(tiled_broadcast.hlo()->dimensions()));
}

absl::StatusOr<TensorValue> EmitIota(mlir::ImplicitLocOpBuilder& b, Value pid,
                                     const ge::TiledHloInstruction& tiled_iota,
                                     const ::xla::IndexingMap& schedule) {
  const HloIotaInstruction* hlo_iota =
      ::xla::Cast<HloIotaInstruction>(tiled_iota.hlo());
  int64_t iota_dim = hlo_iota->iota_dimension();

  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_iota.tile().GetStaticTileSizes());

  // We can treat iota more or less as a parameter load, except that we need to
  // generate the right values in the right place as opposed to loading them.
  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(b, pid, tiled_iota, schedule));

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

absl::StatusOr<TensorValue> EmitPad(
    mlir::ImplicitLocOpBuilder& b, const ge::TiledHloInstruction& tiled_pad,
    absl::flat_hash_map<const ge::TiledHloInstruction*, TensorValue>& values,
    Value pid, const ::xla::IndexingMap& schedule) {
  ASSIGN_OR_RETURN(SmallVector<int64_t> tile_sizes,
                   tiled_pad.tile().GetStaticTileSizes());

  const ge::TiledHloInstruction* tiled_operand = tiled_pad.operand(0);
  const auto& pad_input_shape = tiled_operand->hlo()->shape().dimensions();

  // Compute tile offsets.
  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(b, pid, tiled_pad, schedule));
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
    return values[tiled_operand];
  }
  const ge::TiledHloInstruction* padding_value = tiled_pad.operand(1);

  TensorValue pad_value_splat =
      xtile::Splat(b, values[padding_value], tile_sizes);
  return mlir::cast<TensorValue>(
      arith::SelectOp::create(b, mask, values[tiled_operand], pad_value_splat)
          .getResult());
}

absl::StatusOr<TensorValue> EmitTiledHloInstruction(
    ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const ge::TiledHloInstruction& tiled_hlo,
    const ::xla::IndexingMap& schedule, FunctionOpInterface fn, Value pid,
    absl::flat_hash_map<const ge::TiledHloInstruction*, TensorValue>& values) {
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
    ASSIGN_OR_RETURN(TileInfo tile_info,
                     TileInfo::Construct(b, pid, tiled_hlo, schedule));
    TensorValue parameter =
        EmitParameterExtract(b, tile_info, fn.getArgument(arg_index));

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
            fusion->called_computation()->ToString()));
      }
      parameter =
          mlir::cast<TensorValue>(Cast(b, parameter, expected_element_type));
    }

    return parameter;
  }
  std::vector<Value> operands;
  operands.reserve(hlo->operands().size());
  for (const ge::TiledHloInstruction* operand : tiled_hlo.operands()) {
    operands.push_back(values[operand]);
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
      return EmitBroadcast(b, tiled_hlo, values);
    }
    case HloOpcode::kConstant: {
      if (ShapeUtil::IsEffectiveScalar(hlo->shape())) {
        return EmitConstant(b, *hlo);
      }
      return absl::UnimplementedError(
          absl::StrCat("Unsupported non-scalar constant ", hlo->ToString()));
    }
    case HloOpcode::kIota: {
      return EmitIota(b, pid, tiled_hlo, schedule);
    }
    case HloOpcode::kSlice: {
      return values[tiled_hlo.operand(0)];
    }
    case HloOpcode::kPad: {
      return EmitPad(b, tiled_hlo, values, pid, schedule);
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
    mlir::ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const ::xla::gpu::experimental::TiledHloComputation& tiled_computation,
    const ::xla::IndexingMap& schedule, xtile::EntryFuncOp fn, Value pid,
    absl::flat_hash_map<const ge::TiledHloInstruction*, TensorValue>& values) {
  VLOG(2) << "EmitTiledComputation: " << tiled_computation.ToString();
  for (const auto& tiled_hlo : tiled_computation.tiled_hlo_instructions()) {
    const HloInstruction* hlo = tiled_hlo->hlo();
    ASSIGN_OR_RETURN(TensorValue result,
                     EmitTiledHloInstruction(b, fusion, *tiled_hlo, schedule,
                                             fn, pid, values));
    TF_RET_CHECK(values.insert({tiled_hlo.get(), result}).second)
        << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  auto roots = tiled_computation.roots();
  std::vector<TensorValue> results;
  results.reserve(roots.size());
  for (const auto* root : roots) {
    results.push_back(values[root]);
  }
  return std::move(results);
}

absl::Status EmitGeneric(
    ImplicitLocOpBuilder& b, const HloFusionInstruction* fusion,
    const ::xla::gpu::experimental::TiledHloComputation& tiled_computation,
    const ::xla::IndexingMap& schedule, xtile::EntryFuncOp fn,
    MLIRContext* mlir_context) {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Emitting XTile IR for fusion\n"
            << ExtractInstructionIntoNewModule(*fusion)->ToString();
    VLOG(6) << "Tiled computation: \n" << tiled_computation.ToString();
  }
  Value tile_id = fn.getTileId();
  absl::flat_hash_map<const gpu::experimental::TiledHloInstruction*,
                      TensorValue>
      values;
  ASSIGN_OR_RETURN(auto results,
                   EmitTiledComputation(b, fusion, tiled_computation, schedule,
                                        fn, tile_id, values));
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

    ASSIGN_OR_RETURN(auto tile_info,
                     TileInfo::Construct(b, tile_id, *root, schedule));

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
