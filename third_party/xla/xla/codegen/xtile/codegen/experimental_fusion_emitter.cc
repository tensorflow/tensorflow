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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/tiling/experimental/scheduling.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::xtile {
namespace {

using ::llvm::SmallVector;
using ::mlir::FunctionOpInterface;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::Type;
using ::mlir::Value;
using ::stream_executor::GpuComputeCapability;

namespace ge = ::xla::gpu::experimental;

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
    TF_ASSIGN_OR_RETURN(TileInfo tile_info,
                        TileInfo::Construct(b, pid, tiled_hlo, schedule));
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

  if (hlo->IsElementwise()) {
    std::vector<Value> operands;
    operands.reserve(hlo->operands().size());

    for (const ge::TiledHloInstruction* operand : tiled_hlo.operands()) {
      operands.push_back(values[operand]);
    }
    TF_ASSIGN_OR_RETURN(Value result, EmitElementwise(b, *hlo, operands));
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
    TF_ASSIGN_OR_RETURN(TensorValue result,
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
  TF_ASSIGN_OR_RETURN(
      auto results, EmitTiledComputation(b, fusion, tiled_computation, schedule,
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

    TF_ASSIGN_OR_RETURN(auto tile_info,
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
  TF_ASSIGN_OR_RETURN(SmallVector<Type> fn_arg_types,
                      GetFnArgTypes(b, fusion, opaque_args_types, gpu_cc));
  // Metadata arguments are opaque to the tiling infra.
  llvm::SmallVector<mlir::NamedAttribute> named_attributes{b.getNamedAttr(
      "num_opaque_args", b.getI32IntegerAttr(opaque_args_types.size()))};

  auto fn = xtile::EntryFuncOp::create(b, fn_name, fn_arg_types,
                                       named_attributes, {});
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  TF_ASSIGN_OR_RETURN(auto schedule, Schedule(tiled_computation));
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
