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

#include <optional>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::xtile {
namespace {

using ::llvm::SmallVector;
using ::mlir::MLIRContext;
using ::mlir::Type;

absl::Status EmitGeneric(
    mlir::OpBuilder builder, const HloFusionInstruction* fusion,
    const ::xla::gpu::experimental::TiledHloComputation& tiled_computation,
    xtile::EntryFuncOp fn, MLIRContext* mlir_context) {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Emitting XTile IR for fusion\n"
            << ExtractInstructionIntoNewModule(*fusion)->ToString();
    VLOG(6) << "Tiled computation: \n" << tiled_computation.ToString();
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
    const std::optional<stream_executor::GpuComputeCapability>& gpu_cc) {
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
      TF_ASSIGN_OR_RETURN(ir_type, PrimitiveTypeToMlirType(b, type, gpu_cc));
    }
    fn_arg_types.push_back(GetMemRefType(p->shape(), ir_type));
  }

  for (const auto& [index, shape] : ShapeUtil::GetLeafShapes(fusion->shape())) {
    TF_ASSIGN_OR_RETURN(
        Type ir_type, PrimitiveTypeToMlirType(b, shape.element_type(), gpu_cc));
    fn_arg_types.push_back(GetMemRefType(shape, ir_type));
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

  TF_RETURN_IF_ERROR(
      EmitGeneric(b, fusion, tiled_computation, fn, &mlir_context));

  b.create<xtile::EntryFuncReturnOp>();

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
