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

#include "xla/backends/gpu/codegen/triton/xtile_test_base.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/codegen/experimental_fusion_emitter.h"
#include "xla/codegen/xtile/codegen/fusion_emitter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/tiling_from_block_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla::gpu {

absl::StatusOr<
    std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::unique_ptr<HloModule>>>
XTileTestBase::CreateXTileIrAndFileCheck(std::unique_ptr<HloModule> hlo_module,
                                         absl::string_view triton_fusion_name,
                                         absl::string_view filecheck_pattern) {
  auto* comp = hlo_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr) << absl::StrCat(
      "Computation '", triton_fusion_name, "' is not found in the module");
  auto fusion_backend_config = comp->FusionInstruction()
                                   ->backend_config<GpuBackendConfig>()
                                   ->fusion_backend_config();
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          fusion_backend_config.block_level_fusion_config());
  ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> xtile_dialect_module,
                   CreateXTileIrAndFileCheck(*comp, block_level_parameters,
                                             filecheck_pattern));
  return std::make_pair(std::move(xtile_dialect_module), std::move(hlo_module));
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
CreateXTileIrAndFileCheckLegacy(
    mlir::MLIRContext* mlir_context, const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern) {
  auto* fusion = Cast<HloFusionInstruction>(computation.FusionInstruction());
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          computation, mlir_context,
          TritonEmitterConstraints::GetBuilder(
              TestGpuDeviceInfo::RTXA6000DeviceInfo()));

  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in EmitGeneric: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  const auto& symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(symbolic_tile_analysis_or);

  ASSIGN_OR_RETURN(Tiling tiling,
                   TilingFromAnnotatedFusion(symbolic_tile_analysis,
                                             block_level_parameters));

  ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> xtile_dialect_module,
      xtile::EmitXTileModule("xtile_dialect_fn", *fusion,
                             symbolic_tile_analysis, tiling, *mlir_context));
  return xtile_dialect_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
XTileTestBase::CreateXTileIrAndFileCheck(
    const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern) {
  mlir::OwningOpRef<mlir::ModuleOp> xtile_dialect_module;
  LoadMlirDialectsForTriton(*mlir_context());
  if (computation.parent()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_tiling_propagation()) {
    namespace ge = ::xla::gpu::experimental;
    auto* fusion = Cast<HloFusionInstruction>(computation.FusionInstruction());
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(fusion);
    ASSIGN_OR_RETURN(std::unique_ptr<ge::TilingSpace> tiling_space,
                     ge::TilingSpace::Create(*fusion_adaptor, mlir_context()));
    ASSIGN_OR_RETURN(
        llvm::SmallVector<int64_t> concrete_sizes,
        GetTilingSpaceConcreteSizes(
            *tiling_space, block_level_parameters,
            computation.parent()
                ->config()
                .debug_options()
                .xla_experimental_enable_same_shape_multi_output_fusion()));
    RETURN_IF_ERROR(tiling_space->AssignTileSizes(
        xtile::GetPaddedTileSizes(concrete_sizes)));
    ASSIGN_OR_RETURN(ge::TiledHloComputation tiled_computation,
                     ge::TiledHloComputation::Tile(*fusion_adaptor,
                                                   std::move(tiling_space)));
    tiled_computation.Simplify();
    tiled_computation.SortInstructionsPostOrder();
    if (Decision constraints = ge::VerifyTritonConstraints(
            tiled_computation, TestGpuDeviceInfo::RTXA6000DeviceInfo());
        !constraints) {
      return absl::InternalError(
          absl::StrCat("Triton constraints violated during test codegen: ",
                       constraints.Explain()));
    }
    ASSIGN_OR_RETURN(
        xtile_dialect_module,
        xtile::EmitXTileModule("xtile_dialect_fn", *fusion, tiled_computation,
                               *mlir_context()));
  } else {
    ASSIGN_OR_RETURN(xtile_dialect_module,
                     CreateXTileIrAndFileCheckLegacy(
                         mlir_context(), computation, block_level_parameters,
                         filecheck_pattern));
  }
  std::string out;
  llvm::raw_string_ostream os(out);
  xtile_dialect_module->print(os);
  ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return std::move(xtile_dialect_module);
}

absl::Status XTileTestBase::LowerXTileIrToTritonAndFileCheck(
    mlir::ModuleOp xtile_dialect_module, absl::string_view filecheck_pattern,
    const HloFusionInstruction& fusion) {
  auto fusion_backend_config =
      fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          fusion_backend_config.block_level_fusion_config());
  RETURN_IF_ERROR(ir_emitter_triton_internal::LowerXTileToTriton(
      xtile_dialect_module, *mlir_context(), fusion,
      TestGpuDeviceInfo::H100SXMDeviceInfo(), block_level_parameters));

  std::string out;
  llvm::raw_string_ostream os(out);
  xtile_dialect_module->print(os);
  ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
