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

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/cost_model/block_level_parameters.h"
#include "xla/backends/gpu/cost_model/triton_emitter_constraints.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/codegen/fusion_emitter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
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
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> xtile_dialect_module,
                      CreateXTileIrAndFileCheck(*comp, block_level_parameters,
                                                filecheck_pattern));
  return std::make_pair(std::move(xtile_dialect_module), std::move(hlo_module));
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
XTileTestBase::CreateXTileIrAndFileCheck(
    const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern) {
  auto* fusion = Cast<HloFusionInstruction>(computation.FusionInstruction());
  LoadMlirDialectsForTriton(*mlir_context());

  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          computation, mlir_context(),
          TritonEmitterConstraints::GetBuilder(
              TestGpuDeviceInfo::RTXA6000DeviceInfo()));

  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in EmitGeneric: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  const auto& symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(symbolic_tile_analysis_or);

  TF_ASSIGN_OR_RETURN(
      Tiling tiling,
      ir_emitter_triton_internal::TilingFromAnnotatedFusion(
          fusion, symbolic_tile_analysis, block_level_parameters));

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> xtile_dialect_module,
      xtile::EmitXTileModule("xtile_dialect_fn", fusion, symbolic_tile_analysis,
                             tiling, *mlir_context()));

  std::string out;
  llvm::raw_string_ostream os(out);
  xtile_dialect_module->print(os);
  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return xtile_dialect_module;
}

absl::Status XTileTestBase::LowerXTileIrToTritonAndFileCheck(
    mlir::ModuleOp xtile_dialect_module, absl::string_view filecheck_pattern,
    const HloFusionInstruction& fusion) {
  auto fusion_backend_config =
      fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          fusion_backend_config.block_level_fusion_config());
  TF_RETURN_IF_ERROR(ir_emitter_triton_internal::LowerXTileToTriton(
      xtile_dialect_module, *mlir_context(), fusion,
      TestGpuDeviceInfo::RTXH100SXMDeviceInfo(), block_level_parameters));

  std::string out;
  llvm::raw_string_ostream os(out);
  xtile_dialect_module->print(os);
  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
