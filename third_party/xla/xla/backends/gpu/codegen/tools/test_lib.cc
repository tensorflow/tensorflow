/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/tools/test_lib.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/status_macros.h"
#include "xla/tools/hlo_module_loader.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<HloModule>> LoadTestModule(
    absl::string_view filename) {
  auto module = *xla::LoadModuleFromFile(std::string(filename));
  int num_fusions = absl::c_count_if(
      module->entry_computation()->instructions(),
      [](const HloInstruction* instruction) {
        return instruction->opcode() == xla::HloOpcode::kFusion;
      });
  TF_RET_CHECK(num_fusions <= 1) << "HLO must contain at most one fusion";

  if (num_fusions == 0) {
    // Generate a fusion from the entry computation.
    HloComputation::Builder builder("generated_main");
    std::vector<HloInstruction*> params;
    for (const auto* param :
         module->entry_computation()->parameter_instructions()) {
      params.push_back(*builder.AddParameter(param->Clone(/*suffix=*/"")));
    }
    builder.AddInstruction(HloInstruction::CreateFusion(
        module->entry_computation()->root_instruction()->shape(),
        HloInstruction::FusionKind::kLoop /* irrelevant */, params,
        module->entry_computation()));

    auto* new_entry = module->AddComputationAndUnifyNamesAndIds(
        builder.Build(), /*is_entry=*/false);
    module->ReplaceEntryComputation(new_entry);
    *module->mutable_entry_computation_layout() =
        module->compute_computation_layout();
  }

  return module;
}

absl::StatusOr<std::unique_ptr<EmitterData>> GetEmitter(
    const HloModule& module) {
  auto data = std::make_unique<EmitterData>();
  data->fusion = DynCast<HloFusionInstruction>(
      module.entry_computation()->root_instruction());
  TF_RET_CHECK(data->fusion != nullptr) << "Root instruction must be a fusion";
  data->device.emplace(TestGpuDeviceInfo::RTXA6000DeviceInfo());
  data->analysis.emplace(
      HloFusionAnalysis::Create(*data->fusion, data->device.value()));
  PreBufferAssignmentFusionInfo info(data->analysis.value());
  auto fusion_emitter = GetFusionEmitter(info);

  auto emitter = dynamic_cast<EmitterBase*>(fusion_emitter.get());
  TF_RET_CHECK(emitter != nullptr) << "Expected emitter to be an EmitterBase";

  fusion_emitter.release();
  data->emitter.reset(emitter);
  return data;
}

mlir::MLIRContext GetMlirContextForTest() {
  return mlir::MLIRContext(EmitterBase::GetDialectRegistry());
}

}  // namespace gpu
}  // namespace xla
