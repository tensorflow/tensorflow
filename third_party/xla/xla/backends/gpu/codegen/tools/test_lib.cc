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

#include "absl/status/statusor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/status_macros.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<EmitterData>> GetEmitter(
    const HloModule& module, mlir::MLIRContext& mlir_context) {
  auto data = std::make_unique<EmitterData>();
  {
    auto* inst = module.entry_computation()->root_instruction();
    while (inst && (inst->opcode() == HloOpcode::kTuple ||
                    inst->opcode() == HloOpcode::kGetTupleElement)) {
      inst = inst->mutable_operand(0);
    }
    data->fusion = DynCast<HloFusionInstruction>(inst);
  }
  TF_RET_CHECK(data->fusion != nullptr) << "Root instruction must be a fusion";
  data->device.emplace(TestGpuDeviceInfo::RTXA6000DeviceInfo());
  data->analysis.emplace(
      HloFusionAnalysis::Create(*data->fusion, data->device.value()));
  PreBufferAssignmentFusionInfo info(data->analysis.value());
  auto fusion_emitter = GetFusionEmitter(info, &mlir_context);

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
