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
#include "xla/service/gpu/fusions/custom.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernels/custom_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/runtime3/kernel_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/statusor.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<std::unique_ptr<Thunk>> BuildCustomKernelThunkForFusion(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    mlir::lmhlo::FusionOp fusion_op, CustomKernel custom_kernel) {
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      ir_emitter_context.emit_ir_from_hlo()
                          ? KernelArguments::Create(
                                ir_emitter_context.buffer_assignment(), &fusion)
                          : KernelArguments::Create(
                                ir_emitter_context.allocations(), fusion_op));

  std::variant<mlir::Operation*, const HloInstruction*> instr;
  if (ir_emitter_context.emit_ir_from_hlo()) {
    instr = &fusion;
  } else {
    instr = fusion_op;
  }

  return std::make_unique<CustomKernelThunk>(
      instr, std::move(custom_kernel), std::move(kernel_arguments.args()));
}

}  // namespace

absl::StatusOr<FusionEmissionResult> CustomFusionEmitter::Emit(
    IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
    const HloFusionInstruction& fusion) const {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  const auto& config = backend_config.custom_fusion_config();

  VLOG(3) << "Lower HLO fusion to a custom fusion " << config.name();

  auto* registry = CustomFusionRegistry::Default();
  auto* custom_fusion = registry->Lookup(config.name());

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_fusion == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Custom fusion ", config.name(), " not found in a default registry."));
  }

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(
      std::vector<CustomKernel> kernels,
      custom_fusion->LoadKernels(ir_emitter_context.gpu_device_info(),
                                 fusion.fused_instructions_computation()));

  // This should never happen, it means that compilation pipeline created a
  // fusion operation that is not supported by a given custom fusion.
  if (kernels.empty()) {
    return absl::InternalError(
        absl::StrCat("Custom fusion ", config.name(),
                     " returned empty custom kernels for a fused computation"));
  }

  // TODO(ezhulenev): Add support for auto tuning to select the best kernel.
  if (kernels.size() != 1) {
    return absl::InternalError("Expected exactly one custom kernel");
  }

  TF_ASSIGN_OR_RETURN(auto thunk, BuildCustomKernelThunkForFusion(
                                      ir_emitter_context, fusion, fusion_op,
                                      std::move(kernels[0])));

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

}  // namespace gpu
}  // namespace xla
