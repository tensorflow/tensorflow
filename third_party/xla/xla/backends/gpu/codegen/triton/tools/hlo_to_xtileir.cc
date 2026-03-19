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

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

absl::Status RealMain(absl::string_view input_file) {
  ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   xla::LoadModuleFromFile(std::string(input_file)));

  HloInstruction* fusion = hlo_module->entry_computation()->root_instruction();
  if (!fusion->IsCustomFusion()) {
    return absl::InvalidArgumentError("Instruction is not a custom fusion.");
  }

  ASSIGN_OR_RETURN(auto gpu_config, fusion->backend_config<GpuBackendConfig>());
  const auto* fusion_instr = Cast<HloFusionInstruction>(fusion);
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (!backend_config.has_block_level_fusion_config()) {
    return absl::InvalidArgumentError(
        "Fusion backend config must have block_level_fusion_config.");
  }
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          backend_config.block_level_fusion_config());

  mlir::MLIRContext mlir_context;
  // Note that CreateTritonModule creates an xtile dialect module that
  // CreateTritonXlaPipeline() will lower to TTIR.
  ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> triton_module,
                   CreateTritonModule("triton_fn", fusion_instr,
                                      TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                                      block_level_parameters, mlir_context,
                                      /*use_experimental_tiling=*/false));
  triton_module->print(llvm::outs());
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  CHECK_EQ(argc, 2) << "Must specify an input file";
  CHECK_OK(xla::gpu::RealMain(argv[1]));
  return 0;
}
