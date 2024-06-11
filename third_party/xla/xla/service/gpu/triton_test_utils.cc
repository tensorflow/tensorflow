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

#include "xla/service/gpu/triton_test_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/triton.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

bool TritonTest::SkipBF16Tests() {
  if (std::holds_alternative<stream_executor::RocmComputeCapability>(
          GpuComputeComp())) {
    auto rcc = device_desc().rocm_compute_capability();
    return !rcc.has_bf16_dtype_support();
  }
  return false;
}

stream_executor::GpuComputeCapability TritonTest::CudaAmpereOrRocm() {
  if (std::holds_alternative<stream_executor::RocmComputeCapability>(
          GpuComputeComp())) {
    return stream_executor::GpuComputeCapability{
        device_desc().rocm_compute_capability()};
  } else {
    return stream_executor::GpuComputeCapability{
        stream_executor::CudaComputeCapability{
            stream_executor::CudaComputeCapability::AMPERE, 0}};
  }
}

absl::Status TritonFilecheckTest::CreateTritonIrAndFileCheck(
    absl::string_view hlo_text, const TritonGemmConfig& config,
    std::vector<int64_t> output_tile_sizes, TritonIrEmitter emitter,
    absl::string_view triton_fusion_name, absl::string_view filecheck_pattern) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> verified_module,
                      ParseAndReturnVerifiedModule(hlo_text));
  auto* comp = verified_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr);
  return CreateTritonIrAndFileCheck(*comp, config, output_tile_sizes, emitter,
                                    filecheck_pattern);
}

absl::Status TritonFilecheckTest::CreateTritonIrAndFileCheck(
    const HloComputation& computation, const TritonGemmConfig& config,
    std::vector<int64_t> output_tile_sizes, TritonIrEmitter emitter,
    absl::string_view filecheck_pattern) {
  auto* fusion = Cast<HloFusionInstruction>(computation.FusionInstruction());

  TF_ASSIGN_OR_RETURN(auto analysis,
                      TritonFusionAnalysis::Execute(computation));

  auto fusion_analysis = HloFusionAnalysis::Create(fusion, &device_desc());

  if (fusion_analysis.fusion_backend_config().kind() ==
      kTritonSoftmaxFusionKind) {
    TritonFusion triton_fusion(fusion_analysis);
    if (auto launch_config = triton_fusion.launch_config()) {
      output_tile_sizes = launch_config->output_tile_sizes;
    }
  }

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      auto module,
      CreateTritonModule(analysis, "triton_fn", fusion,
                         TestGpuDeviceInfo::RTXA6000DeviceInfo(), config,
                         output_tile_sizes, emitter, context));

  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os);
  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
