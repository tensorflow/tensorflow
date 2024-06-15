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

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/float_normalization.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/hlo_pass_pipeline.h"
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
  return !GetCudaComputeCapability().IsAtLeast(
      se::CudaComputeCapability::AMPERE);
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
    absl::string_view hlo_text,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view triton_fusion_name, absl::string_view filecheck_pattern) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> verified_module,
                      ParseAndReturnVerifiedModule(hlo_text));
  auto* comp = verified_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr);
  return CreateTritonIrAndFileCheck(*comp, block_level_parameters,
                                    filecheck_pattern);
}

absl::Status TritonFilecheckTest::CreateTritonIrAndFileCheck(
    const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern) {
  auto* fusion = Cast<HloFusionInstruction>(computation.FusionInstruction());

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      auto module, CreateTritonModule("triton_fn", fusion,
                                      TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                                      block_level_parameters, context));

  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os);
  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return absl::OkStatus();
}

absl::Status TritonFilecheckTest::CreateTritonIrAndFileCheckForDot(
    absl::string_view hlo_text, absl::string_view triton_fusion_name,
    absl::string_view filecheck_pattern) {
  return CreateTritonIrAndFileCheck(hlo_text, /*block_level_parameters=*/{},
                                    triton_fusion_name, filecheck_pattern);
}

absl::Status TritonFilecheckTest::CreateTritonIrAndFileCheckForDot(
    const HloComputation& computation, absl::string_view filecheck_pattern) {
  return CreateTritonIrAndFileCheck(computation, /*block_level_parameters=*/{},
                                    filecheck_pattern);
}

absl::StatusOr<bool> TritonSupportTest::ApplyFloatNormalization(
    HloModule* module) {
  const GpuFloatSupport bf16_support(GetCudaComputeCapability(), BF16);
  HloPassPipeline pipeline("hlo float normalization");
  pipeline.AddPass<FloatNormalization>(&bf16_support);
  return pipeline.Run(module);
}

std::string TritonSupportTestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode>>&
        data) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = data.param;
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

absl::StatusOr<TritonSupportTest::TestedInstruction>
TritonSupportTest::ParseTemplateAndGetInstruction(
    absl::string_view hlo_template, xla::PrimitiveType data_type,
    xla::HloOpcode opcode) {
  const std::string hlo_text = absl::Substitute(
      hlo_template, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  if (fusion == nullptr) {
    return absl::InvalidArgumentError(
        "The computation's entry root is not a fusion.");
  }
  if (computation == nullptr) {
    return absl::InvalidArgumentError(
        "No computation with the name `triton_computation` found.");
  }
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (instr == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "No instruction with opcode [%s] found.", HloOpcodeString(opcode)));
  }
  return TestedInstruction(std::move(module), *instr);
}

}  // namespace xla::gpu
