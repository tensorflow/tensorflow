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

#include "xla/service/gpu/fusions/triton/triton_test_utils.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/triton/triton_fusion_emitter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

bool SupportsBF16(const stream_executor::GpuComputeCapability& cc) {
  if (std::holds_alternative<stream_executor::CudaComputeCapability>(cc)) {
    return std::get<stream_executor::CudaComputeCapability>(cc).IsAtLeast(
        se::CudaComputeCapability::AMPERE);
  } else if (std::holds_alternative<stream_executor::RocmComputeCapability>(
                 cc)) {
    return std::get<stream_executor::RocmComputeCapability>(cc)
        .has_bf16_dtype_support();
  }
  CHECK(false);
}

absl::Status CreateTritonIrAndFileCheck(HloTestBase* test,
                                        absl::string_view hlo_text,
                                        absl::string_view triton_fusion_name,
                                        absl::string_view filecheck_pattern) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> verified_module,
                      test->ParseAndReturnVerifiedModule(hlo_text));
  auto* comp = verified_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr);
  auto fusion_backend_config = comp->FusionInstruction()
                                   ->backend_config<GpuBackendConfig>()
                                   ->fusion_backend_config();
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          fusion_backend_config.block_level_fusion_config());
  return CreateTritonIrAndFileCheck(*comp, block_level_parameters,
                                    filecheck_pattern);
}

absl::Status CreateTritonIrAndFileCheck(
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

absl::Status CreateTritonIrAndFileCheckForDot(
    HloTestBase* test, absl::string_view hlo_text,
    absl::string_view triton_fusion_name, absl::string_view filecheck_pattern) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> verified_module,
                      test->ParseAndReturnVerifiedModule(hlo_text));
  auto* comp = verified_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr);
  return CreateTritonIrAndFileCheck(*comp, /*block_level_parameters=*/{},
                                    filecheck_pattern);
}

absl::Status CreateTritonIrAndFileCheckForDot(
    const HloComputation& computation, absl::string_view filecheck_pattern) {
  return CreateTritonIrAndFileCheck(computation, /*block_level_parameters=*/{},
                                    filecheck_pattern);
}

absl::StatusOr<bool> ApplyFloatNormalization(
    HloModule* module, const stream_executor::GpuComputeCapability& cc) {
  const GpuFloatSupport bf16_support(cc, BF16);
  HloPassPipeline pipeline("hlo float normalization");
  pipeline.AddPass<FloatNormalization>(&bf16_support);
  return pipeline.Run(module);
}

namespace {

std::string PrimitiveTypeAndHloOpcodeToString(PrimitiveType data_type,
                                              HloOpcode opcode) {
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

std::string ComputeCapabilityToString(
    const stream_executor::GpuComputeCapability& cc) {
  if (auto cuda_cc = std::get_if<se::CudaComputeCapability>(&cc)) {
    return absl::StrReplaceAll(cuda_cc->ToString(), {{".", ""}});
  } else {
    CHECK(std::holds_alternative<se::RocmComputeCapability>(cc));
    return "rocm";
  }
}

}  // namespace

std::string TritonSupportTestTypeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, se::GpuComputeCapability>>& data) {
  auto [data_type, cc] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_", ComputeCapabilityToString(cc));
}

std::string TritonSupportTestTypeAndOpcodeToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode>>&
        data) {
  auto [data_type, opcode] = data.param;
  return PrimitiveTypeAndHloOpcodeToString(data_type, opcode);
}

std::string TritonSupportTestTypeAndOpcodeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>>& data) {
  auto [data_type, opcode, cc] = data.param;
  return absl::StrCat(PrimitiveTypeAndHloOpcodeToString(data_type, opcode), "_",
                      ComputeCapabilityToString(cc));
}

std::string TritonSupportTestTwoTypesAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, PrimitiveType, se::GpuComputeCapability>>&
        data) {
  auto [data_type_1, data_type_2, cc] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type_1),
                      "_",
                      primitive_util::LowercasePrimitiveTypeName(data_type_2),
                      "_", ComputeCapabilityToString(cc));
}

std::string TritonSupportTestTypeToString(
    const ::testing::TestParamInfo<PrimitiveType>& data) {
  return primitive_util::LowercasePrimitiveTypeName(data.param);
}

namespace {

// This function does nothing if the input module already has an entry
// computation whose root is a fusion. Otherwise, creates a new entry
// computation whose root is a fusion instruction that calls the original entry
// computation. The new fusion instruction uses the generic Triton backend kind.
absl::Status ConvertEntryToTritonFusion(HloModule* module) {
  if (module->entry_computation()->root_instruction()->opcode() ==
      HloOpcode::kFusion) {
    return absl::OkStatus();
  }
  auto builder = HloComputation::Builder("entry");
  std::vector<HloInstruction*> params;
  for (auto& param : module->entry_computation()->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(
        auto param_clone,
        builder.AddParameter(HloInstruction::CreateParameter(
            param->parameter_number(), param->shape(),
            absl::StrCat("param_", param->parameter_number()))));
    params.push_back(param_clone);
  }

  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      module->entry_computation()->root_instruction()->shape(),
      HloInstruction::FusionKind::kCustom, params,
      module->entry_computation()));

  gpu::GpuBackendConfig gpu_config;
  gpu_config.mutable_fusion_backend_config()->set_kind(
      std::string(kTritonFusionKind));
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  auto new_entry =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                /*is_entry=*/false);
  module->ReplaceEntryComputation(new_entry);
  return absl::OkStatus();
}

}  // namespace

DebugOptions TritonSupportTestBase::GetDebugOptionsForTest() const {
  auto options = HloTestBase::GetDebugOptionsForTest();
  // It's necessary to set this manually, because it's disabled in optimized
  // builds and there are some ASAN builds that run on TAP with -c opt.
  options.set_xla_gpu_llvm_verification_level(1);
  return options;
}

absl::StatusOr<TritonSupportTestBase::TestedInstruction>
TritonSupportTestBase::ParseTemplateAndGetInstruction(
    absl::string_view hlo_template, xla::PrimitiveType data_type,
    xla::HloOpcode opcode) {
  const std::string hlo_text = absl::Substitute(
      hlo_template, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnVerifiedModule(hlo_text));
  TF_RETURN_IF_ERROR(ConvertEntryToTritonFusion(module.get()));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  if (computation == module->entry_computation()) {
    return absl::InvalidArgumentError(
        "The `triton_computation` and the module's entry computation cannot be "
        "the same.");
  }
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
