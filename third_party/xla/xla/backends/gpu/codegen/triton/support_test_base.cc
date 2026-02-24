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

#include "xla/backends/gpu/codegen/triton/support_test_base.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

std::string PrimitiveTypeAndHloOpcodeToString(PrimitiveType data_type,
                                              HloOpcode opcode) {
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

}  // namespace

std::string ComputeCapabilityToString(
    const stream_executor::GpuComputeCapability& cc) {
  if (auto* cuda_cc = cc.cuda_compute_capability()) {
    return absl::StrReplaceAll(cuda_cc->ToString(), {{".", ""}});
  }
  CHECK(cc.IsRocm());
  return "rocm";
}

std::string SupportTestTypeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, se::GpuComputeCapability>>& data) {
  auto [data_type, cc] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_", ComputeCapabilityToString(cc));
}

std::string SupportTestTypeAndOpcodeToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode>>&
        data) {
  auto [data_type, opcode] = data.param;
  return PrimitiveTypeAndHloOpcodeToString(data_type, opcode);
}

std::string SupportTestTypeAndOpcodeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>>& data) {
  auto [data_type, opcode, cc] = data.param;
  return absl::StrCat(PrimitiveTypeAndHloOpcodeToString(data_type, opcode), "_",
                      ComputeCapabilityToString(cc));
}

std::string SupportTestTwoTypesAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, PrimitiveType, se::GpuComputeCapability>>&
        data) {
  auto [data_type_1, data_type_2, cc] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type_1),
                      "_",
                      primitive_util::LowercasePrimitiveTypeName(data_type_2),
                      "_", ComputeCapabilityToString(cc));
}

std::string SupportTestDeviceToString(
    const ::testing::TestParamInfo<se::GpuComputeCapability>& data) {
  return ComputeCapabilityToString(data.param);
}

namespace {

bool IsCollectiveFusion(const HloFusionInstruction& fusion) {
  return fusion.fused_expression_root()->opcode() == HloOpcode::kAllReduceDone;
}

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
      IsCollectiveFusion(*xla::Cast<HloFusionInstruction>(fusion))
          ? kTritonCollectiveFusionKind
          : kTritonNestedGemmFusionKind);
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  auto new_entry =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                /*is_entry=*/false);
  module->ReplaceEntryComputation(new_entry);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<SupportTestBase::TestedInstruction>
SupportTestBase::ParseTemplateAndGetInstruction(absl::string_view hlo_template,
                                                xla::PrimitiveType data_type,
                                                xla::HloOpcode opcode) {
  const std::string hlo_text = absl::Substitute(
      hlo_template, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      parse_module_callback_(hlo_text));
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
