/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hlo_module_map.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/profiler/utils/hlo_cost_analysis_wrapper.h"
#include "tensorflow/core/profiler/utils/hlo_module_utils.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tensorflow/core/profiler/utils/hlo_proto_to_module.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace profiler {

HloInstructionWrapper::HloInstructionWrapper(
    const xla::HloInstruction* instr, const xla::HloCostAnalysis* cost_analysis)
    : instr_(instr),
      op_full_name_(
          tsl::profiler::TraceMeOp(Metadata().op_name(), Metadata().op_type())),
      tf_op_name_(tsl::profiler::TfOpFullname(Metadata().op_type(),
                                              Metadata().op_name())),
      category_(instr_->ToCategory()),
      expression_(UncachedExpression(instr_, false, kMaxHlolNameSize)) {
  ProcessXlaCostAnalysis(cost_analysis);
}

HloModuleWrapper::HloModuleWrapper(
    const xla::HloProto& hlo_proto,
    std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper)
    : HloModuleWrapper(ConvertHloProtoToModuleIgnoringErrors(hlo_proto),
                       std::move(cost_analysis_wrapper)) {}

HloModuleWrapper::HloModuleWrapper(
    std::unique_ptr<xla::HloModule> module,
    std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper)
    : module_(std::move(module)) {
  if (module_ == nullptr) return;

  xla::HloCostAnalysis* cost_analysis = nullptr;
  if (cost_analysis_wrapper != nullptr &&
      InitializeHloCostAnalysis(*module_,
                                *cost_analysis_wrapper->GetXlaCostAnalysis())
          .ok()) {
    cost_analysis = cost_analysis_wrapper->GetXlaCostAnalysis();
  }

  // Populate instructions_by_name_ with module.
  for (const xla::HloComputation* computation : module_->computations()) {
    for (const xla::HloInstruction* instr : computation->instructions()) {
      instructions_by_name_.try_emplace(
          instr->name(), HloInstructionWrapper(instr, cost_analysis));
    }
  }
  // Gather nested fusion instructions.
  for (const xla::HloComputation* computation : module_->computations()) {
    // Some modules still seem to have "dead" fusions computations. In this
    // case, IsFusionComputation() = true but there is no parent
    // FusionInstruction().
    if (computation->FusionInstruction() != nullptr) {
      GatherFusionInstructions(computation->FusionInstruction());
    }
  }
}

// Function to gather all the instructions in a fusion computation.
void HloModuleWrapper::GatherFusionInstructions(xla::HloInstruction* inst) {
  HloInstructionWrapper* fused_inst_wrapper =
      GetMutableHloInstruction(inst->name());
  DCHECK(fused_inst_wrapper != nullptr);
  if (!fused_inst_wrapper->FusedChildren().empty()) return;
  for (auto* fused : inst->fused_instructions()) {
    const auto child_inst_wrapper = GetHloInstruction(fused->name());
    DCHECK(child_inst_wrapper != nullptr);
    fused_inst_wrapper->AddFusedChild(child_inst_wrapper);
    if (fused->opcode() == xla::HloOpcode::kFusion) {
      GatherFusionInstructions(fused);
    }
  }
}

HloInstructionWrapper* HloModuleWrapper::GetMutableHloInstruction(
    absl::string_view hlo_name) {
  auto it = instructions_by_name_.find(hlo_name);
  if (it != instructions_by_name_.end()) return &it->second;
  return nullptr;
}

const HloInstructionWrapper* HloModuleWrapper::GetHloInstruction(
    absl::string_view hlo_name) const {
  auto it = instructions_by_name_.find(hlo_name);
  if (it != instructions_by_name_.end()) return &it->second;
  return nullptr;
}

std::string HloInstructionWrapper::source_info() const {
  if (!Metadata().source_file().empty()) {
    return absl::StrCat(io::Basename(Metadata().source_file()), ":",
                        Metadata().source_line());
  } else {
    return std::string();
  }
}

void AddHloProto(HloModuleMap& hlo_module_map, uint64_t program_id,
                 const xla::HloProto& hlo_proto,
                 std::unique_ptr<HloCostAnalysisWrapper> cost_analysis) {
  auto hlo_module = ConvertHloProtoToModule(hlo_proto);
  if (!hlo_module.ok()) {
    LOG(ERROR) << hlo_module.status();
    return;
  }
  hlo_module_map.try_emplace(program_id,
                             HloModuleWrapper(std::move(hlo_module).value(),
                                              std::move(cost_analysis)));
}

void ProcessHloModuleMapFromXSpace(HloModuleMap& hlo_module_map,
                                   const XSpace* space) {
  for (auto& [program_id, hlo_proto] : ParseHloProtosFromXSpace(*space)) {
    AddHloProto(hlo_module_map, program_id, *hlo_proto);
  }
}

absl::Status InitializeHloCostAnalysis(const xla::HloModule& hlo_module,
                                       xla::HloCostAnalysis& cost_analysis) {
  const xla::HloComputation* hlo_computation = hlo_module.entry_computation();
  cost_analysis.ReserveVisitStates(hlo_computation->instruction_count());
  absl::Status analysis_status = hlo_computation->Accept(&cost_analysis);
  if (analysis_status.ok()) {
    cost_analysis.DestroyVisitState();
  }
  return analysis_status;
}

}  // namespace profiler
}  // namespace tensorflow
