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

#include "tensorflow/compiler/xla/service/gpu/hlo_fusion_stats.h"

#include <string>

#include "absl/strings/match.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

class OpcodeCollector : public ConstDfsHloVisitorWithDefault {
 public:
  std::set<std::string> GetUniqueOpcodes() { return opcodes_; }

 protected:
  Status DefaultAction(const xla::HloInstruction* instr) final {
    switch (instr->opcode()) {
      case HloOpcode::kConstant:
        break;
      case HloOpcode::kParameter:
        break;
      default:
        // TODO(tjoerg): Consider to aggregrate ops that are very similar, e.g.
        // cwise ops.
        opcodes_.insert(HloOpcodeString(instr->opcode()));
    }
    return Status::OK();
  }

 private:
  std::set<std::string> opcodes_;
};

std::set<std::string> GetUniqueOpcodes(HloComputation* computation) {
  OpcodeCollector collector;
  if (computation->Accept(&collector) != Status::OK()) {
    return {};
  }
  return collector.GetUniqueOpcodes();
}

}  // namespace

std::string HloOpcodeHistogram::ToString() {
  std::string result;
  for (const auto& entry : *this) {
    absl::StrAppend(&result, "{", absl::StrJoin(entry.first, ", "),
                    "}: ", entry.second, "\n");
  }
  return result;
}

Status HloFusionStatsVisitor::RunOnModule(HloModule* module) {
  TF_RETURN_IF_ERROR(module->entry_computation()->Accept(this));
  return Status::OK();
}

std::string HloFusionStatsVisitor::ToString() {
  return absl::StrCat("HLO Fusion Stats:\n",
                      "Number of fusion ops: ", num_fusions_, "\n",
                      "Number of kLoop fusions: ", num_loop_fusions_, "\n",
                      loop_fusion_opcode_histogram_.ToString(), "\n",
                      "Number of kInput fusions: ", num_input_fusions_, "\n",
                      input_fusion_opcode_histogram_.ToString());
}

Status HloFusionStatsVisitor::DefaultAction(const xla::HloInstruction* instr) {
  return Status::OK();
}

Status HloFusionStatsVisitor::HandleFusion(const HloInstruction* fusion) {
  num_fusions_++;
  std::set<std::string> opcodes =
      GetUniqueOpcodes(fusion->fused_instructions_computation());
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    num_loop_fusions_++;
    loop_fusion_opcode_histogram_[opcodes]++;
  } else if (fusion->fusion_kind() == HloInstruction::FusionKind::kInput) {
    num_input_fusions_++;
    input_fusion_opcode_histogram_[opcodes]++;
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
