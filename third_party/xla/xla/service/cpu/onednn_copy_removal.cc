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

#include "xla/service/cpu/onednn_copy_removal.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/shape.h"

namespace xla {
namespace cpu {

// Individual pre-oneDNN operand cleanup: eliminate redundant kCopy ops
// feeding kDot and kConvolution operations before oneDNN pattern matching.
// This is needed to enable pattern matching for oneDNN quantization support.

absl::Status OneDnnOperandCopyRemovalVisitor::HandleCopy(HloInstruction* copy) {
  bool feeds_contraction = false;
  for (HloInstruction* user : copy->users()) {
    if (user->opcode() == HloOpcode::kConvolution ||
        user->opcode() == HloOpcode::kDot ||
        (user->opcode() == HloOpcode::kTranspose &&
         absl::c_any_of(user->users(), [](HloInstruction* u) {
           return u->opcode() == HloOpcode::kConvolution ||
                  u->opcode() == HloOpcode::kDot;
         }))) {
      feeds_contraction = true;
      break;
    }
  }
  if (!feeds_contraction) {
    return absl::OkStatus();
  }
  if (!copy->has_sharding() &&
      copy->GetModule()->entry_computation()->root_instruction() != copy &&
      Shape::Equal()(copy->shape(), copy->operand(0)->shape())) {
    RETURN_IF_ERROR(ReplaceInstruction(copy, copy->mutable_operand(0)));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> OneDnnOperandCopyRemoval::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config().use_spmd_partitioning() ||
      module->config().replica_count() > 1 ||
      module->config().num_partitions() > 1) {
    return false;
  }
  auto is_collective = [](const HloInstruction* instr) {
    HloOpcode op = instr->opcode();
    if (op == HloOpcode::kAsyncStart || op == HloOpcode::kAsyncDone) {
      op = instr->async_wrapped_opcode();
    }
    return hlo_query::IsCollectiveCommunicationOp(op) ||
           op == HloOpcode::kAllToAll || op == HloOpcode::kRaggedAllToAll;
  };
  bool has_dot_or_conv = false;
  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kDot ||
          instr->opcode() == HloOpcode::kConvolution) {
        has_dot_or_conv = true;
      }
      if (is_collective(instr) || instr->channel_id().has_value()) {
        return false;
      }
    }
  }
  if (!has_dot_or_conv) {
    return false;
  }
  OneDnnOperandCopyRemovalVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla
