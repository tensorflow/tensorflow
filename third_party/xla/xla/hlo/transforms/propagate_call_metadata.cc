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

#include "xla/hlo/transforms/propagate_call_metadata.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/stack_frames.h"
#include "xla/xla_data.pb.h"

namespace xla {

bool PropagateCallMetadata::UpdateOpName(OpMetadata& metadata,
                                         absl::string_view prefix) {
  if (prefix.empty()) {
    return false;
  }
  absl::string_view clean_prefix = absl::StripSuffix(prefix, "/");
  if (clean_prefix.empty()) {
    return false;
  }

  std::string op_name = metadata.op_name();
  absl::string_view clean_name = absl::StripPrefix(op_name, "/");
  clean_name = absl::StripSuffix(clean_name, "/");

  if (absl::StartsWith(clean_name, clean_prefix)) {
    return false;
  }
  if (clean_prefix.size() + clean_name.size() >= kMaxOpNameSize) {
    return false;
  }
  std::string result;
  if (clean_name.empty()) {
    result = std::string(clean_prefix);
  } else {
    result = absl::StrCat(clean_prefix, "/", clean_name);
  }
  metadata.set_op_name(std::move(result));
  return true;
}

bool PropagateCallMetadata::UpdateStackFrame(HloInstruction* hlo,
                                             StackFrameId parent_frame_id) {
  if (!parent_frame_id.valid()) {
    return false;
  }
  HloModule* module = hlo->GetModule();
  OpMetadata metadata = hlo->metadata();
  if (module->stack_frames().IsPrefix(
          parent_frame_id, StackFrameId{metadata.stack_frame_id()})) {
    return false;
  }
  metadata.set_stack_frame_id(
      module->mutable_stack_frames()
          .Concatenate(parent_frame_id, StackFrameId{metadata.stack_frame_id()})
          .value);
  hlo->set_metadata(metadata);
  return true;
}

void PropagateCallMetadata::PropagateMetadataToInstruction(
    HloInstruction* hlo, absl::string_view prefix,
    StackFrameId parent_frame_id) {
  if (prefix.empty() && !parent_frame_id.valid()) {
    return;
  }

  if (GetInstructionCallContext(hlo->opcode()) == CallContext::kControlFlow &&
      hlo->opcode() != HloOpcode::kCall) {
    for (HloComputation* computation : hlo->called_computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        PropagateMetadataToInstruction(instruction, prefix, parent_frame_id);
      }
    }
  }

  OpMetadata metadata = hlo->metadata();
  if (UpdateOpName(metadata, prefix)) {
    hlo->set_metadata(metadata);
  }
  UpdateStackFrame(hlo, parent_frame_id);
}

namespace {

bool PropagateIntoComputation(HloComputation* computation,
                              absl::string_view prefix,
                              StackFrameId parent_frame_id) {
  bool changed = false;
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    OpMetadata metadata = instr->metadata();
    if (PropagateCallMetadata::UpdateOpName(metadata, prefix)) {
      instr->set_metadata(metadata);
      changed = true;
    }
    if (PropagateCallMetadata::UpdateStackFrame(instr, parent_frame_id)) {
      changed = true;
    }

    if (GetInstructionCallContext(instr->opcode()) ==
            CallContext::kControlFlow &&
        instr->opcode() != HloOpcode::kCall) {
      for (HloComputation* sub : instr->called_computations()) {
        changed |= PropagateIntoComputation(sub, prefix, parent_frame_id);
      }
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> PropagateCallMetadata::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  auto computations = module->MakeNonfusionComputations(execution_threads);
  std::reverse(computations.begin(), computations.end());

  for (HloComputation* computation : computations) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kCall) {
        continue;
      }
      const OpMetadata& call_metadata = instr->metadata();
      absl::string_view prefix = call_metadata.op_name();
      StackFrameId parent_frame_id{call_metadata.stack_frame_id()};
      if (prefix.empty() && !parent_frame_id.valid()) {
        continue;
      }
      for (HloComputation* callee : instr->called_computations()) {
        changed |= PropagateIntoComputation(callee, prefix, parent_frame_id);
      }
    }
  }

  return changed;
}

}  // namespace xla
