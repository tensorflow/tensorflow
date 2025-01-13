/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/ir/dfs_hlo_visitor.h"

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {

template <typename HloInstructionPtr>
absl::Status DfsHloVisitorBase<HloInstructionPtr>::HandleElementwiseUnary(
    HloInstructionPtr hlo) {
  return Unimplemented("DfsHloVisitor::HandleElementwiseUnary: %s",
                       HloOpcodeString(hlo->opcode()));
}

template <typename HloInstructionPtr>
absl::Status DfsHloVisitorBase<HloInstructionPtr>::HandleElementwiseBinary(
    HloInstructionPtr hlo) {
  return Unimplemented("DfsHloVisitor::HandleElementwiseBinary: %s",
                       HloOpcodeString(hlo->opcode()));
}

template <typename HloInstructionPtr>
typename DfsHloVisitorBase<HloInstructionPtr>::VisitState
DfsHloVisitorBase<HloInstructionPtr>::GetVisitState(
    const HloInstruction& instruction) {
  return GetVisitState(instruction.unique_id());
}

template <typename HloInstructionPtr>
void DfsHloVisitorBase<HloInstructionPtr>::SetVisiting(
    const HloInstruction& instruction) {
  VLOG(3) << "marking HLO " << &instruction << " as visiting: ";
  DCHECK(NotVisited(instruction));
  visit_state_[instruction.unique_id()] = VisitState::kVisiting;
}

template <typename HloInstructionPtr>
void DfsHloVisitorBase<HloInstructionPtr>::SetVisited(
    const HloInstruction& instruction) {
  VLOG(3) << "marking HLO " << &instruction << " as visited: ";
  DCHECK(NotVisited(instruction) || IsVisiting(instruction));
  visit_state_[instruction.unique_id()] = VisitState::kVisited;
}

template <typename HloInstructionPtr>
absl::Status DfsHloVisitorBase<HloInstructionPtr>::Preprocess(
    HloInstructionPtr) {
  return absl::OkStatus();
}

template <typename HloInstructionPtr>
absl::Status DfsHloVisitorBase<HloInstructionPtr>::Postprocess(
    HloInstructionPtr) {
  return absl::OkStatus();
}

// Explicit instantiations.
template class DfsHloVisitorBase<HloInstruction*>;
template class DfsHloVisitorBase<const HloInstruction*>;

}  // namespace xla
