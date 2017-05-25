/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"

#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

Status DfsHloVisitor::HandleElementwiseUnary(HloInstruction* hlo,
                                             HloOpcode opcode,
                                             HloInstruction* operand) {
  return Unimplemented("DfsHloVisitor::HandleElementwiseUnary: %s",
                       HloOpcodeString(opcode).c_str());
}

Status DfsHloVisitor::HandleElementwiseBinary(HloInstruction* hlo,
                                              HloOpcode opcode,
                                              HloInstruction* lhs,
                                              HloInstruction* rhs) {
  return Unimplemented("DfsHloVisitor::HandleElementwiseBinary: %s",
                       HloOpcodeString(opcode).c_str());
}

void DfsHloVisitor::SetVisiting(const HloInstruction& instruction) {
  VLOG(3) << "marking HLO " << &instruction << " as visiting: ";
  CHECK(NotVisited(instruction));
  visit_state_[&instruction] = VisitState::kVisiting;
}

void DfsHloVisitor::SetVisited(const HloInstruction& instruction) {
  VLOG(3) << "marking HLO " << &instruction << " as visited: ";
  CHECK(NotVisited(instruction) || IsVisiting(instruction));
  visit_state_[&instruction] = VisitState::kVisited;
}

bool DfsHloVisitor::IsVisiting(const HloInstruction& instruction) {
  if (visit_state_.count(&instruction) == 0) {
    return false;
  }
  return visit_state_[&instruction] == VisitState::kVisiting;
}

bool DfsHloVisitor::DidVisit(const HloInstruction& instruction) {
  if (visit_state_.count(&instruction) == 0) {
    return false;
  }
  return visit_state_[&instruction] == VisitState::kVisited;
}

bool DfsHloVisitor::NotVisited(const HloInstruction& instruction) {
  return visit_state_.count(&instruction) == 0 ||
         visit_state_[&instruction] == VisitState::kNotVisited;
}

Status DfsHloVisitor::Preprocess(HloInstruction* hlo) { return Status::OK(); }

Status DfsHloVisitor::Postprocess(HloInstruction* visited) {
  return Status::OK();
}

}  // namespace xla
