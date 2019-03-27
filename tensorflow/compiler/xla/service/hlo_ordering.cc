/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_ordering.h"

#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

bool HloOrdering::ExecutesBefore(const HloInstruction* a,
                                 const HloInstruction* b) const {
  // 'a' and 'b' may be in different computations. In this case, find the
  // callgraph ancestor instructions which call (potentially transitively) the
  // computations containing 'a' and 'b' and use these ancestor instructions to
  // compare order.
  const HloInstruction* a_ancestor;
  const HloInstruction* b_ancestor;
  std::tie(a_ancestor, b_ancestor) =
      call_graph_->NearestAncestorsInSameComputation(
          const_cast<HloInstruction*>(a), const_cast<HloInstruction*>(b));

  if (a_ancestor == nullptr) {
    // Ancestors in a common computation could not be found so consider the
    // instructions 'a' and 'b' to be unordered.
    return false;
  }
  // a_ancestor and b_ancestor must be either both null or both non-null.
  CHECK_NE(b_ancestor, nullptr);
  CHECK_EQ(a_ancestor->parent(), b_ancestor->parent());

  // If the common ancestor is a while instruction there is an additional
  // ordering criteria which may apply. The condition computation is considered
  // to execute before the body computation so if 'a' is in the condition and
  // 'b' is in the body, then 'a' executes before 'b'.
  if (a_ancestor == b_ancestor && a_ancestor->opcode() == HloOpcode::kWhile) {
    const HloComputation* body = a_ancestor->while_body();
    const HloComputation* condition = a_ancestor->while_condition();
    if (call_graph_->InstructionIsNestedIn(a, condition) &&
        call_graph_->InstructionIsNestedIn(b, body)) {
      return true;
    }
  }

  // If the common ancestor is a conditional instruction, even though the branch
  // computations are not really ordered per-se, we define the 0th branch
  // computation to be ordered before the 1st one, before the 2nd and so forth.
  // This ensures that buffers can still be shared among branch computations
  // as they will forcibly have disjoint liveness.
  if (a_ancestor == b_ancestor &&
      (a_ancestor->opcode() == HloOpcode::kConditional)) {
    int a_branch = -1;
    int b_branch = -1;
    for (int j = 0; j < a_ancestor->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              a, a_ancestor->branch_computation(j))) {
        a_branch = j;
      }
      if (call_graph_->InstructionIsNestedIn(
              b, a_ancestor->branch_computation(j))) {
        b_branch = j;
      }
    }
    if (a_branch != -1 && a_branch < b_branch) {
      return true;
    }
    // If 'b' is the conditional ancestor, and 'a' is within a branch
    // computation, 'a' executes before 'b'.
    if (b == a_ancestor && a_branch != -1) {
      return true;
    }
  }

  return ExecutesBeforeInSameComputation(a_ancestor, b_ancestor);
}

bool HloOrdering::IsDefinedBefore(const HloValue& a, const HloValue& b) const {
  // Entry parameter should always be defined before other instructions.
  const HloModule* module = b.defining_instruction()->parent()->parent();
  if (b.defining_instruction()->parent() == module->entry_computation() &&
      b.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return false;
  }

  if (a.defining_instruction()->parent() == module->entry_computation() &&
      a.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return true;
  }

  // Phi values require special handling. Because XLA does not have a phi
  // instruction, the definition instruction of the phis values are
  // placeholders: either the subcomputation parameter (body or condition) or
  // the while instruction. However, the program point where these values are
  // logically defined does not necessarily coincide exactly with program point
  // of these place-holder instructions. So we explicitly define the following
  // order for phi values:
  //
  //   body/condition parameter phi:
  //     Defined before all values defined in its computation excepting other
  //     phis.
  //
  //   while phi:
  //     defined after all values defined in the condition or body.
  //
  auto is_body_or_condition_phi = [](const HloValue& v) {
    return v.is_phi() &&
           v.defining_instruction()->opcode() == HloOpcode::kParameter;
  };
  if (is_body_or_condition_phi(a) && !is_body_or_condition_phi(b) &&
      call_graph_->InstructionIsNestedIn(b.defining_instruction(),
                                         a.defining_instruction()->parent())) {
    return true;
  }
  if (is_body_or_condition_phi(b) &&
      call_graph_->InstructionIsNestedIn(a.defining_instruction(),
                                         b.defining_instruction()->parent())) {
    return false;
  }

  // If 'b' is a while phi and 'a' is in the body or condition, then 'a'
  // executes before 'b'.
  if (b.is_phi() && b.defining_instruction()->opcode() == HloOpcode::kWhile &&
      (call_graph_->InstructionIsNestedIn(
           a.defining_instruction(), b.defining_instruction()->while_body()) ||
       call_graph_->InstructionIsNestedIn(
           a.defining_instruction(),
           b.defining_instruction()->while_condition()))) {
    return true;
  }
  // If 'b' is a conditional phi and 'a' is in some branch computation, then 'a'
  // executes before 'b'.
  if (b.is_phi() &&
      b.defining_instruction()->opcode() == HloOpcode::kConditional) {
    for (int j = 0; j < b.defining_instruction()->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              a.defining_instruction(),
              b.defining_instruction()->branch_computation(j))) {
        return true;
      }
    }
  }
  return ExecutesBefore(a.defining_instruction(), b.defining_instruction());
}

/* static */
bool HloOrdering::UseIsBeforeValueDefinition(
    const HloUse& use, const HloValue& value,
    const HloDataflowAnalysis& dataflow) const {
  VLOG(4) << "UseIsBeforeValueDefinition(use=" << use
          << ", value=" << value.ToShortString() << ")";
  if (ExecutesBefore(use.instruction, value.defining_instruction())) {
    VLOG(4) << "  use instruction executes before value-defining instruction";
    return true;
  }

  // If the use is at the instruction where the value is defined, then the use
  // is before the def if the instruction allows buffer sharing (in place
  // computation).
  if (use.instruction == value.defining_instruction() &&
      dataflow.CanShareOperandBufferWithUser(
          use.instruction->mutable_operand(use.operand_number),
          use.operand_index, value.defining_instruction(),
          value.defining_index())) {
    VLOG(4) << "  use is value def, and instruction can share use buffer";
    return true;
  }

  // The use at a while is an input to a phi, and logically occurs before values
  // are defined in the body or condition computations.
  if (use.instruction->opcode() == HloOpcode::kWhile) {
    const HloInstruction* xla_while = use.instruction;
    if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                           xla_while->while_body()) ||
        call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                           xla_while->while_condition())) {
      VLOG(4) << "  use is while " << use.instruction->name()
              << " and def is in condition or body";
      return true;
    }
  }

  // Similarly if the value is defined at a while, it logically occurs after any
  // uses in the body or condition computations.
  if (value.defining_instruction()->opcode() == HloOpcode::kWhile) {
    CHECK(value.is_phi());
    const HloInstruction* xla_while = value.defining_instruction();
    if (call_graph_->InstructionIsNestedIn(use.instruction,
                                           xla_while->while_body()) ||
        call_graph_->InstructionIsNestedIn(use.instruction,
                                           xla_while->while_condition())) {
      VLOG(4) << "  value is while " << value.defining_instruction()->name()
              << " and use is in condition or body";
      return true;
    }
  }

  // The use at a call occurs before values that are defined in the called
  // computation.
  if (use.instruction->opcode() == HloOpcode::kCall) {
    const HloInstruction* call = use.instruction;
    if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                           call->to_apply())) {
      VLOG(4) << "  use is call " << use.instruction->name()
              << " and def is in called computation";
      return true;
    }
  }

  if (use.instruction->opcode() == HloOpcode::kConditional) {
    const HloInstruction* conditional = use.instruction;
    for (int j = 0; j < conditional->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              value.defining_instruction(),
              conditional->branch_computation(j))) {
        VLOG(4) << "  use is conditional " << use.instruction->name()
                << " and def is in " << j << "th branch computation";
        return true;
      }
    }
    if (value.defining_instruction() == use.instruction) {
      VLOG(4) << "  use is conditional " << use << " and def is "
              << value.ToShortString();
      return true;
    }
  }

  VLOG(4) << "  use is not before value";
  return false;
}

bool HloOrdering::LiveRangeStrictlyBefore(
    const HloValue& a, const HloValue& b,
    const HloDataflowAnalysis& dataflow) const {
  VLOG(4) << "LiveRangeStrictlyBefore(a = " << a.ToShortString()
          << ", b = " << b.ToShortString() << ")";
  if (!IsDefinedBefore(a, b)) {
    VLOG(4) << a << " not defined before " << b;
    return false;
  }

  if (a.live_out_of_module()) {
    VLOG(4) << a << " is live out of module and defined before " << b;
    return false;
  }

  // All uses of 'a' must be before 'b' is defined.
  for (const HloUse& use : a.uses()) {
    if (dataflow.DoesNotUseOperandBuffer(a.instruction(), a.index(),
                                         use.instruction)) {
      continue;
    }
    if (!UseIsBeforeValueDefinition(use, b, dataflow)) {
      VLOG(4) << "use of " << a << " (" << use << ") not before " << b
              << " is defined";
      return false;
    }
  }

  if (a.instruction()->parent() == b.instruction()->parent()) {
    for (const HloPosition& position : a.positions()) {
      if (position.instruction ==
          a.instruction()->parent()->root_instruction()) {
        VLOG(4) << a << " is live out of computation and defined before " << b
                << " which is in same computation";
        return false;
      }
    }
  }

  return true;
}

bool HloOrdering::MayInterfere(const HloValue& a, const HloValue& b,
                               const HloDataflowAnalysis& dataflow) const {
  // Buffers without disjoint liveness may interfere.
  return !LiveRangeStrictlyBefore(a, b, dataflow) &&
         !LiveRangeStrictlyBefore(b, a, dataflow);
}

PredecessorHloOrdering::PredecessorHloOrdering(const HloModule* module)
    : HloOrdering(module) {}

bool PredecessorHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());

  // 'a' executes before 'b' if 'a' is in the strict predecessor set of 'b'.
  return a != b && predecessors_.at(a->parent())->IsReachable(a, b);
}

string PredecessorHloOrdering::ToStringHelper(const string& name) const {
  std::vector<string> pieces;
  pieces.push_back(name);
  for (auto* computation : module_->MakeNonfusionComputations()) {
    pieces.push_back(absl::StrFormat("computation %s:", computation->name()));
    const auto all = computation->MakeInstructionPostOrder();
    for (auto instruction : all) {
      pieces.push_back(
          absl::StrFormat("  %s predecessors:", instruction->name()));
      for (auto predecessor : all) {
        if (predecessors_.at(computation)
                ->IsReachable(predecessor, instruction)) {
          pieces.push_back(absl::StrFormat("    %s", predecessor->name()));
        }
      }
    }
  }
  return absl::StrJoin(pieces, "\n");
}

DependencyHloOrdering::DependencyHloOrdering(const HloModule* module)
    : PredecessorHloOrdering(module) {
  // Compute predecessor relationships between all instructions to determine
  // ordering based on dependencies. ExecutesBefore will return true iff there
  // exists a path in the HLO computation graph from 'a' to 'b'.
  for (auto* computation : module->MakeNonfusionComputations()) {
    predecessors_.emplace(computation, HloReachabilityMap::Build(computation));
  }
}

string DependencyHloOrdering::ToString() const {
  return ToStringHelper("DependencyHloOrdering");
}

SequentialHloOrdering::SequentialHloOrdering(const HloSchedule& schedule)
    : HloOrdering(schedule.module()), schedule_(schedule) {
  Initialize();
}

SequentialHloOrdering::SequentialHloOrdering(HloSchedule&& schedule)
    : HloOrdering(schedule.module()), schedule_(std::move(schedule)) {
  Initialize();
}

void SequentialHloOrdering::Initialize() {
  // Create a map from instruction to its order position.
  TF_DCHECK_OK(schedule_.Verify());
  for (const auto& computation_sequence : schedule_.sequences()) {
    const auto& order = computation_sequence.second.instructions();
    for (int i = 0; i < order.size(); ++i) {
      InsertOrDie(&order_position_, order[i], i);
    }
  }
}

bool SequentialHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());
  // If either instruction is not in the order, then 'a' and 'b' are unordered.
  if (!order_position_.contains(a) || !order_position_.contains(b)) {
    return false;
  }
  return order_position_.at(a) < order_position_.at(b);
}

const HloInstructionSequence* SequentialHloOrdering::SequentialOrder(
    const HloComputation& computation) const {
  return schedule_.is_computation_scheduled(&computation)
             ? &schedule_.sequence(&computation)
             : nullptr;
}

string SequentialHloOrdering::ToString() const {
  return absl::StrCat("SequentialHloOrdering\n", schedule_.ToString());
}

}  // namespace xla
