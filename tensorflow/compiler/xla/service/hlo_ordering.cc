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
  switch (GetExecutionConstraint(a, b)) {
    case ExecutionConstraint::kIsSame:  // a and b are the same instruction;
      return false;
    case ExecutionConstraint::kRunBeforeStart:
    case ExecutionConstraint::kRunBeforeEnd:
    case ExecutionConstraint::kRunExclusiveBefore:
      return true;
    case ExecutionConstraint::kRunExclusiveAfter:
    case ExecutionConstraint::kRunAfter:
    case ExecutionConstraint::kUnordered:
      return false;
  }
}

HloOrdering::ExecutionConstraint HloOrdering::GetExecutionConstraint(
    const HloInstruction* a, const HloInstruction* b) const {
  // 'a' and 'b' may be in different computations. In this case, find the
  // callgraph ancestor instructions which call (potentially transitively) the
  // computations containing 'a' and 'b' and use these ancestor instructions to
  // compare order.
  if (a == b) {
    return ExecutionConstraint::kIsSame;
  }
  const HloInstruction* a_ancestor;
  const HloInstruction* b_ancestor;
  std::tie(a_ancestor, b_ancestor) =
      call_graph_->NearestAncestorsInSameComputation(
          const_cast<HloInstruction*>(a), const_cast<HloInstruction*>(b));

  if (a_ancestor == nullptr) {
    VLOG(4) << "Ancestors in a common computation could not be found between"
            << a->ToString() << "\n and \n"
            << b->ToString() << "\n so consider them to be unordered.\n";
    return ExecutionConstraint::kUnordered;
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
      return ExecutionConstraint::kRunBeforeEnd;
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
    // If neither a nor b is inside the branches they both are the ancestor.
    if (a_branch == -1 && b_branch == -1) {
      CHECK_EQ(a, a_ancestor);
      CHECK_EQ(b, b_ancestor);
      CHECK_EQ(a, b);
      return ExecutionConstraint::kIsSame;
    }
    // If 'b' is the conditional ancestor, and 'a' is within a branch
    // computation, 'a' executes before 'b'.
    if (b_branch == -1) {
      CHECK_EQ(b, a_ancestor);
      return ExecutionConstraint::kRunBeforeEnd;
    }
    if (a_branch == -1) {
      CHECK_EQ(a, a_ancestor);
      return ExecutionConstraint::kRunAfter;
    }
    if (a_branch < b_branch) {
      return ExecutionConstraint::kRunExclusiveBefore;
    }
    if (b_branch < a_branch) {
      return ExecutionConstraint::kRunExclusiveAfter;
    }
  }

  if (ExecutesBeforeInSameComputation(a_ancestor, b_ancestor)) {
    return ExecutionConstraint::kRunBeforeStart;
  }
  if (ExecutesBeforeInSameComputation(b_ancestor, a_ancestor)) {
    return ExecutionConstraint::kRunAfter;
  }
  VLOG(1) << "Cannot determine order between:" << a->ToString() << "\n"
          << "and " << b->ToString() << " which are in the same computation\n";
  return ExecutionConstraint::kUnordered;
}

bool HloOrdering::IsDefinedBefore(const HloValue& a, const HloValue& b) const {
  // Entry parameter should always be defined before other instructions.
  const HloModule* module = b.defining_instruction()->GetModule();
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
bool HloOrdering::UsesBeforeValueDefinition(
    absl::Span<const HloUse* const> uses, const HloValue& value,
    const HloDataflowAnalysis& dataflow,
    bool use_is_always_before_def_in_same_instr) const {
  bool has_use_in_exclusive_branches = false;
  bool has_escaped_use_in_conditional = false;
  auto UseIsBeforeValueDefinition = [&](const HloUse& use) {
    VLOG(4) << "UseIsBeforeValueDefinition(use=" << use
            << ", value=" << value.ToShortString() << ")";
    switch (
        GetExecutionConstraint(use.instruction, value.defining_instruction())) {
      case HloOrdering::ExecutionConstraint::kIsSame:
        // If the use is at the instruction where the value is defined, then the
        // use is before the def if the instruction allows buffer sharing (in
        // place computation).
        if (use_is_always_before_def_in_same_instr ||
            dataflow.CanShareOperandBufferWithUser(
                use.instruction->mutable_operand(use.operand_number),
                use.operand_index, value.defining_instruction(),
                value.defining_index())) {
          VLOG(4)
              << "  use is value def, and instruction can share use buffer.";
          return true;
        }
        break;
      case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
        // If the use is located in a branch that is exclusive to the branch
        // where value is located, in order for them to interfere, there must be
        // an execution path where the value's definition can reach the use, so
        // that the wrong value would reach use if their live ranges are merged.
        // If there is such a path, it would have to pass through the point
        // where the two exclusive branches are joined --- specifically the end
        // of the conditional operation. For the join point to reach back to the
        // use at the other exclusive branch, there has to be a be a surrounding
        // loop, where the result of the conditional is passed back inside the
        // conditional through one of its parameters. This use-def conflict
        // between the parameter of a conditional and one of its branches is
        // caught in the has_escaped_use_in_conditinoal variable.
        VLOG(4) << " use and value def are in exclusive branches.";
        if (!has_escaped_use_in_conditional) {
          has_use_in_exclusive_branches = true;
          VLOG(4) << "Allowing them to share buffer.\n";
          return true;
        }
        VLOG(4) << "value def has escaped use in conditional. \n";
        break;
      case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
      case HloOrdering::ExecutionConstraint::kRunBeforeStart:
      case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
        VLOG(4)
            << "  use instruction executes before value-defining instruction";
        return true;
      case HloOrdering::ExecutionConstraint::kRunAfter:
        // Treat CollectivePermuteDone as a special case as it shares the buffer
        // from its operand (CollectivePermuteStart).
        if (use_is_always_before_def_in_same_instr &&
            use.instruction->opcode() == HloOpcode::kCollectivePermuteDone &&
            use.instruction->operand(0) == value.instruction()) {
          return true;
        }
        break;
      case HloOrdering::ExecutionConstraint::kUnordered:
        break;
    }

    // The use at a while is an input to a phi, and logically occurs before
    // values are defined in the body. Note that the use is *not* before the
    // value if the value is defined in the condition and is not the condition
    // parameter, since the input of a while's live range is only ended at the
    // start the body.
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      const HloInstruction* xla_while = use.instruction;
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             xla_while->while_body())) {
        VLOG(4) << "  use is while " << use.instruction->name()
                << " and def is in body";
        return true;
      }
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             xla_while->while_condition())) {
        if (value.defining_instruction() !=
            xla_while->while_condition()->parameter_instruction(0)) {
          VLOG(4) << "  use is while " << use.instruction->name()
                  << " and def is in condition and is not the parameter";
          return false;
        } else {
          VLOG(4) << "  use is while " << use.instruction->name()
                  << " and def is in condition and is the parameter";
          return true;
        }
      }
    }
    // Similarly if the value is defined at a while, it logically occurs after
    // any uses in the body or condition computations.
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
      // In general the use of a value in the conditional parameter should be
      // considered to be before a definition in one of its branches, and
      // therefore allowed in live range merging, if there is no
      // surrounding loop that creates a backward control flow path that
      // allows the definition in the branch to have its value flow backward
      // into the conditional and then flow into another branch in the
      // conditional that uses the value. This is reflected by checking that
      // the use-def in exclusive branches has not been already allowed.
      // Further, if the def value escapes its branch, we conservatively
      // assume a backward control flow path could exist, and set
      // has_escaped_use_in_conditinoal to disallow any later uses in
      // exclusive branches.
      for (int j = 0; j < conditional->branch_count(); ++j) {
        if (call_graph_->InstructionIsNestedIn(
                value.defining_instruction(),
                conditional->branch_computation(j))) {
          // If the use operand does not create a new value, and the value def
          // is returned by as part of the result of the conditional, it
          // is possible for the branch definition to flow backward through a
          // surrounding loop and then back into the conditional parameter.
          if (!dataflow.ValueIsDefinedAt(
                  use.instruction->operand(use.operand_number), {})) {
            for (auto value_use : value.GetUses()) {
              VLOG(4) << "def have use:" << value_use << "\n";
              if (value_use.instruction ==
                  value_use.instruction->parent()->root_instruction()) {
                VLOG(4) << "def use is conditional root \n";
                has_escaped_use_in_conditional = true;
                break;
              }
            }
          }
          if (!has_use_in_exclusive_branches) {
            VLOG(4) << "  use is conditional " << use.instruction->name()
                    << " and def is in " << j << "th branch computation";
            return true;
          }
        }
      }
      if (value.defining_instruction() == use.instruction) {
        VLOG(4) << "  use is conditional " << use << " and def is "
                << value.ToShortString();
        return true;
      }
    }

    VLOG(4) << "  use is not before value definition";
    return false;
  };
  for (auto* use : uses) {
    if (!UseIsBeforeValueDefinition(*use)) {
      return false;
    }
  }
  return true;
}

bool HloOrdering::LiveRangeStrictlyBefore(
    const HloValue& a, const HloValue& b, const HloDataflowAnalysis& dataflow,
    bool use_is_always_before_def_in_same_instr) const {
  VLOG(4) << "LiveRangeStrictlyBefore(a = " << a.ToShortString()
          << ", b = " << b.ToShortString() << ")";
  VLOG(4) << "Parent:" << a.instruction()->parent()->ToString() << "\n";
  if (!IsDefinedBefore(a, b)) {
    VLOG(4) << a << " not defined before " << b;
    return false;
  }

  if (a.live_out_of_module()) {
    VLOG(4) << a << " is live out of module and not defined before " << b;
    return false;
  }

  // If the root instruction aliases the buffer 'a', the live range of 'a' is
  // until the end of the computation and can never be strictly before another
  // buffer nested in the same computation. This is needed to prevent the root
  // instruction's buffers from being reused by later instructions even when
  // the root is not the last instruction in the schedule.
  for (const HloPosition& pos : a.positions()) {
    if (pos.instruction->parent()->root_instruction() == pos.instruction &&
        call_graph().InstructionIsNestedIn(b.instruction(),
                                           pos.instruction->parent())) {
      return false;
    }
  }

  // All uses of 'a' must be before 'b' is defined.
  std::vector<const HloUse*> uses;
  for (const HloUse& use : a.GetUses()) {
    if (dataflow.DoesNotUseOperandBuffer(a.instruction(), a.index(),
                                         use.instruction)) {
      continue;
    }
    uses.push_back(&use);
  }
  if (!UsesBeforeValueDefinition(uses, b, dataflow,
                                 use_is_always_before_def_in_same_instr)) {
    VLOG(4) << "uses of " << a << "not before " << b << " is defined";
    return false;
  }

  if (a.IsRootOf(b.instruction()->parent())) {
    VLOG(4) << a << " is live out of computation and defined before " << b
            << " which is in same computation";
    return false;
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

std::string PredecessorHloOrdering::ToStringHelper(
    const std::string& name) const {
  std::vector<std::string> pieces;
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

std::string DependencyHloOrdering::ToString() const {
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
  if (a->parent()->root_instruction() == a) {
    // 'a' is the root instruction of the computation, which lives out. So
    // 'a' cannot execute before 'b'.
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

std::string SequentialHloOrdering::ToString() const {
  return absl::StrCat("SequentialHloOrdering\n", schedule_.ToString());
}

}  // namespace xla
