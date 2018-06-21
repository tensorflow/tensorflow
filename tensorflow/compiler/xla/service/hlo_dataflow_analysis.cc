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

#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"

#include <algorithm>
#include <queue>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

// We have this pattern in dynamaic update slice fusion, which should be
// supported:
//
// Parameters: p0, p1
// Fusion
//   ds = DynamicSlice(p0, p1)
//   ROOT DynamicUpdateslice(p0, ds, p1)
//
// In this case, we should be able to reuse p0 and output, although p0 has
// multiple uses.
bool MultiDynamicSliceUseShareSameIndices(
    tensorflow::gtl::ArraySlice<HloUse> uses) {
  if (uses.empty()) {
    return false;
  }
  const HloInstruction* indices = nullptr;
  for (HloUse use : uses) {
    auto user = use.instruction;
    if (user->opcode() == HloOpcode::kDynamicUpdateSlice) {
      if (indices == nullptr) {
        indices = user->operand(2);
      } else if (indices != user->operand(2)) {
        return false;
      }
      if (use.operand_number != 0) {
        return false;
      }
    } else if (user->opcode() == HloOpcode::kDynamicSlice) {
      if (indices == nullptr) {
        indices = user->operand(1);
      } else if (indices != user->operand(1)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

HloDataflowAnalysis::HloDataflowAnalysis(const HloModule& module, bool ssa_form,
                                         bool bitcast_defines_value)
    : module_(module),
      ssa_form_(ssa_form),
      bitcast_defines_value_(bitcast_defines_value),
      call_graph_(CallGraph::Build(&module)) {}

bool HloDataflowAnalysis::AreTransitiveUsesElementwiseOrTuple(
    const HloInstruction* inst) {
  tensorflow::gtl::FlatSet<const HloInstruction*> visited;
  tensorflow::gtl::InlinedVector<const HloInstruction*, 4> stack;
  stack.push_back(inst);
  while (!stack.empty()) {
    const HloInstruction* current = stack.back();
    stack.pop_back();
    visited.insert(current);
    for (const HloInstruction* user : current->users()) {
      // Found a user that is non-elementwise on current instruction.
      for (const int64 use_index : user->OperandIndices(current)) {
        if (!user->IsElementwiseOnOperand(use_index) &&
            user->opcode() != HloOpcode::kTuple) {
          return false;
        }
      }
      if (!visited.count(user)) {
        stack.push_back(user);
      }
    }
  }
  return true;
}

bool HloDataflowAnalysis::ValueIsDefinedAt(const HloInstruction* instruction,
                                           const ShapeIndex& index) const {
  const HloValueSet& value_set = GetValueSet(instruction, index);
  if (value_set.values().size() != 1) {
    return false;
  }
  return value_set.GetUniqueValue().defining_instruction() == instruction;
}

const HloValue& HloDataflowAnalysis::GetValueDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  CHECK(ValueIsDefinedAt(instruction, index));
  return GetUniqueValueAt(instruction, index);
}

HloValue& HloDataflowAnalysis::GetValueDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  CHECK(ValueIsDefinedAt(instruction, index));
  return GetUniqueValueAt(instruction, index);
}

HloValue* HloDataflowAnalysis::NewHloValue(HloInstruction* instruction,
                                           const ShapeIndex& index,
                                           bool is_phi) {
  const int64 value_id = next_value_id_++;
  auto emplaced = values_.emplace(
      std::piecewise_construct, std::forward_as_tuple(value_id),
      std::forward_as_tuple(value_id, instruction, index, is_phi));
  CHECK(emplaced.second);

  VLOG(4) << "NewHloValue = " << emplaced.first->second.ToShortString();

  return &emplaced.first->second;
}

void HloDataflowAnalysis::MarkValueForDeletion(HloValue::Id value_id) {
  HloValue& value = values_.at(value_id);
  VLOG(4) << "MarkValueForDeletion(" << value.ToShortString() << ")";

  value_ids_to_delete_.push_back(value_id);
}

void HloDataflowAnalysis::DeleteMarkedValues() {
#ifndef NDEBUG
  // Verify that no marked-for-deletion values are in any of the value sets.
  tensorflow::gtl::FlatSet<HloValue::Id> id_set(value_ids_to_delete_.begin(),
                                                value_ids_to_delete_.end());
  for (const auto& pair : value_sets_) {
    const HloInstruction* instruction = pair.first;
    const InstructionValueSet& instruction_value_set = pair.second;
    for (const auto& index_value_set : instruction_value_set) {
      const HloValueSet& value_set = index_value_set.second;
      for (const HloValue* value : value_set.values()) {
        DCHECK(!ContainsKey(id_set, value->id()))
            << "Value " << value->ToShortString()
            << " marked for deletion, but still exists in value set for "
               "instruction "
            << instruction->name();
      }
    }
  }
#endif

  for (HloValue::Id value_id : value_ids_to_delete_) {
    values_.erase(value_id);
  }
  value_ids_to_delete_.clear();
}

string HloDataflowAnalysis::ToString() const {
  string out = StrCat("HloDataflowAnalysis, module ", module_.name(), "\n");
  StrAppend(&out, "  Instruction value sets:\n");
  for (const HloComputation* computation : module_.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      StrAppend(&out, "    ", instruction->name(), ":\n");
      if (ShapeUtil::IsTuple(instruction->shape())) {
        GetInstructionValueSet(instruction)
            .ForEachElement([this, &instruction, &out](
                                const ShapeIndex& index,
                                const HloValueSet& value_set) {
              StrAppend(&out, "      tuple index ", index.ToString(), ":\n");
              for (const HloValue* value : value_set.values()) {
                StrAppend(&out, "        ", value->ToShortString(),
                          ValueIsDefinedAt(instruction, index) ? " (def)" : "",
                          "\n");
              }
            });
      } else {
        const HloValueSet& top_level_value_set =
            GetValueSet(instruction, /*index=*/{});
        for (const HloValue* value : top_level_value_set.values()) {
          StrAppend(&out, "      ", value->ToShortString(),
                    ValueIsDefinedAt(instruction) ? " (def)" : "", "\n");
        }
      }
    }
  }
  StrAppend(&out, "  HloValues:\n");
  for (const HloValue* value : values()) {
    StrAppend(&out, value->ToString(/*indent=*/4));
  }
  return out;
}

bool HloDataflowAnalysis::Phi(
    HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<const InstructionValueSet*> inputs) {
  CHECK(ssa_form_);
  VLOG(4) << "Phi(" << instruction->name() << ")";
  VLOG(5) << "instruction value set = "
          << GetInstructionValueSet(instruction).ToString();
  for (const InstructionValueSet* input : inputs) {
    VLOG(5) << "input value set = " << input->ToString();
  }
  for (const InstructionValueSet* input : inputs) {
    DCHECK(ShapeUtil::Compatible(instruction->shape(), input->shape()));
  }

  bool changed = false;
  for (auto& pair : GetInstructionValueSet(instruction)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    // Positions with phi values should never have more than one value in the
    // value set.
    CHECK_LE(value_set.values().size(), 1);
    const HloValue* current_value =
        value_set.values().size() == 1 ? value_set.values()[0] : nullptr;

    // Construct a vector of unique value IDs of the inputs.
    // Don't add value ids where the input is equal to the definition.
    std::vector<HloValue::Id> input_value_ids;
    for (const InstructionValueSet* input : inputs) {
      for (const HloValue* value : input->element(index).values()) {
        if (value->defining_instruction() == instruction &&
            value->defining_index() == index) {
          continue;
        }
        input_value_ids.push_back(value->id());
      }
    }
    std::sort(input_value_ids.begin(), input_value_ids.end());
    input_value_ids.erase(
        std::unique(input_value_ids.begin(), input_value_ids.end()),
        input_value_ids.end());

    // Remove the existing phi value (if it exists). The phi can be its own
    // input, for example, in while body parameters where the body passes
    // through the parameter value.
    bool current_value_defined_here =
        (current_value != nullptr &&
         current_value->defining_instruction() == instruction &&
         current_value->defining_index() == index);
    if (current_value_defined_here) {
      VLOG(5) << "current_value_defined_here: " << current_value->ToString();
      CHECK(current_value->is_phi());
      auto it = std::find(input_value_ids.begin(), input_value_ids.end(),
                          current_value->id());
      if (it != input_value_ids.end()) {
        input_value_ids.erase(it);
      }
    }
    VLOG(5) << "after input_value_ids.size = " << input_value_ids.size();
    if (input_value_ids.empty()) {
      // A value set which has at least one element should never have its value
      // set reduced to zero elements. During dataflow value sets only can go
      // from empty to non-empty, not the reverse.
      CHECK_EQ(value_set.values().size(), 0)
          << "Instruction " << instruction->name() << " at index " << index
          << " previously had non-empty value set. Value set: " << value_set;
    } else if (input_value_ids.size() == 1) {
      // Only a single value reaches this point. There should be no phi, and
      // this value set should contain this single value.
      const HloValue& new_value = GetValue(input_value_ids[0]);
      if (current_value == nullptr) {
        value_set.Clear();
        value_set.AddValue(&new_value);
        changed = true;
      } else if (current_value != &new_value) {
        if (current_value_defined_here) {
          // Remove the existing phi.
          MarkValueForDeletion(current_value->id());
        }
        value_set.Clear();
        value_set.AddValue(&new_value);
        changed = true;
      }
    } else {
      // Multiple distinct values reach this point. A phi value is
      // necessary.
      CHECK_GT(input_value_ids.size(), 1);
      if (current_value == nullptr ||
          !(current_value->is_phi() && current_value_defined_here)) {
        value_set.Clear();
        value_set.AddValue(NewHloValue(instruction, index, /*is_phi=*/true));
        changed = true;
      }
    }
  }
  return changed;
}

const HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) const {
  return values_.at(value_id);
}

HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) {
  return values_.at(value_id);
}

const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return GetInstructionValueSet(instruction).element(index);
}

HloValueSet& HloDataflowAnalysis::GetValueSet(const HloInstruction* instruction,
                                              const ShapeIndex& index) {
  return *GetInstructionValueSet(instruction).mutable_element(index);
}

const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloPosition& position) const {
  return GetValueSet(position.instruction, position.index);
}

HloValueSet& HloDataflowAnalysis::GetValueSet(const HloPosition& position) {
  return GetValueSet(position.instruction, position.index);
}

bool HloDataflowAnalysis::UpdateBitcastValueSet(HloInstruction* bitcast) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcast);
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(bitcast->operand(0));
  InstructionValueSet& bitcast_set = GetInstructionValueSet(bitcast);
  if (!bitcast_defines_value_ && operand_set != bitcast_set) {
    bitcast_set = operand_set;
    return true;
  }
  return false;
}

bool HloDataflowAnalysis::UpdateSliceValueSet(HloInstruction* slice) {
  CHECK_EQ(slice->opcode(), HloOpcode::kSlice);
  if (!slice->IsInPlaceSlice()) {
    return false;
  }
  // If this slice is lowered to an in-place version, then it forwards the
  // operand value to the output.
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(slice->operand(0));
  InstructionValueSet& slice_set = GetInstructionValueSet(slice);
  if (operand_set != slice_set) {
    slice_set = operand_set;
    return true;
  }
  return false;
}

bool HloDataflowAnalysis::UpdateSendValueSet(HloInstruction* send) {
  CHECK_EQ(send->opcode(), HloOpcode::kSend);
  bool changed = false;
  // Send forwards the operand value to the output tuple at {0}.
  for (auto& pair : GetInstructionValueSet(send->operand(0))) {
    const ShapeIndex& operand_index = pair.first;
    const HloValueSet& operand_value_set = pair.second;

    ShapeIndex index = {0};
    for (int64 i : operand_index) {
      index.push_back(i);
    }

    HloValueSet& value_set = GetValueSet(send, index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateRecvDoneValueSet(HloInstruction* recv_done) {
  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  bool changed = false;
  // RecvDone forwards the operand value at {0} to the output.
  for (auto& pair : GetInstructionValueSet(recv_done)) {
    ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    ShapeIndex operand_index = {0};
    for (int64 i : index) {
      operand_index.push_back(i);
    }

    const HloValueSet& operand_value_set =
        GetValueSet(recv_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateCallValueSet(HloInstruction* call) {
  CHECK_EQ(call->opcode(), HloOpcode::kCall);
  InstructionValueSet& value_set = GetInstructionValueSet(call);
  InstructionValueSet& root_value_set =
      GetInstructionValueSet(call->to_apply()->root_instruction());
  if (value_set != root_value_set) {
    value_set = root_value_set;
    return true;
  }
  return false;
}

bool HloDataflowAnalysis::UpdateConditionalValueSet(
    HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);
  const InstructionValueSet* const inputs[] = {
      &GetInstructionValueSet(
          conditional->true_computation()->root_instruction()),
      &GetInstructionValueSet(
          conditional->false_computation()->root_instruction())};
  if (ssa_form_) {
    return Phi(conditional, inputs);
  } else {
    return GetInstructionValueSet(conditional).AssignUnionOf(inputs);
  }
}

bool HloDataflowAnalysis::UpdateCopyValueSet(HloInstruction* copy) {
  CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(copy)) {
    const ShapeIndex& index = pair.first;
    if (index.empty()) {
      // kCopy shallow copies and thus defines the top-level value so nothing to
      // update.
      continue;
    }

    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(copy->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateGetTupleElementValueSet(HloInstruction* gte) {
  CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
  bool changed = false;
  // The GetTupleElement instruction forwards the values from the specified
  // tuple element.
  for (auto& pair : GetInstructionValueSet(gte)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    // The corresponding ShapeIndex of the operand is simply the GTE ShapeIndex
    // with the tuple element number prefixed.
    ShapeIndex operand_index = {gte->tuple_index()};
    for (int64 i : index) {
      operand_index.push_back(i);
    }

    HloValueSet& operand_value_set =
        GetValueSet(gte->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateParameterValueSet(HloInstruction* parameter) {
  CHECK_EQ(parameter->opcode(), HloOpcode::kParameter);
  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(parameter->parent());

  // Subcomputations called in a parallel context (eg, map) do not have dataflow
  // from the caller operands.
  if (call_graph_node.context() == CallContext::kParallel ||
      call_graph_node.caller_callsites().empty()) {
    return false;
  }
  CHECK_EQ(call_graph_node.context(), CallContext::kSequential);

  std::vector<const InstructionValueSet*> inputs;
  bool need_phi = false;
  for (const CallSite& callsite : call_graph_node.caller_callsites()) {
    if (callsite.instruction()->opcode() == HloOpcode::kCall) {
      // The operand values of a call instruction are forwarded to the
      // respective parameter instruction of the subcomputation.
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
      // In a while instruction, the while operand (ie, the init value) and the
      // backedge are dataflow inputs to the parameter instruction. This is the
      // case for parameters of both the body and condition computations.
      CHECK_EQ(parameter->parameter_number(), 0);
      inputs.push_back(
          &GetInstructionValueSet(callsite.instruction()->operand(0)));
      // If the parameter *is* the root, then don't consider it's current state
      // (InstructionValueSet) as we are recomputing its current
      // state. Otherwise, the parameter state would never be updated.
      if (parameter !=
          callsite.instruction()->while_body()->root_instruction()) {
        inputs.push_back(&GetInstructionValueSet(
            callsite.instruction()->while_body()->root_instruction()));
      }
      need_phi = true;
    } else if (callsite.instruction()->opcode() == HloOpcode::kConditional) {
      CHECK_EQ(parameter->parameter_number(), 0);
      auto conditional = callsite.instruction();
      // Conditional has 3 operands. Operand 0 is the predicate, operand 1 is
      // the argument to the true computation and operand 2 is the argument to
      // the false computation.
      //
      // If the parameter belongs to conditional's true computation, then
      // operand 1 is forwarded to this parameter instruction. If the parameter
      // belongs to conditional's false computation, then operand 2 is forwarded
      // to this parameter instruction.
      if (parameter->parent() == conditional->true_computation()) {
        inputs.push_back(&GetInstructionValueSet(conditional->operand(1)));
      } else {
        CHECK_EQ(parameter->parent(), conditional->false_computation());
        inputs.push_back(&GetInstructionValueSet(conditional->operand(2)));
      }
      need_phi = true;
    } else {
      LOG(FATAL) << "CallContext::kSequential computations should only be "
                    "called from call, while, or conditional instructions";
    }
  }

  if (ssa_form_ && need_phi) {
    return Phi(parameter, inputs);
  } else {
    return GetInstructionValueSet(parameter).AssignUnionOf(inputs);
  }
}

bool HloDataflowAnalysis::UpdateSelectValueSet(HloInstruction* select) {
  CHECK_EQ(select->opcode(), HloOpcode::kSelect);
  // A phi value is not defined at a kSelect instruction because kSelect does
  // not create a new value. Rather it forwards a value from its operands. This
  // contrasts with kWhile instruction (which does define a phi value) which has
  // in-place update semantics.
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(select)) {
    const ShapeIndex& index = pair.first;
    if (index.empty()) {
      // kSelect copies (not forwards) the top-level value.
      continue;
    }
    HloValueSet& value_set = pair.second;
    changed |=
        value_set.AssignUnionOf({&GetValueSet(select->operand(1), index),
                                 &GetValueSet(select->operand(2), index)});
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateTupleValueSet(HloInstruction* tuple) {
  CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
  bool changed = false;
  for (int64 i = 0; i < tuple->operands().size(); ++i) {
    // Copy the value set(s) of each operand into the respective position in the
    // kTuple instruction's value sets.
    for (auto& pair : GetInstructionValueSet(tuple->operand(i))) {
      const ShapeIndex& operand_index = pair.first;
      HloValueSet& operand_value_set = pair.second;

      ShapeIndex index = {i};
      for (int64 op_index : operand_index) {
        index.push_back(op_index);
      }
      HloValueSet& value_set = GetValueSet(tuple, index);

      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateWhileValueSet(HloInstruction* xla_while) {
  CHECK_EQ(xla_while->opcode(), HloOpcode::kWhile);
  const InstructionValueSet* const inputs[] = {
      &GetInstructionValueSet(xla_while->while_body()->root_instruction()),
      &GetInstructionValueSet(xla_while->operand(0))};
  if (ssa_form_) {
    return Phi(xla_while, inputs);
  } else {
    return GetInstructionValueSet(xla_while).AssignUnionOf(inputs);
  }
}

bool HloDataflowAnalysis::UpdateInstructionValueSet(
    HloInstruction* instruction) {
  // Recompute from operands.
  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
      return UpdateBitcastValueSet(instruction);
    case HloOpcode::kSlice:
      return UpdateSliceValueSet(instruction);
    case HloOpcode::kCopy:
      return UpdateCopyValueSet(instruction);
    case HloOpcode::kGetTupleElement:
      return UpdateGetTupleElementValueSet(instruction);
    case HloOpcode::kSelect:
      return UpdateSelectValueSet(instruction);
    case HloOpcode::kTuple:
      return UpdateTupleValueSet(instruction);
    case HloOpcode::kParameter:
      return UpdateParameterValueSet(instruction);
    case HloOpcode::kCall:
      return UpdateCallValueSet(instruction);
    case HloOpcode::kWhile:
      return UpdateWhileValueSet(instruction);
    case HloOpcode::kSend:
      return UpdateSendValueSet(instruction);
    case HloOpcode::kRecvDone:
      return UpdateRecvDoneValueSet(instruction);
    case HloOpcode::kConditional:
      return UpdateConditionalValueSet(instruction);
    default:
      // Instruction does not forward HloValues (it defines all values in its
      // output). No update is necessary.
      return false;
  }
}

void HloDataflowAnalysis::Propagate() {
  std::queue<HloInstruction*> worklist;
  tensorflow::gtl::FlatSet<HloInstruction*> workset;
  auto add_to_worklist = [&worklist, &workset](HloInstruction* instruction) {
    if (workset.insert(instruction).second) {
      worklist.push(instruction);
    }
  };

  for (HloComputation* computation : module_.computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      add_to_worklist(instruction);
    }
  }

  while (!worklist.empty()) {
    HloInstruction* instruction = worklist.front();
    worklist.pop();
    workset.erase(workset.find(instruction));

    VLOG(3) << "Worklist top: " << instruction->name();
    VLOG(3) << ToString();

    if (!UpdateInstructionValueSet(instruction)) {
      // No change to the instruction's value set.
      VLOG(4) << "No change.";
      continue;
    }

    VLOG(4) << "New value set for " << instruction->name() << ": "
            << GetInstructionValueSet(instruction);

    // Instruction value was updated. Add users to work list if we haven't
    // already.
    for (HloInstruction* user : instruction->users()) {
      add_to_worklist(user);

      // If user sequentially calls a computation, then the respective
      // parameter(s) of the computation need to be updated.
      if (user->opcode() == HloOpcode::kConditional) {
        // If operand 0 is the use of instruction, then no parameters need to be
        // updated, since that is the predicate of the conditional.
        // If operand 1 is the use of instruction, then the true_computation's
        // parameter need to be updated.
        // If operand 2 is the use of instruction, then the false_computation's
        // parameter need to be updated.
        //
        // Note that the same instruction can be used in both operand 1 and
        // operand 2.
        if (user->operand(1) == instruction) {
          add_to_worklist(user->true_computation()->parameter_instruction(0));
        }
        if (user->operand(2) == instruction) {
          add_to_worklist(user->false_computation()->parameter_instruction(0));
        }
      } else {
        for (HloComputation* called_computation : user->called_computations()) {
          const CallGraphNode& call_graph_node =
              call_graph_->GetNode(called_computation);
          if (call_graph_node.context() == CallContext::kSequential) {
            for (int64 operand_number : user->OperandIndices(instruction)) {
              add_to_worklist(
                  called_computation->parameter_instruction(operand_number));
            }
          }
        }
      }
    }

    // If instruction is a root instruction, then propagate out to any calling
    // instruction and across any while backedge.
    if (instruction == instruction->parent()->root_instruction()) {
      const CallGraphNode& call_graph_node =
          call_graph_->GetNode(instruction->parent());
      for (const CallSite& callsite : call_graph_node.caller_callsites()) {
        if ((callsite.instruction()->opcode() == HloOpcode::kCall) ||
            (callsite.instruction()->opcode() == HloOpcode::kConditional)) {
          add_to_worklist(callsite.instruction());
        } else if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
          // Add the while itself, and the body and condition parameters.
          add_to_worklist(callsite.instruction());
          add_to_worklist(
              callsite.instruction()->while_body()->parameter_instruction(0));
          add_to_worklist(
              callsite.instruction()->while_condition()->parameter_instruction(
                  0));
        }
      }
    }
  }
}

const InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) const {
  return value_sets_.at(instruction);
}

InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) {
  return value_sets_.at(instruction);
}

Status HloDataflowAnalysis::InitializeInstructionValueSets() {
  for (const HloComputation* computation : module_.computations()) {
    const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);
    for (HloInstruction* instruction : computation->instructions()) {
      // Create an empty shape tree.
      value_sets_.emplace(std::piecewise_construct,
                          std::forward_as_tuple(instruction),
                          std::forward_as_tuple(instruction->shape()));

      // Lambda to set the value set to define all values in the output of the
      // instruction.
      auto define_all_values = [this, &instruction](bool is_phi = false) {
        for (auto& pair : GetInstructionValueSet(instruction)) {
          const ShapeIndex& index = pair.first;
          HloValue* value = NewHloValue(instruction, index, /*is_phi=*/false);
          GetValueSet(instruction, index).AddValue(value);
        }
      };

      // Lambda to set the value set to define only the top-level buffer in the
      // output of the instruction. Any other values flow from the operands of
      // the instruction (or from cross-computation dataflow).
      auto define_top_level_only = [this, &instruction]() {
        HloValue* value =
            NewHloValue(instruction, /*index=*/{}, /*is_phi=*/false);
        GetValueSet(instruction, /*index=*/{}).AddValue(value);
      };

      // Lambda to set the value set at the given index of the output.
      auto define_value_at = [this, &instruction](const ShapeIndex& index) {
        HloValue* value = NewHloValue(instruction, index, /*is_phi=*/false);
        GetValueSet(instruction, index).AddValue(value);
      };

      switch (instruction->opcode()) {
        case HloOpcode::kBitcast:
          if (bitcast_defines_value_) {
            define_all_values();
          }
          break;
        case HloOpcode::kSlice:
          if (!instruction->IsInPlaceSlice()) {
            define_all_values();
          }
          break;
        case HloOpcode::kWhile:
        case HloOpcode::kCall:
        case HloOpcode::kConditional:
        case HloOpcode::kGetTupleElement:
          // These instructions define no values. The values in their output
          // flow from their operands or from cross computation dataflow.
          break;
        case HloOpcode::kParameter:
          if (call_graph_node.context() == CallContext::kBoth) {
            // We do not support a subcomputation that is called from both a
            // parallel and sequential context. In this case, the parameter
            // would both define a value and propagate a value from its
            // caller. This limitation is not really a problem because the call
            // graph is typically flattened.
            return Unimplemented(
                "Computation %s is called in both a parallel (eg, kMap) and "
                "sequential (eg, kCall) context",
                computation->name().c_str());
          }
          if (call_graph_node.caller_callsites().empty() ||
              call_graph_node.context() == CallContext::kParallel) {
            // Parameters of computations called in a parallel context (eg, map
            // and reduce) as well as parameters of dead computations define all
            // values in their output. Otherwise the values of the parameter
            // come from the caller (eg, operands to the kCall instruction).
            define_all_values();
          }
          break;
        case HloOpcode::kCopy:
        case HloOpcode::kSelect:
        case HloOpcode::kTuple:
          // These instructions only define their top-level values. Any other
          // values flow from their operands.
          define_top_level_only();
          break;
        case HloOpcode::kRecvDone:
          // RecvDone aliases its input tuple element {0}, therefore does not
          // define any values.
          break;
        case HloOpcode::kSend:
          // Send produces a tuple of {aliased operand, U32 context}, therefore
          // only defines the top-level tuple and the tuple element at {1}.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          break;
        default:
          define_all_values();
          break;
      }
    }
  }

  return Status::OK();
}

/* static */
StatusOr<std::unique_ptr<HloDataflowAnalysis>> HloDataflowAnalysis::Run(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value) {
  VLOG(1) << "HloDataflowAnalysis::Run on module " << module.name();
  XLA_VLOG_LINES(2, module.ToString());

  auto dataflow_analysis = WrapUnique(
      new HloDataflowAnalysis(module, ssa_form, bitcast_defines_value));

  TF_RETURN_IF_ERROR(dataflow_analysis->InitializeInstructionValueSets());
  dataflow_analysis->Propagate();

  // Delete all values marked for deletion.
  dataflow_analysis->DeleteMarkedValues();

  // Gather and set all non-definition positions of all values. Value deletion
  // is rare, so just use a vector indexed by Value::Id rather than a map from
  // Value::Id to positions. There should be very few holes in the vector, and
  // lookup is faster.
  std::vector<std::vector<HloPosition>> value_positions(
      dataflow_analysis->next_value_id_);
  for (const HloComputation* computation : module.computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      for (const auto& pair :
           dataflow_analysis->GetInstructionValueSet(instruction)) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        for (const HloValue* value : value_set.values()) {
          if (value->defining_instruction() != instruction) {
            value_positions[value->id()].push_back(
                HloPosition{instruction, index});
          }
        }
      }
    }
  }
  for (auto& pair : dataflow_analysis->values_) {
    HloValue::Id value_id = pair.first;
    HloValue& value = pair.second;
    value.SetPositionsAndComputeUses(value_positions[value_id]);
  }

  // Construct vector of values.
  dataflow_analysis->values_vector_.reserve(dataflow_analysis->values_.size());
  for (auto& pair : dataflow_analysis->values_) {
    dataflow_analysis->values_vector_.push_back(&pair.second);
  }
  std::sort(dataflow_analysis->values_vector_.begin(),
            dataflow_analysis->values_vector_.end(), HloValue::IdLessThan);

  TF_DCHECK_OK(dataflow_analysis->Verify());

  XLA_VLOG_LINES(1, dataflow_analysis->ToString());

  return std::move(dataflow_analysis);
}

Status HloDataflowAnalysis::Verify() const {
  // Verify each HloValue appears in the value sets that the value's positions()
  // indicate.
  for (const HloValue* value : values()) {
    for (const HloPosition& position : value->positions()) {
      const HloValueSet& value_set = GetValueSet(position);
      TF_RET_CHECK(std::find(value_set.values().begin(),
                             value_set.values().end(),
                             value) != value_set.values().end())
          << "Value set at position " << position << " does not contain value "
          << value->ToShortString();
    }
  }

  // For each value in each value set, verify that the value set's position
  // appears in the value's positions().
  for (const auto& computation : module_.computations()) {
    for (const auto& instruction : computation->instructions()) {
      for (const auto& pair : GetInstructionValueSet(instruction)) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        const HloPosition position{instruction, index};
        for (const HloValue* value : value_set.values()) {
          TF_RET_CHECK(std::find(value->positions().begin(),
                                 value->positions().end(),
                                 position) != value->positions().end())
              << "Value set at position " << position
              << " unexpectedly contains value " << value->ToShortString();
        }
      }
    }
  }

  return Status::OK();
}

bool HloDataflowAnalysis::DoesNotUseOperandBuffer(
    const HloInstruction* operand, const ShapeIndex& index,
    const HloInstruction* user) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (user->opcode() == HloOpcode::kFusion &&
      user->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    // Find fusion parameter associated with 'operand'.
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));
    // Iterate through all users of all uses of the fusion parameter value.
    // Return false if any uses are detected, returns true otherwise.
    const HloValue& value = GetValueDefinedAt(fusion_param, index);
    return value.uses().empty();
  } else {
    // Return false if no value at 'operand' and 'index' is used at 'user'.
    for (const HloValue* value : GetValueSet(operand, index).values()) {
      for (const HloUse& use : value->uses()) {
        if (use.instruction == user) {
          return false;
        }
      }
    }
  }

  return true;
}

bool HloDataflowAnalysis::CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  const Shape& operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);

  // Check that operand and user emit the same shape and layout.
  if (!ShapeUtil::Equal(operand_subshape, user_subshape)) {
    return false;
  }

  if (user->opcode() == HloOpcode::kFusion) {
    // Get the parameter associated with 'operand';
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));

    const HloValue& value = GetValueDefinedAt(fusion_param, operand_index);
    if (value.uses().size() != 1) {
      if (MultiDynamicSliceUseShareSameIndices(value.uses())) {
        return true;
      }
      return false;
    }
    const HloUse& use = value.uses()[0];

    if (user->fusion_kind() == HloInstruction::FusionKind::kLoop ||
        user->fusion_kind() == HloInstruction::FusionKind::kInput) {
      if (user->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice) {
        // Loop fusion with kDynamicUpdateSlice fused root.
        //
        // Returns true iff there is exactly one use of 'operand' at shape index
        // 'operand_index', and this singleton use is the fused root at operand
        // index 0.
        return use.instruction == user->fused_expression_root() &&
               use.operand_number == 0;
      } else {
        return AreTransitiveUsesElementwiseOrTuple(fusion_param);
      }
    } else if (user->fusion_kind() == HloInstruction::FusionKind::kOutput &&
               user->fused_expression_root()->opcode() == HloOpcode::kAdd) {
      // Output fusion with kAdd fused root.

      // Check if one operand of kAdd fused root is kDot or kConvolution.
      auto* add = user->fused_expression_root();
      auto add_operand_it =
          std::find_if(add->operands().begin(), add->operands().end(),
                       [&](HloInstruction* operand) {
                         return operand->opcode() == HloOpcode::kConvolution ||
                                operand->opcode() == HloOpcode::kDot;
                       });
      if (add_operand_it == add->operands().end()) {
        return false;
      }
      auto* matched_add_operand = *add_operand_it;
      // Calculate operand index of 'add' operand which was not matched above.
      const int64 other_add_operand_index =
          matched_add_operand == add->operand(0) ? 1 : 0;
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root (at operand
      // index 'other_add_operand_index').
      return use.instruction == user->fused_expression_root() &&
             use.operand_number == other_add_operand_index;
    }
  }

  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kWhile) {
    // We eliminated other users in BufferLiveness::live_range_strictly_before,
    // so here we just need to check that the use is at operand index 0.
    std::vector<int64> operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && operand_indices[0] == 0;
  }
  if (user->opcode() == HloOpcode::kCall) {
    // Get all uses of value defined by 'operand' at 'operand_index'.
    const auto& uses = GetValueDefinedAt(operand, operand_index).uses();
    // Return true iff:
    // *) There exists two uses of 'operand'.
    // *) One use is by 'user' (caller).
    // *) One use is by root instruction of called computation (callee root).
    //    (Note: we check the root of the called computation, because the
    //     root result buffer is required to alias with the Call result buffer).
    // *) The root instruction of the called computation is element-wise on
    //    'operand'.
    const bool found_caller_use =
        std::find_if(uses.begin(), uses.end(), [user](const HloUse& use) {
          return use.instruction == user;
        }) != uses.end();
    auto* callee_root = user->to_apply()->root_instruction();
    const bool found_elementwise_callee_use =
        std::find_if(
            uses.begin(), uses.end(), [callee_root](const HloUse& use) {
              return use.instruction == callee_root &&
                     callee_root->IsElementwiseOnOperand(use.operand_number);
            }) != uses.end();
    return uses.size() == 2 && found_caller_use && found_elementwise_callee_use;
  }

  // Loop fusions that contain transposing copies won't reach here as they have
  // different layouts, which fails the check in the beginning of this function.
  return user->IsElementwiseOnOperand(user->operand_index(operand));
}

}  // namespace xla
