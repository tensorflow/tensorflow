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
#include "tensorflow/compiler/xla/service/liveness_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

HloDataflowAnalysis::HloDataflowAnalysis(HloModule* module, bool ssa_form,
                                         bool bitcast_defines_value)
    : module_(module),
      ssa_form_(ssa_form),
      bitcast_defines_value_(bitcast_defines_value),
      call_graph_(CallGraph::Build(module)) {}

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

string HloDataflowAnalysis::ToString() const {
  string out = StrCat("HloDataflowAnalysis, module ", module_->name(), "\n");
  StrAppend(&out, "  Instruction value sets:\n");
  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      StrAppend(&out, "    ", instruction->name(), ":\n");
      if (ShapeUtil::IsTuple(instruction->shape())) {
        GetInstructionValueSet(instruction.get())
            .ForEachElement([this, &instruction, &out](
                                const ShapeIndex& index,
                                const HloValueSet& value_set) {
              StrAppend(&out, "      tuple index ", index.ToString(), ":\n");
              for (const HloValue* value : value_set.values()) {
                StrAppend(
                    &out, "        ", value->ToShortString(),
                    ValueIsDefinedAt(instruction.get(), index) ? " (def)" : "",
                    "\n");
              }
            });
      } else {
        const HloValueSet& top_level_value_set =
            GetValueSet(instruction.get(), /*index=*/{});
        for (const HloValue* value : top_level_value_set.values()) {
          StrAppend(&out, "      ", value->ToShortString(),
                    ValueIsDefinedAt(instruction.get()) ? " (def)" : "", "\n");
        }
      }
    }
  }
  StrAppend(&out, "  HloValues:\n");
  for (const HloValue& value : values()) {
    StrAppend(&out, value.ToString(/*indent=*/4));
  }
  StrAppend(&out, "  Phi resolutions:\n");
  for (const HloValue& value : values()) {
    if (value.is_phi()) {
      const HloValue* resolved_value = ResolvePhi(value);
      StrAppend(&out, "    ", value.ToShortString(), " => ",
                resolved_value == nullptr ? "UNKNOWN"
                                          : resolved_value->ToShortString(),
                "\n");
    }
  }
  return out;
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

void HloDataflowAnalysis::UpdateAfterChangingOperand(
    HloInstruction* instruction, HloInstruction* old_operand,
    HloInstruction* new_operand) {
  CHECK(std::find(instruction->operands().begin(),
                  instruction->operands().end(),
                  new_operand) != instruction->operands().end());
  VLOG(1) << "UpdateAfterChangingOperand(" << instruction->name() << ", "
          << old_operand->name() << " => " << new_operand->name() << ")";

  std::vector<HloInstruction*> to_update = {instruction};

  // If the instruction calls any computations then add the parameters of called
  // computation to capture any changes to the dataflow into the subcomputation
  // introduced by the new operand.
  for (HloComputation* computation : instruction->called_computations()) {
    to_update.insert(to_update.end(),
                     computation->parameter_instructions().begin(),
                     computation->parameter_instructions().end());
  }

  UpdateInstructionsAndPropagate(to_update);

  // The uses of the values in the old and new operand may have changed. Uses of
  // other HloValues are updated in UpdateInstructionsAndPropagate.
  for (auto& pair : GetInstructionValueSet(old_operand)) {
    for (const HloValue* value : pair.second.values()) {
      GetValue(value->id()).RecomputeUses();
    }
  }
  for (auto& pair : GetInstructionValueSet(new_operand)) {
    for (const HloValue* value : pair.second.values()) {
      GetValue(value->id()).RecomputeUses();
    }
  }

  TF_DCHECK_OK(VerifyAgainstReference());
}

void HloDataflowAnalysis::UpdateAfterChangingRoot(HloInstruction* old_root,
                                                  HloInstruction* new_root) {
  VLOG(1) << "UpdateAfterChangingRoot(" << old_root->name() << " => "
          << new_root->name() << ")";

  CHECK_EQ(new_root, new_root->parent()->root_instruction());
  CHECK_EQ(new_root->parent(), old_root->parent());

  std::vector<HloInstruction*> to_update = {old_root, new_root};

  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(new_root->parent());
  for (const CallSite& callsite : call_graph_node.caller_callsites()) {
    if (callsite.instruction()->opcode() == HloOpcode::kCall) {
      to_update.push_back(callsite.instruction());
    } else if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
      // Add the while itself, and the body and condition parameters.
      to_update.push_back(callsite.instruction());
      to_update.push_back(
          callsite.instruction()->while_body()->parameter_instruction(0));
      to_update.push_back(
          callsite.instruction()->while_condition()->parameter_instruction(0));
    }
  }

  UpdateInstructionsAndPropagate(to_update);

  TF_DCHECK_OK(VerifyAgainstReference());
}

const HloValue* HloDataflowAnalysis::ResolvePhi(const HloValue& phi) const {
  CHECK(phi.is_phi());

  tensorflow::gtl::FlatSet<const HloValue*> visited;
  std::queue<const HloValue*> worklist;
  auto add_to_worklist = [&worklist, &visited](const HloValue* v) {
    if (visited.insert(v).second) {
      // 'v' was not previously in visited.
      worklist.push(v);
    }
  };
  add_to_worklist(&phi);

  const HloValue* resolved_value = nullptr;
  while (!worklist.empty()) {
    const HloValue* value = worklist.front();
    worklist.pop();

    if (!value->is_phi()) {
      if (resolved_value == nullptr) {
        resolved_value = value;
      } else if (resolved_value != value) {
        return nullptr;
      }
    } else {
      for (const HloValue* input : phi_inputs_.at(value)) {
        add_to_worklist(input);
      }
    }
  }
  return resolved_value;
}

void HloDataflowAnalysis::UpdatePhiInputs(
    const HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<const InstructionValueSet*> inputs) {
  CHECK(ssa_form_);
  for (auto& pair : GetInstructionValueSet(instruction)) {
    const ShapeIndex& index = pair.first;
    const HloValue& phi_value = GetUniqueValueAt(instruction, index);
    auto& phi_inputs = phi_inputs_.at(&phi_value);
    phi_inputs.clear();
    for (const InstructionValueSet* input : inputs) {
      for (const HloValue* value : input->element(index).values()) {
        // The number of phi inputs is typically 2, and virtually always very
        // small.
        if (std::find(phi_inputs.begin(), phi_inputs.end(), value) ==
            phi_inputs.end()) {
          phi_inputs.push_back(value);
        }
      }
    }
  }
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
  bool called_from_while = false;
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
      called_from_while = true;
    } else {
      LOG(FATAL) << "CallContext::kSequential computations should only be "
                    "called from call or while instructions";
    }
  }

  if (ssa_form_ && called_from_while) {
    UpdatePhiInputs(parameter, inputs);
    return false;
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
  std::vector<const InstructionValueSet*> inputs = {
      &GetInstructionValueSet(xla_while->while_body()->root_instruction()),
      &GetInstructionValueSet(xla_while->operand(0))};
  if (ssa_form_) {
    UpdatePhiInputs(xla_while, inputs);
    return false;
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
    default:
      // Instruction does not forward HloValues (it defines all values in its
      // output). No update is necessary.
      return false;
  }
}

void HloDataflowAnalysis::UpdateInstructionsAndPropagate(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions) {
  std::queue<HloInstruction*> worklist;
  for (HloInstruction* instruction : instructions) {
    worklist.push(instruction);
  }

  while (!worklist.empty()) {
    HloInstruction* instruction = worklist.front();
    worklist.pop();

    VLOG(3) << "Worklist top: " << instruction->name();
    VLOG(3) << ToString();

    // The updating of the instruction value set below in
    // UpdateInstructionValueSet does not update HloValue::positions(). To
    // perform the positions() update remove all positions in 'instruction' from
    // the HloValues in 'instruction's value set prior to the update, then after
    // the update add the new positions back in. There is likely a more
    // efficient way of doing this.
    for (auto& pair : GetInstructionValueSet(instruction)) {
      const ShapeIndex& index = pair.first;
      HloValueSet& value_set = pair.second;
      for (const HloValue* value : value_set.values()) {
        if (value->defining_instruction() != instruction) {
          // Use GetValue for a non-const HloValue reference.
          GetValue(value->id()).RemovePosition(instruction, index);
        }
      }
    }

    bool changed = UpdateInstructionValueSet(instruction);

    // Add the positions back in.
    for (auto& pair : GetInstructionValueSet(instruction)) {
      const ShapeIndex& index = pair.first;
      HloValueSet& value_set = pair.second;
      for (const HloValue* value : value_set.values()) {
        if (value->defining_instruction() != instruction) {
          // Use GetValue for a non-const HloValue reference.
          GetValue(value->id()).AddPosition(instruction, index);
        }
      }
    }

    if (!changed) {
      // No change to the instruction's value set.
      VLOG(4) << "No change.";
      continue;
    }

    VLOG(4) << "New value set for " << instruction->name() << ": "
            << GetInstructionValueSet(instruction);

    // Instruction value was updated. Add users to work list.
    for (HloInstruction* user : instruction->users()) {
      worklist.push(user);

      // If user calls a computation, then the respective parameter(s) of the
      // computation need to be updated.
      for (HloComputation* called_computation : user->called_computations()) {
        for (int64 operand_number : user->OperandIndices(instruction)) {
          worklist.push(
              called_computation->parameter_instruction(operand_number));
        }
      }
    }

    // If instruction is a root instruction, then propagate out to any calling
    // instruction and across any while backedge.
    if (instruction == instruction->parent()->root_instruction()) {
      const CallGraphNode& call_graph_node =
          call_graph_->GetNode(instruction->parent());
      for (const CallSite& callsite : call_graph_node.caller_callsites()) {
        if (callsite.instruction()->opcode() == HloOpcode::kCall) {
          worklist.push(callsite.instruction());
        } else if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
          // Add the while itself, and the body and condition parameters.
          worklist.push(callsite.instruction());
          worklist.push(
              callsite.instruction()->while_body()->parameter_instruction(0));
          worklist.push(
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
  // Gather the values to create before creating them. This is done because we
  // want to allocate the vector of values only once so references to elements
  // are stable.
  struct ValueToCreate {
    HloInstruction* instruction;
    ShapeIndex index;
    bool is_phi;
  };
  std::vector<ValueToCreate> values_to_create;

  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    const CallGraphNode& call_graph_node =
        call_graph_->GetNode(computation.get());
    bool called_from_while = std::any_of(
        call_graph_node.caller_callsites().begin(),
        call_graph_node.caller_callsites().end(), [](const CallSite& cs) {
          return cs.instruction()->opcode() == HloOpcode::kWhile;
        });

    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      // Create an empty shape tree.
      value_sets_.emplace(std::piecewise_construct,
                          std::forward_as_tuple(instruction.get()),
                          std::forward_as_tuple(instruction->shape()));

      // Lambda to set the value set to define all values in the output of the
      // instruction.
      auto define_all_values = [this, &instruction,
                                &values_to_create](bool is_phi = false) {
        for (auto& pair : GetInstructionValueSet(instruction.get())) {
          const ShapeIndex& index = pair.first;
          values_to_create.push_back({instruction.get(), index, is_phi});
        }
      };

      // Lambda to set the value set to define only the top-level buffer in the
      // output of the instruction. Any other values flow from the operands of
      // the instruction (or from cross-computation dataflow).
      auto define_top_level_only = [this, &instruction, &values_to_create]() {
        values_to_create.push_back(
            {instruction.get(), /*index=*/{}, /*is_phi=*/false});
      };

      switch (instruction->opcode()) {
        case HloOpcode::kBitcast:
          if (bitcast_defines_value_) {
            define_all_values();
          }
          break;
        case HloOpcode::kWhile:
          if (ssa_form_) {
            define_all_values(/*is_phi=*/true);
          }
          break;
        case HloOpcode::kCall:
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
          } else if (call_graph_node.context() == CallContext::kSequential &&
                     called_from_while && ssa_form_) {
            // Parameters of while bodies and conditions are phis.
            define_all_values(/*is_phi=*/true);
          }
          break;
        case HloOpcode::kCopy:
        case HloOpcode::kSelect:
        case HloOpcode::kTuple:
          // These instructions only define their top-level values. Any other
          // values flow from their operands.
          define_top_level_only();
          break;
        default:
          define_all_values();
          break;
      }
    }
  }

  // Reserve the vector ahead of time so references to elements are stable.
  values_.reserve(values_to_create.size());
  for (int64 i = 0; i < values_to_create.size(); ++i) {
    const ValueToCreate& to_create = values_to_create[i];
    values_.emplace_back(/*id=*/i, to_create.instruction, to_create.index,
                         to_create.is_phi);
    const HloValue& value = values_.back();
    GetValueSet(to_create.instruction, to_create.index).AddValue(&value);
    if (value.is_phi()) {
      phi_inputs_[&value] = {};
    }
  }
  return Status::OK();
}

bool HloDataflowAnalysis::IsDefinedBefore(const HloValue& a, const HloValue& b,
                                          const HloOrdering& ordering) const {
  // If 'b' is an entry param then 'a' cannot be defined before 'b' because 'b'
  // is live into the module.
  if (b.defining_instruction()->parent() == module_->entry_computation() &&
      b.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return false;
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

  return ordering.ExecutesBefore(a.defining_instruction(),
                                 b.defining_instruction());
}

bool HloDataflowAnalysis::UseIsBeforeValueDefinition(
    const HloUse& use, const HloValue& value,
    const HloOrdering& ordering) const {
  if (ordering.ExecutesBefore(use.instruction, value.defining_instruction())) {
    return true;
  }

  // If the use is at the instruction where the value is defined, then the use
  // is before the def if the instruction allows buffer sharing (in place
  // computation).
  if (use.instruction == value.defining_instruction() &&
      CanShareOperandBufferWithUser(
          use.instruction->mutable_operand(use.operand_number),
          use.operand_index, value.defining_instruction(),
          value.defining_index())) {
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
      return true;
    }
  }

  // Similarly if the value is defined at a while, it logically occurs after any
  // uses in the body or condition computations.
  if (value.defining_instruction()->opcode() == HloOpcode::kWhile) {
    CHECK(ssa_form_);
    const HloInstruction* xla_while = value.defining_instruction();
    if (call_graph_->InstructionIsNestedIn(use.instruction,
                                           xla_while->while_body()) ||
        call_graph_->InstructionIsNestedIn(use.instruction,
                                           xla_while->while_condition())) {
      return true;
    }
  }
  return false;
}

bool HloDataflowAnalysis::LiveRangeStrictlyBefore(
    const HloValue& a, const HloValue& b, const HloOrdering& ordering) const {
  VLOG(4) << "LiveRangeStrictlyBefore(a = " << a.ToShortString()
          << ", b = " << b.ToShortString() << ")";
  if (!IsDefinedBefore(a, b, ordering)) {
    VLOG(4) << "a not defined before b";
    return false;
  }

  // Live-out values from the module can never have ranges strictly before any
  // other value.
  if (a.live_out_of_module()) {
    VLOG(4) << "a is live out of module";
    return false;
  }

  // Live-out values of computations can never have ranges strictly before any
  // other value in the computation (including values nested in
  // subcomputations).
  if (a.live_out_of_computation() &&
      call_graph_->InstructionIsNestedIn(b.defining_instruction(),
                                         a.defining_instruction()->parent())) {
    VLOG(4) << "a is live out of computation containing b";
    return false;
  }

  // All uses of 'a' must be before 'b' is defined.
  for (const HloUse& use : a.uses()) {
    if (!UseIsBeforeValueDefinition(use, b, ordering)) {
      VLOG(4) << "use of a (" << use << ") not before b is defined";
      return false;
    }
  }

  return true;
}

bool HloDataflowAnalysis::MayInterfere(const HloValue& a, const HloValue& b,
                                       const HloOrdering& ordering) const {
  // Buffers without disjoint liveness may interfere.
  return !LiveRangeStrictlyBefore(a, b, ordering) &&
         !LiveRangeStrictlyBefore(b, a, ordering);
}

/* static */
StatusOr<std::unique_ptr<HloDataflowAnalysis>> HloDataflowAnalysis::Run(
    HloModule* module, bool ssa_form, bool bitcast_defines_value) {
  VLOG(1) << "HloDataflowAnalysis::Run on module " << module->name();
  XLA_VLOG_LINES(2, module->ToString());

  auto dataflow_analysis = WrapUnique(
      new HloDataflowAnalysis(module, ssa_form, bitcast_defines_value));

  TF_RETURN_IF_ERROR(dataflow_analysis->InitializeInstructionValueSets());

  // Construct list of all instructions to initialize the worklist to propagate
  // the data flow. For efficiency sort the instruction in post order so
  // producers appear before consumers.
  std::vector<HloInstruction*> all_instructions;
  for (const HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      all_instructions.push_back(instruction);
    }
  }
  dataflow_analysis->UpdateInstructionsAndPropagate(all_instructions);

  TF_DCHECK_OK(dataflow_analysis->Verify());

  XLA_VLOG_LINES(1, dataflow_analysis->ToString());

  return std::move(dataflow_analysis);
}

Status HloDataflowAnalysis::Verify() const {
  // Verify each HloValue appears in the value sets that the value's positions()
  // indicate.
  for (const HloValue& value : values()) {
    for (const HloPosition& position : value.positions()) {
      const HloValueSet& value_set = GetValueSet(position);
      TF_RET_CHECK(std::find(value_set.values().begin(),
                             value_set.values().end(),
                             &value) != value_set.values().end())
          << "Value set at position " << position << " does not contain value "
          << value.ToShortString();
    }
  }

  // For each value in each value set, verify that the value set's position
  // appears in the value's positions().
  for (const auto& computation : module_->computations()) {
    for (const auto& instruction : computation->instructions()) {
      for (const auto& pair : GetInstructionValueSet(instruction.get())) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        const HloPosition position{instruction.get(), index};
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

Status HloDataflowAnalysis::VerifyAgainstReference() const {
  TF_RETURN_IF_ERROR(Verify());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> reference,
                      Run(module_, ssa_form_, bitcast_defines_value_));
  TF_RETURN_IF_ERROR(reference->Verify());

  VLOG(2) << "This analysis:";
  XLA_VLOG_LINES(2, ToString());
  VLOG(2) << "Reference:";
  XLA_VLOG_LINES(2, reference->ToString());

  // Verify value sets in each position are identical.
  for (const auto& computation : module_->computations()) {
    for (const auto& instruction : computation->instructions()) {
      for (const auto& pair : GetInstructionValueSet(instruction.get())) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        const HloValueSet& reference_value_set =
            reference->GetValueSet(instruction.get(), index);

        auto value_in_set = [](const HloValue& v, const HloValueSet& vset) {
          return std::find_if(vset.values().begin(), vset.values().end(),
                              [&v](const HloValue* w) { return *w == v; }) !=
                 vset.values().end();
        };

        for (const HloValue* value : value_set.values()) {
          TF_RET_CHECK(value_in_set(*value, reference_value_set))
              << "Value " << value->ToShortString()
              << " does not exist in reference";
        }
        for (const HloValue* reference_value : reference_value_set.values()) {
          TF_RET_CHECK(value_in_set(*reference_value, value_set))
              << "Value " << reference_value->ToShortString()
              << " only exists in reference";
        }
      }
    }
  }

  // Verify all phis resolve identically and uses are identical.
  for (const HloValue& value : values()) {
    const HloValue& reference_value = reference->GetValueDefinedAt(
        value.defining_instruction(), value.defining_index());
    TF_RET_CHECK(value.is_phi() == reference_value.is_phi());
    if (value.is_phi()) {
      const HloValue* resolved_value = ResolvePhi(value);
      const HloValue* reference_resolved_value =
          reference->ResolvePhi(reference_value);
      if (resolved_value == nullptr) {
        TF_RET_CHECK(reference_resolved_value == nullptr);
      } else {
        TF_RET_CHECK(reference_resolved_value != nullptr);
        TF_RET_CHECK(*reference_resolved_value == *resolved_value);
      }
    }

    for (const HloUse& use : value.uses()) {
      TF_RET_CHECK(std::find(reference_value.uses().begin(),
                             reference_value.uses().end(),
                             use) != reference_value.uses().end());
    }
    for (const HloUse& reference_use : reference_value.uses()) {
      TF_RET_CHECK(std::find(value.uses().begin(), value.uses().end(),
                             reference_use) != value.uses().end());
    }
  }
  return Status::OK();
}

}  // namespace xla
