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

HloValue* HloDataflowAnalysis::NewHloValue(HloInstruction* instruction,
                                           const ShapeIndex& index,
                                           bool is_phi) {
  const int64 value_id = next_value_id_++;
  auto emplaced = values_.emplace(
      std::piecewise_construct, std::forward_as_tuple(value_id),
      std::forward_as_tuple(value_id, instruction, index, is_phi));
  CHECK(emplaced.second);

  return &emplaced.first->second;
}

void HloDataflowAnalysis::DeleteHloValue(HloValue::Id value_id) {
  values_.erase(value_id);
}

string HloDataflowAnalysis::ToString() const {
  string out = StrCat("HloDataflowAnalysis, module ", module_->name(), "\n");
  StrAppend(&out, "  Instruction value sets:\n");
  for (const HloComputation* computation : module_->computations()) {
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
    std::vector<HloValue::Id> input_value_ids;
    for (const InstructionValueSet* input : inputs) {
      for (const HloValue* value : input->element(index).values()) {
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
      CHECK(current_value->is_phi());
      auto it = std::find(input_value_ids.begin(), input_value_ids.end(),
                          current_value->id());
      if (it != input_value_ids.end()) {
        input_value_ids.erase(it);
      }
    }

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
          DeleteHloValue(current_value->id());
        }
        value_set.Clear();
        value_set.AddValue(&new_value);
        changed = true;
      }
    } else {
      // Multiple distinct values reach this point. A phi value is
      // necessary.
      CHECK_GT(input_value_ids.size(), 1);
      if (current_value == nullptr || !current_value->is_phi()) {
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
  std::vector<const InstructionValueSet*> inputs = {
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

    if (!UpdateInstructionValueSet(instruction)) {
      // No change to the instruction's value set.
      VLOG(4) << "No change.";
      continue;
    }

    VLOG(4) << "New value set for " << instruction->name() << ": "
            << GetInstructionValueSet(instruction);

    // Instruction value was updated. Add users to work list.
    for (HloInstruction* user : instruction->users()) {
      worklist.push(user);

      // If user sequentially calls a computation, then the respective
      // parameter(s) of the computation need to be updated.
      for (HloComputation* called_computation : user->called_computations()) {
        const CallGraphNode& call_graph_node =
            call_graph_->GetNode(called_computation);
        if (call_graph_node.context() == CallContext::kSequential) {
          for (int64 operand_number : user->OperandIndices(instruction)) {
            worklist.push(
                called_computation->parameter_instruction(operand_number));
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
  for (const HloComputation* computation : module_->computations()) {
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

      switch (instruction->opcode()) {
        case HloOpcode::kBitcast:
          if (bitcast_defines_value_) {
            define_all_values();
          }
          break;
        case HloOpcode::kWhile:
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

  return Status::OK();
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

  // Add in positions to all values.
  for (const HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      for (const auto& pair :
           dataflow_analysis->GetInstructionValueSet(instruction)) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        for (const HloValue* value : value_set.values()) {
          if (value->defining_instruction() != instruction) {
            dataflow_analysis->GetValue(value->id())
                .AddPosition(instruction, index);
          }
        }
      }
    }
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
  for (const auto& computation : module_->computations()) {
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

}  // namespace xla
