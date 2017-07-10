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
  if (value_set.value_ids().size() != 1) {
    return false;
  }
  return GetValue(value_set.GetUniqueValueId()).defining_instruction() ==
         instruction;
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

HloValue::Id HloDataflowAnalysis::NewHloValue(HloInstruction* instruction,
                                              const ShapeIndex& index,
                                              bool is_phi) {
  int64 value_id = next_value_id_++;
  auto it_added = values_.emplace(
      std::piecewise_construct, std::forward_as_tuple(value_id),
      std::forward_as_tuple(value_id, instruction, index, is_phi));
  CHECK(it_added.second);

  // Clear the vector of values as it is now stale. It will be lazily
  // reconstructed if needed when HloDataflowAnalysis::values() is called.
  values_vector_.clear();

  return value_id;
}

void HloDataflowAnalysis::DeleteHloValue(HloValue::Id value_id) {
  values_.erase(value_id);

  // Clear the vector of values as it is now stale. It will be lazily
  // reconstructed if needed when HloDataflowAnalysis::values() is called.
  values_vector_.clear();
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
              for (HloValue::Id value_id : value_set.value_ids()) {
                StrAppend(
                    &out, "        ", GetValue(value_id).ToShortString(),
                    ValueIsDefinedAt(instruction.get(), index) ? " (def)" : "",
                    "\n");
              }
            });
      } else {
        const HloValueSet& top_level_value_set =
            GetValueSet(instruction.get(), /*index=*/{});
        for (HloValue::Id value_id : top_level_value_set.value_ids()) {
          StrAppend(&out, "      ", GetValue(value_id).ToShortString(),
                    ValueIsDefinedAt(instruction.get()) ? " (def)" : "", "\n");
        }
      }
    }
  }
  StrAppend(&out, "  HloValues:\n");
  for (const auto& pair : values_) {
    StrAppend(&out, pair.second.ToString(/*indent=*/4));
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

const std::vector<const HloValue*>& HloDataflowAnalysis::values() const {
  if (values_vector_.empty()) {
    // Lazily construct vector of values.
    values_vector_.reserve(values_.size());
    for (auto& pair : values_) {
      values_vector_.push_back(&pair.second);
    }
    std::sort(
        values_vector_.begin(), values_vector_.end(),
        [](const HloValue* a, const HloValue* b) { return a->id() < b->id(); });
  } else {
    CHECK_EQ(values_vector_.size(), values_.size());
    for (const HloValue* value : values_vector_) {
      DCHECK(ContainsKey(values_, value->id()));
      DCHECK(&GetValue(value->id()) == value);
    }
  }
  return values_vector_;
}

/* static */
InstructionValueSet HloDataflowAnalysis::Phi(
    HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<const InstructionValueSet*> inputs,
    bool skip_top_level) {
  CHECK(ssa_form_);

  for (const InstructionValueSet* input : inputs) {
    CHECK(ShapeUtil::Compatible(instruction->shape(), input->shape()));
  }
  InstructionValueSet new_value_set(instruction->shape());
  new_value_set.ForEachMutableElement(
      [this, instruction, &inputs, skip_top_level](const ShapeIndex& index,
                                                   HloValueSet* value_set) {
        // If we're skipping the top level, just copy over the existing
        // HloValueSet.
        if (skip_top_level && index.empty()) {
          *value_set = GetInstructionValueSet(instruction).element(index);
          return;
        }

        // Identify the existing phi value at this index if it exists.
        const HloValue* existing_phi_value = nullptr;
        if (ValueIsDefinedAt(instruction, index) &&
            GetUniqueValueAt(instruction, index).is_phi()) {
          existing_phi_value = &GetUniqueValueAt(instruction, index);
        }

        // Construct a vector of unique value IDs of the inputs.
        std::vector<HloValue::Id> input_value_ids;
        for (const InstructionValueSet* input : inputs) {
          for (HloValue::Id value_id : input->element(index).value_ids()) {
            input_value_ids.push_back(value_id);
          }
        }
        std::sort(input_value_ids.begin(), input_value_ids.end());
        input_value_ids.erase(
            std::unique(input_value_ids.begin(), input_value_ids.end()),
            input_value_ids.end());

        // Remove the existing phi value (if it exists). The phi can be its own
        // input, for example, in while body parameters where the body passes
        // through the parameter value.
        if (existing_phi_value != nullptr) {
          auto it = std::find(input_value_ids.begin(), input_value_ids.end(),
                              existing_phi_value->id());
          if (it != input_value_ids.end()) {
            input_value_ids.erase(it);
          }
        }

        if (input_value_ids.size() <= 1) {
          if (input_value_ids.size() == 1) {
            *value_set = HloValueSet({input_value_ids[0]});
          }
          if (existing_phi_value) {
            // The merge point does not have multiple distinct inputs (which are
            // not the phi value itself). Therefore there is no need to insert a
            // phi value because there is a single reaching definition (or no
            // reaching definition).
            DeleteHloValue(existing_phi_value->id());
          }
        } else if (input_value_ids.size() > 1) {
          // Multiple distinct values reach this point. A phi value is
          // necessary.
          if (existing_phi_value) {
            // A phi value already exists so reuse it in the new
            // InstructionValueSet.
            *value_set = HloValueSet({existing_phi_value->id()});
          } else {
            // Create a new phi value.
            *value_set =
                HloValueSet({NewHloValue(instruction, index, /*is_phi=*/true)});
          }
        }
      });
  return new_value_set;
}

void HloDataflowAnalysis::UpdateLocationsOfValuesAt(
    HloInstruction* instruction, const InstructionValueSet& new_value_set,
    const InstructionValueSet* prev_value_set) {
  if (prev_value_set != nullptr) {
    // Remove locations from the old value set.
    prev_value_set->ForEachElement(
        [this, instruction](const ShapeIndex& index,
                            const HloValueSet& value_set) {
          for (HloValue::Id value_id : value_set.value_ids()) {
            // HloValues in the previous value set may have been deleted.
            if (!ContainsKey(values_, value_id)) {
              continue;
            }
            // Don't remove the defining location of the value.
            HloValue& value = GetValue(value_id);
            if (instruction == value.defining_instruction()) {
              CHECK_EQ(index, value.defining_index());
            } else {
              value.RemoveLocation(instruction, index);
            }
          }
        });
  }
  // Add locations in the new value set.
  new_value_set.ForEachElement(
      [this, instruction](const ShapeIndex& index,
                          const HloValueSet& value_set) {
        for (HloValue::Id value_id : value_set.value_ids()) {
          HloValue& value = GetValue(value_id);
          if (instruction == value.defining_instruction()) {
            CHECK_EQ(index, value.defining_index());
          } else {
            value.AddLocation(instruction, index);
          }
        }
      });
}

InstructionValueSet HloDataflowAnalysis::RecomputeBitcastValueSet(
    HloInstruction* bitcast) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcast);
  if (bitcast_defines_value_) {
    return GetInstructionValueSet(bitcast);
  } else {
    return GetInstructionValueSet(bitcast->operand(0));
  }
}

InstructionValueSet HloDataflowAnalysis::RecomputeCopyValueSet(
    HloInstruction* copy) {
  CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
  InstructionValueSet new_value_set = GetInstructionValueSet(copy);
  if (ShapeUtil::IsTuple(copy->shape())) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(copy->shape()); ++i) {
      new_value_set.CopySubtreeFrom(GetInstructionValueSet(copy->operand(0)),
                                    /*source_base_index=*/{i},
                                    /*target_base_index=*/{i});
    }
  }
  return new_value_set;
}

InstructionValueSet HloDataflowAnalysis::RecomputeGetTupleElementValueSet(
    HloInstruction* gte) {
  CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
  InstructionValueSet new_value_set(gte->shape());
  new_value_set.CopySubtreeFrom(GetInstructionValueSet(gte->operand(0)),
                                /*source_base_index=*/{gte->tuple_index()},
                                /*target_base_index=*/{});
  return new_value_set;
}

InstructionValueSet HloDataflowAnalysis::RecomputeSelectValueSet(
    HloInstruction* select) {
  CHECK_EQ(select->opcode(), HloOpcode::kSelect);
  std::vector<const InstructionValueSet*> inputs = {
      &GetInstructionValueSet(select->operand(1)),
      &GetInstructionValueSet(select->operand(2))};
  // A phi value is not defined at a kSelect instruction because kSelect does
  // not create a new value. Rather it forwards a value from its operands. This
  // contrasts with kWhile instruction (which does define a phi value) which has
  // in-place update semantics.
  InstructionValueSet new_value_set = InstructionValueSet::Union(inputs);
  *new_value_set.mutable_element(/*index=*/{}) =
      GetInstructionValueSet(select).element(/*index=*/{});
  return new_value_set;
}

InstructionValueSet HloDataflowAnalysis::RecomputeTupleValueSet(
    HloInstruction* tuple) {
  CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
  InstructionValueSet new_value_set(tuple->shape());
  *new_value_set.mutable_element(/*index=*/{}) =
      GetInstructionValueSet(tuple).element(/*index=*/{});
  for (int64 i = 0; i < tuple->operands().size(); ++i) {
    new_value_set.CopySubtreeFrom(GetInstructionValueSet(tuple->operand(i)),
                                  /*source_base_index=*/{},
                                  /*target_base_index=*/{i});
  }
  return new_value_set;
}

InstructionValueSet HloDataflowAnalysis::RecomputeWhileValueSet(
    HloInstruction* xla_while) {
  CHECK_EQ(xla_while->opcode(), HloOpcode::kWhile);
  std::vector<const InstructionValueSet*> inputs = {
      &GetInstructionValueSet(xla_while->while_body()->root_instruction()),
      &GetInstructionValueSet(xla_while->operand(0))};
  if (ssa_form_) {
    return Phi(xla_while, inputs);
  } else {
    return InstructionValueSet::Union(inputs);
  }
}

void HloDataflowAnalysis::UpdateInstructionValueSet(
    HloInstruction* instruction) {
  // Recompute from operands.
  InstructionValueSet& value_set = GetInstructionValueSet(instruction);
  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
      value_set = RecomputeBitcastValueSet(instruction);
      break;
    case HloOpcode::kCopy:
      value_set = RecomputeCopyValueSet(instruction);
      break;
    case HloOpcode::kGetTupleElement:
      value_set = RecomputeGetTupleElementValueSet(instruction);
      break;
    case HloOpcode::kSelect:
      value_set = RecomputeSelectValueSet(instruction);
      break;
    case HloOpcode::kTuple:
      value_set = RecomputeTupleValueSet(instruction);
      break;
    case HloOpcode::kParameter:
      value_set = RecomputeParameterValueSet(instruction);
      break;
    case HloOpcode::kCall:
      // The output of a kCall instruction is exactly the output of the root of
      // the subcomputation.
      value_set =
          GetInstructionValueSet(instruction->to_apply()->root_instruction());
      break;
    case HloOpcode::kWhile:
      value_set = RecomputeWhileValueSet(instruction);
      break;
    default:
      // Instruction does not forward HloValues (it defines all values in its
      // output). No update is necessary.
      return;
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

    // Save old value for recomputing uses and live out.
    InstructionValueSet old_value = GetInstructionValueSet(instruction);
    UpdateInstructionValueSet(instruction);

    if (GetInstructionValueSet(instruction) == old_value) {
      // No change to the instruction's value set.
      VLOG(4) << "No change.";
      continue;
    }

    VLOG(4) << "New value set for " << instruction->name() << ": "
            << GetInstructionValueSet(instruction);
    VLOG(4) << "Previously: " << old_value;

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

    // Update uses. First clear all of the old uses at the particular
    // operands. Then add the new uses. There may be overlap between the old
    // uses and new uses.
    UpdateLocationsOfValuesAt(instruction, GetInstructionValueSet(instruction),
                              &old_value);
  }
}

InstructionValueSet HloDataflowAnalysis::RecomputeParameterValueSet(
    HloInstruction* parameter) {
  CHECK_EQ(parameter->opcode(), HloOpcode::kParameter);
  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(parameter->parent());

  // Subcomputations called in a parallel context (eg, map) do not have dataflow
  // from the caller operands.
  if (call_graph_node.context() == CallContext::kParallel ||
      call_graph_node.caller_callsites().empty()) {
    return GetInstructionValueSet(parameter);
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
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->while_body()->root_instruction()));
      called_from_while = true;
    } else {
      LOG(FATAL) << "CallContext::kSequential computations should only be "
                    "called from call or while instructions";
    }
  }

  if (ssa_form_ && called_from_while) {
    return Phi(parameter, inputs);
  } else {
    return InstructionValueSet::Union(inputs);
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
  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    const CallGraphNode& call_graph_node =
        call_graph_->GetNode(computation.get());
    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      // Create an empty shape tree.
      value_sets_.emplace(std::piecewise_construct,
                          std::forward_as_tuple(instruction.get()),
                          std::forward_as_tuple(instruction->shape()));

      // Lambda to set the value set to define all values in the output of the
      // instruction.
      auto define_all_values = [this, &instruction]() {
        GetInstructionValueSet(instruction.get())
            .ForEachMutableElement([this, &instruction](
                                       const ShapeIndex& index,
                                       HloValueSet* value_set) {
              *value_set = HloValueSet({NewHloValue(instruction.get(), index)});
            });
      };

      // Lambda to set the value set to define only the top-level buffer in the
      // output of the instruction. Any other values flow from the operands of
      // the instruction (or from cross-computation dataflow).
      auto define_top_level_only = [this, &instruction]() {
        GetValueSet(instruction.get(), /*index=*/{}) =
            HloValueSet({NewHloValue(instruction.get(), /*index=*/{})});
      };

      switch (instruction->opcode()) {
        case HloOpcode::kBitcast:
          if (bitcast_defines_value_) {
            define_all_values();
          }
          break;
        case HloOpcode::kCall:
        case HloOpcode::kWhile:
        case HloOpcode::kGetTupleElement:
          // These instructions define no values. The values in their output
          // flow from their operands or from cross computation dataflow.
          break;
        case HloOpcode::kParameter:
          if (call_graph_node.caller_callsites().empty() ||
              call_graph_node.context() == CallContext::kParallel) {
            // Parameters of computations called in a parallel context (eg, map
            // and reduce) as well as parameters of dead computations define all
            // values in their output. Otherwise the values of the parameter
            // come from the caller (eg, operands to the kCall instruction).
            define_all_values();
          } else if (call_graph_node.context() == CallContext::kBoth) {
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
      UpdateLocationsOfValuesAt(instruction.get(),
                                GetInstructionValueSet(instruction.get()));
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

  VLOG(1) << dataflow_analysis->ToString();
  return std::move(dataflow_analysis);
}

}  // namespace xla
