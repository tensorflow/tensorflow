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
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::StrAppend;
using absl::StrCat;

HloDataflowAnalysis::HloDataflowAnalysis(const HloModule& module, bool ssa_form,
                                         bool bitcast_defines_value,
                                         const CanShareBuffer& can_share_buffer)
    : module_(module),
      ssa_form_(ssa_form),
      bitcast_defines_value_(bitcast_defines_value),
      call_graph_(CallGraph::Build(&module)),
      can_share_buffer_(can_share_buffer) {}

bool HloDataflowAnalysis::AreTransitiveUsesElementwiseOrTuple(
    const HloInstruction* inst) {
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 4> stack;
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
      if (!visited.contains(user)) {
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
  CHECK(ValueIsDefinedAt(instruction, index)) << instruction->ToString();
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
  // Use a set to prevent deleting an id twice.
  absl::flat_hash_set<HloValue::Id> id_set(value_ids_to_delete_.begin(),
                                           value_ids_to_delete_.end());
#ifndef NDEBUG
  // Verify that no marked-for-deletion values are in any of the value sets.
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

  for (HloValue::Id value_id : id_set) {
    values_.erase(value_id);
  }
  value_ids_to_delete_.clear();
}

string HloDataflowAnalysis::ToString() const {
  string out = StrCat("HloDataflowAnalysis, module ", module_.name(), "\n");
  StrAppend(&out, "  Instruction value sets:\n");
  for (const HloComputation* computation : module_.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      StrAppend(&out, "Instruction: \n  ", instruction->name(), ":\n");
      if (instruction->shape().IsTuple()) {
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
    absl::Span<const InstructionValueSet* const> inputs) {
  CHECK(ssa_form_);
  VLOG(4) << "Phi(" << instruction->name() << ")";
  VLOG(5) << "instruction value set = "
          << GetInstructionValueSet(instruction).ToString();
  for (const InstructionValueSet* input : inputs) {
    VLOG(5) << "input value set = " << input->ToString();
  }

  if (bitcast_defines_value_) {
    absl::c_for_each(inputs, [&](const InstructionValueSet* input) {
      DCHECK(ShapeUtil::Compatible(instruction->shape(), input->shape()));
    });
  } else {
    const Shape& shape = instruction->shape();
    PrimitiveType ty = shape.element_type();
    bool is_array = shape.IsArray();
    absl::c_for_each(inputs, [&](const InstructionValueSet* input) {
      DCHECK(ty == input->shape().element_type() &&
             (!is_array || ShapeUtil::ElementsIn(shape) ==
                               ShapeUtil::ElementsIn(input->shape())));
    });
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

    // Construct a vector of value IDs of the inputs.
    std::vector<HloValue::Id> input_value_ids;
    for (const InstructionValueSet* input : inputs) {
      for (const HloValue* value : input->element(index).values()) {
        input_value_ids.push_back(value->id());
      }
    }

    // Remove the existing phi value (if it exists). The phi can be its own
    // input, for example, in while body parameters where the body passes
    // through the parameter value.
    bool current_value_defined_here =
        (current_value != nullptr &&
         current_value->defining_instruction() == instruction &&
         current_value->defining_index() == index);

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
      bool phi_defined_here =
          current_value_defined_here && current_value->is_phi();
      if (current_value == nullptr || !phi_defined_here) {
        value_set.Clear();
        value_set.AddValue(NewHloValue(instruction, index, /*is_phi=*/true));

        std::vector<HloValue*> inputs;
        inputs.reserve(input_value_ids.size());
        for (HloValue::Id id : input_value_ids) {
          inputs.push_back(&GetValue(id));
        }
        // Register the phi into phi graph.
        phi_graph_.RegisterPhi(*value_set.values()[0], inputs);
        changed = true;
      } else if (phi_defined_here) {
        std::vector<HloValue*> new_inputs;
        new_inputs.reserve(input_value_ids.size());
        for (HloValue::Id id : input_value_ids) {
          new_inputs.push_back(&GetValue(id));
        }

        if (!phi_graph_.InputsEqualTo(*current_value, new_inputs)) {
          VLOG(1) << current_value->ToShortString() << " has new phi inputs: ";
          // Update phi inputs.
          phi_graph_.RegisterPhi(*current_value, new_inputs);
          changed = true;
        }
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

HloValueSet HloDataflowAnalysis::GetFlattenedValueSet(
    const HloInstruction* instruction) const {
  HloValueSet value_set;

  const InstructionValueSet& value_set_tree =
      GetInstructionValueSet(instruction);

  std::vector<const HloValueSet*> all_sets;
  for (auto& pair : value_set_tree) {
    const HloValueSet& value_set = pair.second;
    all_sets.push_back(&value_set);
  }
  value_set.AssignUnionOf(all_sets);

  return value_set;
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

bool HloDataflowAnalysis::UpdateSetDimensionSizeValueSet(
    HloInstruction* set_dimension_size) {
  CHECK_EQ(set_dimension_size->opcode(), HloOpcode::kSetDimensionSize);
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(set_dimension_size->operand(0));
  InstructionValueSet& set_dimension_size_set =
      GetInstructionValueSet(set_dimension_size);
  if (operand_set != set_dimension_size_set) {
    set_dimension_size_set = operand_set;
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

bool HloDataflowAnalysis::UpdateCopyStartValueSet(HloInstruction* copy_start) {
  CHECK_EQ(copy_start->opcode(), HloOpcode::kCopyStart);
  bool changed = false;
  // CopyStart forwards the operand value to element {1} of its output.
  const HloValueSet& operand_value_set = GetValueSet(copy_start->operand(0));
  HloValueSet& value_set = GetValueSet(copy_start, {1});
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateCopyDoneValueSet(HloInstruction* copy_done) {
  CHECK_EQ(copy_done->opcode(), HloOpcode::kCopyDone);
  bool changed = false;
  // CopyDone forwards the operand value at {0} to element {} of its output.
  const HloValueSet& operand_value_set =
      GetValueSet(copy_done->operand(0), {0});
  HloValueSet& value_set = GetValueSet(copy_done);
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateRecvDoneValueSet(HloInstruction* recv_done) {
  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  bool changed = false;
  // RecvDone forwards the operand value at {0} to element {0} of its output.
  for (auto& pair : GetInstructionValueSet(recv_done)) {
    ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    if (index.empty() || index[0] != 0) {
      continue;
    }

    const HloValueSet& operand_value_set =
        GetValueSet(recv_done->operand(0), index);
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
  std::vector<const InstructionValueSet*> inputs(conditional->branch_count());
  for (int j = 0; j < conditional->branch_count(); ++j) {
    inputs[j] = &GetInstructionValueSet(
        conditional->branch_computation(j)->root_instruction());
  }
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

bool HloDataflowAnalysis::UpdateDomainValueSet(HloInstruction* domain) {
  // Domain instructions just forward their operand. Given that domains can have
  // a tuple operand, we iterate through its indexes, like for copies.
  // Unlike copies though we also propagate the top-level value.
  CHECK_EQ(domain->opcode(), HloOpcode::kDomain);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(domain)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(domain->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateAddDependencyValueSet(
    HloInstruction* add_dependency) {
  // AddDependency just forwards the value of its zero-th operand.
  CHECK_EQ(add_dependency->opcode(), HloOpcode::kAddDependency);
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(add_dependency->operand(0));
  InstructionValueSet& add_dependency_set =
      GetInstructionValueSet(add_dependency);
  if (operand_set != add_dependency_set) {
    add_dependency_set = operand_set;
    return true;
  }
  return false;
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
      // If the parameter *is not* the root, parameter state would be
      // updated by the root, otherwise don't consider it's current state
      // (InstructionValueSet) as we are recomputing its current state.
      if (parameter !=
          callsite.instruction()->while_body()->root_instruction()) {
        inputs.push_back(&GetInstructionValueSet(
            callsite.instruction()->while_body()->root_instruction()));
      }
      need_phi = true;
    } else if (callsite.instruction()->opcode() == HloOpcode::kConditional) {
      CHECK_EQ(parameter->parameter_number(), 0);
      auto conditional = callsite.instruction();
      // Conditional has branch_count+1 operands. Operand 0 is the branch_index,
      // operands 1 and onward are the arguments to the branch computations.
      //
      // If the parameter belongs to conditional's branch 0 computation, then
      // operand 1 is forwarded to this parameter instruction. If the parameter
      // belongs to conditional's branch 5 computation, then operand 6 is
      // forwarded to this parameter instruction.
      bool found_parent = false;
      for (int j = 0; j < conditional->branch_count(); ++j) {
        if (parameter->parent() == conditional->branch_computation(j)) {
          inputs.push_back(
              &GetInstructionValueSet(conditional->operand(j + 1)));
          found_parent = true;
          break;
        }
      }
      CHECK(found_parent);
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

bool HloDataflowAnalysis::UpdateTupleSelectValueSet(HloInstruction* select) {
  CHECK_EQ(select->opcode(), HloOpcode::kTupleSelect);
  // A phi value is not defined at a kTupleSelect instruction because
  // kTupleSelect does not create a new value. Rather it forwards a value from
  // its operands. This contrasts with kWhile instruction (which does define a
  // phi value) which has in-place update semantics.
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(select)) {
    const ShapeIndex& index = pair.first;
    if (index.empty()) {
      // kTupleSelect copies (not forwards) the top-level value.
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

bool HloDataflowAnalysis::UpdateCollectivePermuteStartValueSet(
    HloInstruction* collective_permute_start) {
  CHECK_EQ(collective_permute_start->opcode(),
           HloOpcode::kCollectivePermuteStart);
  bool changed = false;
  // CollectivePermuteStart forwards the operand value to element {0} of its
  // output.
  const HloValueSet& operand_value_set =
      GetValueSet(collective_permute_start->operand(0));
  HloValueSet& value_set = GetValueSet(collective_permute_start, {0});
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateCollectivePermuteDoneValueSet(
    HloInstruction* collective_permute_done) {
  CHECK_EQ(collective_permute_done->opcode(),
           HloOpcode::kCollectivePermuteDone);
  bool changed = false;
  // CollectivePermuteDone forwards the operand value at {0} to its output.
  const HloValueSet& operand_value_set =
      GetValueSet(collective_permute_done->operand(0), {1});
  HloValueSet& value_set = GetValueSet(collective_permute_done);
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateInstructionValueSet(
    HloInstruction* instruction) {
  // Recompute from operands.
  switch (instruction->opcode()) {
    case HloOpcode::kAddDependency:
      return UpdateAddDependencyValueSet(instruction);
    case HloOpcode::kBitcast:
      return UpdateBitcastValueSet(instruction);
    case HloOpcode::kSetDimensionSize:
      return UpdateSetDimensionSizeValueSet(instruction);
    case HloOpcode::kDomain:
      return UpdateDomainValueSet(instruction);
    case HloOpcode::kCopy:
      return UpdateCopyValueSet(instruction);
    case HloOpcode::kGetTupleElement:
      return UpdateGetTupleElementValueSet(instruction);
    case HloOpcode::kTupleSelect:
      return UpdateTupleSelectValueSet(instruction);
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
    case HloOpcode::kCopyStart:
      return UpdateCopyStartValueSet(instruction);
    case HloOpcode::kCopyDone:
      return UpdateCopyDoneValueSet(instruction);
    case HloOpcode::kConditional:
      return UpdateConditionalValueSet(instruction);
    case HloOpcode::kCollectivePermuteStart:
      return UpdateCollectivePermuteStartValueSet(instruction);
    case HloOpcode::kCollectivePermuteDone:
      return UpdateCollectivePermuteDoneValueSet(instruction);
    default:
      // Instruction does not forward HloValues (it defines all values in its
      // output). No update is necessary.
      return false;
  }
}

void HloDataflowAnalysis::Propagate() {
  std::queue<HloInstruction*> worklist;
  absl::flat_hash_set<HloInstruction*> workset;
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
  VLOG(1) << "SSA_FORM_: " << ssa_form_;

  while (!worklist.empty()) {
    HloInstruction* instruction = worklist.front();
    auto add_to_worklist = [&](HloInstruction* todo) {
      if (workset.insert(todo).second) {
        VLOG(1) << "  Adding todo : " << todo->name();
        worklist.push(todo);
      }
    };
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
        // updated, since that is the branch_index of the conditional.
        // If operand n+1 is the use of instruction, then the branch_computation
        // n's parameter need to be updated.
        //
        // Note that the same instruction can be used in multiple branches'
        // operands.
        for (int j = 0; j < user->branch_count(); ++j) {
          if (user->operand(j + 1) == instruction) {
            add_to_worklist(
                user->branch_computation(j)->parameter_instruction(0));
          }
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
        if (callsite.instruction()->opcode() == HloOpcode::kCall ||
            callsite.instruction()->opcode() == HloOpcode::kConditional) {
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

      // For each sub-shape of the instruction shape, add a new HloValue to its
      // HloValueSet.
      auto define_all_values = [this, &instruction]() {
        for (auto& pair : GetInstructionValueSet(instruction)) {
          const ShapeIndex& index = pair.first;
          HloValue* value = NewHloValue(instruction, index, /*is_phi=*/false);
          GetValueSet(instruction, index).AddValue(value);
        }
      };

      // Add a new HloValue to the HloValueSet corresponding to the given index
      // of the instruction shape.
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
        case HloOpcode::kSetDimensionSize:
        case HloOpcode::kAddDependency:
        case HloOpcode::kWhile:
        case HloOpcode::kCall:
        case HloOpcode::kConditional:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kDomain:
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
                computation->name());
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
        case HloOpcode::kTupleSelect:
        case HloOpcode::kTuple:
          // These instructions only define their top-level values. Any other
          // values flow from their operands.
          define_value_at(/*index=*/{});
          break;
        case HloOpcode::kCopyStart:
          // CopyStart produces a tuple of {destination buffer, aliased operand,
          // U32 context}.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{0});
          define_value_at(/*index=*/{2});
          break;
        case HloOpcode::kCopyDone:
          // CopyDone consumes a tuple produced by CopyStart and produces an
          // element. Its output aliases its input tuple element {0}.
          break;
        case HloOpcode::kRecvDone:
          // RecvDone produces a two-element tuple. Element zero aliases its
          // input tuple element {0}; element one is a token.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          break;
        case HloOpcode::kSend:
          // Send produces a tuple of {aliased operand, U32 context, token},
          // therefore only defines the top-level tuple and the tuple elements
          // at {1} and {2}.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          define_value_at(/*index=*/{2});
          break;
        default:
          define_all_values();
          break;
      }
    }
  }

  return Status::OK();
}

void HloDataflowAnalysis::OptimizePhiValues() {
  // Only applicable to SSA form where phis are defined.
  if (!ssa_form_) {
    return;
  }

  VLOG(1) << "Before phi graph optimization";
  XLA_VLOG_LINES(1, phi_graph_.ToString());
  phi_graph_.Optimize();
  VLOG(1) << "After phi graph optimization";
  XLA_VLOG_LINES(1, phi_graph_.ToString());

  for (const HloComputation* computation : module_.computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      InstructionValueSet& instruction_value_set =
          GetInstructionValueSet(instruction);
      VLOG(1) << "inst: " << instruction->name();
      VLOG(1) << instruction_value_set.ToString();
      instruction_value_set.ForEachMutableElement(
          [&](const xla::ShapeIndex& index, HloValueSet* value_set) {
            auto values = value_set->values();
            if (!(values.size() == 1 && values[0]->is_phi())) {
              return;
            }
            HloValue::Id phi_id = values[0]->id();
            HloValue::Id new_id = phi_graph_.FindOptimizedValue(phi_id);
            if (new_id != phi_id) {
              value_set->Clear();
              const HloValue& new_value = GetValue(new_id);
              value_set->AddValue(&new_value);
              MarkValueForDeletion(phi_id);
            }
          });
    }
  }
}

/* static */
StatusOr<std::unique_ptr<HloDataflowAnalysis>> HloDataflowAnalysis::Run(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value,
    const CanShareBuffer& can_share_buffer) {
  VLOG(1) << "HloDataflowAnalysis::Run on module " << module.name();
  XLA_VLOG_LINES(2, module.ToString());

  auto dataflow_analysis = absl::WrapUnique(new HloDataflowAnalysis(
      module, ssa_form, bitcast_defines_value, can_share_buffer));

  TF_RETURN_IF_ERROR(dataflow_analysis->InitializeInstructionValueSets());
  dataflow_analysis->Propagate();
  dataflow_analysis->OptimizePhiValues();

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
  absl::c_sort(dataflow_analysis->values_vector_, HloValue::IdLessThan);

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
      TF_RET_CHECK(absl::c_linear_search(value_set.values(), value))
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
          TF_RET_CHECK(absl::c_linear_search(value->positions(), position))
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
  // Return false if no value at 'operand' and 'index' is used at 'user'.
  for (const HloValue* value : GetValueSet(operand, index).values()) {
    for (const HloUse& use : value->uses()) {
      if (use.instruction == user) {
        if (user->IsLoopFusion()) {
          HloInstruction* fusion_param =
              user->fused_parameter(use.operand_number);
          const HloValue& value =
              GetValueDefinedAt(fusion_param, use.operand_index);
          return value.uses().empty();
        }
        return false;
      }
    }
  }
  return true;
}

// Given a fusion whose root is a dynamic-update-slice op, determines whether
// the fusion's output buffer can be shared with the buffer of fusion_param,
// which must be a fused parameter of the fusion.
//
// Preconditions:
//
//  - fusion's root is a dynamic-update-slice op.
//  - fusion_param is a parameter within the fusion.
//
// fusion_param may point to a subelement of the actual parameter instruction if
// the param is a tuple; i.e. fusion_param->index() need not be the empty list.
//
// Returns true if:
//
//  * fusion_param is used by the root of dynamic-update-slice as the "base" of
//    the update, i.e. the thing being updated, AND
//  * all other uses of fusion_param are dynamic-slices that slice the same
//    indices as are overwritten in the dynamic-update-slice.
//
// In the case that there are no other uses of fusion_param (last bullet point
// is vacuously true) it's easy to see why an in-place DUS is safe; this is just
// the "natural" implementation of DUS.  If there are other users, in-place DUS
// is safe on the assumption that the thread which writes element i of the
// output will be the only one to read element i of fusion_param (via the
// dynamic-slice ops).
static bool CanDoInPlaceDynamicUpdateSlice(HloInstruction* fusion,
                                           const HloValue& fusion_param_value) {
  auto* root =
      Cast<HloDynamicUpdateSliceInstruction>(fusion->fused_expression_root());
  auto* fusion_param = fusion_param_value.instruction();
  CHECK_EQ(fusion_param->opcode(), HloOpcode::kParameter);
  CHECK_EQ(fusion_param->parent(), fusion->fused_instructions_computation());

  // fusion_param must be used by the root as the "base" of the
  // dynamic-update-slice.  The natural way to check this would be
  //
  //   `if (root->operand(0) != fusion_param)`
  //
  // but we also have to handle the case where the fusion parameter is
  // tuple-shaped and we're considering just one element of that tuple, i.e.
  // fusion_param.index() != {}.
  if (absl::c_count_if(fusion_param_value.uses(), [&](const HloUse& use) {
        return use.instruction == root;
      }) != 1) {
    return false;
  }

  // All other uses of fusion_param must be dynamic-slices that slice the same
  // indices as are overwritten by the dynamic-update-slice.
  for (const HloUse& use : fusion_param_value.uses()) {
    auto* user = use.instruction;
    if (user == root) {
      continue;
    }

    // Check that `user` is a dynamic-slice op and has the same slice indices as
    // `root`.
    auto* ds = DynCast<HloDynamicSliceInstruction>(user);
    if (!ds || ds->index_operands() != root->index_operands()) {
      return false;
    }
  }
  return true;
}

bool HloDataflowAnalysis::CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (operand->opcode() == HloOpcode::kConstant) {
    return false;
  }
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

    const HloValue& fusion_param_value =
        GetValueDefinedAt(fusion_param, operand_index);

    // TODO(b/80315712): This code is in a bit of a weird intermediate state
    // at the moment. The in-place DUS check really needs to be common to all
    // backends, so it runs first. Then we run the backend-specific check if
    // provided, or go through the target-independent check if not.
    // Unfortunately, the notionally "target-independent" path actually contains
    // some target-specific code, so we can't run all of it *in addition* to the
    // target-specific function, like the interface documentation says.
    if (user->fused_expression_root()->opcode() ==
        HloOpcode::kDynamicUpdateSlice) {
      return CanDoInPlaceDynamicUpdateSlice(user, fusion_param_value);
    }
  }

  if (can_share_buffer_ != nullptr) {
    if (absl::optional<bool> hint =
            can_share_buffer_(user, operand, user_index)) {
      return *hint;
    }
  }

  if (user->opcode() == HloOpcode::kFusion) {
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));
    const HloValue& fusion_param_value =
        GetValueDefinedAt(fusion_param, operand_index);

    if (user->IsLoopFusion() || user->IsInputFusion()) {
      return AreTransitiveUsesElementwiseOrTuple(fusion_param);
    }

    if (user->IsOutputFusion() &&
        user->fused_expression_root()->opcode() == HloOpcode::kAdd) {
      // Output fusion with kAdd fused root.

      // Check if one operand of kAdd fused root is kDot or kConvolution.
      auto* add = user->fused_expression_root();
      auto add_operand_it =
          absl::c_find_if(add->operands(), [&](HloInstruction* operand) {
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
      if (fusion_param_value.uses().size() == 1) {
        const HloUse& use = fusion_param_value.uses()[0];
        return use.instruction == user->fused_expression_root() &&
               use.operand_number == other_add_operand_index;
      }
      return false;
    }
  }

  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kScatter ||
      user->opcode() == HloOpcode::kTriangularSolve ||
      user->opcode() == HloOpcode::kWhile) {
    // We eliminated other users in HloOrdering::LiveRangeStrictlyBefore
    // so here we just need to check that the use is at the right operand index.
    const auto operand_indices = user->OperandIndices(operand);
    int64 operand_no = user->opcode() == HloOpcode::kTriangularSolve ? 1 : 0;
    return operand_indices.size() == 1 && operand_indices[0] == operand_no;
  }
  if (user->opcode() == HloOpcode::kSort) {
    // Only valid if there are no other users.
    if (operand->users().size() != 1) {
      return false;
    }
    // If we only sort keys, the output of sort is not a tuple, so we can always
    // share the buffer.
    if (user->operand_count() == 1) {
      return true;
    }
    CHECK(!user_index.empty());
    // Only share with the right tuple element buffer.
    const auto operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && user_index[0] == operand_indices[0];
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
        absl::c_find_if(uses, [user](const HloUse& use) {
          return use.instruction == user;
        }) != uses.end();
    auto* callee_root = user->to_apply()->root_instruction();
    const bool found_elementwise_callee_use =
        absl::c_find_if(uses, [callee_root](const HloUse& use) {
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
