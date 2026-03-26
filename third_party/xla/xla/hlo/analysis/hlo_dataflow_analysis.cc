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

#include "xla/hlo/analysis/hlo_dataflow_analysis.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/map_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {
// CalculatePostOrderSchedule traverses a module and assign a ordinal to each
// instruction based the postorder dependency.
int64_t CalculatePostOrderScheduleHelper(
    const HloComputation* comp, int64_t start_ordinal,
    absl::flat_hash_map<HloInstruction*, int64_t>* ordinal_map) {
  int64_t ordinal = start_ordinal;
  for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall ||
        instruction->opcode() == HloOpcode::kAsyncStart ||
        instruction->opcode() == HloOpcode::kConditional) {
      for (const HloComputation* called_computation :
           instruction->called_computations()) {
        ordinal = CalculatePostOrderScheduleHelper(called_computation, ordinal,
                                                   ordinal_map);
      }
    }
    if (instruction->opcode() == HloOpcode::kWhile) {
      ordinal = CalculatePostOrderScheduleHelper(instruction->while_condition(),
                                                 ordinal, ordinal_map);
      ordinal = CalculatePostOrderScheduleHelper(instruction->while_body(),
                                                 ordinal, ordinal_map);
    }
    // It's possible that in some unit tests the computation graph is not
    // flatten (meaning we could have multiple callers for one computation). In
    // that case the oridinal_map will see the instruction multiple times. We
    // consider that case to be ok as it only shows up in unit tests.
    VLOG(4) << "Add instruction " << instruction->name()
            << " to ordinal map with ordinal " << ordinal;
    ordinal_map->insert({instruction, ordinal++});
  }
  return ordinal;
}

absl::flat_hash_map<HloInstruction*, int64_t> CalculatePostOrderSchedule(
    const HloModule& module) {
  absl::flat_hash_map<HloInstruction*, int64_t> map;
  CalculatePostOrderScheduleHelper(module.entry_computation(), 0, &map);
  return map;
}

}  // namespace
using absl::StrAppend;
using absl::StrCat;

HloDataflowAnalysis::HloDataflowAnalysis(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value,
    absl::flat_hash_set<absl::string_view> execution_threads)
    : module_(module),
      execution_threads_(std::move(execution_threads)),
      ssa_form_(ssa_form),
      bitcast_defines_value_(bitcast_defines_value),
      call_graph_(CallGraph::Build(&module)) {}

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
      for (const int64_t use_index : user->OperandIndices(current)) {
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
  const int64_t value_id = NewValueId();
  auto result =
      values_.insert({value_id, std::make_unique<HloValue>(
                                    value_id, instruction, index, is_phi)});
  CHECK(result.second);

  VLOG(4) << "NewHloValue = " << result.first->second->ToShortString();

  return result.first->second.get();
}

void HloDataflowAnalysis::MarkValueForDeletion(HloValue::Id value_id) {
  const HloValue& value = GetValue(value_id);
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
    const InstructionValueSet& instruction_value_set = *pair.second;
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

std::string HloDataflowAnalysis::ToString() const {
  std::string out =
      StrCat("HloDataflowAnalysis, module ", module_.name(), "\n");
  StrAppend(&out, "  Instruction value sets:\n");
  for (const HloComputation* computation : module_.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
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
      DCHECK(
          ty == input->shape().element_type() &&
          (!is_array ||
           ShapeUtil::ElementsIn(shape) ==
               ShapeUtil::ElementsIn(input->shape()) ||
           ShapeUtil::ArraySize(shape) == ShapeUtil::ArraySize(input->shape())))
          << shape.ToString() << " vs." << input->shape().ToString();
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
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
}

HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) {
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
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

HloValueSet& HloDataflowAnalysis::GetMutableValueSet(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return *GetInstructionValueSet(instruction).mutable_element(index);
}

const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloPosition& position) const {
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

bool HloDataflowAnalysis::UpdateSendValueSet(HloInstruction* send) {
  CHECK_EQ(send->opcode(), HloOpcode::kSend);
  bool changed = false;
  // Send forwards the operand value to the output tuple at {0}.
  for (auto& pair : GetInstructionValueSet(send->operand(0))) {
    const ShapeIndex& operand_index = pair.first;
    const HloValueSet& operand_value_set = pair.second;

    ShapeIndex index = {0};
    for (int64_t i : operand_index) {
      index.push_back(i);
    }

    HloValueSet& value_set = GetMutableValueSet(send, index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateAsyncStartValueSet(
    HloInstruction* async_start) {
  CHECK_EQ(async_start->opcode(), HloOpcode::kAsyncStart);
  bool changed = false;
  // AsyncStart forwards the operand values to element {0} of its output.
  for (int64_t i = 0; i < async_start->operand_count(); ++i) {
    const HloInstruction* operand = async_start->operand(i);
    ShapeUtil::ForEachSubshape(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!subshape.IsArray()) {
            return;
          }
          const HloValueSet& operand_value_set = GetValueSet(operand, index);

          ShapeIndex output_index = {0, i};
          output_index.insert(output_index.end(), index.begin(), index.end());

          HloValueSet& value_set =
              GetMutableValueSet(async_start, output_index);
          if (value_set != operand_value_set) {
            value_set = operand_value_set;
            changed = true;
          }
        });
  }
  if (!HloInstruction::IsThreadIncluded(async_start->async_execution_thread(),
                                        execution_threads_)) {
    return changed;
  }
  // AsyncStart forwards the async wrapped computation root values to element
  // {1} of its output.
  HloInstruction* root =
      async_start->async_wrapped_computation()->root_instruction();
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloValueSet& root_value_set = GetValueSet(root, index);

        ShapeIndex output_index = {1};
        output_index.insert(output_index.end(), index.begin(), index.end());

        HloValueSet& value_set = GetMutableValueSet(async_start, output_index);
        if (value_set != root_value_set) {
          value_set = root_value_set;
          changed = true;
        }
      });
  return changed;
}

bool HloDataflowAnalysis::UpdateAsyncUpdateValueSet(
    HloInstruction* async_update) {
  CHECK_EQ(async_update->opcode(), HloOpcode::kAsyncUpdate);
  CHECK_EQ(async_update->shape(), async_update->operand(0)->shape());
  bool changed = false;
  HloInstruction* root =
      HloInstruction::IsThreadIncluded(async_update->async_execution_thread(),
                                       execution_threads_)
          ? async_update->async_wrapped_computation()->root_instruction()
          : nullptr;
  // AsyncUpdate forwards all of the operand values to corresponding elements of
  // its output.
  ShapeUtil::ForEachSubshape(
      async_update->operand(0)->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloValueSet& operand_value_set =
            GetValueSet(async_update->operand(0), index);

        HloValueSet& value_set = GetMutableValueSet(async_update, index);
        CHECK_GE(index.size(), 0);
        if (index[0] == 1 && root != nullptr) {
          // If this subshape is an output (index {1}), we need to create the
          // union with the async wrapped computation root.
          ShapeIndex root_index(index.begin() + 1, index.end());
          const HloValueSet& root_value_set = GetValueSet(root, root_index);
          changed |=
              value_set.AssignUnionOf({&operand_value_set, &root_value_set});
        } else if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      });
  return changed;
}

bool HloDataflowAnalysis::UpdateAsyncDoneValueSet(HloInstruction* async_done) {
  CHECK_EQ(async_done->opcode(), HloOpcode::kAsyncDone);
  bool changed = false;
  HloInstruction* root =
      HloInstruction::IsThreadIncluded(async_done->async_execution_thread(),
                                       execution_threads_)
          ? async_done->async_wrapped_computation()->root_instruction()
          : nullptr;
  // AsyncDone creates a union of the operand values at {1} and the async
  // wrapped computation root to element {} of its output.
  ShapeUtil::ForEachSubshape(
      async_done->operand(0)->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray() || index.front() != 1) {
          return;
        }
        const HloValueSet& operand_value_set =
            GetValueSet(async_done->operand(0), index);

        ShapeIndex output_index(index.begin() + 1, index.end());
        HloValueSet& value_set = GetMutableValueSet(async_done, output_index);
        if (root != nullptr) {
          const HloValueSet& root_value_set = GetValueSet(root, output_index);
          changed |=
              value_set.AssignUnionOf({&operand_value_set, &root_value_set});
        } else if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      });
  return changed;
}

bool HloDataflowAnalysis::UpdateCopyStartValueSet(HloInstruction* copy_start) {
  CHECK_EQ(copy_start->opcode(), HloOpcode::kCopyStart);
  bool changed = false;
  // CopyStart forwards the operand value to element {1} of its output.
  const HloValueSet& operand_value_set = GetValueSet(copy_start->operand(0));
  HloValueSet& value_set = GetMutableValueSet(copy_start, {1});
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
  HloValueSet& value_set = GetMutableValueSet(copy_done);
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
  if (!HloInstruction::IsThreadIncluded(call->to_apply()->execution_thread(),
                                        execution_threads_)) {
    return false;
  }
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
  }
  return GetInstructionValueSet(conditional).AssignUnionOf(inputs);
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
    HloValueSet& operand_value_set =
        GetMutableValueSet(copy->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateOptimizationBarrierValueSet(
    HloInstruction* barrier) {
  // Optimization Barriers just forward their operand. Given that barriers can
  // have a tuple operand, we iterate through its indexes, like for copies.
  // Unlike copies though we also propagate the top-level value.
  CHECK_EQ(barrier->opcode(), HloOpcode::kOptimizationBarrier);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(barrier)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set =
        GetMutableValueSet(barrier->operand(0), index);
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
    HloValueSet& operand_value_set =
        GetMutableValueSet(domain->operand(0), index);
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
    for (int64_t i : index) {
      operand_index.push_back(i);
    }

    HloValueSet& operand_value_set =
        GetMutableValueSet(gte->operand(0), operand_index);
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
  if (call_graph_node.context() == CallContext::kEmbedded ||
      call_graph_node.caller_callsites().empty()) {
    return false;
  }
  CHECK_EQ(call_graph_node.context(), CallContext::kControlFlow);

  std::vector<const InstructionValueSet*> inputs;
  bool need_phi = false;
  for (const CallSite& callsite : call_graph_node.caller_callsites()) {
    const HloOpcode& opcode = callsite.instruction()->opcode();
    if (opcode == HloOpcode::kCall) {
      // The operand values of a call instruction are forwarded to the
      // respective parameter instruction of the subcomputation.
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (opcode == HloOpcode::kWhile) {
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
    } else if (opcode == HloOpcode::kConditional) {
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
    } else if (opcode == HloOpcode::kAsyncStart) {
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (opcode == HloOpcode::kAsyncUpdate ||
               opcode == HloOpcode::kAsyncDone) {
      return GetInstructionValueSet(parameter).AssignUnionOf(
          GetInstructionValueSet(callsite.instruction()->operand(0)),
          {0, parameter->parameter_number()});
    } else {
      LOG(FATAL) << "CallContext::kSequential computations should only be "
                    "called from call, while, or conditional instructions";
    }
  }
  if (ssa_form_ && need_phi) {
    return Phi(parameter, inputs);
  }
  return GetInstructionValueSet(parameter).AssignUnionOf(inputs);
}

bool HloDataflowAnalysis::UpdateTupleValueSet(HloInstruction* tuple) {
  CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
  bool changed = false;
  for (int64_t i = 0; i < tuple->operands().size(); ++i) {
    // Copy the value set(s) of each operand into the respective position in the
    // kTuple instruction's value sets.
    for (auto& pair : GetInstructionValueSet(tuple->operand(i))) {
      const ShapeIndex& operand_index = pair.first;
      HloValueSet& operand_value_set = pair.second;

      ShapeIndex index = {i};
      for (int64_t op_index : operand_index) {
        index.push_back(op_index);
      }
      HloValueSet& value_set = GetMutableValueSet(tuple, index);

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
  }
  return GetInstructionValueSet(xla_while).AssignUnionOf(inputs);
}

bool HloDataflowAnalysis::UpdateAllGatherStartValueSet(
    HloInstruction* all_gather_start) {
  CHECK_EQ(all_gather_start->opcode(), HloOpcode::kAllGatherStart);
  bool changed = false;
  // AllGatherStart forwards the operand values to element {0} of its output.
  for (int64_t i = 0; i < all_gather_start->operand_count(); ++i) {
    const HloValueSet& operand_value_set =
        GetValueSet(all_gather_start->operand(i));

    ShapeIndex output_index = {0};
    if (all_gather_start->operand_count() > 1) {
      output_index.push_back(i);
    }

    HloValueSet& value_set = GetMutableValueSet(all_gather_start, output_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateAllGatherDoneValueSet(
    HloInstruction* all_gather_done) {
  CHECK_EQ(all_gather_done->opcode(), HloOpcode::kAllGatherDone);
  bool changed = false;
  // AllGatherDone forwards the operand value at {1} to its output. If the
  // output is a tuple, then that tuple is defined by all-gather-done, so
  // only update the value set for tuple leaf elements (arrays).
  for (auto& pair : GetInstructionValueSet(all_gather_done)) {
    const ShapeIndex& output_index = pair.first;
    HloValueSet& value_set = pair.second;

    if (!ShapeUtil::GetSubshape(all_gather_done->shape(), output_index)
             .IsArray()) {
      continue;
    }
    ShapeIndex operand_index = {1};
    for (int64_t i : output_index) {
      operand_index.push_back(i);
    }

    const HloValueSet& operand_value_set =
        GetValueSet(all_gather_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateAllReduceDoneValueSet(
    HloInstruction* all_reduce_done) {
  CHECK_EQ(all_reduce_done->opcode(), HloOpcode::kAllReduceDone);
  bool changed = false;
  // AllReduceDone forwards its only operand.
  for (auto& pair : GetInstructionValueSet(all_reduce_done)) {
    const ShapeIndex& output_index = pair.first;
    HloValueSet& value_set = pair.second;

    ShapeIndex operand_index = {};
    for (int64_t i : output_index) {
      operand_index.push_back(i);
    }

    const HloValueSet& operand_value_set =
        GetValueSet(all_reduce_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateCollectivePermuteStartValueSet(
    HloInstruction* collective_permute_start) {
  CHECK_EQ(collective_permute_start->opcode(),
           HloOpcode::kCollectivePermuteStart);
  bool changed = false;
  // CollectivePermuteStart forwards the operand value to element {0} of its
  // output.
  if (!Cast<HloCollectivePermuteInstruction>(collective_permute_start)
           ->inplace() &&
      collective_permute_start->operands().size() > 1) {
    for (int oprd_idx = 0;
         oprd_idx < collective_permute_start->operands().size(); ++oprd_idx) {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_start->operand(oprd_idx));
      HloValueSet& value_set =
          GetMutableValueSet(collective_permute_start, {0, oprd_idx});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  } else {
    // TODO support multi-operand in-place collective-permute and unify in-place
    // collective-permute with normal ones
    if (collective_permute_start->operand(0)->shape().IsTuple()) {
      for (int i = 0; i < ShapeUtil::TupleElementCount(
                              collective_permute_start->operand(0)->shape());
           ++i) {
        const HloValueSet& operand_value_set =
            GetValueSet(collective_permute_start->operand(0), {i});
        HloValueSet& value_set =
            GetMutableValueSet(collective_permute_start, {0, i});
        if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      }
    } else {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_start->operand(0));
      HloValueSet& value_set =
          GetMutableValueSet(collective_permute_start, {0});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateCollectivePermuteDoneValueSet(
    HloInstruction* collective_permute_done) {
  CHECK_EQ(collective_permute_done->opcode(),
           HloOpcode::kCollectivePermuteDone);
  bool changed = false;
  // CollectivePermuteDone forwards the operand value at {1} to its output.
  if (collective_permute_done->shape().IsTuple()) {
    for (int i = 0;
         i < ShapeUtil::TupleElementCount(collective_permute_done->shape());
         ++i) {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_done->operand(0), {1, i});
      HloValueSet& value_set = GetMutableValueSet(collective_permute_done, {i});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  } else {
    const HloValueSet& operand_value_set =
        GetValueSet(collective_permute_done->operand(0), {1});
    HloValueSet& value_set = GetMutableValueSet(collective_permute_done);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloDataflowAnalysis::UpdateInstructionValueSet(
    HloInstruction* instruction) {
  // Recompute from operands.
  switch (instruction->opcode()) {
    case HloOpcode::kAddDependency:
      return UpdateAddDependencyValueSet(instruction);
    case HloOpcode::kAllGatherStart:
      return UpdateAllGatherStartValueSet(instruction);
    case HloOpcode::kAllGatherDone:
      return UpdateAllGatherDoneValueSet(instruction);
    case HloOpcode::kAsyncStart:
      return UpdateAsyncStartValueSet(instruction);
    case HloOpcode::kAsyncUpdate:
      return UpdateAsyncUpdateValueSet(instruction);
    case HloOpcode::kAsyncDone:
      return UpdateAsyncDoneValueSet(instruction);
    case HloOpcode::kBitcast:
      return UpdateBitcastValueSet(instruction);
    case HloOpcode::kDomain:
      return UpdateDomainValueSet(instruction);
    case HloOpcode::kCopy:
      return UpdateCopyValueSet(instruction);
    case HloOpcode::kGetTupleElement:
      return UpdateGetTupleElementValueSet(instruction);
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
    case HloOpcode::kAllReduceDone:
      return UpdateAllReduceDoneValueSet(instruction);
    case HloOpcode::kCollectivePermuteStart:
      return UpdateCollectivePermuteStartValueSet(instruction);
    case HloOpcode::kCollectivePermuteDone:
      return UpdateCollectivePermuteDoneValueSet(instruction);
    case HloOpcode::kOptimizationBarrier:
      return UpdateOptimizationBarrierValueSet(instruction);
    default:
      break;
  }

  return false;
}

void HloDataflowAnalysis::Propagate() {
  using Work = std::pair<int64_t, HloInstruction*>;
  // Avoid duplicating work by preferring work items early in the post order
  // schedule. Intuitively, we start from entry parameters and propagate buffers
  // updates throughout the module only once.
  std::priority_queue<Work, std::vector<Work>, std::greater<Work>> worklist;
  absl::flat_hash_set<HloInstruction*> workset;
  auto priority_map = CalculatePostOrderSchedule(module_);
  auto add_to_worklist = [&priority_map, &worklist,
                          &workset](HloInstruction* instruction) {
    if (workset.insert(instruction).second) {
      VLOG(4) << "Add " << instruction->name() << " to worklist with priority "
              << priority_map[instruction];
      worklist.emplace(priority_map[instruction], instruction);
    }
  };

  auto comps = module_.MakeComputationPostOrder();
  for (HloComputation* computation : comps) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      add_to_worklist(instruction);
    }
  }
  VLOG(1) << "SSA_FORM_: " << ssa_form_;

  while (!worklist.empty()) {
    HloInstruction* instruction = worklist.top().second;
    worklist.pop();

    workset.erase(workset.find(instruction));

    VLOG(4) << "Worklist top: " << instruction->name();
    XLA_VLOG_LINES(3, ToString());

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
      } else if (user->opcode() == HloOpcode::kAsyncUpdate ||
                 user->opcode() == HloOpcode::kAsyncDone) {
        if (HloInstruction::IsThreadIncluded(user->async_execution_thread(),
                                             execution_threads_)) {
          // For async update and async done, we cannot distinguish which
          // parameter needs to be updated so add all to the worklist.
          for (int64_t parameter_number = 0;
               parameter_number <
               user->async_wrapped_computation()->num_parameters();
               ++parameter_number) {
            add_to_worklist(
                user->async_wrapped_computation()->parameter_instruction(
                    parameter_number));
          }
        }
      } else {
        for (HloComputation* called_computation : user->called_computations()) {
          if (!HloInstruction::IsThreadIncluded(
                  called_computation->execution_thread(), execution_threads_)) {
            continue;
          }
          const CallGraphNode& call_graph_node =
              call_graph_->GetNode(called_computation);
          if (call_graph_node.context() == CallContext::kControlFlow) {
            for (int64_t operand_number : user->OperandIndices(instruction)) {
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
        if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
          // Add the while itself, and the body and condition parameters.
          add_to_worklist(callsite.instruction());
          add_to_worklist(
              callsite.instruction()->while_body()->parameter_instruction(0));
          add_to_worklist(
              callsite.instruction()->while_condition()->parameter_instruction(
                  0));
        } else if (call_graph_node.context() == CallContext::kControlFlow ||
                   callsite.instruction()->opcode() ==
                       HloOpcode::kConditional) {
          add_to_worklist(callsite.instruction());
        }
      }
    }
  }
}

const InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) const {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}

InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}

absl::Status HloDataflowAnalysis::InitializeInstructionValueSets() {
  for (const HloComputation* computation : module_.MakeComputationPostOrder()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      // Create an empty shape tree.
      value_sets_.insert({instruction, std::make_unique<InstructionValueSet>(
                                           &instruction->shape())});

      // For each sub-shape of the instruction shape, add a new HloValue to its
      // HloValueSet. should_define may be provided to define a subset of
      // values.
      auto define_all_values =
          [this, &instruction](
              absl::FunctionRef<bool(const ShapeIndex&)> should_define =
                  [](const ShapeIndex&) { return true; }) {
            for (auto& pair : GetInstructionValueSet(instruction)) {
              const ShapeIndex& index = pair.first;
              if (should_define(index)) {
                HloValue* value =
                    NewHloValue(instruction, index, /*is_phi=*/false);
                GetMutableValueSet(instruction, index).AddValue(value);
              }
            }
          };

      // Add a new HloValue to the HloValueSet corresponding to the given index
      // of the instruction shape.
      auto define_value_at = [this, &instruction](const ShapeIndex& index) {
        HloValue* value = NewHloValue(instruction, index, /*is_phi=*/false);
        GetMutableValueSet(instruction, index).AddValue(value);
      };

      switch (instruction->opcode()) {
        case HloOpcode::kBitcast:
          if (bitcast_defines_value_) {
            define_all_values();
          }
          break;
        case HloOpcode::kAddDependency:
        case HloOpcode::kWhile:
        case HloOpcode::kCall:
        case HloOpcode::kConditional:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kDomain:
        case HloOpcode::kOptimizationBarrier:
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
              call_graph_node.context() == CallContext::kEmbedded) {
            // Parameters of computations called in a parallel context (eg, map
            // and reduce) as well as parameters of dead computations define all
            // values in their output. Otherwise the values of the parameter
            // come from the caller (eg, operands to the kCall instruction).
            define_all_values();
          }
          break;
        case HloOpcode::kCopy:
        case HloOpcode::kTuple:
          // These instructions only define their top-level values. Any other
          // values flow from their operands.
          define_value_at(/*index=*/{});
          break;
        case HloOpcode::kAsyncStart: {
          // AsyncStart produces a tuple of {{aliased operands}, {destination},
          // contexts}. It defines all of the tuple-shaped values and the
          // contexts.
          // If the thread is excluded, then we don't track the contained
          // dataflow, and define the destination values too.
          bool thread_included = HloInstruction::IsThreadIncluded(
              instruction->async_execution_thread(), execution_threads_);
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                       .IsTuple() ||
                   (!thread_included && index.front() == 1) ||
                   (index.front() > 1);
          });
          break;
        }
        case HloOpcode::kAsyncUpdate:
          // AsyncUpdate produces a tuple of {{aliased operands}, {destination},
          // contexts} where all of the array-typed values alias with the
          // operand. So, only tuple-shaped values are defined by AsyncUpdate.
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                .IsTuple();
          });
          break;
        case HloOpcode::kAsyncDone:
          // AsyncDone's output aliases its output. It defines all remaining
          // tuple-shaped values.
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                .IsTuple();
          });
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
        case HloOpcode::kAllGatherStart:
          // AllGatherStart produces a tuple of
          // {aliased operands, destination buffers}. If there is more than
          // one operand, then both aliased operands and destination buffers
          // will be tuples themselves. all-gather-start will define all tuples
          // and all tuple leaves (arrays) in tuple sub-index 1 (destination
          // buffers).
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                       .IsTuple() ||
                   index.front() == 1;
          });
          break;
        case HloOpcode::kAllGatherDone:
          // AllGatherDone's output aliases its input tuple element {1}.
          if (instruction->shape().IsTuple()) {
            define_value_at(/*index=*/{});
          }
          break;
        case HloOpcode::kAllReduceDone:
          // AllReduceDone's output aliases its input.
          break;
        case HloOpcode::kCollectivePermuteStart:
          // CollectivePermuteStart produces a tuple of {{aliased operand(s)},
          // {destination buffer(s)}, contexts}, where the context data are
          // optional.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          for (int i = 2; i < instruction->shape().tuple_shapes().size(); ++i) {
            define_value_at(/*index=*/{i});
          }

          if (Cast<HloCollectivePermuteInstruction>(instruction)->inplace()) {
            CHECK_EQ(instruction->operand_count(), 4);
            if (instruction->operand(1)->shape().IsTuple()) {
              for (int i = 0; i < ShapeUtil::TupleElementCount(
                                      instruction->operand(1)->shape());
                   ++i) {
                define_value_at(/*index=*/{1, i});
              }
            }
          } else if (instruction->operand_count() > 1) {
            for (int i = 0; i < instruction->operand_count(); ++i) {
              define_value_at(/*index=*/{1, i});
            }
          }
          break;
        case HloOpcode::kCollectivePermuteDone:
          // CollectivePermuteDone's output aliases its input tuple element {1}.
          if (instruction->shape().IsTuple()) {
            define_value_at(/*index=*/{});
          }
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

  return absl::OkStatus();
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
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (HloInstruction* instruction : computation->instructions()) {
      InstructionValueSet& instruction_value_set =
          GetInstructionValueSet(instruction);
      VLOG(1) << "inst: " << instruction->name();
      VLOG(1) << instruction_value_set.ToString();
      instruction_value_set.ForEachMutableElement(
          [&](const xla::ShapeIndex& index, HloValueSet* value_set) {
            auto values = value_set->values();
            bool changed = false;
            std::vector<const HloValue*> new_values;
            for (const HloValue* value : values) {
              if (value->is_phi()) {
                HloValue::Id phi_id = value->id();
                HloValue::Id new_id = phi_graph_.FindOptimizedValue(phi_id);
                if (new_id != phi_id) {
                  VLOG(1) << "Replacing " << value->ToShortString() << " with "
                          << GetValue(new_id).ToShortString();
                  const HloValue& new_value = GetValue(new_id);
                  new_values.push_back(&new_value);
                  MarkValueForDeletion(phi_id);
                  changed = true;
                  continue;
                }
              }
              new_values.push_back(value);
            }
            if (changed) {
              value_set->Clear();
              for (const HloValue* new_value : new_values) {
                value_set->AddValue(new_value);
              }
            }
          });
    }
  }
}

/* static */
absl::StatusOr<std::unique_ptr<HloDataflowAnalysis>> HloDataflowAnalysis::Run(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value,
    absl::flat_hash_set<absl::string_view> execution_threads) {
  VLOG(1) << "HloDataflowAnalysis::Run on module " << module.name();
  XLA_VLOG_LINES(2, module.ToString());

  auto dataflow_analysis = absl::WrapUnique(new HloDataflowAnalysis(
      module, ssa_form, bitcast_defines_value, execution_threads));
  TF_RETURN_IF_ERROR(dataflow_analysis->RunImpl());
  return dataflow_analysis;
}

absl::Status HloDataflowAnalysis::RunImpl() {
  TF_RETURN_IF_ERROR(InitializeInstructionValueSets());
  Propagate();
  OptimizePhiValues();

  // Delete all values marked for deletion.
  DeleteMarkedValues();

  // Gather and set all non-definition positions of all values. Value deletion
  // is rare, so just use a vector indexed by Value::Id rather than a map from
  // Value::Id to positions. There should be very few holes in the vector, and
  // lookup is faster.
  std::vector<std::vector<HloPosition>> value_positions(next_value_id_);
  for (const HloComputation* computation : module_.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (HloInstruction* instruction : computation->instructions()) {
      for (const auto& pair : GetInstructionValueSet(instruction)) {
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
  for (auto& pair : values_) {
    HloValue::Id value_id = pair.first;
    HloValue& value = *pair.second;
    value.SetPositions(value_positions[value_id]);
  }

  // Construct vector of values.
  values_vector_.reserve(values_.size());
  for (const auto& pair : values_) {
    values_vector_.push_back(pair.second.get());
  }
  absl::c_sort(values_vector_, HloValue::IdLessThan);

  DCHECK_OK(Verify());

  XLA_VLOG_LINES(1, ToString());

  return absl::OkStatus();
}

absl::Status HloDataflowAnalysis::Verify() const {
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
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (const auto& instruction : computation->instructions()) {
      // TODO(b/361618355): Teach HloDataflowAnalysis how to handle input/output
      // aliasing for async calls.
      if (instruction->opcode() == HloOpcode::kAsyncStart &&
          (instruction->async_wrapped_opcode() == HloOpcode::kCall ||
           instruction->async_wrapped_opcode() == HloOpcode::kCustomCall)) {
        continue;
      }
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

  return absl::OkStatus();
}

bool HloDataflowAnalysis::DoesNotUseOperandBuffer(
    const HloInstruction* operand, const ShapeIndex& index,
    const HloInstruction* user) const {
  // Return false if no value at 'operand' and 'index' is used at 'user'.
  for (const HloValue* value : GetValueSet(operand, index).values()) {
    for (const HloUse& use : value->GetUses()) {
      if (use.instruction == user) {
        if (user->IsLoopFusion()) {
          HloInstruction* fusion_param =
              user->fused_parameter(use.operand_number);
          const HloValue& value =
              GetValueDefinedAt(fusion_param, use.operand_index);
          return value.GetUses().empty();
        }
        return false;
      }
    }
  }
  return true;
}

/*static*/ bool HloDataflowAnalysis::IsAsynchronousOperationStart(
    HloOpcode opcode) {
  return opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
         opcode == HloOpcode::kCopyStart ||
         opcode == HloOpcode::kAllReduceStart ||
         opcode == HloOpcode::kAllGatherStart ||
         opcode == HloOpcode::kCollectivePermuteStart ||
         opcode == HloOpcode::kAsyncStart;
}

/*static*/ bool HloDataflowAnalysis::IsAsynchronousOperationDone(
    HloOpcode opcode) {
  return opcode == HloOpcode::kSendDone || opcode == HloOpcode::kRecvDone ||
         opcode == HloOpcode::kCopyDone ||
         opcode == HloOpcode::kAllReduceDone ||
         opcode == HloOpcode::kAllGatherDone ||
         opcode == HloOpcode::kCollectivePermuteDone ||
         opcode == HloOpcode::kAsyncDone;
}

/*static*/ std::vector<std::pair<HloOperandIndex, ShapeIndex>>
HloDataflowAnalysis::GetInPlaceInputOutputPairs(
    const HloInstruction* instruction) {
  // TODO(b/424109294): For now, we can use the default AliasInfo here, as there
  // are no backend specific MustAlias rules yet. But we will need to migrate
  // users of this method when we want to support backend specific MustAlias
  // rules.
  static AliasInfo* alias_info = new AliasInfo();
  return alias_info->GetInPlaceInputOutputPairs(instruction);
}

bool HloDataflowAnalysis::CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index,
    const AliasInfo* alias_info) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();

  if (user->opcode() == HloOpcode::kCustomCall) {
    if (user->custom_call_target() == "__cublas$lt$matmul" ||
        user->custom_call_target() == "__cublas$lt$matmul$f8") {
      if (user_index == ShapeIndex({1})) {
        VLOG(2) << "Prohibiting buffer sharing for cuBLASLt workspace: "
                << user->name();
        return false;
      }
    }
  }

  if (operand->opcode() == HloOpcode::kConstant) {
    VLOG(2) << "Prohibiting buffer sharing for constant operand: "
            << operand->name();
    return false;
  }

  const Shape& operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);

  // During tiling assignment, we can add no-op instructions which appear to
  // change tiling (and memory space) of the operand, but don't.
  if (hlo_query::IsChangeTilingCopyFusion(user) ||
      hlo_query::IsChangeTilingCopyFusion(operand)) {
    VLOG(2) << "Allowing buffer sharing for ChangeTilingCopyFusion: "
            << user->name() << " or " << operand->name();
    return true;
  }
  const bool shapes_equal = ShapeUtil::Equal(operand_subshape, user_subshape);
  // Check that operand and user emit the same shape and layout.
  if (shapes_equal) {
    // Must-alias relationship returns true for in-place operations (DUS and DUS
    // fusions), regardless of the backend.
    // Cache uses of value and alias_info->GetInPlaceInputOutputPairs to speed
    // up repeated calls.
    auto [operand_it, operand_inserted] =
        cache_share_buffer_with_operand_.try_emplace(
            std::make_pair(operand, operand_index),
            absl::flat_hash_set<HloUse>());
    if (operand_inserted) {
      auto uses = GetUniqueValueAt(operand, operand_index).GetUses();
      operand_it->second.insert(uses.begin(), uses.end());
    }
    auto [user_it, user_inserted] = cache_share_buffer_with_user_.try_emplace(
        user, absl::flat_hash_map<ShapeIndex, std::vector<HloOperandIndex>>());
    if (user_inserted) {
      auto& pairs = user_it->second;
      for (const auto& operand_and_output_index :
           alias_info->GetInPlaceInputOutputPairs(user)) {
        pairs[operand_and_output_index.second].push_back(
            operand_and_output_index.first);
      }
    }
    auto& uses = operand_it->second;
    auto& user_pairs = user_it->second;
    auto pairs_it = user_pairs.find(user_index);
    if (pairs_it != user_pairs.end()) {
      for (const auto& operand_index : pairs_it->second) {
        if (uses.contains(HloUse{user, operand_index.operand_number,
                                 operand_index.operand_index})) {
          VLOG(2) << "Allowing buffer sharing for must-alias: " << user->name();
          return true;
        }
      }
    }
  }

  if (std::optional<bool> hint =
          alias_info->MayAlias(operand, operand_index, user, user_index)) {
    VLOG(2) << "Buffer sharing for alias info hint: " << *hint;
    return *hint;
  }

  if (!shapes_equal) {
    VLOG(2) << "Prohibiting buffer sharing for different shapes/layouts: "
            << operand_subshape.ToString() << " vs "
            << user_subshape.ToString();
    return false;
  }

  if (user->opcode() == HloOpcode::kFusion) {
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));
    const HloValue& fusion_param_value =
        GetValueDefinedAt(fusion_param, operand_index);

    if (user->IsLoopFusion() || user->IsInputFusion()) {
      bool result = AreTransitiveUsesElementwiseOrTuple(fusion_param);
      VLOG(2) << "Buffer sharing for loop/input fusion: " << result;
      return result;
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
        VLOG(2) << "Prohibiting buffer sharing for output fusion kAdd root "
                   "without dot/convolution operand";
        return false;
      }
      auto* matched_add_operand = *add_operand_it;
      // Calculate operand index of 'add' operand which was not matched above.
      const int64_t other_add_operand_index =
          matched_add_operand == add->operand(0) ? 1 : 0;
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root (at operand
      // index 'other_add_operand_index').
      if (fusion_param_value.GetUses().size() == 1) {
        const HloUse& use = fusion_param_value.GetUses()[0];
        bool result = use.instruction == user->fused_expression_root() &&
                      use.operand_number == other_add_operand_index;
        VLOG(2) << "Buffer sharing for output fusion kAdd root: " << result;
        return result;
      }
      VLOG(2) << "Prohibiting buffer sharing for output fusion kAdd root: "
                 "fusion param has multiple uses or other mismatch";
      return false;
    }
  }

  // There is nothing inherently wrong with while and conditional ops to have
  // input/output buffers to alias with each other, even when the indices are
  // different in the while case. It is a problem when this aliasing causes HLO
  // ops inside these while or conditional to have input/output buffer aliasing
  // that isn't allowed. So allow while and conditional to share buffers with
  // operands and we will discover any problematic sharing when we explore the
  // ops inside these computations.
  if (user->opcode() == HloOpcode::kWhile ||
      user->opcode() == HloOpcode::kConditional) {
    VLOG(2) << "Allowing buffer sharing for while/conditional: "
            << user->name();
    return true;
  }

  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kScatter ||
      user->opcode() == HloOpcode::kTriangularSolve ||
      user->opcode() == HloOpcode::kSetDimensionSize) {
    // We eliminated other users in HloOrdering::LiveRangeStrictlyBefore
    // so here we just need to check that the use is at the right operand index.
    const auto operand_indices = user->OperandIndices(operand);
    int64_t operand_no = user->opcode() == HloOpcode::kTriangularSolve ? 1 : 0;
    bool result =
        operand_indices.size() == 1 && operand_indices[0] == operand_no;
    VLOG(2) << "Buffer sharing for " << HloOpcodeString(user->opcode()) << ": "
            << result;
    return result;
  }
  if (user->opcode() == HloOpcode::kSort) {
    // Only valid if there are no other users.
    if (operand->users().size() != 1) {
      VLOG(2) << "Prohibiting buffer sharing for sort: operand has multiple "
                 "users";
      return false;
    }
    // If we only sort keys, the output of sort is not a tuple, so we can always
    // share the buffer.
    if (user->operand_count() == 1) {
      VLOG(2) << "Allowing buffer sharing for sort: single operand";
      return true;
    }
    CHECK(!user_index.empty());
    // Only share with the right tuple element buffer.
    const auto operand_indices = user->OperandIndices(operand);
    bool result =
        operand_indices.size() == 1 && user_index[0] == operand_indices[0];
    VLOG(2) << "Buffer sharing for sort: " << result;
    return result;
  }
  if (user->opcode() == HloOpcode::kCall) {
    // Get all uses of value defined by 'operand' at 'operand_index'.
    auto uses = GetValueDefinedAt(operand, operand_index).GetUses();
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
    bool result =
        uses.size() == 2 && found_caller_use && found_elementwise_callee_use;
    VLOG(2) << "Buffer sharing for call: " << result;
    return result;
  }

  // Loop fusions that contain transposing copies won't reach here as they have
  // different layouts, which fails the check in the beginning of this function.
  bool result = user->IsElementwiseOnOperand(user->operand_index(operand));
  VLOG(2) << "Buffer sharing for elementwise on operand: " << result;
  return result;
}

}  // namespace xla
