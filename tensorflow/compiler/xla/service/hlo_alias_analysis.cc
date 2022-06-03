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

#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::StrAppend;

namespace {

using FlatValueSet = absl::flat_hash_set<const HloValue*>;

void ComputeInputOutputAliasedValues(const HloValue& value,
                                     const HloDataflowAnalysis& dataflow,
                                     FlatValueSet& aliased_values) {
  const HloModule& module = dataflow.module();
  const HloComputation& entry_computation = *module.entry_computation();
  const HloInputOutputAliasConfig& io_alias_config =
      module.input_output_alias_config();

  // If the value shows up in a root instruction, alias it with parameter
  // instruction.
  for (const HloPosition& pos : value.positions()) {
    if (pos.instruction == entry_computation.root_instruction()) {
      absl::optional<HloInputOutputAliasConfig::Alias> aliased_input =
          io_alias_config.GetAliasedParameter(pos.index);
      if (aliased_input) {
        aliased_values.insert(
            &dataflow.GetUniqueValueAt(entry_computation.parameter_instruction(
                                           aliased_input->parameter_number),
                                       aliased_input->parameter_index));
      }
    }
  }
}

void ComputeWhileAliasedValues(const HloValue& value,
                               const HloDataflowAnalysis& dataflow,
                               FlatValueSet& aliased_values) {
  VLOG(3) << "Compute kWhile aliases";
  // Value is init of a while (use is while).
  for (const HloUse& use : value.GetUses()) {
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      // Determine the while value that this shares a buffer with.
      const HloValue& while_value =
          dataflow.GetUniqueValueAt(use.instruction, use.operand_index);
      aliased_values.insert(&while_value);
      VLOG(3) << "  value is init value to a while; must share buffer with "
                 "while value "
              << while_value;
    }
  }
  // Value is a parameter of a while body/condition.
  if (value.defining_instruction()->opcode() == HloOpcode::kParameter) {
    const HloComputation* computation = value.defining_instruction()->parent();
    const CallGraphNode& call_graph_node =
        dataflow.call_graph().GetNode(computation);
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
        // Call graph must have been flattened.
        CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

        const HloValue& while_value = dataflow.GetUniqueValueAt(
            callsite.instruction(), value.defining_index());
        VLOG(3) << "  value is parameter value of the body or condition of a "
                   "while; must share buffer with while value "
                << while_value;
        aliased_values.insert(&while_value);
      }
    }
  }
  // Value is the root of a while body.
  for (const HloPosition& position : value.positions()) {
    if (!position.instruction->IsRoot()) continue;

    const HloComputation* computation = position.instruction->parent();
    const CallGraphNode& call_graph_node =
        dataflow.call_graph().GetNode(computation);

    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kWhile &&
          callsite.instruction()->while_body() == computation) {
        // Call graph must have been flattened.
        CHECK_EQ(call_graph_node.caller_callsites().size(), 1)
            << "Call graph must have been flattened.";

        const HloValue& while_value =
            dataflow.GetUniqueValueAt(callsite.instruction(), position.index);
        VLOG(3) << "  value @ " << position << " is root of "
                << callsite.instruction()->name()
                << "; body root and while value root must share buffer "
                   "among them: "
                << while_value;
        aliased_values.insert(&while_value);
      }
    }
  }
}

void ComputeConditionalAliasedValues(const HloValue& value,
                                     const HloDataflowAnalysis& dataflow,
                                     FlatValueSet& aliased_values) {
  VLOG(3) << "Compute kConditional aliases";
  // Aliases the buffers of the true/false computations roots, with the one of
  // the conditional.
  for (const HloPosition& position : value.positions()) {
    if (!position.instruction->IsRoot()) continue;

    const HloComputation* computation = position.instruction->parent();
    const CallGraphNode& call_graph_node =
        dataflow.call_graph().GetNode(computation);
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kConditional) {
        // Call graph must have been flattened.
        CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

        const HloValue& cond_value =
            dataflow.GetUniqueValueAt(callsite.instruction(), position.index);
        VLOG(3) << "  value @ " << position << " is root of "
                << callsite.instruction()->name()
                << "; branch computation roots must share buffer among them : "
                << cond_value;
        aliased_values.insert(&cond_value);
      }
    }
  }
}

void ComputeInPlaceOperationAliasedValues(const HloValue& value,
                                          const HloDataflowAnalysis& dataflow,
                                          FlatValueSet& aliased_values) {
  VLOG(3) << "Compute aliases for in-place operations (e.g. "
             "kDynamicUpdateSlice and kScatter)";
  for (const HloPosition& position : value.positions()) {
    HloInstruction* instruction = position.instruction;
    for (const auto& operand_and_output_index :
         HloDataflowAnalysis::GetInPlaceInputOutputPairs(instruction)) {
      if (position.index == operand_and_output_index.second) {
        const HloOperandIndex& operand_index = operand_and_output_index.first;
        const HloValue& operand_value = dataflow.GetUniqueValueAt(
            instruction->operand(operand_index.operand_number),
            operand_index.operand_index);
        VLOG(3) << " operand value " << operand_value << " aliases.";
        aliased_values.insert(&operand_value);
      }
    }
  }

  for (const HloUse& use : value.GetUses()) {
    for (const auto& operand_and_output_index :
         HloDataflowAnalysis::GetInPlaceInputOutputPairs(use.instruction)) {
      const HloOperandIndex& operand_index = operand_and_output_index.first;
      if (use.operand_number == operand_index.operand_number &&
          use.operand_index == operand_index.operand_index) {
        const HloValue& use_value = dataflow.GetUniqueValueAt(
            use.instruction, operand_and_output_index.second);
        VLOG(3) << " use value " << use_value << " aliases.";
        aliased_values.insert(&use_value);
      }
    }
  }
}

// Compute and return a set of values that the given value must be aliased
// with due to HLO aliasing rules (including the value itself).
FlatValueSet ComputeAliasedValues(const HloValue& value,
                                  const HloDataflowAnalysis& dataflow) {
  if (VLOG_IS_ON(2)) {
    for (const HloUse& use : value.GetUses()) {
      VLOG(2) << "Use of value " << value << ": " << use;
    }
  }

  FlatValueSet aliased_values{&value};
  ComputeInputOutputAliasedValues(value, dataflow, aliased_values);
  ComputeWhileAliasedValues(value, dataflow, aliased_values);
  ComputeConditionalAliasedValues(value, dataflow, aliased_values);
  ComputeInPlaceOperationAliasedValues(value, dataflow, aliased_values);
  return aliased_values;
}

std::vector<HloBuffer> CreateBuffers(const HloDataflowAnalysis& dataflow) {
  const std::vector<HloValue*>& values = dataflow.values();
  size_t num_buffers = values.size();
  // The sets of values contained in each buffer.
  std::vector<FlatValueSet> buffer_values(values.size());
  // Maps values to the set of values with which they are aliased.
  absl::flat_hash_map<const HloValue*, FlatValueSet*> value_to_set;
  value_to_set.reserve(values.size());

  for (size_t i = 0; i < values.size(); ++i) {
    buffer_values[i].insert(values[i]);
    value_to_set[values[i]] = &buffer_values[i];
  }

  // Merge together sets of HloValues which must be in the same HloBuffer
  // because of aliasing rules (e.g. in-place kWhile instruction).
  for (const HloValue* value : values) {
    VLOG(3) << "Merging colocated values, value: " << *value;

    FlatValueSet aliased_values = ComputeAliasedValues(*value, dataflow);
    if (aliased_values.size() < 2) continue;  // Fast path.

    // The sets of values that are transitively aliased together.
    std::vector<std::pair<FlatValueSet*, HloValue::Id>> aliased_sets;
    aliased_sets.reserve(aliased_values.size());
    for (const HloValue* aliased : aliased_values) {
      aliased_sets.push_back({value_to_set[aliased], aliased->id()});
    }

    // Use the largest set to collect the union of the aliased sets (as it is
    // more efficient to merge smaller sets into larger). Break ties using
    // value ID to maintain determinism.
    auto key = [](const auto& set_and_id) {
      return std::make_pair(set_and_id.first->size(), -set_and_id.second);
    };
    FlatValueSet* union_set =
        absl::c_max_element(aliased_sets, LessThanByKey(key))->first;

    for (auto& aliased_set_and_id : aliased_sets) {
      FlatValueSet* aliased_set = aliased_set_and_id.first;
      if ((aliased_set != union_set) && !aliased_set->empty()) {
        for (const HloValue* aliased_value : *aliased_set) {
          CHECK(union_set->insert(aliased_value).second);
          value_to_set[aliased_value] = union_set;
        }
        aliased_set->clear();
        --num_buffers;
      }
    }
  }

  // Create a vector of HloBuffers, one for each non-empty set of values.
  std::vector<HloBuffer> buffers;
  buffers.reserve(num_buffers);

  for (const FlatValueSet& value_set : buffer_values) {
    if (!value_set.empty()) {
      HloBuffer::Id id = buffers.size();
      buffers.push_back({id, HloValueSet(value_set).TakeValues()});
    }
  }

  CHECK_EQ(buffers.size(), num_buffers);
  return buffers;
}

}  // namespace

HloAliasAnalysis::HloAliasAnalysis(const HloModule* module) : module_(module) {}

const HloBuffer& HloAliasAnalysis::GetUniqueBufferAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::vector<const HloBuffer*> buffers = ComputeBuffersAt(instruction, index);
  CHECK_EQ(buffers.size(), 1);
  return *buffers[0];
}

HloBuffer& HloAliasAnalysis::GetUniqueBufferAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return GetBuffer(const_cast<const HloAliasAnalysis*>(this)
                       ->GetUniqueBufferAt(instruction, index)
                       .id());
}

std::vector<const HloBuffer*> HloAliasAnalysis::ComputeBuffersAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const HloValueSet& value_set =
      dataflow_analysis_->GetValueSet(instruction, index);
  std::vector<const HloBuffer*> buffers;
  buffers.reserve(value_set.values().size());
  for (const HloValue* value : value_set.values()) {
    buffers.push_back(&GetBufferContainingValue(*value));
  }

  // Sort and uniquify vector before returning.
  absl::c_sort(buffers, HloBuffer::IdLessThan);
  buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());

  return buffers;
}

Status HloAliasAnalysis::Verify() const {
  // Verify consistency between the value_to_buffer_ map and
  // HloBuffer::values().
  for (const auto& pair : value_to_buffer_) {
    const HloValue* value = pair.first;
    const HloBuffer& buffer = *pair.second;
    TF_RET_CHECK(absl::c_linear_search(buffer.values(), value));
  }

  for (HloBuffer::Id id = 0; id < buffers_.size(); ++id) {
    const HloBuffer& buffer = buffers_[id];
    TF_RET_CHECK(buffer.id() == id);

    HloValue::Id last_value_id = -1;
    for (const HloValue* value : buffer.values()) {
      TF_RET_CHECK(GetBufferContainingValue(*value) == buffer);

      // Also verify the values in HloBuffer are unique and sorted by id.
      TF_RET_CHECK(value->id() > last_value_id);
      last_value_id = value->id();
    }
  }

  return OkStatus();
}

std::string HloAliasAnalysis::ToString() const {
  std::string out =
      absl::StrCat("HloAliasAnalysis, module ", module_->name(), "\n");
  StrAppend(&out, "  Buffers at each position:\n");
  for (const HloComputation* computation : module_->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      StrAppend(&out, "    ", instruction->name(), ":\n");
      if (instruction->shape().IsTuple()) {
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&out, &instruction, this](const Shape&, const ShapeIndex& index) {
              StrAppend(&out, "      tuple index ", index.ToString(), ":\n");
              for (const HloBuffer* buffer :
                   ComputeBuffersAt(instruction, index)) {
                StrAppend(&out, "        ", buffer->ToString(), "\n");
              }
            });
      } else {
        for (const HloBuffer* buffer :
             ComputeBuffersAt(instruction, /*index=*/{})) {
          StrAppend(&out, "      ", buffer->ToString(), "\n");
        }
      }
    }
  }

  StrAppend(&out, "  Buffers:\n");
  for (const HloBuffer& buffer : buffers()) {
    StrAppend(&out, "    ", buffer.ToString(), "\n");
    StrAppend(&out, "      positions:\n");
    for (const HloPosition& position : buffer.ComputePositions()) {
      StrAppend(&out, "        ", position.ToString(), "\n");
    }
  }

  return out;
}

/* static */
StatusOr<std::unique_ptr<HloAliasAnalysis>> HloAliasAnalysis::Run(
    const HloModule* module,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer) {
  VLOG(2) << "HloAliasAnalysis::Run on module " << module->name();
  XLA_VLOG_LINES(2, module->ToString());

  auto alias_analysis = absl::WrapUnique(new HloAliasAnalysis(module));
  TF_ASSIGN_OR_RETURN(alias_analysis->dataflow_analysis_,
                      HloDataflowAnalysis::Run(*module, /*ssa_form=*/true,
                                               /*bitcast_defines_value=*/false,
                                               can_share_buffer));

  size_t num_values = alias_analysis->dataflow_analysis_->values().size();
  alias_analysis->buffers_ = CreateBuffers(alias_analysis->dataflow_analysis());
  alias_analysis->value_to_buffer_.reserve(num_values);

  for (HloBuffer& buffer : alias_analysis->buffers_) {
    for (const HloValue* value : buffer.values()) {
      alias_analysis->value_to_buffer_[value] = &buffer;
    }
  }

  CHECK_EQ(alias_analysis->value_to_buffer_.size(), num_values);
  TF_DCHECK_OK(alias_analysis->Verify());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ShapeUtil::ForEachSubshape(root->shape(), [&](const Shape& /*subshape*/,
                                                const ShapeIndex& index) {
    std::vector<const HloBuffer*> buffers =
        alias_analysis->ComputeBuffersAt(root, index);
    alias_analysis->live_out_buffers_.insert(buffers.begin(), buffers.end());
  });

  XLA_VLOG_LINES(2, alias_analysis->ToString());
  return std::move(alias_analysis);
}

}  // namespace xla
