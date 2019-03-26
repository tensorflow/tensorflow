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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::StrAppend;

// Data structure used to construct the alias analysis. Thrown away after alias
// analysis is complete. This data structure keeps track of which sets of
// HloValues must be in the same HloBuffer. This is maintained as a map from a
// buffer identifier (BufferNumber) to set of HLoValues.
//
// Initially each value is its own buffer. In MergeAliasedBuffers, sets of
// values which must share the same buffer are merged together. The end result
// is a partitioning of all HloValues into sets where each set needs its own
// HloBuffer. By performing this analysis without constructing HloBuffers on the
// fly, we can after-the-fact construct a vector of contiguously numbered
// HloBuffers after the buffer requirement has been determined.
class BufferValueMap {
 public:
  // A unique identifier for a set of colocated values which must share the same
  // buffer. This is not necessarily the same as the HloBuffer::Id which will
  // ultimately contain the values. The reason is that HloBuffer::Id's are
  // contiguous, while BufferNumbers may not be. BufferNumbers may not be
  // dense because buffers may be created and destroyed during the analysis
  // construction process.
  using BufferNumber = int64;

  explicit BufferValueMap(HloModule* module,
                          const HloDataflowAnalysis& dataflow)
      : module_(module), dataflow_(dataflow) {
    buffers_.reserve(dataflow_.values().size());
    value_to_buffer_number_.reserve(dataflow_.values().size());
    for (const HloValue* value : dataflow_.values()) {
      BufferNumber buffer_number = next_buffer_number_++;
      buffers_[buffer_number].insert(value);
      value_to_buffer_number_[value] = buffer_number;
    }
  }

  // Merge together sets of HloValues which must be in the same HloBuffer
  // because of aliasing rules (eg, in-place kWhile instruction).
  void MergeAliasedBuffers() {
    for (const HloValue* value : dataflow_.values()) {
      VLOG(3) << "Merging colocated values, value: " << value->ToShortString();

      // Gather the set of buffers with aliasing rules (eg, kWhile) which this
      // value must be contained in.
      std::vector<BufferNumber> aliased_buffers = ComputeAliasedBuffers(*value);

      BufferNumber current_buffer = value_to_buffer_number_.at(value);
      if (aliased_buffers.empty()) {
        // The buffer containing 'value' aliases no other buffers. If the buffer
        // containing 'value' already only contains 'value', then no change is
        // necessary. If the buffer containing 'value' does contain other
        // values, then remove 'value' from the buffer and create a new buffer
        // containing only 'value'
        if (buffers_.at(current_buffer).size() == 1) {
          CHECK_EQ(*buffers_.at(current_buffer).begin(), value);
        } else {
          MoveValueToNewBuffer(*value);
        }
      } else {
        // If multiple buffers are aliased merge these buffers together into a
        // single buffer (arbitrarily chosen as the first buffer in the vector).
        if (aliased_buffers.size() > 1) {
          for (int64 i = 1; i < aliased_buffers.size(); ++i) {
            MergeBuffers(/*from=*/aliased_buffers[i],
                         /*to=*/aliased_buffers[0]);
          }
        }
        BufferNumber new_buffer = aliased_buffers[0];
        if (current_buffer != new_buffer) {
          MoveValueToBuffer(*value, new_buffer);
        }
      }
    }
  }

  // Compute and return a sorted vector of all BufferNumbers. Can be used to
  // iterate through all buffers stabily.
  std::vector<BufferNumber> ComputeSortedBufferNumbers() const {
    std::vector<BufferNumber> buffer_numbers;
    for (const auto& pair : buffers_) {
      buffer_numbers.push_back(pair.first);
    }
    absl::c_sort(buffer_numbers);
    return buffer_numbers;
  }

  // Return a set of all the values in the given buffer.
  const absl::flat_hash_set<const HloValue*>& GetValuesInBuffer(
      BufferNumber buffer_number) const {
    return buffers_.at(buffer_number);
  }

 private:
  // Create a new buffer.
  void NewBuffer(const HloValue& value) {
    BufferNumber buffer_number = next_buffer_number_++;
    buffers_[buffer_number].insert(&value);
    value_to_buffer_number_[&value] = buffer_number;
  }

  // Move the given value into a new buffer containing only the value.
  void MoveValueToNewBuffer(const HloValue& value) {
    BufferNumber new_buffer_number = next_buffer_number_++;
    buffers_[new_buffer_number];
    MoveValueToBuffer(value, new_buffer_number);
  }

  // Move the given value into the given buffer.
  void MoveValueToBuffer(const HloValue& value, BufferNumber buffer_number) {
    BufferNumber old_buffer_number = value_to_buffer_number_.at(&value);
    absl::flat_hash_set<const HloValue*>& old_value_set =
        buffers_.at(old_buffer_number);
    old_value_set.erase(&value);
    if (old_value_set.empty()) {
      buffers_.erase(old_buffer_number);
    }

    buffers_.at(buffer_number).insert(&value);
    value_to_buffer_number_.at(&value) = buffer_number;
  }

  // Merge the buffer 'from' into the buffer 'to'.
  void MergeBuffers(BufferNumber from, BufferNumber to) {
    auto& from_value_set = buffers_.at(from);
    buffers_.at(to).insert(from_value_set.begin(), from_value_set.end());
    // NOTE: using a union-find algorithm to hold the colocated values might be
    // faster.
    for (const HloValue* value : from_value_set) {
      value_to_buffer_number_.at(value) = to;
    }
    buffers_.erase(from);
  }

  BufferNumber GetBufferForValue(const HloValue& value) {
    return value_to_buffer_number_.at(&value);
  }

  void ComputeInputOutputAliasedBuffers(
      const HloValue& value, std::vector<BufferNumber>* aliased_buffers) {
    // Get parameter value from an aliased_input object.
    const auto get_parameter_value =
        [this](const HloInputOutputAliasConfig::Alias& aliased_input)
        -> const HloValue& {
      return dataflow_.GetUniqueValueAt(
          module_->entry_computation()->parameter_instruction(
              aliased_input.parameter_number),
          aliased_input.parameter_index);
    };

    // If the value shows up in a root instruction, alias it with parameter
    // intruction.
    for (const HloPosition& pos : value.positions()) {
      if (pos.instruction == module_->entry_computation()->root_instruction()) {
        ShapeIndex output_index = pos.index;

        auto aliased_input =
            module_->input_output_alias_config().GetAliasedParameter(
                output_index);
        if (aliased_input) {
          aliased_buffers->push_back(
              GetBufferForValue(get_parameter_value(*aliased_input)));
        }
      }
    }

    // If the value is parameter instruction itself, alias it with itself.
    if (value.instruction()->opcode() == HloOpcode::kParameter &&
        value.instruction()->parent() == module_->entry_computation()) {
      aliased_buffers->push_back(GetBufferForValue(value));
    }
  }

  void ComputeWhileAliasedBuffers(const HloValue& value,
                                  std::vector<BufferNumber>* aliased_buffers) {
    VLOG(3) << "Compute kWhile aliases";
    // Value is init of a while (use is while).
    for (const HloUse& use : value.uses()) {
      if (use.instruction->opcode() == HloOpcode::kWhile) {
        // Determine the while value that this shares a buffer with.
        const HloValue& while_value =
            dataflow_.GetUniqueValueAt(use.instruction, use.operand_index);
        aliased_buffers->push_back(GetBufferForValue(while_value));
        VLOG(3) << "  value is init value to a while; must share buffer with "
                   "while value "
                << while_value.ToShortString();
      }
    }
    // Value is a parameter of a while body/condition.
    if (value.defining_instruction()->opcode() == HloOpcode::kParameter) {
      const HloComputation* computation =
          value.defining_instruction()->parent();
      const CallGraphNode& call_graph_node =
          dataflow_.call_graph().GetNode(computation);
      for (const CallSite& callsite : call_graph_node.caller_callsites()) {
        if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
          // Call graph must have been flattened.
          CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

          const HloValue& while_value = dataflow_.GetUniqueValueAt(
              callsite.instruction(), value.defining_index());
          VLOG(3) << "  value is parameter value of the body or condition of a "
                     "while; must share buffer with while value "
                  << while_value.ToShortString();
          aliased_buffers->push_back(GetBufferForValue(while_value));
        }
      }
    }
    // Value is the root of a while body.
    for (const HloPosition& position : value.positions()) {
      const HloComputation* computation = position.instruction->parent();
      const CallGraphNode& call_graph_node =
          dataflow_.call_graph().GetNode(computation);
      if (position.instruction == computation->root_instruction()) {
        for (const CallSite& callsite : call_graph_node.caller_callsites()) {
          if (callsite.instruction()->opcode() == HloOpcode::kWhile &&
              callsite.instruction()->while_body() == computation) {
            // Call graph must have been flattened.
            CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

            const HloValue& while_value = dataflow_.GetUniqueValueAt(
                callsite.instruction(), position.index);
            VLOG(3) << "  value @ " << position << " is root of "
                    << callsite.instruction()->name()
                    << "; body root and while value root must share buffer "
                       "among them : "
                    << while_value.ToShortString();
            aliased_buffers->push_back(GetBufferForValue(while_value));
          }
        }
      }
    }
    // Value is the output of the while instruction itself.
    if (value.defining_instruction()->opcode() == HloOpcode::kWhile) {
      VLOG(3) << "  value is output of a while instruction";
      aliased_buffers->push_back(GetBufferForValue(value));
    }
  }

  void ComputeConditionalAliasedBuffers(
      const HloValue& value, std::vector<BufferNumber>* aliased_buffers) {
    VLOG(3) << "Compute kConditional aliases";
    // Aliases the buffers of the true/false computations roots, with the one of
    // the conditional.
    for (const HloPosition& position : value.positions()) {
      const HloComputation* computation = position.instruction->parent();
      const CallGraphNode& call_graph_node =
          dataflow_.call_graph().GetNode(computation);
      if (position.instruction == computation->root_instruction()) {
        for (const CallSite& callsite : call_graph_node.caller_callsites()) {
          if (callsite.instruction()->opcode() == HloOpcode::kConditional) {
            // Call graph must have been flattened.
            CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

            const HloValue& cond_value = dataflow_.GetUniqueValueAt(
                callsite.instruction(), position.index);
            VLOG(3)
                << "  value @ " << position << " is root of "
                << callsite.instruction()->name()
                << "; branch computation roots must share buffer among them : "
                << cond_value.ToShortString();
            aliased_buffers->push_back(GetBufferForValue(cond_value));
          }
        }
      }
    }
    // Value is the output of the conditional instruction itself.
    if (value.defining_instruction()->opcode() == HloOpcode::kConditional) {
      VLOG(3) << "  value is output of a conditional instruction";
      aliased_buffers->push_back(GetBufferForValue(value));
    }
  }

  // Compute and return a vector of buffers that the given value must be
  // contained in due to HLO aliasing rules.
  std::vector<BufferNumber> ComputeAliasedBuffers(const HloValue& value) {
    for (const HloUse& use : value.uses()) {
      VLOG(2) << "Use of value " << value.ToShortString() << ": " << use;
    }
    std::vector<BufferNumber> aliased_buffers;
    ComputeInputOutputAliasedBuffers(value, &aliased_buffers);
    ComputeWhileAliasedBuffers(value, &aliased_buffers);
    ComputeConditionalAliasedBuffers(value, &aliased_buffers);
    // Uniquify aliased buffers.
    absl::c_sort(aliased_buffers);
    aliased_buffers.erase(
        std::unique(aliased_buffers.begin(), aliased_buffers.end()),
        aliased_buffers.end());
    return aliased_buffers;
  }

  HloModule* module_;

  // Dataflow analysis used to construct the buffer map.
  const HloDataflowAnalysis& dataflow_;

  // A map containing the set of values contained in each buffer.
  absl::flat_hash_map<BufferNumber, absl::flat_hash_set<const HloValue*>>
      buffers_;

  // A map indicating which buffer each value is contained in.
  absl::flat_hash_map<const HloValue*, BufferNumber> value_to_buffer_number_;

  // The buffer number of the next buffer to be created.
  BufferNumber next_buffer_number_ = 0;
};

HloAliasAnalysis::HloAliasAnalysis(HloModule* module) : module_(module) {}

const HloBuffer& HloAliasAnalysis::GetUniqueBufferAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::vector<const HloBuffer*> buffers = ComputeBuffersAt(instruction, index);
  CHECK_EQ(buffers.size(), 1);
  return *buffers[0];
}

HloBuffer& HloAliasAnalysis::GetUniqueBufferAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return GetBuffer(static_cast<const HloAliasAnalysis*>(this)
                       ->GetUniqueBufferAt(instruction, index)
                       .id());
}

std::vector<const HloBuffer*> HloAliasAnalysis::ComputeBuffersAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::vector<const HloBuffer*> buffers;
  for (const HloValue* value :
       dataflow_analysis_->GetValueSet(instruction, index).values()) {
    buffers.push_back(&GetBufferContainingValue(*value));
  }

  // Sort and uniquify vector before returning.
  absl::c_sort(buffers, HloBuffer::IdLessThan);
  buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());

  return buffers;
}

bool HloAliasAnalysis::InstructionBuffersAreAmbiguous(
    const HloInstruction* instruction) const {
  for (const auto& pair :
       dataflow_analysis_->GetInstructionValueSet(instruction)) {
    const HloValueSet& value_set = pair.second;
    const HloBuffer* buffer = nullptr;
    for (const HloValue* value : value_set.values()) {
      if (buffer == nullptr) {
        buffer = &GetBufferContainingValue(*value);
      } else if (buffer != &GetBufferContainingValue(*value)) {
        return true;
      }
    }
  }
  return false;
}

bool HloAliasAnalysis::InstructionBuffersAreDistinct(
    const HloInstruction* instruction) const {
  absl::flat_hash_set<const HloBuffer*> buffers_seen;
  for (const auto& pair :
       dataflow_analysis_->GetInstructionValueSet(instruction)) {
    const HloValueSet& value_set = pair.second;
    if (value_set.values().size() == 1) {
      if (!buffers_seen
               .insert(&GetBufferContainingValue(value_set.GetUniqueValue()))
               .second) {
        return false;
      }
    } else {
      // It's possible for multiple values at this index to have the same
      // HloBuffer. This does not result in non-distictness. To account for
      // this case, add all of the buffers at this index after checking
      // whether each buffer exists at an earlier index. This is a corner
      // case, however, as the number of values at an index is almost always
      // one.
      std::vector<const HloBuffer*> buffers_at_this_index;
      for (const HloValue* value : value_set.values()) {
        const HloBuffer* buffer = &GetBufferContainingValue(*value);
        if (ContainsKey(buffers_seen, buffer)) {
          return false;
        }
        buffers_at_this_index.push_back(buffer);
      }
      buffers_seen.insert(buffers_at_this_index.begin(),
                          buffers_at_this_index.end());
    }
  }
  return true;
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

  return Status::OK();
}

string HloAliasAnalysis::ToString() const {
  string out = absl::StrCat("HloAliasAnalysis, module ", module_->name(), "\n");
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
    HloModule* module, const HloDataflowAnalysis::FusionCanShareBufferFunction&
                           fusion_can_share_buffer) {
  VLOG(2) << "HloAliasAnalysis::Run on module " << module->name();
  XLA_VLOG_LINES(2, module->ToString());

  auto alias_analysis = absl::WrapUnique(new HloAliasAnalysis(module));
  TF_ASSIGN_OR_RETURN(alias_analysis->dataflow_analysis_,
                      HloDataflowAnalysis::Run(*module, /*ssa_form=*/true,
                                               /*bitcast_defines_value=*/false,
                                               fusion_can_share_buffer));

  BufferValueMap buffer_map(module, alias_analysis->dataflow_analysis());
  buffer_map.MergeAliasedBuffers();

  // Create a vector of HloBuffers, one for each set of values in the
  // BufferValueMap. Create the HloBuffers as a vector of contiguously numbered
  // buffers.
  std::vector<BufferValueMap::BufferNumber> sorted_buffer_numbers =
      buffer_map.ComputeSortedBufferNumbers();
  alias_analysis->buffers_.reserve(sorted_buffer_numbers.size());
  HloBuffer::Id next_id = 0;
  for (BufferValueMap::BufferNumber buffer_number : sorted_buffer_numbers) {
    auto& value_set = buffer_map.GetValuesInBuffer(buffer_number);
    std::vector<const HloValue*> sorted_values(value_set.begin(),
                                               value_set.end());
    absl::c_sort(sorted_values, HloValue::IdLessThan);
    alias_analysis->buffers_.emplace_back(next_id++, sorted_values);
    for (const HloValue* value : sorted_values) {
      alias_analysis->value_to_buffer_[value] =
          &alias_analysis->buffers_.back();
    }
  }

  TF_DCHECK_OK(alias_analysis->Verify());

  XLA_VLOG_LINES(2, alias_analysis->ToString());
  return std::move(alias_analysis);
}

bool HloAliasAnalysis::HasLiveRangeInterference(
    const HloOrdering& ordering) const {
  for (const HloBuffer& buffer : buffers()) {
    CHECK(!buffer.values().empty());
    if (buffer.values().front()->shape().IsToken()) {
      // Tokens have no on-device representation and cannot interfere.
      for (const HloValue* value : buffer.values()) {
        // If one of the values is a token, all values must be a token.
        DCHECK(value->shape().IsToken());
      }
      continue;
    }

    // Check that the values in the buffer are totally ordered with respect to
    // 'ordering'. Begin by sorting the values with respect to 'ordering' with a
    // tie-break using value ID. The tie-break is necessary because we need a
    // strict weak order for std::sort.
    std::vector<const HloValue*> values = buffer.values();
    absl::c_sort(values, [&ordering](const HloValue* a, const HloValue* b) {
      if (ordering.IsDefinedBefore(*a, *b)) {
        return true;
      } else if (ordering.IsDefinedBefore(*b, *a)) {
        return false;
      } else {
        return a->id() < b->id();
      }
    });

    // Walk through the ordered vector of values. First verify that the values
    // are totally ordered with respect to 'ordering', then check that no
    // adjacent values have overlapping live ranges. Only adjacent values must
    // be checked because of the property of live range interference. For
    // example, if you have values A, B, and C (in program order) contained in
    // a buffer and A interferes with C, then necessarily A also interferes
    // with B. So to check interference you only need to check interference
    // between A and B, and between B and C.
    for (int i = 1; i < values.size(); ++i) {
      if (!ordering.IsDefinedBefore(*values[i - 1], *values[i])) {
        VLOG(1) << values[i - 1]->ToShortString() << " and "
                << values[i]->ToShortString() << " are not ordered";
        return true;
      }
      if (ordering.MayInterfere(*values[i - 1], *values[i],
                                dataflow_analysis())) {
        VLOG(1) << "In buffer " << buffer.id() << " containing values:\n  "
                << absl::StrJoin(values, ", ",
                                 [](string* out, const HloValue* value) {
                                   StrAppend(out, value->ToShortString());
                                 })

                << "\nValue " << values[i - 1]->ToShortString()
                << " may interfere with value " << values[i]->ToShortString();
        return true;
      }
    }
  }

  return false;
}

}  // namespace xla
