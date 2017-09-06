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

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

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
  std::sort(buffers.begin(), buffers.end(), HloBuffer::IdLessThan);
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
  tensorflow::gtl::FlatSet<const HloBuffer*> buffers_seen;
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
      // HloBuffer. This does not result in non-distictness. To account for this
      // case, add all of the buffers at this index after checking whether each
      // buffer exists at an earlier index. This is a corner case, however, as
      // the number of values at an index is almost always one.
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

void HloAliasAnalysis::InitializeBufferSets() {
  // Initially define a buffer for every HloValue in the module.
  for (const HloValue& value : dataflow_analysis_->values()) {
    HloBuffer& buffer = NewHloBuffer();
    buffer.AddValue(value);
    value_to_buffer_[&value] = &buffer;
  }
}

Status HloAliasAnalysis::Verify() const {
  // Verify consistency between the value_to_buffer_ map and
  // HloBuffer::values().
  for (const auto& pair : value_to_buffer_) {
    const HloValue* value = pair.first;
    const HloBuffer& buffer = *pair.second;
    TF_RET_CHECK(std::find(buffer.values().begin(), buffer.values().end(),
                           value) != buffer.values().end());
  }

  for (const auto& pair : buffers_) {
    const HloBuffer::Id id = pair.first;
    const HloBuffer& buffer = pair.second;
    TF_RET_CHECK(buffer.id() == id);

    HloValue::Id last_value_id = -1;
    for (const HloValue* value : buffer.values()) {
      TF_RET_CHECK(GetBufferContainingValue(*value) == buffer);

      // Also verify the values in HloBuffer are unique and sorted by id.
      TF_RET_CHECK(value->id() > last_value_id);
      last_value_id = value->id();
    }
  }

  if (!buffers_vector_.empty()) {
    // buffers_vector_ should be a vector of all HloBuffers sorted by id.
    std::vector<const HloBuffer*> buffers;
    for (const auto& id_buffer : buffers_) {
      buffers.push_back(&id_buffer.second);
    }
    std::sort(buffers.begin(), buffers.end(), HloBuffer::IdLessThan);
    TF_RET_CHECK(buffers_vector_ == buffers);
  }

  return Status::OK();
}

Status HloAliasAnalysis::VerifyAgainstReference() const {
  TF_RETURN_IF_ERROR(Verify());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> reference,
                      Run(module_));
  TF_RETURN_IF_ERROR(reference->Verify());

  VLOG(2) << "This analysis:";
  XLA_VLOG_LINES(2, ToString());
  VLOG(2) << "Reference:";
  XLA_VLOG_LINES(2, reference->ToString());

  // Create map from HloValue in the reference analysis to HloValue in this
  // analysis and vice versa.
  tensorflow::gtl::FlatMap<const HloValue*, const HloValue*> reference_to_this;
  tensorflow::gtl::FlatMap<const HloValue*, const HloValue*> this_to_reference;
  for (const HloValue& value : dataflow_analysis().values()) {
    const HloValue& reference_value =
        reference->dataflow_analysis().GetValueDefinedAt(
            value.defining_instruction(), value.defining_index());
    reference_to_this[&reference_value] = &value;
    this_to_reference[&value] = &reference_value;
  }

  TF_RET_CHECK(buffers_.size() == reference->buffers_.size())
      << "Different number of buffers (" << buffers_.size()
      << " != " << reference->buffers_.size() << ")";
  for (const auto& pair : reference->buffers_) {
    const HloBuffer& reference_buffer = pair.second;

    // Find the corresponding buffer in the reference by taking the first value
    // in the buffer, finding the corresponding value in the reference, and then
    // finding the buffer holding that value.
    TF_RET_CHECK(!reference_buffer.values().empty());
    const HloValue* reference_value = reference_buffer.values()[0];
    const HloValue* value = reference_to_this.at(reference_value);
    const HloBuffer& buffer = GetBufferContainingValue(*value);

    // The buffer and the reference should have the exact same values. To make
    // comparison easy, sort the values in the reference buffer identically to
    // the values in the non-reference buffer (ie, by the corresponding id of
    // the non-reference value).
    std::vector<const HloValue*> reference_values = reference_buffer.values();
    std::sort(reference_values.begin(), reference_values.end(),
              [&reference_to_this](const HloValue* a, const HloValue* b) {
                return reference_to_this.at(a)->id() <
                       reference_to_this.at(b)->id();
              });
    TF_RET_CHECK(reference_values.size() == buffer.values().size());
    for (int i = 0; i < buffer.values().size(); ++i) {
      TF_RET_CHECK(*reference_values[i] == *buffer.values()[i])
          << "Buffer:\n  " << buffer
          << "\ndoes not have the same values as reference buffer:\n  "
          << reference_buffer;
    }
  }

  return Status::OK();
}

HloBuffer& HloAliasAnalysis::NewHloBuffer() {
  HloBuffer::Id buffer_id = next_buffer_id_++;
  auto emplaced = buffers_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(buffer_id),
                                   std::forward_as_tuple(buffer_id));
  CHECK(emplaced.second);

  buffers_vector_.clear();

  return emplaced.first->second;
}

void HloAliasAnalysis::MoveValueToNewBuffer(const HloValue& value) {
  HloBuffer& new_buffer = NewHloBuffer();
  MoveValueToBuffer(value, &new_buffer);

  VLOG(3) << "Moved value " << value.ToShortString() << " into new buffer "
          << new_buffer.id();
}

void HloAliasAnalysis::MoveValueToBuffer(const HloValue& value,
                                         HloBuffer* buffer) {
  HloBuffer& old_buffer = GetBufferContainingValue(value);
  CHECK_NE(buffer, &old_buffer);
  VLOG(3) << "Moved value " << value.ToShortString() << " from buffer "
          << old_buffer.id() << " into buffer " << buffer->id();
  old_buffer.RemoveValue(value);
  if (old_buffer.values().empty()) {
    VLOG(3) << "Buffer " << old_buffer.id() << " now empty. Removing.";
    buffers_.erase(old_buffer.id());
    buffers_vector_.clear();
  }

  buffer->AddValue(value);
  value_to_buffer_[&value] = buffer;
}

string HloAliasAnalysis::ToString() const {
  string out = StrCat("HloAliasAnalysis, module ", module_->name(), "\n");
  StrAppend(&out, "  Buffers at each position:\n");
  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      StrAppend(&out, "    ", instruction->name(), ":\n");
      if (ShapeUtil::IsTuple(instruction->shape())) {
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&out, &instruction, this](const Shape&, const ShapeIndex& index) {
              StrAppend(&out, "      tuple index ", index.ToString(), ":\n");
              for (const HloBuffer* buffer :
                   ComputeBuffersAt(instruction.get(), index)) {
                StrAppend(&out, "        ", buffer->ToString(), "\n");
              }
            });
      } else {
        for (const HloBuffer* buffer :
             ComputeBuffersAt(instruction.get(), /*index=*/{})) {
          StrAppend(&out, "      ", buffer->ToString(), "\n");
        }
      }
    }
  }

  StrAppend(&out, "  Buffers:\n");
  for (const HloBuffer* buffer : buffers()) {
    StrAppend(&out, "    ", buffer->ToString(), "\n");
    StrAppend(&out, "      positions:\n");
    for (const HloPosition& position : buffer->ComputePositions()) {
      StrAppend(&out, "        ", position.ToString(), "\n");
    }
  }

  return out;
}

const std::vector<const HloBuffer*>& HloAliasAnalysis::buffers() const {
  if (buffers_vector_.empty()) {
    // Lazily construct vector of buffers.
    buffers_vector_.reserve(buffers_.size());
    for (auto& pair : buffers_) {
      buffers_vector_.push_back(&pair.second);
    }
    std::sort(buffers_vector_.begin(), buffers_vector_.end(),
              HloBuffer::IdLessThan);
  } else {
    CHECK_EQ(buffers_vector_.size(), buffers_.size());
    for (const HloBuffer* buffer : buffers_vector_) {
      DCHECK(ContainsKey(buffers_, buffer->id()));
      DCHECK(&GetBuffer(buffer->id()) == buffer);
    }
  }
  return buffers_vector_;
}

void HloAliasAnalysis::UpdateAtInstructions(
    tensorflow::gtl::ArraySlice<const HloInstruction*> instructions) {
  VLOG(4) << "Updated HLO module:";
  XLA_VLOG_LINES(4, module_->ToString());

  VLOG(3) << "Before update:";
  XLA_VLOG_LINES(3, ToString());

  std::vector<const HloValue*> values_to_update;
  for (const HloInstruction* instruction : instructions) {
    for (auto& pair : dataflow_analysis().GetInstructionValueSet(instruction)) {
      for (const HloValue* value : pair.second.values()) {
        values_to_update.push_back(value);
      }
    }
  }

  UpdateBuffersForValues(values_to_update);

  VLOG(3) << "After update:";
  XLA_VLOG_LINES(3, ToString());
}

void HloAliasAnalysis::UpdateAfterChangingOperand(HloInstruction* instruction,
                                                  HloInstruction* old_operand,
                                                  HloInstruction* new_operand) {
  VLOG(1) << "UpdateAfterChangingOperand(" << instruction->name() << ", "
          << old_operand->name() << " => " << new_operand->name() << ")";

  dataflow_analysis_->UpdateAfterChangingOperand(instruction, old_operand,
                                                 new_operand);
  TF_DCHECK_OK(dataflow_analysis_->VerifyAgainstReference());

  VLOG(4) << "Updated dataflow:";
  XLA_VLOG_LINES(4, dataflow_analysis_->ToString());

  UpdateAtInstructions({instruction, old_operand, new_operand});
}

void HloAliasAnalysis::UpdateAfterChangingRoot(HloInstruction* old_root,
                                               HloInstruction* new_root) {
  VLOG(1) << "UpdateAfterChangingRoot(" << old_root->name() << " => "
          << new_root->name() << ")";

  dataflow_analysis_->UpdateAfterChangingRoot(old_root, new_root);
  TF_DCHECK_OK(dataflow_analysis_->VerifyAgainstReference());

  VLOG(4) << "Updated dataflow:";
  XLA_VLOG_LINES(4, dataflow_analysis_->ToString());

  UpdateAtInstructions({old_root, new_root});
}

std::vector<HloBuffer*> HloAliasAnalysis::ComputeAliasedBuffers(
    const HloValue& value) {
  std::vector<HloBuffer*> aliased_buffers;

  // Value is init of a while (use is while).
  for (const HloUse& use : value.uses()) {
    VLOG(1) << "use of value " << value.ToShortString() << ": " << use;
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      // Determine the while value that this shares a buffer with.
      const HloValue& while_value = dataflow_analysis().GetUniqueValueAt(
          use.instruction, use.operand_index);
      aliased_buffers.push_back(&GetBufferContainingValue(while_value));
      VLOG(3) << "  value is init value to a while; must share buffer with "
                 "while value "
              << while_value.ToShortString();
    }
  }

  // Value is a parameter of a while body/condition.
  if (value.defining_instruction()->opcode() == HloOpcode::kParameter) {
    const HloComputation* computation = value.defining_instruction()->parent();
    const CallGraphNode& call_graph_node =
        dataflow_analysis().call_graph().GetNode(computation);
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
        // Call graph must have been flattened.
        CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

        const HloValue& while_value = dataflow_analysis().GetUniqueValueAt(
            callsite.instruction(), value.defining_index());
        VLOG(3) << "  value is parameter value of the body or condition of a "
                   "while; must share buffer with while value "
                << while_value.ToShortString();
        aliased_buffers.push_back(&GetBufferContainingValue(while_value));
      }
    }
  }

  // Value is the root of a while body.
  for (const HloPosition& position : value.positions()) {
    const HloComputation* computation = position.instruction->parent();
    const CallGraphNode& call_graph_node =
        dataflow_analysis().call_graph().GetNode(computation);
    if (position.instruction == computation->root_instruction()) {
      for (const CallSite& callsite : call_graph_node.caller_callsites()) {
        if (callsite.instruction()->opcode() == HloOpcode::kWhile &&
            callsite.instruction()->while_body() == computation) {
          // Call graph must have been flattened.
          CHECK_EQ(call_graph_node.caller_callsites().size(), 1);

          // If the value appears in the root of a while body, then
          // necessarily the value is defined in the body as well.
          CHECK_EQ(value.defining_instruction()->parent(), computation);

          const HloValue& while_value = dataflow_analysis().GetUniqueValueAt(
              callsite.instruction(), position.index);
          VLOG(3) << "  value is root the body computation of a while; must "
                     "share buffer with while value "
                  << while_value.ToShortString();
          aliased_buffers.push_back(&GetBufferContainingValue(while_value));
        }
      }
    }
  }

  // Value is in the while instruction itself.
  if (value.defining_instruction()->opcode() == HloOpcode::kWhile) {
    VLOG(3) << "  value is output of a while instruction";
    aliased_buffers.push_back(&GetUniqueBufferAt(value.defining_instruction(),
                                                 value.defining_index()));
  }

  // Uniquify aliased buffers.
  std::sort(aliased_buffers.begin(), aliased_buffers.end(),
            HloBuffer::IdLessThan);
  aliased_buffers.erase(
      std::unique(aliased_buffers.begin(), aliased_buffers.end()),
      aliased_buffers.end());

  return aliased_buffers;
}

// This method recomputes the HloBuffer for each of the given HloValues. The
// method does not necessarily update the HloBuffer of values which share a
// buffer with the given values, but are not explicitly passed in
// 'values'. Therefore, the caller must pass in all values which may require an
// update according to the kind of HLO graph change which occurred: operand
// changed (UpdateAfterChangingOperand), or root of computation changed
// (UpdateAfterChangingRoot).
void HloAliasAnalysis::UpdateBuffersForValues(
    tensorflow::gtl::ArraySlice<const HloValue*> values) {
  for (const HloValue* value : values) {
    VLOG(3) << "Updating buffer for value: " << value->ToShortString();

    // Gather the set of buffer with aliasing rules (eg, kWhile) which this
    // value must be contained in due.
    std::vector<HloBuffer*> aliased_buffers = ComputeAliasedBuffers(*value);

    HloBuffer& current_buffer = GetBufferContainingValue(*value);
    if (aliased_buffers.empty()) {
      // The buffer containing 'value' aliases no other buffers. If the buffer
      // containing 'value' already only contains 'value', then no change is
      // necessary. If the buffer containing 'value' does contain other values,
      // then remove 'value' from the buffer and create a new buffer containing
      // only 'value'
      if (current_buffer.values().size() == 1) {
        CHECK_EQ(current_buffer.values()[0], value);
      } else {
        MoveValueToNewBuffer(*value);
      }
    } else {
      // If multiple buffers are aliased merge these buffers together into a
      // single buffer (arbitrarily chosen as the first buffer in the vector).
      if (aliased_buffers.size() > 1) {
        for (int64 i = 1; i < aliased_buffers.size(); ++i) {
          // Make copy of values vector because MoveValueToBuffer invalidates
          // the values iterator. The could be done more efficiently by moving
          // all values and once.
          std::vector<const HloValue*> values = aliased_buffers[i]->values();
          for (const HloValue* value : values) {
            MoveValueToBuffer(*value, aliased_buffers[0]);
          }
        }
        aliased_buffers.resize(1);
      }

      CHECK_EQ(aliased_buffers.size(), 1);
      HloBuffer* new_buffer = aliased_buffers[0];

      if (&current_buffer != new_buffer) {
        MoveValueToBuffer(*value, new_buffer);
      }
    }

    VLOG(4) << "Analysis after update:";
    XLA_VLOG_LINES(4, ToString());
  }
}

/* static */
StatusOr<std::unique_ptr<HloAliasAnalysis>> HloAliasAnalysis::Run(
    HloModule* module) {
  VLOG(1) << "HloAliasAnalysis::Run on module " << module->name();
  XLA_VLOG_LINES(2, module->ToString());

  auto alias_analysis = WrapUnique(new HloAliasAnalysis(module));
  TF_ASSIGN_OR_RETURN(
      alias_analysis->dataflow_analysis_,
      HloDataflowAnalysis::Run(module, /*ssa_form=*/true,
                               /*bitcast_defines_value=*/false));

  alias_analysis->InitializeBufferSets();

  VLOG(3) << "After initialization:";
  XLA_VLOG_LINES(3, alias_analysis->ToString());

  std::vector<const HloValue*> all_values;
  for (const HloValue& value : alias_analysis->dataflow_analysis().values()) {
    all_values.push_back(&value);
  }

  alias_analysis->UpdateBuffersForValues(all_values);

  TF_DCHECK_OK(alias_analysis->Verify());

  XLA_VLOG_LINES(1, alias_analysis->ToString());
  return std::move(alias_analysis);
}

}  // namespace xla
