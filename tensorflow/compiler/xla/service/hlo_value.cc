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

#include "tensorflow/compiler/xla/service/hlo_value.h"

#include <algorithm>
#include <utility>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
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

using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

const Shape& HloPosition::shape() const {
  return ShapeUtil::GetSubshape(instruction->shape(), index);
}

string HloPosition::ToString() const {
  string index_str =
      ShapeUtil::IsTuple(instruction->shape()) ? (" " + index.ToString()) : "";
  return StrCat(instruction->name(), index_str);
}

std::ostream& operator<<(std::ostream& out, const HloPosition& position) {
  out << position.ToString();
  return out;
}

string HloUse::ToString() const {
  string index_str =
      ShapeUtil::IsTuple(instruction->operand(operand_number)->shape())
          ? (" " + operand_index.ToString())
          : "";
  return StrCat(instruction->name(), ", operand ", operand_number, index_str);
}

std::ostream& operator<<(std::ostream& out, const HloUse& use) {
  out << use.ToString();
  return out;
}

HloValue::HloValue(HloValue::Id id, HloInstruction* instruction,
                   const ShapeIndex& index, bool is_phi)
    : id_(id), is_phi_(is_phi) {
  // The defining position is always the first element in the positions_ vector.
  AddPosition(instruction, index);
}

bool HloValue::operator==(const HloValue& other) const {
  bool equal = defining_instruction() == other.defining_instruction() &&
               defining_index() == other.defining_index();
  // If the values are equal they most both be phi (or non phi).
  CHECK(!(equal && is_phi() != other.is_phi()));
  return equal;
}

bool HloValue::operator!=(const HloValue& other) const {
  return !(*this == other);
}

string HloValue::ToShortString() const {
  string index_str = ShapeUtil::IsTuple(defining_instruction()->shape())
                         ? defining_index().ToString()
                         : "";
  return StrCat(id_, " ", is_phi_ ? "PHI " : "", defining_instruction()->name(),
                index_str);
}

string HloValue::ToString(int indent) const {
  string indentation(indent, ' ');
  string out = StrCat(indentation, ToShortString(), ", positions:\n");
  for (const HloPosition& position : positions()) {
    StrAppend(&out, indentation, "  ", position.ToString(), "\n");
  }
  StrAppend(&out, indentation, " uses:\n");
  for (const HloUse& use : uses()) {
    StrAppend(&out, indentation, "  ", use.ToString(), "\n");
  }
  return out;
}

namespace {

// Returns true if the instruction 'user' may use the value at the given
// ShapeIndex in the given operand. Generally, instruction which pass through
// values transparently without reading the value are not considered to use the
// value.
bool MayUseOperandValue(int64 operand_number, const ShapeIndex& index,
                        const HloInstruction* user) {
  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kCopy:
      // These instructions only access the top-level values of their
      // operand. Non-top-level (nested) values are passed through
      // transparently.
      CHECK_EQ(operand_number, 0);
      return index.empty();
    case HloOpcode::kSelect:
      // Select does not use any nested elements of its selected-from operands
      // (operand 1 and 2)
      CHECK_GE(operand_number, 0);
      CHECK_LE(operand_number, 2);
      return operand_number == 0 || index.empty();

    case HloOpcode::kCall:
    case HloOpcode::kTuple:
      // These instructions always pass through their operands transparently.
      return false;

    case HloOpcode::kWhile:
      // Though the while instructions passes through its operands, we return
      // true because in SSA form there may be a Phi at the parameter of the
      // while which is considered a use of its incoming value because the Phi
      // input values are not passed through into the body computation. Because
      // this function is used in both SSA and non-SSA forms of the analysis
      // conservatively return true.
      return true;

    default:
      return true;
  }
}

}  // namespace

void HloValue::AddPosition(HloInstruction* instruction,
                           const ShapeIndex& index) {
  HloPosition new_position{instruction, index};

  // The new position must not already exist in positions_.
  for (const HloPosition& position : positions_) {
    DCHECK_NE(position, new_position);
  }
  // The shape of the new position must match existing positions.
  if (!positions_.empty()) {
    CHECK(
        ShapeUtil::Compatible(positions_.front().shape(), new_position.shape()))
        << "front: " << positions_.front() << " new: " << new_position;
  }

  positions_.push_back(std::move(new_position));

  // Update uses.
  for (HloInstruction* user : instruction->users()) {
    for (int64 operand_number : user->OperandIndices(instruction)) {
      if (MayUseOperandValue(operand_number, index, user)) {
        HloUse new_use{user, operand_number, index};

        // The new use must not already exist in uses_.
        for (const HloUse& use : uses_) {
          DCHECK_NE(use, new_use);
        }

        uses_.push_back(std::move(new_use));
      }
    }
  }

  // Update liveout status of this HloValue.
  const HloModule& module = *instruction->parent()->parent();
  if (instruction == module.entry_computation()->root_instruction()) {
    live_out_of_module_ = true;
  }

  if (instruction == instruction->parent()->root_instruction()) {
    live_out_of_computation_ = true;
  }
}

void HloValue::RemovePosition(HloInstruction* instruction,
                              const ShapeIndex& index) {
  // The defining position cannot be removed.
  CHECK(!(instruction == defining_instruction() && index == defining_index()));

  int64 size_before = positions_.size();
  positions_.erase(
      std::remove_if(positions_.begin(), positions_.end(),
                     [instruction, &index](const HloPosition& position) {
                       return position.instruction == instruction &&
                              position.index == index;
                     }),
      positions_.end());
  // Only a single position should have been removed.
  CHECK_EQ(positions_.size(), size_before - 1);

  //  Update uses which referred to this position.
  uses_.erase(std::remove_if(uses_.begin(), uses_.end(),
                             [instruction, &index](const HloUse& use) {
                               return use.instruction->operand(
                                          use.operand_number) == instruction &&
                                      use.operand_index == index;
                             }),
              uses_.end());

  // Returns whether this value is contained in the given instruction's output.
  auto is_contained_in = [this](const HloInstruction* instruction) {
    for (const HloPosition& position : positions()) {
      if (position.instruction == instruction) {
        return true;
      }
    }
    return false;
  };

  const HloModule& module = *instruction->parent()->parent();
  if (instruction == module.entry_computation()->root_instruction()) {
    // Value has been removed from a position in the entry root instruction.
    live_out_of_module_ =
        is_contained_in(module.entry_computation()->root_instruction());
  }
  if (instruction == defining_instruction()->parent()->root_instruction()) {
    // Value has been removed from the root of the computation the value has
    // been defined in.
    live_out_of_computation_ =
        is_contained_in(defining_instruction()->parent()->root_instruction());
  }
}

void HloValue::RecomputeUses() {
  uses_.clear();
  for (const HloPosition& position : positions()) {
    for (HloInstruction* user : position.instruction->users()) {
      for (int64 operand_number : user->OperandIndices(position.instruction)) {
        if (MayUseOperandValue(operand_number, position.index, user)) {
          uses_.push_back(HloUse{user, operand_number, position.index});
        }
      }
    }
  }
}

std::ostream& operator<<(std::ostream& out, const HloValue& value) {
  out << value.ToShortString();
  return out;
}

void HloValueSet::SortAndUniquifyValues() {
  std::sort(values_.begin(), values_.end(), HloValue::IdLessThan);
  values_.erase(std::unique(values_.begin(), values_.end(), HloValue::IdEqual),
                values_.end());
}

string HloValueSet::ToString() const {
  return StrCat("HloValueSet: ",
                Join(values_, ", ", [](string* result, const HloValue* value) {
                  result->append(value->ToShortString());
                }));
}

bool HloValueSet::AssignUnionOf(
    tensorflow::gtl::ArraySlice<const HloValueSet*> inputs) {
  HloValueSet union_set;
  for (const HloValueSet* input : inputs) {
    for (const HloValue* value : input->values()) {
      union_set.values_.push_back(value);
    }
  }
  union_set.SortAndUniquifyValues();
  if (*this != union_set) {
    *this = union_set;
    return true;
  }
  return false;
}

bool HloValueSet::AddValue(const HloValue* value) {
  auto it = std::lower_bound(values_.begin(), values_.end(), value,
                             HloValue::IdLessThan);
  if (it == values_.end() || (*it)->id() != value->id()) {
    values_.insert(it, value);
    return true;
  }
  return false;  // already exists
}

std::ostream& operator<<(std::ostream& out, const HloValueSet& value_set) {
  out << value_set.ToString();
  return out;
}

bool InstructionValueSet::AssignUnionOf(
    tensorflow::gtl::ArraySlice<const InstructionValueSet*> inputs) {
  CHECK_GT(inputs.size(), 0);
  for (int i = 1; i < inputs.size(); ++i) {
    DCHECK(ShapeUtil::Compatible(inputs[0]->shape(), inputs[i]->shape()));
  }
  bool changed = false;
  for (auto& pair : *this) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    std::vector<const HloValueSet*> input_value_sets;
    for (const InstructionValueSet* input : inputs) {
      input_value_sets.push_back(&input->element(index));
    }
    changed |= value_set.AssignUnionOf(input_value_sets);
  }

  return changed;
}

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set) {
  out << instruction_value_set.ToString();
  return out;
}

string InstructionValueSet::ToString() const {
  string out =
      StrCat("InstructionValueSet(", ShapeUtil::HumanString(shape()), ")\n");
  ForEachElement([this, &out](const ShapeIndex& index,
                              const HloValueSet& value_set) {
    StrAppend(&out, "  ", index.ToString(), " : ", value_set.ToString(), "\n");
  });
  return out;
}

}  // namespace xla
