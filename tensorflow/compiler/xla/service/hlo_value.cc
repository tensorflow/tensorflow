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

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

using absl::StrAppend;
using absl::StrCat;

const Shape& HloPosition::shape() const {
  return ShapeUtil::GetSubshape(instruction->shape(), index);
}

string HloPosition::ToString() const {
  string index_str =
      instruction->shape().IsTuple() ? (" " + index.ToString()) : "";
  return StrCat(instruction->name(), index_str);
}

std::ostream& operator<<(std::ostream& out, const HloPosition& position) {
  out << position.ToString();
  return out;
}

string HloUse::ToString() const {
  string index_str = instruction->operand(operand_number)->shape().IsTuple()
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
    : BufferValue(instruction, index, id), is_phi_(is_phi) {
  // The defining position is always the first element in the positions_ vector.
  positions_.push_back(HloPosition{instruction, index});
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
  string index_str = defining_instruction()->shape().IsTuple()
                         ? defining_index().ToString()
                         : "";
  return StrCat(id(), " ", is_phi_ ? "PHI " : "",
                defining_instruction()->name(), index_str);
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
    case HloOpcode::kTupleSelect:
      // Select does not use any nested elements of its selected-from operands
      // (operand 1 and 2)
      CHECK_GE(operand_number, 0);
      CHECK_LE(operand_number, 2);
      return operand_number == 0 || index.empty();

    case HloOpcode::kDomain:
    case HloOpcode::kTuple:
      // These instructions always pass through their operands transparently.
      return false;

    case HloOpcode::kCall:
    case HloOpcode::kWhile:
      // Although call and while instructions pass through their operands, they
      // are considered uses.
      return true;

    default:
      return true;
  }
}

}  // namespace

void HloValue::SetPositionsAndComputeUses(
    absl::Span<const HloPosition> positions) {
  CHECK_EQ(positions_.size(), 1) << "SetPositions should only be called once.";

  // The positions must be unique and should not contain the defining position
  // as this is added at construction time.
  for (const HloPosition& position_a : positions) {
    DCHECK_NE(position_a, defining_position());
    for (const HloPosition& position_b : positions) {
      if (&position_a != &position_b) {
        DCHECK_NE(position_a, position_b);
      }
    }
  }

  positions_.insert(positions_.end(), positions.begin(), positions.end());

  // Gather the computation roots at which this value appears.
  absl::flat_hash_set<HloInstruction*> root_positions;
  for (const HloPosition& position : positions_) {
    if (position.instruction ==
        position.instruction->parent()->root_instruction()) {
      root_positions.insert(position.instruction);
    }
  }

  // Build vector of HloUses for the value.
  for (const HloPosition& position : positions_) {
    for (HloInstruction* user : position.instruction->users()) {
      for (int64 operand_number : user->OperandIndices(position.instruction)) {
        // Root instructions of computations are considered to be uses whether
        // or not the root instruction itself actually uses the value.
        if (MayUseOperandValue(operand_number, position.index, user) ||
            ContainsKey(root_positions, user)) {
          HloUse new_use{user, operand_number, position.index};

          // The new use must not already exist in uses_.
          for (const HloUse& use : uses_) {
            DCHECK_NE(use, new_use);
          }

          uses_.push_back(std::move(new_use));
        }
      }
    }

    // Update liveout status of this HloValue.
    const HloModule& module = *position.instruction->parent()->parent();
    if (position.instruction ==
        module.entry_computation()->root_instruction()) {
      live_out_of_module_ = true;
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
  return StrCat(
      "HloValueSet: ",
      absl::StrJoin(values_, ", ", [](string* result, const HloValue* value) {
        result->append(value->ToShortString());
      }));
}

bool HloValueSet::AssignUnionOf(absl::Span<const HloValueSet* const> inputs) {
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
    absl::Span<const InstructionValueSet* const> inputs) {
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
  ForEachElement([&out](const ShapeIndex& index, const HloValueSet& value_set) {
    StrAppend(&out, "  ", index.ToString(), " : ", value_set.ToString(), "\n");
  });
  return out;
}

}  // namespace xla
