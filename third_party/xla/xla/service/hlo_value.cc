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

#include "xla/service/hlo_value.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {

using absl::StrAppend;
using absl::StrCat;

const Shape& HloPosition::shape() const {
  return ShapeUtil::GetSubshape(instruction->shape(), index);
}

std::string HloPosition::ToString() const {
  std::string index_str =
      instruction->shape().IsTuple() ? (" " + index.ToString()) : "";
  return StrCat(instruction->name(), index_str);
}

std::ostream& operator<<(std::ostream& out, const HloPosition& position) {
  out << position.ToString();
  return out;
}

std::string HloUse::ToString() const {
  std::string index_str =
      instruction->operand(operand_number)->shape().IsTuple()
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
    : BufferValue(instruction, index, id),
      is_phi_(is_phi),
      live_out_of_module_(false) {
  // The defining position is always the first element in the positions_ vector.
  positions_.push_back(HloPosition{instruction, index});
}

std::string HloValue::ToShortString() const {
  return absl::StrFormat(
      "<%d %s%s%s%s>", id(), instruction()->name(),
      instruction()->shape().IsTuple() ? index().ToString() : "",
      is_phi() ? " (phi)" : "", has_color() ? StrCat(" @", color()) : "");
}

std::string HloValue::ToString(int indent) const {
  std::string indentation(indent, ' ');
  std::string out =
      StrCat(indentation, ToShortString(), "\n", indentation, " positions:\n");
  for (const HloPosition& position : positions()) {
    StrAppend(&out, indentation, "  ", position.ToString(), "\n");
  }
  if (uses_ != nullptr) {
    StrAppend(&out, indentation, " uses:\n");
    if (GetUses().empty()) {
      StrAppend(&out, indentation, "  (none)\n");
    } else {
      for (const HloUse& use : GetUses()) {
        StrAppend(&out, indentation, "  ", use.ToString(), "\n");
      }
    }
  } else {
    StrAppend(&out, indentation, " uses are not initialized yet.\n");
  }
  StrAppend(&out, indentation, " from instruction: ", instruction()->ToString(),
            "\n");
  return out;
}

namespace {

// Returns true if the instruction 'user' may use the value at the given
// ShapeIndex in the given operand. Generally, instruction which pass through
// values transparently without reading the value are not considered to use the
// value.
bool MayUseOperandValue(const ShapeIndex& index, const HloInstruction* user) {
  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kCopy:
      // These instructions only access the top-level values of their
      // operand. Non-top-level (nested) values are passed through
      // transparently.
      return index.empty();
    case HloOpcode::kAsyncDone:
      return index.empty() || index[0] == 1;
    case HloOpcode::kDomain:
    case HloOpcode::kTuple:
      // These instructions always pass through their operands transparently.
      return false;

    default:
      // Although call (HloOpcode::kCall) and while (HloOpcode::kWhile)
      // instructions pass through their operands as are all other opcode types,
      // they are considered uses.
      return true;
  }
}

// The shared list of a tuple shaped instruction's nested value users
// amortizes one scan of a long user list across the many values nested in
// that instruction, which otherwise each scan it. Below this many users the
// scan is about two loads per user, cheaper than the hash lookup and the one
// time build, so the list is scanned directly; both branches yield the same
// uses in the same order.
constexpr int64_t kMaxUsersToScan = 8;

// Users with at least this many operands get an operand number table in the
// UseCache; narrower users are scanned. Without the table each of the many
// values flowing into a wide root tuple or fusion rescans all of its operands.
constexpr int64_t kMinOperandCountForTable = 32;

}  // namespace

// Filled on first request, from the module as it is at that moment, so it is
// only meaningful while the module is not mutated. The tables are node based,
// so the views handed out below stay valid while later requests add entries.
class HloValue::UseCache {
 public:
  // Returns the users of tuple shaped `instruction`, in users() order, that
  // MayUseOperandValue says may use a value nested inside its output, plus
  // every computation root (roots count as uses of everything they return).
  const absl::InlinedVector<HloInstruction*, 2>& NestedValueUsers(
      const HloInstruction* instruction) {
    auto [it, inserted] = nested_value_users_.try_emplace(instruction);
    if (inserted) {
      // MayUseOperandValue only distinguishes the top level index from nested
      // ones, so any nested index stands for all of them.
      const ShapeIndex nested_index({0});
      for (HloInstruction* const user : instruction->users()) {
        if (MayUseOperandValue(nested_index, user) || user->IsRoot()) {
          it->second.push_back(user);
        }
      }
    }
    return it->second;
  }

  // Returns the ascending operand numbers at which `user` has `operand`.
  absl::Span<const int64_t> OperandNumbers(const HloInstruction* user,
                                           const HloInstruction* operand) {
    auto [it, inserted] = operand_numbers_.try_emplace(user);
    if (inserted) {
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        it->second[user->operand(i)].push_back(i);
      }
    }
    auto numbers_it = it->second.find(operand);
    if (numbers_it == it->second.end()) {
      return {};
    }
    return numbers_it->second;
  }

 private:
  absl::node_hash_map<const HloInstruction*,
                      absl::InlinedVector<HloInstruction*, 2>>
      nested_value_users_;
  absl::node_hash_map<const HloInstruction*,
                      absl::flat_hash_map<const HloInstruction*,
                                          absl::InlinedVector<int64_t, 1>>>
      operand_numbers_;
};

/* static */ void HloValue::PrecomputeUses(absl::Span<HloValue* const> values) {
  UseCache use_cache;
  for (HloValue* value : values) {
    DCHECK(value->uses_ == nullptr) << value->ToShortString();
    value->uses_ = std::make_unique<Uses>(value->ComputeUses(&use_cache));
  }
}

void HloValue::SetPositions(absl::Span<const HloPosition> positions) {
  CHECK_EQ(positions_.size(), 1) << "SetPositions should only be called once.";

  // The positions must be unique and should not contain the defining position
  // as this is added at construction time.
#ifndef NDEBUG
  for (const HloPosition& position_a : positions) {
    DCHECK_NE(position_a, defining_position());
    for (const HloPosition& position_b : positions) {
      if (&position_a != &position_b) {
        DCHECK_NE(position_a, position_b);
      }
    }
  }
#endif  // NDEBUG

  positions_.insert(positions_.end(), positions.begin(), positions.end());
  // Update liveout status of this HloValue.
  live_out_of_module_ |=
      IsRootOf(defining_instruction()->GetModule()->entry_computation());
}

HloValue::Uses HloValue::ComputeUses(UseCache* use_cache) const {
  // Gather the computation roots at which this value appears.
  absl::flat_hash_set<HloInstruction*> root_positions;
  for (const HloPosition& position : positions_) {
    if (position.instruction->IsRoot()) {
      root_positions.insert(position.instruction);
    }
  }

  Uses uses;
  // Build vector of HloUses for the value.
  for (const HloPosition& position : positions_) {
    HloInstruction* const instruction = position.instruction;
    // Appends the uses of this position by `user`, if any.
    auto add_uses_by = [&](HloInstruction* const user) {
#ifndef NDEBUG
      // If user is in the root positions of this value, it must be a root.
      if (root_positions.contains(user)) {
        CHECK(user->IsRoot());
      }
#endif  // NDEBUG
      // Root instructions of computations are considered to be uses whether
      // or not the root instruction itself actually uses the value.
      if (!MayUseOperandValue(position.index, user) &&
          !(user->IsRoot() && root_positions.contains(user))) {
        return;
      }

      if (use_cache != nullptr &&
          user->operand_count() >= kMinOperandCountForTable) {
        for (int64_t operand_number :
             use_cache->OperandNumbers(user, instruction)) {
          uses.emplace_back(user, operand_number, position.index);
        }
        return;
      }

      int i = -1;
      for (const auto& operand : user->operands()) {
        ++i;

        if (operand != instruction) {
          continue;
        }

        uses.emplace_back(user, i, position.index);
#ifndef NDEBUG
        // The new use must not already exist in uses.
        for (int index = 0; index + 1 < uses.size(); ++index) {
          DCHECK_NE(uses[index], uses.back());
        }
#endif  // NDEBUG
      }
      // In case of HloOpcode::kGetTupleElement or HloOpcode::kCopy instruction,
      // ensure that user has at most one operand.
      if (user->opcode() == HloOpcode::kGetTupleElement ||
          user->opcode() == HloOpcode::kCopy) {
        CHECK_LE(i, 0);
      }
    };

    // For a nested value the shared list leaves out the users that cannot
    // pass the test in add_uses_by, so that the values of a wide tuple do not
    // each scan all of its get-tuple-element users. A short user list is
    // scanned directly (see kMaxUsersToScan).
    if (use_cache != nullptr && !position.index.empty() &&
        instruction->user_count() > kMaxUsersToScan) {
      for (HloInstruction* const user :
           use_cache->NestedValueUsers(instruction)) {
        add_uses_by(user);
      }
    } else {
      for (HloInstruction* const user : instruction->users()) {
        add_uses_by(user);
      }
    }
  }
  return uses;
}

bool HloValue::IsRootOf(const HloComputation* computation) const {
  const HloInstruction* root = computation->root_instruction();

  return absl::c_any_of(positions_, [root](const HloPosition& position) {
    return position.instruction == root;
  });
}

std::ostream& operator<<(std::ostream& out, const HloValue& value) {
  out << value.ToShortString();
  return out;
}

HloValueSet::HloValueSet(absl::Span<const HloValue* const> values)
    : values_(values.begin(), values.end()) {
  SortAndUniquifyValues();
}

HloValueSet::HloValueSet(const absl::flat_hash_set<const HloValue*>& values)
    : values_(values.begin(), values.end()) {
  // Values are already unique, so only need to sort.
  absl::c_sort(values_, HloValue::IdLessThan);
}

void HloValueSet::SortAndUniquifyValues() {
  absl::c_sort(values_, HloValue::IdLessThan);
  values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
}

std::string HloValueSet::ToString() const {
  return StrCat("HloValueSet: ",
                absl::StrJoin(values_, ", ",
                              [](std::string* result, const HloValue* value) {
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

bool InstructionValueSet::IsAmbiguous() const {
  bool ambiguous = false;
  for (auto& iter : *this) {
    ambiguous |= iter.second.values().size() > 1;
  }
  return ambiguous;
}

bool InstructionValueSet::AssignUnionOf(
    absl::Span<const InstructionValueSet* const> inputs) {
  CHECK_GT(inputs.size(), 0);
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

bool InstructionValueSet::AssignUnionOf(const InstructionValueSet& input,
                                        ShapeIndexView input_index) {
  bool changed = false;
  for (auto& [index, value_set] : *this) {
    ShapeIndex source_index(input_index);
    for (auto i : index) {
      source_index.push_back(i);
    }
    changed |= value_set.AssignUnionOf({&input.element(source_index)});
  }

  return changed;
}

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set) {
  out << instruction_value_set.ToString();
  return out;
}

std::string InstructionValueSet::ToString() const {
  std::string out =
      StrCat("InstructionValueSet(", ShapeUtil::HumanString(shape()), ")\n");
  ForEachElement([&out](const ShapeIndex& index, const HloValueSet& value_set) {
    StrAppend(&out, "  ", index.ToString(), " : ", value_set.ToString(), "\n");
  });
  return out;
}

}  // namespace xla
