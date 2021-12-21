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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Abstraction which identifies a specific point in the XLA graph. An
// HloPosition specifies a ShapeIndex within the output of a specific
// instruction.
struct HloPosition {
  HloInstruction* instruction;
  ShapeIndex index;

  // Returns the shape at this position.
  const Shape& shape() const;

  std::string ToString() const;

  bool operator==(const HloPosition& other) const {
    return instruction == other.instruction && index == other.index;
  }
  bool operator!=(const HloPosition& other) const { return !(*this == other); }

  // Stable less-than operator using instruction id and index.
  bool operator<(const HloPosition& other) const {
    return instruction->unique_id() < other.instruction->unique_id() ||
           (instruction->unique_id() == other.instruction->unique_id() &&
            index < other.index);
  }

  template <typename H>
  friend H AbslHashValue(H h, const HloPosition& pos) {
    return H::combine(std::move(h), pos.instruction->Hash(), pos.index);
  }
};

std::ostream& operator<<(std::ostream& out, const HloPosition& position);

// Defines a single use of an HLO value.
struct HloUse {
  // Instruction at which the value is used.
  HloInstruction* instruction;

  // The operand number in which the value is appears.
  int64_t operand_number;

  // The shape index within the operand in which the value appears.
  ShapeIndex operand_index;

  std::string ToString() const;

  bool operator==(const HloUse& other) const {
    return instruction == other.instruction &&
           operand_number == other.operand_number &&
           operand_index == other.operand_index;
  }

  bool operator!=(const HloUse& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const HloUse& use) {
    return H::combine(std::move(h), use.instruction, use.operand_index,
                      use.operand_number);
  }
};

std::ostream& operator<<(std::ostream& out, const HloUse& use);

// HloDataflowAnalysis uses this subclass of BufferValue.
class HloValue : public BufferValue {
 public:
  // Predicate comparing HloValues by increasing id, useful for std::sort.
  static bool IdLessThan(const HloValue* a, const HloValue* b) {
    return a->id() < b->id();
  }

  // Predicate comparing HloValues by equal id, useful for std::unique.
  static bool IdEqual(const HloValue* a, const HloValue* b) {
    return a->id() == b->id();
  }

  // Construct an HloValue defined by 'instruction' at shape index 'index'. If
  // is_phi is true, then this value is a phi value, for example, at the
  // parameter of a while body computation. Phi values are only used in the SSA
  // dataflow analysis (HloDataflowAnalysis::ssa_form_ is true).
  HloValue(Id id, HloInstruction* instruction, const ShapeIndex& index,
           bool is_phi = false);
  ~HloValue() override {}

  // Sets the positions in the module at which the HloValue appears. Updates
  // uses. Should be called once and only once. The defining position should not
  // be included in 'positions' as this is set at construction time.
  void SetPositionsAndComputeUses(absl::Span<const HloPosition> positions);

  // Returns whether this value is a phi value.
  bool is_phi() const { return is_phi_; }

  // Return the position where this value is defined.
  const HloPosition& defining_position() const { return positions_[0]; }

  // Return the instruction which defines this HloValue.
  HloInstruction* defining_instruction() const {
    return defining_position().instruction;
  }

  HloInstruction* instruction() const override {
    return defining_instruction();
  }

  // Return the shape index at which this HloValue is defined in the output of
  // its defining instruction.
  const ShapeIndex& defining_index() const { return defining_position().index; }

  const ShapeIndex& index() const override { return defining_index(); }

  // Return the shape of this HloValue.
  const Shape& shape() const override { return defining_position().shape(); }

  // Return all positions of the HloValue in the module.
  const std::vector<HloPosition>& positions() const { return positions_; }

  // Return all uses of the HloValue.
  const std::vector<HloUse>& uses() const { return uses_; }

  // Get whether this HloValue is live out of the module.
  bool live_out_of_module() const { return live_out_of_module_; }

  bool operator==(const HloValue& other) const;
  bool operator!=(const HloValue& other) const;

  // Return a single-line string representation of the value.
  std::string ToShortString() const;

  std::string ToString(int indent) const;

  std::string ToString() const override { return ToString(0); }

 private:
  // Whether this instruction is a phi value.
  const bool is_phi_;

  // The set of positions of this HloValue. The first element is always the
  // position of the definition.
  std::vector<HloPosition> positions_;

  // The set of uses of this HloValue.
  std::vector<HloUse> uses_;

  // Whether this value is live out of the HLO module.
  bool live_out_of_module_ = false;
};

std::ostream& operator<<(std::ostream& out, const HloValue& hlo_value);

// A class representing the possible set of HloValues at a particular point
// (shape index in the output of an instruction) in the XLA graph. This set
// contains the set of reaching HloValue definitions. For a simple array-shaped
// instruction like Add, the HloValueSet of the top-level of the instruction's
// output trivially contains only the HloValue defined by the instruction. For
// instructions which have non-trivial dataflow such as Tuple or Select, the
// HloValueSets of the instruction's output contains one or more HloValues
// defined by the instruction's operands or defined further up in the XLA graph.
class HloValueSet {
 public:
  HloValueSet() = default;

  explicit HloValueSet(absl::Span<const HloValue* const> values)
      : values_(values.begin(), values.end()) {
    SortAndUniquifyValues();
  }

  // Sets this value set to the union of the given value sets. Returns whether
  // this value set changed.
  bool AssignUnionOf(absl::Span<const HloValueSet* const> inputs);

  // Return the vector of HloValues in the set. Values in the vector are unique
  // and stably sorted by value id.
  const std::vector<const HloValue*>& values() const { return values_; }

  // Adds the value to the set.  Returns true iff the value was added and didn't
  // already exist in the set.
  bool AddValue(const HloValue* value);

  // Clear all values from the set.
  void Clear() { values_.clear(); }

  // Return the unique HLO value in the set. CHECKs if the set does not contain
  // exactly one value.
  const HloValue& GetUniqueValue() const {
    CHECK_EQ(values_.size(), 1);
    return *values_[0];
  }

  bool operator==(const HloValueSet& other) const {
    if (values_.size() != other.values_.size()) return false;
    for (size_t i = 0; i < values_.size(); ++i) {
      if (values_[i]->id() != other.values_[i]->id()) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const HloValueSet& other) const { return !(*this == other); }

  std::string ToString() const;

 private:
  // Sorts value_ and removes duplicates. This should be called after adding any
  // elements to values_.
  void SortAndUniquifyValues();

  // HloValues sorted by HloValue::Id.
  std::vector<const HloValue*> values_;
};

std::ostream& operator<<(std::ostream& out, const HloValueSet& hlo_value);

// A class collecting the HloValues which might be contained in the output of
// an HLO instruction. For array-shaped instructions, an InstructionValueSet
// trivially holds a single HloValueSet. Tuple-shaped InstructionValueSets
// hold multiple HloValueSets.
class InstructionValueSet : public ShapeTree<HloValueSet> {
 public:
  explicit InstructionValueSet(const Shape& shape)
      : ShapeTree<HloValueSet>(shape) {}

  // Sets this value set to the union of the given value sets. Returns whether
  // this value set changed.
  bool AssignUnionOf(absl::Span<const InstructionValueSet* const> inputs);

  // Returns true if any value sets for any subshape element is not a
  // singleton.
  bool IsAmbiguous() const;

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_H_
