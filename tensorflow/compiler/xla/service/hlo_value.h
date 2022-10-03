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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"

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

  // Sort by instruction ID, then index.
  bool operator<(const HloPosition& other) const {
    return std::forward_as_tuple(instruction->unique_id(), index) <
           std::forward_as_tuple(other.instruction->unique_id(), other.index);
  }

  template <typename H>
  friend H AbslHashValue(H h, const HloPosition& pos) {
    return H::combine(std::move(h), *pos.instruction, pos.index);
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

  // Construct an HloValue defined by 'instruction' at shape index 'index'. If
  // is_phi is true, then this value is a phi value, for example, at the
  // parameter of a while body computation. Phi values are only used in the SSA
  // dataflow analysis (HloDataflowAnalysis::ssa_form_ is true).
  HloValue(Id id, HloInstruction* instruction, const ShapeIndex& index,
           bool is_phi = false);

  // Sets the positions in the module at which the HloValue appears. Should be
  // called once and only once. The defining position should not be included in
  // 'positions' as this is set at construction time.
  void SetPositions(absl::Span<const HloPosition> positions);

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

  // Return all uses of the HloValue. This computes the uses lazily, and the
  // overhead could be non-trivial for the first invocation. Therefore even
  // though it is marked `const`, it actually can mutate its data members. It is
  // kept this way to allow passing around const references.
  absl::Span<const HloUse> GetUses() const {
    return uses_.MaybeInitAndGet(
        [this](std::vector<HloUse>& uses) { ComputeUses(uses); });
  }

  // Returns true if this has a position that is the root of the given
  // computation.
  bool IsRootOf(const HloComputation* computation) const;

  // Get whether this HloValue is live out of the module.
  bool live_out_of_module() const { return live_out_of_module_; }

  bool operator==(const HloValue& other) const { return this == &other; }
  bool operator!=(const HloValue& other) const { return !(*this == other); }

  // Return a single-line string representation of the value.
  std::string ToShortString() const;
  std::string ToString(int indent) const;
  std::string ToString() const override { return ToString(0); }

 private:
  template <typename T>
  class Lazy {
   public:
    Lazy() = default;
    const T& MaybeInitAndGet(absl::FunctionRef<void(T&)> func) const {
      if (!initialized_) {
        func(uses_);
        initialized_ = true;
      }
      return uses_;
    }

   private:
    mutable T uses_;
    mutable bool initialized_ = false;
  };
  // Called when lazily computing the uses.
  void ComputeUses(std::vector<HloUse>& uses) const;

  // The set of positions of this HloValue. The first element is always the
  // position of the definition.
  std::vector<HloPosition> positions_;

  // The set of uses of this HloValue. This is lazily constructed until getting
  // accessed.
  Lazy<std::vector<HloUse>> uses_;

  // Whether this instruction is a phi value.
  const bool is_phi_;

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

  explicit HloValueSet(absl::Span<const HloValue* const> values);
  explicit HloValueSet(const absl::flat_hash_set<const HloValue*>& values);

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

  std::vector<const HloValue*> TakeValues() { return std::move(values_); }

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

  // Sets this value set to the input value set at the given index. Returns
  // whether this value set changed.
  bool AssignUnionOf(const InstructionValueSet& input,
                     ShapeIndexView input_index);

  // Returns true if any value sets for any subshape element is not a
  // singleton.
  bool IsAmbiguous() const;

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_H_
