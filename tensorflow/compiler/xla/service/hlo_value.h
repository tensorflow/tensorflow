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

#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Abstraction which identifies a specific point in the XLA graph. An
// HloLocation specifies a ShapeIndex within the output of a specific
// instruction.
struct HloLocation {
  HloInstruction* instruction;
  ShapeIndex index;

  // Returns the shape at this location.
  const Shape& shape() const;

  string ToString() const;

  bool operator==(const HloLocation& other) const {
    return instruction == other.instruction && index == other.index;
  }
  bool operator!=(const HloLocation& other) const { return !(*this == other); }
};

std::ostream& operator<<(std::ostream& out, const HloLocation& location);

// Defines a single use of an HLO value.
struct HloUse {
  // Instruction at which the value is used.
  HloInstruction* instruction;

  // The operand number in which the value is appears.
  int64 operand_number;

  // The shape index within the operand in which the value appears.
  ShapeIndex operand_index;

  string ToString() const;

  bool operator==(const HloUse& other) const {
    return instruction == other.instruction &&
           operand_number == other.operand_number &&
           operand_index == other.operand_index;
  }

  bool operator!=(const HloUse& other) const { return !(*this == other); }
};

std::ostream& operator<<(std::ostream& out, const HloUse& use);

// Class describing a value used by the dataflow analysis. XLA arrays are
// trivially a single HloValue. Tuples are made up of more than one HloValue: an
// HloValue for the pointer vector, and an HloValue for each child element.
//
// Every HloValue is defined by a particular instruction and most instructions
// define only a single HloValue. Instructions which define a single HloValue
// include array-shaped instructions such as Add but also includes Tuple-shaped
// instructions such as Tuple. The Tuple instruction defines a single HloValue
// which is a vector of pointers to the values containing the Tuple
// instruction's operands. Though the result of the Tuple instruction includes
// multiple values only the top-level HloValue (the vector of pointers) is
// defined by the Tuple instruction. The values containing the tuple elements
// are defined by earlier instructions, usually the operands of the Tuple
// instruction.
//
// Instructions which construct both the tuple *and* the tuple elements define
// more than one HloValue. This includes (at least) tuple-shaped Constant,
// Parameter, Infeed and While instructions. These tuple-shaped instructions do
// not assemble a tuple from existing HloValues like the Tuple instruction does,
// but rather define all the HloValues in the tuple.
class HloValue {
 public:
  using Id = int64;

  // Construct an HloValue defined by 'instruction' at shape index 'index'. If
  // is_phi is true, then this value is a phi value, for example, at the
  // parameter of a while body computation. Phi values are only used in the SSA
  // dataflow analysis (HloDataflowAnalysis::ssa_form_ is true).
  HloValue(Id id, HloInstruction* instruction, const ShapeIndex& index,
           bool is_phi = false);

  // Return a unique identifier for this HloValue. This value is used for stable
  // sorting and iteration
  Id id() const { return id_; }

  // Returns whether this value is a phi value.
  bool is_phi() const { return is_phi_; }

  // Return the location where this value is defined.
  const HloLocation& defining_location() const { return locations_[0]; }

  // Return the instruction which defines this HloValue.
  HloInstruction* defining_instruction() const {
    return defining_location().instruction;
  }

  // Return the shape index at which this HloValue is defined in the output of
  // its defining instruction.
  const ShapeIndex& defining_index() const { return defining_location().index; }

  // Return the shape of this HloValue.
  const Shape& shape() const { return defining_location().shape(); }

  // Add or remove a location at which the HloValue appears. The definition
  // location can not be removed. The uses of the HloValue are updated.
  void AddLocation(HloInstruction* instruction, const ShapeIndex& index);
  void RemoveLocation(HloInstruction* instruction, const ShapeIndex& index);

  // Return all locations of the HloValue in the module.
  const std::vector<HloLocation>& locations() const { return locations_; }

  // Return all uses of the HloValue.
  const std::vector<HloUse>& uses() const { return uses_; }

  // Get whether this HloValue is live out of the module.
  bool live_out_of_module() const { return live_out_of_module_; }

  // Get whether this HloValue is live out of the computation it is defined in.
  bool live_out_of_computation() const { return live_out_of_computation_; }

  bool operator==(const HloValue& other) const;
  bool operator!=(const HloValue& other) const;

  // Return a single-line string representation of the value.
  string ToShortString() const;

  string ToString(int indent = 0) const;

 private:
  // Unique identifier for this HloValue. Used for stable sorting and iteration.
  const Id id_;

  // Whether this instruction is a phi value.
  const bool is_phi_;

  // The set of locations of this HloValue. The first element is always the
  // location of the definition.
  std::vector<HloLocation> locations_;

  // The set of uses of this HloValue.
  std::vector<HloUse> uses_;

  // Whether this value is live out of the HLO module.
  bool live_out_of_module_ = false;

  // Whether this value is live out of its computation.
  bool live_out_of_computation_ = false;
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

  explicit HloValueSet(tensorflow::gtl::ArraySlice<HloValue::Id> value_ids)
      : value_ids_(value_ids.begin(), value_ids.end()) {
    SortAndUniquifyValues();
  }

  // Return the union of the given HloValueSets.
  static HloValueSet Union(
      tensorflow::gtl::ArraySlice<const HloValueSet*> inputs);

  // Return the vector of the IDs of all HloValues in the set. Values in the
  // vector are unique and sorted.
  const std::vector<HloValue::Id>& value_ids() const { return value_ids_; }

  // Return the unique HLO value in the set. CHECKs if the set does not contain
  // exactly one value.
  HloValue::Id GetUniqueValueId() const {
    CHECK_EQ(value_ids().size(), 1);
    return value_ids()[0];
  }

  bool operator==(const HloValueSet& other) const {
    return value_ids() == other.value_ids();
  }
  bool operator!=(const HloValueSet& other) const { return !(*this == other); }

  string ToString() const;

 private:
  // Sorts value_ and removes duplicates. This should be called after adding any
  // elements to values_.
  void SortAndUniquifyValues();

  // HloValues sorted by HloValue::Id.
  std::vector<HloValue::Id> value_ids_;
};

std::ostream& operator<<(std::ostream& out, const HloValueSet& hlo_value);

// A class collecting the HloValues which might be contained in the output of
// an HLO instruction. For array-shaped instructions, an InstructionValueSet
// trivially holds a single HloValueSet. Tuple-shaped InstructionValueSets
// hold multiple HloValueSets.
class InstructionValueSet : public ShapeTree<HloValueSet> {
 public:
  InstructionValueSet(const Shape& shape) : ShapeTree<HloValueSet>(shape) {}

  // Return the union of the given InstructionValueSets.
  static InstructionValueSet Union(
      tensorflow::gtl::ArraySlice<const InstructionValueSet*> inputs);

  string ToString() const;
};

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_H_
