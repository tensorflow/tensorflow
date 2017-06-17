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

// Analysis for determining the possible set of values for all locations
// (instructions and ShapeIndexes) in the HLO module. Analysis is module-scoped
// tracking values across computation boundaries.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Abstraction which identifies a specific point in the XLA graph. An
// HloLocation specifies a ShapeIndex within the output of a specific
// instruction.
struct HloLocation {
  HloInstruction* instruction;
  ShapeIndex index;

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
  HloValue(HloValue::Id id, HloInstruction* instruction,
           const ShapeIndex& index, bool is_phi = false);

  // Return a unique identifier for this HloValue. This value is used for stable
  // sorting and iteration
  Id id() const { return id_; }

  // Returns whether this value is a phi value.
  bool is_phi() const { return is_phi_; }

  // Return the location where this value is defined.
  const HloLocation& DefinitionLocation() const { return locations_[0]; }

  // Return the instruction which defines this HloValue.
  HloInstruction* instruction() const {
    return DefinitionLocation().instruction;
  }

  // Return the shape index at which this HloValue is defined in the output of
  // instruction().
  const ShapeIndex& index() const { return DefinitionLocation().index; }

  // Add or remove a location at which the HloValue appears. The definition
  // location can not be removed. The uses of the HloValue are updated.
  void AddLocation(HloInstruction* instruction, const ShapeIndex& index);
  void RemoveLocation(HloInstruction* instruction, const ShapeIndex& index);

  // Return all locations of the HloValue in the module.
  const std::vector<HloLocation>& locations() const { return locations_; }

  // Return all uses of the HloValue.
  const std::vector<HloUse>& uses() const { return uses_; }

  // Set/get whether this HloValue is live out of the module.
  bool live_out_of_module() const { return live_out_of_module_; }

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

// Analysis which identifies all HLO values and their uses in an HLO module.
class HloDataflowAnalysis {
 public:
  // Run dataflow analysis on the given module. Parameters:
  //
  //   ssa_form : If true then new values are defined at the merge points of
  //     kWhile instructions. Abusing nomenclature somewhat, we call these "phi
  //     values".  The merge is formed by the init value and loop backedge. The
  //     SSA form is minimal in that a new phi value is defined only if the
  //     merge point is reachable by multiple different values. The SSA form is
  //     also in loop-closed form in that no values defined inside of a loop
  //     (while body) is used outside of the loop.
  //
  //     If ssa_form is false, then merge points do not define new
  //     values. Rather, the HloValueSet for the merge point contains the union
  //     of the merged HloValues.
  //
  //   bitcast_defines_value : If true then the Bitcast HLO instruction defines
  //     a new HLO value in the analysis. If false then Bitcast forwards the
  //     value of its operand.
  static StatusOr<std::unique_ptr<HloDataflowAnalysis>> Run(
      HloModule* module, bool ssa_form = false,
      bool bitcast_defines_value = false);

  // Returns true if 'instruction' defines an HLO value at the given shape index
  // of its output.
  bool ValueIsDefinedAt(const HloInstruction* instruction,
                        const ShapeIndex& index = {}) const;

  // Return the HloValue defined by 'instruction' at the given shape index of
  // its output.
  //
  // Precondition: ValueIsDefinedAt is true for this instruction and index.
  const HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                                    const ShapeIndex& index = {}) const;
  HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                              const ShapeIndex& index = {});

  // Return the InstructionValueSet for the given instruction.
  const InstructionValueSet& GetInstructionValueSet(
      const HloInstruction* instruction) const;
  InstructionValueSet& GetInstructionValueSet(
      const HloInstruction* instruction);

  // Return the HloValueSet for the given instruction at the given index.
  const HloValueSet& GetValueSet(const HloInstruction* instruction,
                                 const ShapeIndex& index = {}) const;
  HloValueSet& GetValueSet(const HloInstruction* instruction,
                           const ShapeIndex& index = {});

  // Return the unique value in the HloValueSet at the given instruction and
  // shape index. CHECKs if the value set does not contain a exactly one value.
  const HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                                   const ShapeIndex& index = {}) const {
    return GetValue(GetValueSet(instruction, index).GetUniqueValueId());
  }
  HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                             const ShapeIndex& index = {}) {
    return GetValue(GetValueSet(instruction, index).GetUniqueValueId());
  }

  // Return the HloValue with the given Id.
  const HloValue& GetValue(HloValue::Id value_id) const;
  HloValue& GetValue(HloValue::Id value_id);

  // Return the total number of HloValues.
  int64 value_count() const { return values_.size(); }

  // Return a vector of all HloValues stabily sorted by HloValue::Id. This
  // vector is lazily computed. Mutating operations on HloDataflowAnalysis may
  // invalidate the underlying vector requiring recomputation.
  const std::vector<const HloValue*>& values() const;

  string ToString() const;

 protected:
  HloDataflowAnalysis(HloModule* module, bool ssa_form,
                      bool bitcast_defines_value = false);

  // Creates a new HloValue defined at the given instruction and shape index and
  // return its ID.
  HloValue::Id NewHloValue(HloInstruction* instruction, const ShapeIndex& index,
                           bool is_phi = false);

  // Delete the HloValue with the given ID.
  void DeleteHloValue(HloValue::Id value_id);

  // Constructs and initializes the InstructionValueSets of all instructions to
  // contain exactly the HloValues defined by each instruction. These values can
  // then propagated throughout the HLO graph by calling
  // UpdateInstructionsAndPropagate.
  Status InitializeInstructionValueSets();

  // Updates the value set of the given instruction based on the values flowing
  // into the instruction (operands and cross-computation dataflow).
  void UpdateInstructionValueSet(HloInstruction* instruction);

  // Recomputes and returns the value set for the given parameter instruction.
  InstructionValueSet RecomputeBitcastValueSet(HloInstruction* bitcast);
  InstructionValueSet RecomputeCopyValueSet(HloInstruction* copy);
  InstructionValueSet RecomputeGetTupleElementValueSet(HloInstruction* gte);
  InstructionValueSet RecomputeParameterValueSet(HloInstruction* parameter);
  InstructionValueSet RecomputeSelectValueSet(HloInstruction* select);
  InstructionValueSet RecomputeTupleValueSet(HloInstruction* tuple);
  InstructionValueSet RecomputeWhileValueSet(HloInstruction* xla_while);

  // Update the value sets of the given instructions and propagate the
  // changes to fixed point.
  void UpdateInstructionsAndPropagate(
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions);

  // Return the result of the SSA Phi function applied to the given inputs at
  // the given instruction. If skip_top_level is true, then the top level of the
  // value set of 'instruction' is not modified.
  InstructionValueSet Phi(
      HloInstruction* instruction,
      tensorflow::gtl::ArraySlice<const InstructionValueSet*> inputs,
      bool skip_top_level = false);

  // Updates the locations of the HloValues in the output of the given
  // instruction. This should be called after the instruction value set of
  // 'instruction' has been changed. 'prev_value_set' must point to the previous
  // state of the value set prior to the change. 'prev_value_set' may be null if
  // this is the first time locations are being computed. The previous state is
  // necessary to efficiently remove locations which have been eliminated due to
  // changes in the instructions' InstructionValueSet.
  void UpdateLocationsOfValuesAt(
      HloInstruction* instruction, const InstructionValueSet& new_value_set,
      const InstructionValueSet* prev_value_set = nullptr);

  HloModule* const module_;
  const bool ssa_form_;
  const bool bitcast_defines_value_;

  std::unique_ptr<CallGraph> call_graph_;

  // The map of all HloValues in the module.
  std::unordered_map<HloValue::Id, HloValue> values_;

  // A map from instruction to InstructionValueSet.
  std::unordered_map<const HloInstruction*, InstructionValueSet> value_sets_;

  // A lazily constructed vector containing all HloValues sorted by
  // HloValue::Id.
  mutable std::vector<const HloValue*> values_vector_;

  // The Id to use for the next HloValue.
  HloValue::Id next_value_id_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_
