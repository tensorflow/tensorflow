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

// Analysis for determining the possible set of values for all positions
// (instructions and ShapeIndexes) in the HLO module. Analysis is module-scoped
// tracking values across computation boundaries.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

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

  // Returns whether the given values interfere assuming the given HLO
  // ordering. Two values interfere if they may both be simultaneously live.
  bool MayInterfere(const HloValue& a, const HloValue& b,
                    const HloOrdering& ordering) const;

  // Overload which takes HloValue:Ids.
  bool MayInterfere(HloValue::Id a, HloValue::Id b,
                    const HloOrdering& ordering) const {
    return MayInterfere(GetValue(a), GetValue(b), ordering);
  }

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

  // Updates the positions of the HloValues in the output of the given
  // instruction. This should be called after the instruction value set of
  // 'instruction' has been changed. 'prev_value_set' must point to the previous
  // state of the value set prior to the change. 'prev_value_set' may be null if
  // this is the first time positions are being computed. The previous state is
  // necessary to efficiently remove positions which have been eliminated due to
  // changes in the instructions' InstructionValueSet.
  void UpdatePositionsOfValuesAt(
      HloInstruction* instruction, const InstructionValueSet& new_value_set,
      const InstructionValueSet* prev_value_set = nullptr);

  // Returns true if the live range of the given value 'a' is strictly before
  // the live range of value 'b' using the given HLO ordering.
  bool LiveRangeStrictlyBefore(const HloValue& a, const HloValue& b,
                               const HloOrdering& ordering) const;

  // Returns whether the value 'a' is defined before the value 'b' under the
  // given ordering.
  bool IsDefinedBefore(const HloValue& a, const HloValue& b,
                       const HloOrdering& ordering) const;

  // Returns whether the given use is before the given value definition.
  bool UseIsBeforeValueDefinition(const HloUse& use, const HloValue& value,
                                  const HloOrdering& ordering) const;

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
