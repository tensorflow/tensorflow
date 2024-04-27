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

// Analysis for determining the possible set of values for all positions
// (instructions and ShapeIndexes) in the HLO module. Analysis is module-scoped
// tracking values across computation boundaries.

#ifndef XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_
#define XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_phi_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Identifies one array input of an HloInstruction.
struct HloOperandIndex {
  // The operand number in which the array value appears.
  int64_t operand_number;

  // The shape index within the operand in which the array value appears.
  ShapeIndex operand_index;

  bool operator==(const HloOperandIndex& other) const {
    return operand_number == other.operand_number &&
           operand_index == other.operand_index;
  }

  bool operator!=(const HloOperandIndex& other) const {
    return !(*this == other);
  }
};

// Analysis which identifies all HLO values and their uses in an HLO module.
class HloDataflowAnalysis {
 public:
  // Infrastructure for passing may-alias hints: HLO passes can populate the
  // may-alias table. If an empty optional is returned, default rules are used.
  //
  // Must-alias rules (as defined by GetInPlaceInputOutputPairs) cannot be
  // overriden using backend-specific overrides.
  //
  // The first parameter of the function should be the instruction, the
  // second parameter should be an operand of the instruction. The third
  // parameter should be the output index of the instruction.
  using CanShareBuffer = std::function<std::optional<bool>(
      const HloInstruction* instr, const HloInstruction* operand,
      const ShapeIndex& user_index)>;

  // Infrastructure for overriding whether an instruction defines a new value.
  //
  // The first parameter is the instruction and the second parameter is the
  // output index. If an empty optional is used, default rules are used. If a
  // ForwardedOperand object is returned, the value at the corresponding
  // operand's index is used for the output, overriding all default logic.
  struct ForwardedOperand {
    int64_t operand_number;
    ShapeIndex operand_index;
  };
  using ForwardsValue = std::function<std::optional<ForwardedOperand>(
      const HloInstruction* instr, const ShapeIndex& index)>;

  // Runs dataflow analysis on the given module. Parameters:
  //
  //   ssa_form : If true then new values are defined at the merge points of
  //     kWhile instructions. Abusing nomenclature somewhat, we call these "phi
  //     values".  The merge is formed by the init value and loop backedge. The
  //     SSA form is minimal in that a new phi value is defined only if the
  //     merge point is reachable by multiple different values. The SSA form is
  //     also in loop-closed form in that no values defined inside of a loop
  //     (while body) is used outside of the loop. Example use of this ssa_form
  //     mode is to reason about live range interference of buffers.
  //
  //     If ssa_form is false, then merge points do not define new
  //     values. Rather, the HloValueSet for the merge point contains the union
  //     of the merged HloValues.
  //
  //   bitcast_defines_value : If true then the Bitcast HLO instruction defines
  //     a new HLO value in the analysis. If false then Bitcast forwards the
  //     value of its operand.
  static absl::StatusOr<std::unique_ptr<HloDataflowAnalysis>> Run(
      const HloModule& module, bool ssa_form = false,
      bool bitcast_defines_value = false,
      const CanShareBuffer& can_share_buffer = nullptr,
      const ForwardsValue& forwards_value = nullptr,
      absl::flat_hash_set<absl::string_view> execution_threads = {});

  // Returns true if 'instruction' defines an HLO value at the given shape index
  // of its output.
  bool ValueIsDefinedAt(const HloInstruction* instruction,
                        const ShapeIndex& index = {}) const;

  // Returns the HloValue defined by 'instruction' at the given shape index of
  // its output.
  //
  // Precondition: ValueIsDefinedAt is true for this instruction and index.
  const HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                                    const ShapeIndex& index = {}) const;
  HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                              const ShapeIndex& index = {});

  // Returns the InstructionValueSet for the given instruction.
  const InstructionValueSet& GetInstructionValueSet(
      const HloInstruction* instruction) const;
  InstructionValueSet& GetInstructionValueSet(
      const HloInstruction* instruction);

  // Returns all values that are contained in the output of this instruction in
  // a flattened set.
  HloValueSet GetFlattenedValueSet(const HloInstruction* instruction) const;

  // Returns the HloValueSet for the given instruction at the given index or the
  // given position.
  const HloValueSet& GetValueSet(const HloInstruction* instruction,
                                 const ShapeIndex& index = {}) const;
  const HloValueSet& GetValueSet(const HloPosition& position) const;
  HloValueSet& GetValueSet(const HloPosition& position);
  HloValueSet& GetValueSet(const HloInstruction* instruction,
                           const ShapeIndex& index = {});

  // Returns the unique value in the HloValueSet at the given instruction and
  // shape index. CHECKs if the value set does not contain a exactly one value.
  const HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                                   const ShapeIndex& index = {}) const {
    return GetValueSet(instruction, index).GetUniqueValue();
  }
  HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                             const ShapeIndex& index = {}) {
    return GetValue(GetValueSet(instruction, index).GetUniqueValue().id());
  }

  // Returns the HloValue with the given Id.
  const HloValue& GetValue(HloValue::Id value_id) const;
  HloValue& GetValue(HloValue::Id value_id);

  // Returns the total number of HloValues.
  int64_t value_count() const { return values_.size(); }

  // Returns a vector of all HloValues stabily sorted by HloValue::Id.
  const std::vector<HloValue*>& values() const { return values_vector_; }

  // Returns the call graph used for computing the dataflow.
  const CallGraph& call_graph() const { return *call_graph_; }

  std::string ToString() const;

  // Returns true if 'user' cannot possibly use the buffer at 'index' in
  // 'operand'. Returns false otherwise.
  //
  // 'operand' does not have to be an operand of 'user'. This can be the
  // case with indirect uses.
  bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                               const ShapeIndex& index,
                               const HloInstruction* user) const;

  // Returns true if 'user' (at 'user_index') can share a buffer with its
  // operand 'operand' (at 'operand_index'). Returns false otherwise.
  //
  // REQUIRES: 'operand' is an operand of 'user'.
  bool CanShareOperandBufferWithUser(HloInstruction* operand,
                                     const ShapeIndex& operand_index,
                                     HloInstruction* user,
                                     const ShapeIndex& user_index) const;

  const HloModule& module() const { return module_; }

  // Returns true if the operation is an in-place operation and its operand 0
  // must alias with the output.
  static bool IsInPlaceOperation(HloOpcode opcode);

  // Returns true if the operation is the start/done of an asynchronous
  // operation, where the buffer used/produced by the op needs to stay alive
  // until the asynchronous operation completes.
  static bool IsAsynchronousOperationStart(HloOpcode opcode);
  static bool IsAsynchronousOperationDone(HloOpcode opcode);

  // Returns the pairs of inputs and outputs that must share the same buffer,
  // according to the aliasing rules for that instruction.
  //
  // This function only considers array values as inputs and outputs, so
  // when tuples are present it "sees through" to the array values inside. The
  // HloUse describing the input parameter contains not only the operand number
  // but also a shape index describing its position inside a nested tuple shape
  // (if any). Similarly, the output parameter is described by a shape index
  // into the nested tuple shape (if any) of the output value.
  //
  // For example, for this hypothetical op:
  //   %foo = (f32[1], (f32[2], f32[3]))
  //              op((f32[4], f32[5]) %arg0, f32[6] %arg1)
  //
  // ... the results can include any of the 3 * 3 = 9 possible pairs of
  // input and output arrays.
  static std::vector<std::pair<HloOperandIndex, ShapeIndex>>
  GetInPlaceInputOutputPairs(const HloInstruction* instruction);

  // Verifies various invariants of the dataflow analysis.
  Status Verify() const;

 private:
  static bool AreTransitiveUsesElementwiseOrTuple(const HloInstruction* inst);

  HloDataflowAnalysis(const HloModule& module, bool ssa_form,
                      bool bitcast_defines_value,
                      const CanShareBuffer& can_share_buffer,
                      const ForwardsValue& forwards_value,
                      absl::flat_hash_set<absl::string_view> execution_threads);

  // 1. During value propagation (Propagate function), always create phi
  // values once it see multiple inputs merging at the same point. It then
  // records those phi values as well as their inputs in a phi graph.
  //
  // 2. Post value propagation, Dataflow analysis can then do certain
  // optimization(OptimizePhiValues) on the phi graph to prune uncessary phi
  // nodes.
  //
  // Note that this applies in SSA form, and Both of the functions are
  // guaranteed to exit.
  //
  void OptimizePhiValues();

  // Returns a new HloValue defined at the given instruction and shape index.
  HloValue* NewHloValue(HloInstruction* instruction, const ShapeIndex& index,
                        bool is_phi);

  // Marks the HloValue with the given ID for deletion.
  void MarkValueForDeletion(HloValue::Id value_id);

  // Deletes all HloValues marked for deletion. Should be called after
  // propagation is complete.
  void DeleteMarkedValues();

  // Constructs and initializes the InstructionValueSets of all instructions to
  // contain exactly the HloValues defined by each instruction. These values can
  // then propagated throughout the HLO graph by calling Propagate.
  Status InitializeInstructionValueSets();

  // Updates the value set of the given instruction based on the values flowing
  // into the instruction (operands and cross-computation dataflow).
  bool UpdateInstructionValueSet(HloInstruction* instruction);

  // Updates the value set for a particular instruction type. Returns whether
  // the instruction value set changed.
  bool UpdateBitcastValueSet(HloInstruction* bitcast);
  bool UpdateCallValueSet(HloInstruction* call);
  bool UpdateConditionalValueSet(HloInstruction* conditional);
  bool UpdateCopyValueSet(HloInstruction* copy);
  bool UpdateCustomCallValueSet(HloInstruction* custom_call);
  bool UpdateDomainValueSet(HloInstruction* domain);
  bool UpdateGetTupleElementValueSet(HloInstruction* gte);
  bool UpdateParameterValueSet(HloInstruction* parameter);
  // Async op propagation rules:
  //  - Operand of async-start to parameter of async wrapped computation and at
  //    index {0, operand_number} of async-start and async-update outputs.
  //  - Root of async wrapped computation to index {1} of async-start and
  //    async-update and index {} of async-done.
  //  - The contexts in indices {2+} of async-start to the same indices of
  //    async-update.
  //
  // As a result of this, the operands/outputs of async-start and async-done
  // instructions share the same values as the parameters/roots of the async
  // wrapped computation.
  bool UpdateAsyncStartValueSet(HloInstruction* async_start);
  bool UpdateAsyncUpdateValueSet(HloInstruction* async_update);
  bool UpdateAsyncDoneValueSet(HloInstruction* async_done);
  bool UpdateCopyStartValueSet(HloInstruction* copy_start);
  bool UpdateCopyDoneValueSet(HloInstruction* copy_done);
  bool UpdateOptimizationBarrierValueSet(HloInstruction* barrier);
  bool UpdateRecvDoneValueSet(HloInstruction* recv_done);
  bool UpdateSendValueSet(HloInstruction* send);
  bool UpdateTupleValueSet(HloInstruction* tuple);
  bool UpdateWhileValueSet(HloInstruction* xla_while);
  bool UpdateAddDependencyValueSet(HloInstruction* add_dependency);
  bool UpdateAllGatherStartValueSet(HloInstruction* all_gather_start);
  bool UpdateAllGatherDoneValueSet(HloInstruction* all_gather_done);
  bool UpdateAllReduceDoneValueSet(HloInstruction* all_reduce_done);
  bool UpdateCollectivePermuteStartValueSet(
      HloInstruction* collective_permute_start);
  bool UpdateCollectivePermuteDoneValueSet(
      HloInstruction* collective_permute_done);

  // Propagates the dataflow through the module. In particular, it propagates
  // the HloValueSet from its defining instruction to the users of the
  // instructions.
  void Propagate();

  // Returns the result of the SSA Phi function applied to the given inputs at
  // the given instruction.
  bool Phi(HloInstruction* instruction,
           absl::Span<const InstructionValueSet* const> inputs);

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

  const HloModule& module_;
  const absl::flat_hash_set<absl::string_view> execution_threads_;
  const bool ssa_form_;
  const bool bitcast_defines_value_;

  std::unique_ptr<CallGraph> call_graph_;

  // The map of all HloValues in the module. We pass around pointers to the
  // mapped HloValues, so the underlying container must keep them valid despite
  // mutations touching other map entries.
  absl::flat_hash_map<HloValue::Id, std::unique_ptr<HloValue>> values_;

  // A map from instruction to InstructionValueSet.
  absl::flat_hash_map<const HloInstruction*,
                      std::unique_ptr<InstructionValueSet>>
      value_sets_;

  // Values marked for deletion during construction. We don't delete them
  // immediately because references to them may remain in ValueSets temporarily
  // during propagation. After construction, these values are deleted.
  std::vector<HloValue::Id> value_ids_to_delete_;

  // A vector containing all HloValues sorted by HloValue::Id.
  std::vector<HloValue*> values_vector_;

  // The Id to use for the next HloValue.
  HloValue::Id next_value_id_ = 0;

  // An explicit graph holding phi values and edges.
  PhiGraph phi_graph_;

  // Backend specific function that decides whether an instruction can share
  // a buffer with its operand.
  CanShareBuffer can_share_buffer_ = nullptr;

  ForwardsValue forwards_value_ = nullptr;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_DATAFLOW_ANALYSIS_H_
