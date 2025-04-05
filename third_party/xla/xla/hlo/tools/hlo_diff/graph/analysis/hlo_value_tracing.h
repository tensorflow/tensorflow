/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_GRAPH_ANALYSIS_HLO_VALUE_TRACING_H_
#define XLA_HLO_TOOLS_HLO_DIFF_GRAPH_ANALYSIS_HLO_VALUE_TRACING_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Analysis that traces the defining HLO instructions of HLO values used by
// any instruction. This is largely based on HloDataflowAnalysis with
// primary difference that the HLO values are traced back through copy and
// fusion instructions.
class HloValueTracing {
 public:
  // Runs dataflow analysis on the given module.
  static absl::StatusOr<std::unique_ptr<HloValueTracing>> Run(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Returns true if 'instruction' defines an HLO value at the given shape index
  // of its output.
  bool ValueIsDefinedAt(const HloInstruction* instruction,
                        const ShapeIndex& index = {}) const;

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


  const HloModule& module() const { return module_; }

 private:
  HloValueTracing(const HloModule& module,
                  absl::flat_hash_set<absl::string_view> execution_threads);

  // Returns a new HloValue defined at the given instruction and shape index.
  HloValue* NewHloValue(HloInstruction* instruction, const ShapeIndex& index,
                        bool is_phi);

  // Deletes all HloValues marked for deletion. Should be called after
  // propagation is complete.
  void DeleteMarkedValues();

  // Constructs and initializes the InstructionValueSets of all instructions to
  // contain exactly the HloValues defined by each instruction. These values can
  // then propagated throughout the HLO graph by calling Propagate.
  absl::Status InitializeInstructionValueSets();

  // Updates the value set of the given instruction based on the values flowing
  // into the instruction (operands and cross-computation dataflow).
  bool UpdateInstructionValueSet(HloInstruction* instruction);

  // Updates the value set for a particular instruction type. Returns whether
  // the instruction value set changed.
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
  bool UpdateFusionValueSet(HloInstruction* fusion);
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

  const HloModule& module_;
  const absl::flat_hash_set<absl::string_view> execution_threads_;

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
};
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_ANALYSIS_HLO_VALUE_TRACING_H_
