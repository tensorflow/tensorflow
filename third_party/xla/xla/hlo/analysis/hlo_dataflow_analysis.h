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

#ifndef XLA_HLO_ANALYSIS_HLO_DATAFLOW_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_HLO_DATAFLOW_ANALYSIS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_phi_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Analysis which identifies all HLO values and their uses in an HLO module.
class HloDataflowAnalysis {
 public:
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
  //
  //   propagate_through_calls : If false, kCall instructions are treated as
  //     opaque instructions that define their own output values, and dataflow
  //     across kCall boundaries is ignored.
  //
  //   precompute_uses : If set, the values this predicate accepts get their
  //     uses (HloValue::GetUses) before Run returns, in one pass that shares
  //     the per instruction work between them. Every other value computes its
  //     uses on the first GetUses, one value at a time, which is quadratic in
  //     the width of a tuple. Set it when the uses of many values are read
  //     before the module is mutated.
  static absl::StatusOr<std::unique_ptr<HloDataflowAnalysis>> Run(
      const HloModule& module, bool ssa_form = false,
      bool bitcast_defines_value = false,
      absl::flat_hash_set<absl::string_view> execution_threads = {},
      bool propagate_through_calls = true,
      std::optional<absl::FunctionRef<bool(const HloValue&)>> precompute_uses =
          std::nullopt,
      bool propagate_through_control_flow = true);

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

  // Returns the unique value in the HloValueSet at the given instruction and
  // shape index. CHECKs if the value set does not contain a exactly one value.
  const HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                                   const ShapeIndex& index = {}) const {
    const HloValueSet& value_set = GetValueSet(instruction, index);
    if (value_set.values().size() != 1) {
      LOG(FATAL) << "GetUniqueValueAt failed on instruction: "
                 << instruction->name() << " at index " << index
                 << " with value set size " << value_set.values().size() << ": "
                 << value_set;
    }
    return value_set.GetUniqueValue();
  }
  HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                             const ShapeIndex& index = {}) {
    const HloValueSet& value_set = GetValueSet(instruction, index);
    if (value_set.values().size() != 1) {
      LOG(FATAL) << "GetUniqueValueAt failed on instruction: "
                 << instruction->name() << " at index " << index
                 << " with value set size " << value_set.values().size() << ": "
                 << value_set;
    }
    return GetValue(value_set.GetUniqueValue().id());
  }

  // Returns the HloValue with the given Id.
  const HloValue& GetValue(HloValue::Id value_id) const;
  HloValue& GetValue(HloValue::Id value_id);

  // Returns the total number of HloValues.
  int64_t value_count() const { return values_.size(); }

  // Returns a vector of all HloValues stabily sorted by HloValue::Id.
  const std::vector<HloValue*>& values() const { return values_vector_; }

  // Returns a new value Id to use.
  HloValue::Id NewValueId() { return next_value_id_++; }

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
                                     const ShapeIndex& user_index,
                                     const AliasInfo* alias_info) const;

  const HloModule& module() const { return module_; }

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
  // TODO(b/424109294): Move this to AliasInfo class.
  static std::vector<std::pair<HloOperandIndex, ShapeIndex>>
  GetInPlaceInputOutputPairs(const HloInstruction* instruction);

  // Verifies various invariants of the dataflow analysis.
  absl::Status Verify() const;

 private:
  static bool AreTransitiveUsesElementwiseOrTuple(const HloInstruction* inst);

  HloDataflowAnalysis(const HloModule& module, bool ssa_form,
                      bool bitcast_defines_value,
                      absl::flat_hash_set<absl::string_view> execution_threads,
                      bool propagate_through_calls = true,
                      bool propagate_through_control_flow = true);

  // Runs dataflow analysis on the module attached to this HloDataflowAnalysis.
  absl::Status RunImpl();

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
  absl::Status InitializeInstructionValueSets();

  // Updates the value set of the given instruction based on the values flowing
  // into the instruction (operands and cross-computation dataflow).
  bool UpdateInstructionValueSet(HloInstruction* instruction);

  // Returns the HloValueSet for the given instruction at the given index.
  HloValueSet& GetMutableValueSet(const HloInstruction* instruction,
                                  const ShapeIndex& index = {});

  // Updates the value set for a particular instruction type. Returns whether
  // the instruction value set changed.
  bool UpdateBitcastValueSet(HloInstruction* bitcast);
  bool UpdateCallValueSet(HloInstruction* call);
  bool UpdateConditionalValueSet(HloInstruction* conditional);
  bool UpdateCopyValueSet(HloInstruction* copy);
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
  // Updates the value set at `operand_index` with the value set of
  // `operand` for the async_op in the async chain (only for
  // async-start/async-update).
  bool UpdateAsyncChainOperandValueSet(HloInstruction* async_op,
                                       int64_t operand_index,
                                       const HloInstruction* operand);
  // Updates the value set for element {1} of the async operation's output,
  // which corresponds to the wrapped computation's root.
  bool UpdateAsyncChainOutputValueSet(HloInstruction* async_op);
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
  bool propagate_through_calls_ = true;
  bool propagate_through_control_flow_ = true;

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

  // Caches for CanShareOperandBufferWithUser.
  mutable absl::flat_hash_map<
      HloInstruction*,
      absl::flat_hash_map<ShapeIndex, std::vector<HloOperandIndex>>>
      cache_share_buffer_with_user_;
  mutable absl::flat_hash_map<std::pair<HloInstruction*, ShapeIndex>,
                              absl::flat_hash_set<HloUse>>
      cache_share_buffer_with_operand_;
};

// Options controlling which computation boundaries are visited by
// HloDataflowPropagation.
struct HloCallBoundaryOptions {
  // Include kCall instructions.
  bool include_calls = true;
  // Include kWhile, kConditional, and kAsyncStart instructions.
  bool include_control_flow = true;
  // Include kFusion instructions.
  bool include_fusions = true;
  // Include associative kScan instructions.
  bool include_associative_scans = true;
  // Optional execution thread filter applied to caller and callee computations
  // across all boundary instruction kinds (and async_execution_thread() on
  // kAsyncStart). When nullptr or empty, all threads are included.
  const absl::flat_hash_set<absl::string_view>* execution_threads = nullptr;

  constexpr HloCallBoundaryOptions() = default;
  constexpr HloCallBoundaryOptions(
      bool include_calls_in, bool include_control_flow_in,
      bool include_fusions_in, bool include_associative_scans_in = false,
      const absl::flat_hash_set<absl::string_view>* execution_threads_in =
          nullptr)
      : include_calls(include_calls_in),
        include_control_flow(include_control_flow_in),
        include_fusions(include_fusions_in),
        include_associative_scans(include_associative_scans_in),
        execution_threads(execution_threads_in) {}
};

// Describes the dataflow boundary between a callsite instruction and one of its
// called computations.
struct HloCallBoundary {
  // The calling instruction (kCall, kWhile, kConditional, kAsyncStart, kFusion,
  // or associative kScan).
  const HloInstruction* callsite = nullptr;
  // The called computation.
  HloComputation* callee = nullptr;
  // Contiguous span of caller operands feeding `callee`'s parameters in order:
  // `caller_operands[param_no]` feeds
  // `callee->parameter_instruction(param_no)`.
  absl::Span<HloInstruction* const> caller_operands;
  // Index of `caller_operands[0]` in `callsite->operands()` (b + 1 for
  // conditional branch b, 0 for all other instructions).
  int64_t first_operand_index = 0;
  // True if `callee->root_instruction()` flows into `callsite`'s output (true
  // for all boundary types except `while_condition()`).
  bool root_feeds_callsite = true;
  // ShapeIndex prefix on `callsite->shape()` where `callee->root_instruction()`
  // appears ({1} for kAsyncStart, {} for all other instructions).
  ShapeIndex callsite_output_prefix;

  HloCallBoundary() = default;
  HloCallBoundary(const HloInstruction* callsite_in, HloComputation* callee_in,
                  absl::Span<HloInstruction* const> caller_operands_in,
                  int64_t first_operand_index_in, bool root_feeds_callsite_in,
                  ShapeIndex callsite_output_prefix_in)
      : callsite(callsite_in),
        callee(callee_in),
        caller_operands(caller_operands_in),
        first_operand_index(first_operand_index_in),
        root_feeds_callsite(root_feeds_callsite_in),
        callsite_output_prefix(std::move(callsite_output_prefix_in)) {}

  int64_t num_parameters() const {
    return std::min<int64_t>(caller_operands.size(), callee->num_parameters());
  }

  int64_t caller_operand_index(int64_t param_no) const {
    return first_operand_index + param_no;
  }

  HloInstruction* caller_operand(int64_t param_no) const {
    return caller_operands[param_no];
  }

  HloInstruction* callee_parameter(int64_t param_no) const {
    return callee->parameter_instruction(param_no);
  }

  HloInstruction* callee_root() const { return callee->root_instruction(); }

  // Maps a ShapeIndex on `callee->root_instruction()` to the corresponding
  // ShapeIndex on `callsite->shape()`.
  ShapeIndex CallerOutputIndex(const ShapeIndex& root_index) const {
    ShapeIndex result = callsite_output_prefix;
    result.insert(result.end(), root_index.begin(), root_index.end());
    return result;
  }

  // Maps a ShapeIndex on `callsite->shape()` to the corresponding ShapeIndex on
  // `callee->root_instruction()`, or returns std::nullopt if
  // `!root_feeds_callsite` or `caller_output_index` does not match
  // `callsite_output_prefix`.
  std::optional<ShapeIndex> CalleeRootIndex(
      const ShapeIndex& caller_output_index) const {
    if (!root_feeds_callsite ||
        caller_output_index.size() < callsite_output_prefix.size()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < callsite_output_prefix.size(); ++i) {
      if (caller_output_index[i] != callsite_output_prefix[i]) {
        return std::nullopt;
      }
    }
    return ShapeIndex(
        caller_output_index.begin() + callsite_output_prefix.size(),
        caller_output_index.end());
  }
};

// Base class and helper for propagating dataflow properties across computation
// call boundaries (kCall, kWhile, kConditional, kAsyncStart, kFusion, and
// associative kScan) over an HloDataflowAnalysis.
//
// Subclasses implement `HasValueAt` and `PropagateAcrossEdge` (plus optional
// policy hooks for while loops, conditionals, and computation flushing) and
// invoke `Run(computations)` for full module fixed point propagation or
// `PropagateForValue(value)` for incremental single value propagation.
class HloDataflowPropagation {
 public:
  explicit HloDataflowPropagation(
      const HloDataflowAnalysis* dataflow_analysis = nullptr,
      const HloCallBoundaryOptions& options = {})
      : dataflow_analysis_(dataflow_analysis), options_(options) {}
  virtual ~HloDataflowPropagation() = default;

  void set_dataflow_analysis(const HloDataflowAnalysis* dataflow_analysis) {
    dataflow_analysis_ = dataflow_analysis;
  }

  // Runs iterative bottom up (callee to caller) and top down (caller to callee)
  // propagation across `computations` until convergence.
  absl::Status Run(absl::Span<HloComputation* const> computations);

  // Propagates a single newly constrained HloValue across any call boundaries
  // its positions touch.
  absl::Status PropagateForValue(const HloValue& value);

  // Propagates constrained parameter and root positions of `computation` to its
  // caller instructions' operands and outputs.
  absl::Status PropagateCalleeToCallers(HloComputation* computation,
                                        bool* changed);

  // Propagates constrained operand and output positions of callsite
  // instructions in `computation` into their called computations' parameters
  // and roots.
  absl::Status PropagateCallerToCallees(HloComputation* computation,
                                        bool* changed);

  // Directional boundary propagation steps.
  absl::Status PropagateParameterToCallers(const HloInstruction* param,
                                           const ShapeIndex& index,
                                           bool allow_override, bool* changed);
  absl::Status PropagateRootToCallers(const HloInstruction* root,
                                      const ShapeIndex& index,
                                      bool allow_override, bool* changed);
  absl::Status PropagateCallerOutputToCallees(const HloInstruction* caller,
                                              const ShapeIndex& index,
                                              bool allow_override,
                                              bool* changed);
  absl::Status PropagateCallerOperandToCallees(const HloInstruction* caller,
                                               const HloInstruction* operand,
                                               const ShapeIndex& index,
                                               bool allow_override,
                                               bool* changed);
  absl::Status PropagateWhileBoundary(const HloInstruction* while_inst,
                                      const ShapeIndex& index,
                                      const HloInstruction* fallback_source,
                                      bool allow_override, bool* changed,
                                      bool check_caller_to_body);

  // Invokes `fn(index)` for each subshape index of `instruction` where
  // `HasValueAt(instruction, index)` holds.
  absl::Status ForEachConstrainedSubshape(
      const HloInstruction* instruction,
      absl::FunctionRef<absl::Status(const ShapeIndex&)> fn) const;

  // Visits each HloCallBoundary of `callsite` enabled by `options`.
  static absl::Status ForEachCallBoundaryWithStatus(
      const HloInstruction* callsite,
      absl::FunctionRef<absl::Status(const HloCallBoundary&)> fn,
      const HloCallBoundaryOptions& options = {});

  static void ForEachCallBoundary(
      const HloInstruction* callsite,
      absl::FunctionRef<void(const HloCallBoundary&)> fn,
      const HloCallBoundaryOptions& options = {});

  // Visits each HloCallBoundary in caller instructions that invokes `callee`.
  static absl::Status ForEachCallerBoundaryWithStatus(
      const HloComputation* callee,
      absl::FunctionRef<absl::Status(const HloCallBoundary&)> fn,
      const HloCallBoundaryOptions& options = {});

  static void ForEachCallerBoundary(
      const HloComputation* callee,
      absl::FunctionRef<void(const HloCallBoundary&)> fn,
      const HloCallBoundaryOptions& options = {});

  // Visits each parameter instruction in computations called by `callsite` that
  // receives `callsite->operand(operand_number)`.
  static absl::Status ForEachCalledParameterWithStatus(
      const HloInstruction* callsite, int64_t operand_number,
      absl::FunctionRef<absl::Status(HloInstruction*)> fn,
      const HloCallBoundaryOptions& options = {});

  static void ForEachCalledParameter(
      const HloInstruction* callsite, int64_t operand_number,
      absl::FunctionRef<void(HloInstruction*)> fn,
      const HloCallBoundaryOptions& options = {});

  // Visits all boundary instructions that must agree at a given ShapeIndex for
  // `while_inst`: the init operand, the body and condition parameters (when
  // present), the body root, and `while_inst` itself.
  static absl::Status ForEachWhileBoundaryInstructionWithStatus(
      const HloInstruction* while_inst,
      absl::FunctionRef<absl::Status(const HloInstruction*)> fn);

  static void ForEachWhileBoundaryInstruction(
      const HloInstruction* while_inst,
      absl::FunctionRef<void(const HloInstruction*)> fn);

  // Returns the output ShapeIndex of `use.instruction` corresponding to `use`
  // when `use.instruction` forwards or preserves subshape structure (kTuple,
  // tuple shaped kAllReduce, kGetTupleElement, or same shape ops).
  static ShapeIndex GetForwardedUseOutputIndex(const HloUse& use);

 protected:
  const HloDataflowAnalysis& dataflow_analysis() const {
    return *dataflow_analysis_;
  }
  const HloCallBoundaryOptions& options() const { return options_; }

  // Returns true if `computation` participates in propagation.
  virtual bool IsComputationIncluded(const HloComputation* computation) const {
    return true;
  }

  // Returns true if `(instruction, index)` currently has a value or constraint
  // ready to propagate across a call boundary.
  virtual bool HasValueAt(const HloInstruction* instruction,
                          const ShapeIndex& index) const = 0;

  // Transfers the value or constraint at `(src_instruction, src_index)` to
  // `(dst_instruction, dst_index)`. When `allow_override` is true, may also
  // update a previously propagated non mandatory constraint. Sets `*changed`
  // (when non null) if any destination value was updated.
  virtual absl::Status PropagateAcrossEdge(
      const HloInstruction* src_instruction, const ShapeIndex& src_index,
      const HloInstruction* dst_instruction, const ShapeIndex& dst_index,
      bool allow_override, bool* changed) = 0;

  // Policy hook: returns true if root values may propagate across `boundary`.
  virtual bool ShouldPropagateAcrossRootBoundary(
      const HloCallBoundary& boundary) const {
    return true;
  }

  // Policy hook: for a `kConditional` caller, returns the preferred branch
  // computation whose root should drive the conditional output, or nullptr if
  // no branch is preferred.
  virtual const HloComputation* PreferredConditionalBranch(
      const HloInstruction* conditional) const {
    return nullptr;
  }

  // Policy hook: returns true if a value on the while body parameter or root
  // `body_source` at `index` should yield to `caller_source`.
  virtual bool IsStaleWhileBodyValue(const HloInstruction* while_inst,
                                     const ShapeIndex& index,
                                     const HloInstruction* body_source,
                                     const HloInstruction* caller_source) {
    return false;
  }

  // Policy hook: returns the instruction whose value at `index` should drive
  // `while_inst`'s boundary at `index`, preferring non stale while body
  // parameter or root values over caller values.
  virtual const HloInstruction* PreferredWhileBoundarySource(
      const HloInstruction* while_inst, const ShapeIndex& index,
      const HloInstruction* fallback_source);

  // Policy hook: returns true if a caller value from `source_instruction` at
  // `index` should propagate into `while_inst`'s body.
  virtual bool ShouldPropagateCallerToWhileBody(
      const HloInstruction* while_inst, const ShapeIndex& index,
      const HloInstruction* source_instruction) {
    return true;
  }

  // Hook invoked at the start of `Run` to reset per-pass state.
  virtual void ResetPropagationState() {}

  // Hook invoked when `*dirty` is true to flush intra computation propagation.
  virtual absl::Status FlushComputationPropagation() {
    return absl::OkStatus();
  }

 private:
  absl::Status FlushPending(bool* dirty, bool* changed);

  const HloDataflowAnalysis* dataflow_analysis_;
  HloCallBoundaryOptions options_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_DATAFLOW_ANALYSIS_H_
