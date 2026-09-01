/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_BFLOAT16_PROPAGATION_H_
#define XLA_HLO_TRANSFORMS_BFLOAT16_PROPAGATION_H_

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// HLO pass which reduces the precision of some HLO instructions to BF16
// according to the backend-specific FloatSupport rule provided by the
// caller.
//
// This pass can be used to reduce instruction precision without affecting the
// numerical accuracy of the module, i.e., the final output of the module would
// be bitwise identical to that without this pass; this is possible if the
// backend already reduces precision to BF16 on some HLO instructions.
//
// This pass will not modify the signature of a computation, unless it is a
// fusion computation or its only caller is a while.
//
// !!! WARNING !!! This pass can introduce mixed precision in individual HLOs,
// which has two issues:
//
// 1) It does not guarantee to respect the passed-in FloatSupport
// specification in terms of mixed precision, so the backend may not support an
// HLO that has mixed precision produced by this pass. To address this issue,
// run FloatNormalization with the same FloatSupport after this pass.
//
// 2) In general, mixed precision may break the assumptions of some other HLO
// passes even if the specific backend supports the individual HLOs. Such
// assumptions include that there are no HLOs using mixed precision, or that the
// precision of an HLO's output is determined by its inputs. It should be used
// at the end of the HLO optimization pipeline but before
// BFloat16ConversionFolding. If other passes are needed after this pass, run
// BFloat16MixedPrecisionRemoval first to undo some of the changes made by this
// pass.
class BFloat16Propagation : public HloModulePass {
 public:
  BFloat16Propagation(const FloatSupport* bfloat16_support,
                      const AliasInfo* alias_info);

  ~BFloat16Propagation() override = default;

  static constexpr absl::string_view kName = "bfloat16-propagation";
  absl::string_view name() const override { return kName; }

  // Returns whether we should avoid changing the precision of inst regardless
  // of the producers and users.
  virtual bool ShouldKeepPrecisionUnchanged(const HloInstruction* inst);

  // Determines whether we should consider changing the precision of the given
  // instruction in the forward pass.
  virtual bool InstructionIsCandidateForBF16Output(HloInstruction* hlo);

 protected:
  const FloatSupport* bfloat16_support_;

  const AliasInfo* alias_info_;

  // Returns the original element type of the HLO instruction before
  // RunImpl starts mutating shapes in-place with changes_to_bf16_.
  PrimitiveType UnmutatedElementType(const HloInstruction* hlo) const {
    if (hlo->shape().element_type() == BF16 &&
        changes_to_bf16_.contains(const_cast<HloInstruction*>(hlo))) {
      return F32;
    }
    return hlo->shape().element_type();
  }

  // Runs the pass on the given module. Returns whether the module was changed
  // (precision reductions were added).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // ***************************
  // Function called and state produced by the forward analysis pass (from
  // parameters to root) that determines the candidate HLOs to use BF16 outputs.

  // The set of instructions to consider using bfloat16, computed in the forward
  // pass.
  absl::flat_hash_set<const HloInstruction*> consider_using_bfloat16_;

  // ***************************
  // Functions called and state produced by the backward pass (from root to
  // parameters) that finds opportunities to use BF16.

  // Determines the precision for the given instruction in the
  // opportunity-finding pass.
  void DetermineInstructionPrecision(HloInstruction* hlo, bool skip_parameters);

  // Special handling in the opportunity-finding pass for fusion computations.
  //
  // Precondition: hlo->opcode() == kFusion
  void DetermineFusionComputationPrecision(HloInstruction* fusion);

  // Reverts changes to BF16 that will not propagate outside a fusion
  // computation. This avoids BF16 casts overhead inside a fusion which won't
  // save memory bandwidth.
  //
  // Precondition: hlo->opcode() == kFusion
  void RevertIfFusionInternalBF16Changes(HloInstruction* fusion);

  // Special handling in the opportunity-finding pass for while computations.
  //
  // Precondition: hlo->opcode() == kWhile
  void DetermineWhileComputationsPrecision(HloInstruction* while_hlo);

  // Special handling in the opportunity-finding pass for associative scans.
  // Mirrors DetermineWhileComputationsPrecision because scan carries form a
  // loop-carried precision equivalence chain across the IR (carry init ->
  // body parameter -> body root -> result carry).
  //
  // Precondition: hlo->opcode() == kScan and the scan is_associative() ==
  // TRI_STATE_TRUE. Non-associative scans are expanded to while loops by the
  // ScanExpander pass and are handled by the kWhile path instead.
  void DetermineScanComputationPrecision(HloInstruction* scan_hlo);

  // Special handling in the opportunity-finding pass for conditional branches.
  //
  // Precondition: hlo->opcode() == kConditional
  void DetermineConditionalComputationsPrecision(HloInstruction* cond);

  // Special handling in the opportunity-finding pass for async computations.
  //
  // Precondition: hlo->opcode() == kAsyncStart
  void DetermineAsyncComputationsPrecision(HloInstruction* async_start);

  // Special handling in the opportunity-finding pass for called computations.
  //
  // Precondition: hlo->opcode() == kCall
  void DetermineCalledComputationsPrecision(HloInstruction* call);

  // The set of HloInstructions that have been visited in the
  // opportunity-finding pass.
  absl::flat_hash_set<const HloInstruction*>
      instructions_visited_in_backward_pass_;

  // The set of HloComputations that have been visited in the
  // opportunity-finding pass.
  absl::flat_hash_set<const HloComputation*>
      computations_visited_in_backward_pass_;

  // ***************************
  // Functions called by the final inconsistency resolving pass.
  //
  // The resolving pass runs a single BFS over the F32 constraint graph. Its
  // nodes are the HLO values and positions of the module; an edge u -> v
  // means "if u must be F32, then v must be F32":
  //  * A value constrains its positions and the in-place output positions
  //    aliased with it.
  //  * A position constrains the values whose AllUsersConsumeBF16 check
  //    reads it, and the F32 values it aliases.
  //  * A call site position constrains the same position of the roots it
  //    calls; an operand position constrains the same position of the
  //    parameters called with it. These edges are called pushes.
  //  * A scan carry init position constrains the other three carry slots.
  // The BFS starts from the seeds, the nodes that must be F32 up front;
  // every position it does not reach becomes BF16. Pushes are one way: F32
  // crossing a call boundary is final, and a BF16 call site never overrides
  // a callee that must be F32. Remaining call boundary mismatches are
  // patched with converts by ResolveInconsistentFusions/Scans.

  // Adjusts the output shapes of HloInstructions such that if two
  // HloInstructions have aliasing buffers in their outputs, they must have the
  // same precision.
  void ResolveInconsistencyOfAliasingBuffers(HloModule* module);

  // Builds the seeds and edges of the F32 constraint graph.
  void BuildF32ConstraintGraph(HloModule* module);

  // Fills push_roots_, push_params_ and bf16_pushable_positions_.
  void BuildCallBoundaryPushIndex(
      absl::Span<HloComputation* const> computations);

  // Registers the parameters of a called computation as push targets of the
  // corresponding call site operands.
  void AddPushableParams(HloComputation* callee,
                         absl::Span<HloInstruction* const> operands);

  // Registers the root of a called computation as a push target of the call
  // site.
  void AddPushableRoot(HloInstruction* call_site, HloComputation* callee);

  // Adds value -> position edges for `hlo`'s in-place operand/output pairs.
  void AddInPlaceEdges(HloInstruction* hlo);

  // Seeds the F32 positions of a ShouldKeepPrecisionUnchanged instruction;
  // nothing may mark them BF16, except a pending call boundary push.
  void AddKeepPrecisionSeeds(HloInstruction* hlo);

  // Adds carry init -> {body parameter, body root slot, scan result slot}
  // edges for an associative scan. HloVerifier requires the four slots to
  // agree, and no dataflow alias connects them.
  void AddScanCarryEdges(HloInstruction* hlo);

  // Seeds `value` if it must be F32 up front, and records which output
  // position each of its uses reads.
  void AddValueSeedsAndUseEdges(const HloValue* value);

  // Adds the position -> `value` edge for one use, mirroring
  // AllUsersConsumeBF16: a called computation use reads the callee
  // parameter, a forwarding user reads its own output. A statically failing
  // use seeds the value instead.
  void AddEdgesForUse(const HloValue* value, const HloUse& use);

  // BFS from the seeds: finds every value and position that must be F32.
  void PropagateF32Constraints();

  // Records that the value / position must be F32 and enqueues it.
  void ConstrainValueToF32(const HloValue* value);
  void ConstrainPositionToF32(const HloPosition& position);

  // Constrains the targets of the dequeued node's outgoing edges.
  void PropagateFromValue(const HloValue* value);
  void PropagateFromPosition(const HloPosition& position);

  // Applies `position`'s push edges: the same position of the roots it
  // calls and of the parameters called with it. The caller ensures
  // `position` is an F32 array leaf.
  void PushAcrossCallBoundaries(const HloPosition& position);

  // Makes every adjustable F32 position BF16 iff the BFS did not reach it.
  void MaterializeResolvedPrecisions(HloModule* module);

  // ***************************
  // Functions called after changes in changes_to_bf16_ are applied.

  // Resolves inconsistencies introduced by this pass for fusions with
  // tuple-type output.
  absl::Status ResolveInconsistentFusions(HloModule* module);

  // Resolves inconsistencies introduced by this pass for associative scans
  // where the body root tuple slot precision diverged from the scan output
  // slot precision (e.g. when the underlying body root op had multiple uses
  // with different precision demands and could not be unilaterally lowered).
  // Inserts precision-changing converts on the affected body root slots so
  // that the body root shape matches the scan output shape, satisfying
  // HloVerifier::HandleScan.
  absl::Status ResolveInconsistentScans(HloModule* module);

  // Converts the literals in kConstant HLOs which have their types changed to
  // BF16 by this pass.
  absl::Status ResolveConvertedConstants(HloModule* module);

  // Skips no-op conversions (same source and target shapes) that can be
  // produced this pass, i.e., replaces them in their uses with their operands.
  absl::Status SkipNoopConversions(HloModule* module);

  // ***************************
  // Functions called and state used by two or more passes.

  // Returns whether all uses of the given HloInstruction can consume BF16
  // input.
  bool AllUsersConsumeBF16(const HloInstruction& hlo,
                           const ShapeIndex& index) const;

  // Memoized ShouldKeepPrecisionUnchanged. Valid while shapes are
  // unmutated; the loop that applies changes_to_bf16_ in RunImpl must call
  // the virtual method directly.
  bool ShouldKeepPrecisionUnchangedCached(const HloInstruction* inst);

  // Memoized AliasInfo::GetInPlaceInputOutputPairs.
  const std::vector<std::pair<HloOperandIndex, ShapeIndex>>&
  GetInPlaceInputOutputPairsCached(const HloInstruction* hlo);

  // The output element type of the HLO at the given shape index after changes
  // in changes_to_bf16_ are applied.
  PrimitiveType OutputTypeAfterChange(HloInstruction* hlo,
                                      const ShapeIndex& index) const;

  // If target_type == BF16, adds the HLO at the given index to
  // changes_to_bf16_; otherwise, target_type must be F32 and this function
  // removes the HLO at the given index from changes_to_bf16_ if it was earlier
  // added.
  void AddToOrRemoveFromBF16ChangeSet(HloInstruction* hlo,
                                      const ShapeIndex& index,
                                      PrimitiveType target_type);

  // The set of F32 HLO values that must be kept in F32. Insert only.
  absl::flat_hash_set<const HloValue*> values_that_must_be_kept_as_f32_;

  // The computations this run operates on. Positions outside are left
  // unchanged.
  absl::flat_hash_set<const HloComputation*> included_computations_;

  // BFS state: the nodes that must be F32, and the queues of nodes whose
  // edges have not been followed yet.
  std::deque<const HloValue*> value_queue_;
  std::deque<HloPosition> position_queue_;
  absl::flat_hash_set<const HloValue*> f32_values_;
  absl::flat_hash_set<HloPosition> f32_positions_;

  // Position -> values whose AllUsersConsumeBF16 check reads it (see
  // AddEdgesForUse).
  absl::flat_hash_map<HloPosition, std::vector<const HloValue*>> use_edges_;

  // Value -> in-place output positions aliased with it.
  absl::flat_hash_map<const HloValue*, std::vector<HloPosition>>
      value_to_inplace_outputs_;

  // The BFS seeds. A value: defining position unmarked and not pushable, a
  // statically failing use, or pinned by the backward pass. A position:
  // unmarked on a keep precision instruction, or holding a value that is
  // neither F32 nor BF16.
  std::vector<const HloValue*> static_f32_seed_values_;
  std::vector<HloPosition> static_f32_seed_positions_;

  // Positions a call boundary push can still mark BF16. Their unmarked
  // defining values are not seeded.
  absl::flat_hash_set<HloPosition> bf16_pushable_positions_;

  // Push targets: an F32 position of a call site forces the same position
  // of push_roots_[site]; an F32 operand position forces push_params_
  // [operand]. Precomputed to avoid rescanning users per leaf.
  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      push_roots_;
  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      push_params_;

  // Carry init position -> the other three carry slot positions.
  absl::flat_hash_map<HloPosition, std::vector<std::array<HloPosition, 3>>>
      scan_carry_edges_;

  // Cache for ShouldKeepPrecisionUnchangedCached. Only valid while the module
  // is unmutated (i.e., until changes_to_bf16_ is applied in RunImpl).
  absl::flat_hash_map<const HloInstruction*, bool>
      keep_precision_unchanged_cache_;

  // Cache for GetInPlaceInputOutputPairsCached.
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<HloOperandIndex, ShapeIndex>>>
      inplace_input_output_pairs_cache_;

  // Mapping from each HloComputation to the number of callers to it in the
  // module. Populated at the beginning of this pass.
  absl::flat_hash_map<const HloComputation*, int64_t> caller_counts_;

  // We first store the potential F32-to-BF16 changes to changes_to_bf16_, which
  // are subject to further adjustment, then finally applied to the HLOs. This
  // avoids setting changed_ to true but all changes are reverted during
  // adjustment.
  //
  // For each HloInstruction, changes_to_bf16_ stores the affected buffers in
  // the output as a map from in-place pointers to subshapes to shape indices.
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_map<Shape*, ShapeIndex>>
      changes_to_bf16_;

  // Whether the last processed HLO module has been changed by this pass.
  bool changed_ = false;

  std::unique_ptr<HloDataflowAnalysis> dataflow_;

  absl::flat_hash_set<absl::string_view> execution_threads_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_BFLOAT16_PROPAGATION_H_
