/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_PROPAGATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_PROPAGATION_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// HLO pass which reduces the precision of some HLO instructions to BF16
// according to the backend-specific BFloat16Support rule provided by the
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
// 1) It does not guarantee to respect the passed-in BFloat16Support
// specification in terms of mixed precision, so the backend may not support an
// HLO that has mixed precision produced by this pass. To address this issue,
// run BFloat16Normalization with the same BFloat16Support after this pass.
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
  explicit BFloat16Propagation(const BFloat16Support* bfloat16_support);

  ~BFloat16Propagation() override = default;

  absl::string_view name() const override { return "bfloat16-propagation"; }

  // Runs the pass on the given module. Returns whether the module was changed
  // (precision reductions were added).
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Returns whether we should avoid changing the precision of inst regardless
  // of the producers and users.
  virtual bool ShouldKeepPrecisionUnchanged(const HloInstruction* inst);

  // Determines whether we should consider changing the precision of the given
  // instruction in the forward pass.
  virtual bool InstructionIsCandidateForBF16Output(HloInstruction* hlo);

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

  // Special handling in the opportunity-finding pass for conditional branches.
  //
  // Precondition: hlo->opcode() == kConditional
  void DetermineConditionalComputationsPrecision(HloInstruction* cond);

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

  // Adjusts the output shapes of HloInstructions such that if two
  // HloInstructions have aliasing buffers in their outputs, they must have the
  // same precision.
  void ResolveInconsistencyOfAliasingBuffers(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Resolves inconsistency of aliasing buffers for the given computation, and
  // recursively runs on a while instruction's condition and body until a fixed
  // point is reached.
  bool ResolveInconsistencyOfAliasingBuffersHelper(
      HloComputation* computation,
      absl::flat_hash_set<const HloComputation*>* visited_computations);

  // Makes the parameters of called computations match how they are called by
  // the given HLO.
  void AdjustCalledComputationParameters(HloInstruction* hlo);

  // Makes the root instructions of called computations match how they are used
  // by the given HLO.
  void AdjustCalledComputationRoot(HloInstruction* hlo);

  // ***************************
  // Functions called after changes in changes_to_bf16_ are applied.

  // Resolves inconsistencies introduced by this pass for fusions with
  // tuple-type output.
  Status ResolveInconsistentFusions(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Converts the literals in kConstant HLOs which have their types changed to
  // BF16 by this pass.
  Status ResolveConvertedConstants(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Skips no-op conversions (same source and target shapes) that can be
  // produced this pass, i.e., replaces them in their uses with their operands.
  Status SkipNoopConversions(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // ***************************
  // Functions called and state used by two or more passes.

  // Returns whether all uses of the given HloInstruction can consume BF16
  // input.
  bool AllUsersConsumeBF16(const HloInstruction& hlo,
                           const ShapeIndex& index) const;

  // The output element type of the HLO at the given shape index after changes
  // in changes_to_bf16_ are applied.
  PrimitiveType OutputTypeAfterChange(HloInstruction* hlo,
                                      const ShapeIndex& index) const;

  // The element type of the HLO value after changes in changes_to_bf16_ are
  // applied.
  PrimitiveType ValueTypeAfterChange(const HloValue* value) const;

  // If target_type == BF16, adds the HLO at the given index to
  // changes_to_bf16_; otherwise, target_type must be F32 and this function
  // removes the HLO at the given index from changes_to_bf16_ if it was earlier
  // added.
  void AddToOrRemoveFromBF16ChangeSet(HloInstruction* hlo,
                                      const ShapeIndex& index,
                                      PrimitiveType target_type);

  // The set of F32 HLO values that must be kept in F32.
  absl::flat_hash_set<const HloValue*> values_that_must_be_kept_as_f32_;

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

  const BFloat16Support* bfloat16_support_;
  std::unique_ptr<HloDataflowAnalysis> dataflow_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_PROPAGATION_H_
