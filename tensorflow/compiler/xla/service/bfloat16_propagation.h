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
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
class BFloat16Propagation : public HloPassInterface {
 public:
  explicit BFloat16Propagation(const BFloat16Support* bfloat16_support);

  ~BFloat16Propagation() override = default;

  tensorflow::StringPiece name() const override {
    return "bfloat16-propagation";
  }

  // Runs the pass on the given module. Returns whether the module was changed
  // (precision reductions were added).
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // ***************************
  // Function called and state produced by the forward analysis pass (from
  // parameters to root) that determines the candidate HLOs to use BF16 outputs.

  // Determines whether we should consider changing the precision of the given
  // instruction in the forward pass.
  bool InstructionIsCandidateForBF16Output(HloInstruction* hlo);

  // The set of instructions to consider using bfloat16, computed in the forward
  // pass.
  tensorflow::gtl::FlatSet<const HloInstruction*> consider_using_bfloat16_;

  // ***************************
  // Functions called and state produced by the backward mutation pass (from
  // root to parameters).

  // Determines the precision for the given instruction in the mutation pass.
  void DetermineAndMutateInstructionPrecision(HloInstruction* hlo,
                                              bool skip_parameters);

  // Special handling in the mutation pass for fusion computations.
  //
  // Precondition: hlo->opcode() == kFusion
  void DetermineAndMutateFusionComputationPrecision(HloInstruction* fusion);

  // Special handling in the mutation pass for while computations.
  //
  // Precondition: hlo->opcode() == kWhile
  void DetermineAndMutateWhileComputationsPrecision(HloInstruction* while_hlo);

  // The set of HloInstructions that have been visited in the mutation pass.
  tensorflow::gtl::FlatSet<const HloInstruction*>
      instructions_visited_in_mutation_pass_;

  // The set of HloComputations that have been visited in the mutation pass.
  tensorflow::gtl::FlatSet<const HloComputation*>
      computations_visited_in_mutation_pass_;

  // ***************************
  // Functions called by the final inconsistency resolving pass.

  // Adjusts the output shapes of HloInstructions such that if two
  // HloInstructions have aliasing buffers in their outputs, they must have the
  // same precision.
  Status ResolveInconsistencyOfAliasingBuffers(HloModule* module);

  // Resolves inconsistency of aliasing buffers for the given computation, and
  // recursively runs on a while instruction's condition and body until a fixed
  // point is reached.
  bool ResolveInconsistencyOfAliasingBuffersHelper(
      HloComputation* computation,
      tensorflow::gtl::FlatSet<const HloComputation*>* visited_computations);

  // Makes the parameters of called computations match how they are called by
  // the given HLO.
  void AdjustCalledComputationParameters(HloInstruction* hlo);

  // Makes the root instructions of called computations match how they are used
  // by the given HLO.
  void AdjustCalledComputationRoot(HloInstruction* hlo);

  // ***************************
  // Functions called and state used by two or more passes.

  // Returns whether all uses of the given HloInstruction can consume BF16
  // input.
  bool AllUsersConsumeBF16(const HloInstruction& hlo,
                           const ShapeIndex& index) const;

  // The set of F32 HLO values that must be kept in F32.
  tensorflow::gtl::FlatSet<const HloValue*> values_that_must_be_kept_as_f32_;

  // Mapping from each HloComputation to the number of callers to it in the
  // module. Populated at the beginning of this pass.
  tensorflow::gtl::FlatMap<const HloComputation*, int64> caller_counts_;

  const BFloat16Support* bfloat16_support_;
  std::unique_ptr<HloDataflowAnalysis> dataflow_;

  bool changed_ = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_PROPAGATION_H_
