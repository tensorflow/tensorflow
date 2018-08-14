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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Copy insertion is a legalization HLO pass which inserts copies (kCopy
// instructions) to eliminate several kinds of problems in the HLO module.
//
//   (1) Entry parameter or a constant live out of the entry computation.  Entry
//       computation arguments and constants have different lifetimes than the
//       computation result and cannot share the same allocation. Parameters and
//       constants live out of non-entry computations do not need copies.
//
//   (2) Different values which are simultaneously live and which must be held
//       in the same buffer. This can occur in while bodies. Specifically, the
//       while loop state (the arguments to the while instruction) is updated
//       in-place and the update may clobber the value from the previous
//       iteration before the previous value is dead. Computations called from
//       kCall instructions do not need such copies because kCall has no update
//       in-place semantics.
//
//   (3) The buffer set of the root instruction of the entry computation must be
//       unambiguous and distinct. That is, InstructionAliasSet::IsAmbiguous and
//       InstructionAliasSet::IsDistinct return true.
class CopyInsertion : public HloPassInterface {
 public:
  tensorflow::StringPiece name() const override { return "copy-insertion"; }

  // fusion_can_share_buffer: backend specific function that decides whether a
  // fusion can share buffer with its operand.
  //
  // TODO(b/80315712): Find a better way to tell whether a fusion can share
  // buffer.
  CopyInsertion(const HloDataflowAnalysis::FusionCanShareBufferFunction&
                    fusion_can_share_buffer = nullptr)
      : fusion_can_share_buffer_(fusion_can_share_buffer) {}

  // Run the pass on the given module. Returns whether the module was changed
  // (copies were inserted).
  StatusOr<bool> Run(HloModule* module) override;

  // The CPU and GPU backend need additional copies added due to deficiencies in
  // buffer assignment. Specifically, copies are needed for constants live-out
  // of computations, and for values which are live-in and live-out of the same
  // computation. These copies are needed because buffer-assignment uses a
  // computation-scoped analyis (TuplePointsToAnalysis) and has limited
  // visibility across computation boundaries. This method adds these necessary
  // copies. Returns whether the module was modified.
  //
  // TODO(b/62548313): Remove this when buffer assignment is module-scoped.
  static StatusOr<bool> AddCopiesForBufferAssignment(HloModule* module);

  // Try to remove as many copies from the module as possible without
  // introducing live range interference. Only copy instructions that are
  // eligible for copy elision are considered for removal.
  Status RemoveUnnecessaryCopies(const HloOrdering& ordering,
                                 HloModule* module);

 private:
  // Verifies that no HLO values have interfering live ranged assuming the
  // ordering used by copy insertion.
  Status VerifyNoLiveRangeInterference(HloModule* module);

  Status AddCopiesToResolveInterference(HloModule* module);

  Status AddSpecialCaseCopies(const CallGraph& call_graph, HloModule* module);

  // Backend specific function that decides whether a fusion can share buffer
  // with its operand.
  HloDataflowAnalysis::FusionCanShareBufferFunction fusion_can_share_buffer_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
