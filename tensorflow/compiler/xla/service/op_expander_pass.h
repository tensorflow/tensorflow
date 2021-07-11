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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_OP_EXPANDER_PASS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_OP_EXPANDER_PASS_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// This pass is an abstract superclass for passes that replace operations that
// match a pattern. It is intended to be subclassed, not used directly.
//
// This pass is useful for legalizing HLO instructions that a particular backend
// does not support into other HLO instructions.
class OpExpanderPass : public HloModulePass {
 public:
  using PatternExtraFilter = std::function<bool(const HloInstruction*)>;

  StatusOr<bool> Run(HloModule* module) override;

  // extra_filter: Optional extra filtering criteria for matching instructions,
  // used in conjunction with InstructionMatchesPattern.
  explicit OpExpanderPass(PatternExtraFilter extra_filter = nullptr)
      : extra_filter_(std::move(extra_filter)) {}

 protected:
  // Returns `true` if `instruction` should be expanded by this pass.
  virtual bool InstructionMatchesPattern(HloInstruction* instruction) = 0;

  // Returns a replacement for `instruction`, or nullptr if no replacement is
  // needed (e.g. only the to_apply subcomputation of the instruction was
  // modified).
  virtual StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) = 0;

  PatternExtraFilter extra_filter_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OP_EXPANDER_PASS_H_
