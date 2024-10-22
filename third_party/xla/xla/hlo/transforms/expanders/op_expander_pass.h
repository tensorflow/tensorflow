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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_OP_EXPANDER_PASS_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_OP_EXPANDER_PASS_H_

#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// This pass is an abstract superclass for passes that replace operations that
// match a pattern. It is intended to be subclassed, not used directly.
//
// This pass is useful for legalizing HLO instructions that a particular backend
// does not support into other HLO instructions.
class OpExpanderPass : public HloModulePass {
 public:
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // extra_filter: Optional extra filtering criteria for matching instructions,
  // used in conjunction with InstructionMatchesPattern.
  // preserve_sharding and relay_control_dependency: If we preserve sharding and
  // relay control dependency when replacing the matched instructions.
  explicit OpExpanderPass(HloPredicate extra_filter = nullptr,
                          bool preserve_sharding = false,
                          bool relay_control_dependency = false)
      : extra_filter_(std::move(extra_filter)),
        preserve_sharding_(preserve_sharding),
        relay_control_dependency_(relay_control_dependency) {}

 protected:
  // Returns `true` if `instruction` should be expanded by this pass.
  virtual bool InstructionMatchesPattern(HloInstruction* instruction) = 0;

  // Returns a replacement for `instruction`, or nullptr if no replacement is
  // needed (e.g. only the to_apply subcomputation of the instruction was
  // modified).
  virtual absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) = 0;

  HloPredicate extra_filter_;
  const bool preserve_sharding_;
  const bool relay_control_dependency_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_OP_EXPANDER_PASS_H_
