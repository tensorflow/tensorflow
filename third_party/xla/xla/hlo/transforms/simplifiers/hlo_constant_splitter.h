/* Copyright 2022 The OpenXLA Authors.
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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_SPLITTER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_SPLITTER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Splits the constant instructions such that they have a single user.
// This is typically used before domain placement, to make sure a shared
// constant does not short-circuit domains. It is also used before sharding
// propagation to prevent unintended propagation of sharding due to shared used
// of constants.
//
// CSE passes after domain placements will ensure that all the sharable
// constants within the same domain, will be rejoined back.
//
// This pass may generate dead instructions. Thus, HloDCE is recommended after
// this pass.
class HloConstantSplitter : public HloModulePass {
 public:
  explicit HloConstantSplitter(
      bool split_expressions = false,
      absl::FunctionRef<bool(const HloInstruction*)> extra_constraints =
          [](const HloInstruction* instruction) { return true; })
      : split_expressions_(split_expressions),
        extra_constraints_(extra_constraints) {}
  absl::string_view name() const override { return "hlo-constant-splitter"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool split_expressions_;
  absl::FunctionRef<bool(const HloInstruction*)> extra_constraints_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_SPLITTER_H_
