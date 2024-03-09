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

#ifndef XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_
#define XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// Sinks while loop invariant values that happen to be constants into the while
// loop body and conditional. This is probably not a win in isolation but may
// unlock further optimizations like constant folding.
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(v)
//     state = (..., v, ...)
//   }
//
// =>
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(const)
//     state = (..., v, ...)
//   }
//
// Note that it leaves the `v` in place to keep that component of the state
// tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
// `v`.
//
class WhileLoopConstantSinking : public HloModulePass {
 public:
  explicit WhileLoopConstantSinking(bool sink_broadcast_of_constants = false)
      : sink_broadcast_of_constants_(sink_broadcast_of_constants) {}

  ~WhileLoopConstantSinking() override = default;

  absl::string_view name() const override {
    return "while-loop-constant-sinking";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> TrySinkingConstantsIntoWhileLoop(HloInstruction* while_instr);

  const bool sink_broadcast_of_constants_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_
