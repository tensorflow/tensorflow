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

#ifndef XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_
#define XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// Sinks while loop invariant values that happen to be fusibles into the while
// loop body and conditional. This is probably not a win in isolation but may
// unlock further optimizations like fusible folding.
//
//   state = (..., fusible_graph, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(v)
//     state = (..., v, ...)
//   }
//
// =>
//
//   state = (..., fusbile_graph, ..., fusible_graph_operands)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(fusibile_graph)
//     state = (..., v, ...)
//   }
//
// Note that it leaves the `v` in place to keep that component of the state
// tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
// `v`.
//
class WhileLoopFusibleSinking : public HloModulePass {
 public:
  WhileLoopFusibleSinking() = default;

  ~WhileLoopFusibleSinking() override = default;

  absl::string_view name() const override {
    return "while-loop-fusible-sinking";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Sink a fusible subgraph into a while loop.
  absl::StatusOr<bool> TrySinkingFusiblesIntoWhileLoop(
      HloInstruction* while_instr);

  // Creates a loop fusion instruction containing the computation to move into
  // the while loop to avoid conflicts with actual instruction fusion, the loop
  // fusion will be defused.
  bool IsSinkableFusion(HloInstruction* while_operand);
  HloInstruction* CreateSinkableFusion(HloInstruction* while_operand);

  absl::flat_hash_map<HloComputation*, int> call_counts_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_
