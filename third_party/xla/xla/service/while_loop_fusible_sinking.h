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
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Sinks values into the while loop body and conditional that fusibles. This is
// probably not a win in isolation but may unlock further optimizations like
// fusible folding. There are two categories:

// 1. Sinks while loop invariant values into the while
// loop body and conditional.
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
// 2. Sinks constant-initialized value, i.e., broadcast(constant) into the while
// body. The high level idea is that we don't want to leave any element of the
// buffer after loop execution as undefined. Therefore, all the elements of the
// buffer must be written to in the body. For element-wise operation 'instr' we
// have:
// forall index i in output shape: instr[i] = f(operand1[i], ...),  where
// f is the elementwise operation.
// We can see that all the indices of the output shape is written to. These
// values can sink into the loop and fused later.
//
//   state = (..., broadcast(constant), ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     value = f(v) // f writes to the entire shape of v.
//     state = (..., value, ...)
//   }
//
// =>
//
//   state = (..., allocate-buffer(), ...)
//   while (pred(state)) {
//     i = iteration_var
//     (..., v, ...) = state
//     new_v = select(i == 0, broadcast(constant), v)
//     value = f(new_v)
//     state = (..., value, ...)
//   }
//
//   This transformation replaces the broadcast with a free AllocateBuffer
//   outside the while loop with the hope that the broadcast inside the loop
//   will be fused.
class WhileLoopFusibleSinking : public HloModulePass {
 public:
  explicit WhileLoopFusibleSinking(bool sink_broadcast_of_constant = true)
      : sink_broadcast_of_constant_(sink_broadcast_of_constant) {}

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
  const bool sink_broadcast_of_constant_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_
