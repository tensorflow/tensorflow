/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_SIMPLIFIER_H_
#define XLA_SERVICE_WHILE_LOOP_SIMPLIFIER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Tries to remove elements in a while loop's tuple that aren't used within the
// loop.
//
// Specifically, if a loop is tuple-shaped, and there exists some element of
// that tuple that is not used by the loop condition and is not used by the loop
// body except to pass it to the next iteration of the loop, then we can remove
// that element from the loop's tuple.
absl::StatusOr<bool> TryRemoveDeadWhileParams(HloInstruction* while_op);

// HLO pass that makes the following transformations on while loops:
//
//  - A while loop with static trip count of 0 is deleted.
//
//  - A while loop with static trip count of 1 is replaced by its body (sans
//    loop).
//
//  - Elements of a while loop's tuple that the loop doesn't use are removed
//    from the tuple.
//
//  - If the while loop's parameter is a nested tuple, it's flattened to a
//    single-level tuple.  This is good because it usually reduces the number of
//    kTuple instructions, but also because it unlocks additional optimizations
//    (e.g. removing unused loop parameters).
//
//  - Removing trivial compare instructions inside while bodies. Assuming a
//    while loop with known trip count, k, loop induction variable i, and the
//    initial loop induction value c, a compare(i,x) instruction is trivial if:
//      1) x is a constant and x >= k + c.
//      2) x is a constant x <= c.
//
// Flattening nested while loop tuples adds a whole mess of likely unnecessary
// kGetTupleElement and kTuple operations to the graph.  We expect that tuple
// simplifier will be run afterwards.
//
class WhileLoopSimplifier : public HloModulePass {
 public:
  explicit WhileLoopSimplifier(bool simplify_compare_instrs = false)
      : simplify_compare_instrs_(simplify_compare_instrs) {}

  ~WhileLoopSimplifier() override = default;
  absl::string_view name() const override { return "simplify-while-loops"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Whether to remove trivial compare instructions inside while loops.
  const bool simplify_compare_instrs_;
};

}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_SIMPLIFIER_H_
