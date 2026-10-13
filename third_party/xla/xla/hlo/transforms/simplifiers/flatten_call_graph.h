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

// Flatten the call graph for an HLO module into a tree.

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_FLATTEN_CALL_GRAPH_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_FLATTEN_CALL_GRAPH_H_

#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Flattening associates each call site with a unique computation (for
// sequential calling contexts) This simplifies buffer assignment and
// points-to analysis (see b/36865746 for details).
//
// Keeps the single version of a computation on the entire module that
// `skip_cloning_handler_` predicate is true.
//
// If `skip_cloning_if_does_not_contain_handler_` is provided, keeps the single
// version of a computation on the entire module that does not transitively
// contain an instruction that `skip_cloning_if_does_not_contain_handler_`
// predicate is true.
//
// Note that the two predicates for skipping cloning combines as an OR. That is,
// a computation skips being cloned if it holds either of the two predicates.
//
// TODO(b/260601110): Combine the two skip handler.
class FlattenCallGraph : public HloModulePass {
 public:
  FlattenCallGraph()
      : skip_cloning_handler_([](const HloComputation&) { return false; }),
        skip_cloning_if_does_not_contain_handler_(std::nullopt) {}
  explicit FlattenCallGraph(
      absl::AnyInvocable<bool(const HloComputation&)> skip_cloning_handler,
      std::optional<absl::AnyInvocable<bool(const HloInstruction&)>>
          skip_cloning_if_does_not_contain_handler = std::nullopt)
      : skip_cloning_handler_(std::move(skip_cloning_handler)),
        skip_cloning_if_does_not_contain_handler_(
            std::move(skip_cloning_if_does_not_contain_handler)) {}

  // A skip cloning handler that skips cloning for computations that are only
  // called by kCall instructions.
  static bool SkipCloningForCalls(const HloComputation& computation);

  // A skip cloning handler that only clones asynchronous computations (i.e.,
  // computations called by kAsyncStart, kAsyncUpdate, or kAsyncDone).
  static bool SkipCloningForNonAsync(const HloComputation& computation);

  absl::string_view name() const override { return "flatten-call-graph"; }

 protected:
  // Duplicates computations called from multiple call- or while-nodes to
  // flatten the call graph.
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::AnyInvocable<bool(const HloComputation&)> skip_cloning_handler_;
  std::optional<absl::AnyInvocable<bool(const HloInstruction&)>>
      skip_cloning_if_does_not_contain_handler_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_FLATTEN_CALL_GRAPH_H_
