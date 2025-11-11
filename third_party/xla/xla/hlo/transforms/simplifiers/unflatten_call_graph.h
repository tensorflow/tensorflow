/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_UNFLATTEN_CALL_GRAPH_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_UNFLATTEN_CALL_GRAPH_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Unflatten a call graph. This pass will find called computations that are
// identical and replace them with calls to a single computation.
// Only computations called by kCall instructions will be unflattened.
class UnflattenCallGraph : public HloModulePass {
 public:
  explicit UnflattenCallGraph(bool check_hash_collision = false)
      : check_hash_collision_(check_hash_collision) {}

  absl::string_view name() const override { return "unflatten-call-graph"; }

 protected:
  // Find called computations that are identical and replace them with calls to
  // a single computation. Returns true if the module was changed.
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Whether to check for hash collisions. If true, the pass will check that
  // computations with the same hash are identical to prevent incorrect merging
  // due to hash collisions. This is expensive, so it should only be enabled
  // when hash collisions are suspected.
  bool check_hash_collision_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_UNFLATTEN_CALL_GRAPH_H_
