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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_DEAD_DYNAMIC_UPDATE_SLICE_ELIMINATION_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_DEAD_DYNAMIC_UPDATE_SLICE_ELIMINATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass that removes dynamic-update-slice (DUS) instructions if the region
// they modify is never accessed by any downstream operations.
//
// This optimization applies if all users of a DUS are slice instructions, its
// indices are constant, and none of its slice users read from the region
// updated by the DUS. If these conditions are met, the pass replaces all uses
// of the DUS with its input operand and removes the DUS instruction. The
// optimization is applied from root to top, so that if any DUS is removed that
// makes certain upstream DUSs removable, those will also be removed in the same
// pass.
class DeadDynamicUpdateSliceElimination : public HloModulePass {
 public:
  DeadDynamicUpdateSliceElimination() = default;
  ~DeadDynamicUpdateSliceElimination() override = default;
  absl::string_view name() const override { return "dead-dus-elimination"; }

  // Run the pass on the given module. Returns whether the module was changed
  // (instructions were removed).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_DEAD_DYNAMIC_UPDATE_SLICE_ELIMINATION_H_
