/* Copyright 2026 The OpenXLA Authors.

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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_inliner.h"

#ifndef XLA_SERVICE_CALL_MARKER_H_
#define XLA_SERVICE_CALL_MARKER_H_

namespace xla {

// The target name of the custom call marking the beginning of an outlinable
// block.
inline constexpr absl::string_view kCallMarkerBeforeTarget =
    "__xla_internal_call_marker_before";

// The target name of the custom call marking the end of an outlinable block.
inline constexpr absl::string_view kCallMarkerAfterTarget =
    "__xla_internal_call_marker_after";

// The key used in the frontend attributes of the call markers to store the name
// of the computation that is being marked for outlining.
inline constexpr absl::string_view kCallMarkedComputationAttribute =
    "xla_call_marked_computation";

// This pass marks the call instructions in the module by wrapping them with
// custom call instructions, which will be used to identify the call
// instructions during the call outliner pass.
//
// For example, the following HLO:
//
//   %param_0 = ...
//   %param_1 = ...
//   %call = target_shape call(%param_0, %param_1), to_apply=my_computation
//   ...
//
// Will be transformed into:
//
//   %param_0 = ...
//   %param_1 = ...
//   %call_before = (op_0_shape, op_1_shape) custom-call(%param_0, %param_1),
//     custom_call_target="__xla_internal_call_marker_before",
//     frontend_attributes={xla_call_marked_computation="my_computation"}
//   %gte_0 = op_0_shape get-tuple-element(%call_before), index=0
//   %gte_1 = op_1_shape get-tuple-element(%call_before), index=1
//   %call = target_shape call(%gte_0, %gte_1), to_apply=my_computation
//   %call_after = target_shape custom-call(%call),
//     custom_call_target="__xla_internal_call_marker_after",
//     frontend_attributes={xla_call_marked_computation="my_computation"}
//   ...
//
// Metadata (sharding, HLO metadata, backend config, original value,
// and frontend attributes) from the original call instruction is copied to
// the 'after' marker. Control dependencies are adjusted so that control
// predecessors execute before the 'before' marker, and control successors
// execute after the 'after' marker.
//
class CallMarker : public HloModulePass {
 public:
  explicit CallMarker(const CallInliner& inliner) : inliner_(inliner) {}
  absl::string_view name() const override { return "call-marker"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const CallInliner& inliner_;
};
}  // namespace xla

#endif  // XLA_SERVICE_CALL_MARKER_H_
