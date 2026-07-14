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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_CALL_PARAMETER_CLEANUP_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_CALL_PARAMETER_CLEANUP_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {

// This pass:
// a) Removes dead (unused) parameters.
// b) Rewrites calls that pass a parameter through s.t. the users of the
//    pass-through parameter instead directly use the operand to the call. If
//    this transformation would make the parameter dead, it is removed.
class CallParameterCleanup : public HloModulePass {
 public:
  CallParameterCleanup() = default;
  ~CallParameterCleanup() override = default;

  static constexpr absl::string_view kName = "call-parameter-cleanup";
  absl::string_view name() const override { return kName; }

 protected:
  // Runs the pass on the given module. Returns whether the module was changed
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_CALL_PARAMETER_CLEANUP_H_
