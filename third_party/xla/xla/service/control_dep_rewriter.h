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

#ifndef XLA_SERVICE_CONTROL_DEP_REWRITER_H_
#define XLA_SERVICE_CONTROL_DEP_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Replaces "control_dep" custom calls with actual control dependencies.
//
// "control_dep" is a custom call that's used to add control dependencies
// between HLO instructions. For example, in the snippet of HLO below, the call
// to "control_dep" signals that there should be a control dependency between
// `exp` and `cos`.
//
//     exp = f64[67,67]{1,0} exponential(x)
//     cos = f64[67,67]{1,0} cosine(y)
//     _ = () custom-call(exp, cos), custom_call_target="control_dep"
//
// The ControlDepRewriter pass adds the appropriate control dependencies and
// erases these custom calls.

class ControlDepRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "control-dep-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_CONTROL_DEP_REWRITER_H_
