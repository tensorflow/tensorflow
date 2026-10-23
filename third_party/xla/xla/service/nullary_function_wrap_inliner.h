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

#ifndef XLA_SERVICE_NULLARY_FUNCTION_WRAP_INLINER_H_
#define XLA_SERVICE_NULLARY_FUNCTION_WRAP_INLINER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Inlines call instructions that take a single token operand and nothing else
// (used to wrap nullary instructions for program order preservation) and
// attaches a control dependency from the token operand to the inlined
// instructions.
class NullaryFunctionWrapInliner : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "nullary-function-wrap-inliner";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_NULLARY_FUNCTION_WRAP_INLINER_H_
