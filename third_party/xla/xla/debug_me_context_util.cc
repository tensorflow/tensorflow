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

#include "xla/debug_me_context_util.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/tsl/platform/debug_me_context.h"

namespace xla {
namespace debug_me_context_util {

std::string DebugMeContextToErrorMessageString() {
  std::string error_message = "DebugMeContext:\n";
  {
    const std::vector<std::string> compiler_values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(
            DebugMeContextKey::kCompiler);
    absl::StrAppend(&error_message,
                    "Compiler: ", absl::StrJoin(compiler_values, "/"), "\n");
  }
  {
    const std::vector<std::string> hlo_pass_values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(
            DebugMeContextKey::kHloPass);
    absl::StrAppend(&error_message,
                    "HLO Passes: ", absl::StrJoin(hlo_pass_values, "/"), "\n");
  }
  {
    const std::vector<std::string> hlo_instruction_values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(
            DebugMeContextKey::kHloInstruction);
    absl::StrAppend(&error_message, "HLO Instructions: ",
                    absl::StrJoin(hlo_instruction_values, "/"), "\n");
  }
  return error_message;
}

HloPassDebugMeContext::HloPassDebugMeContext(const HloPassInterface* pass)
    : tsl::DebugMeContext<DebugMeContextKey>(DebugMeContextKey::kHloPass,
                                             std::string(pass->name())) {}

}  // namespace debug_me_context_util
}  // namespace xla
