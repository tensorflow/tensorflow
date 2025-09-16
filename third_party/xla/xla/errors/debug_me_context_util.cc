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

#include "xla/errors/debug_me_context_util.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/tsl/platform/debug_me_context.h"

namespace xla::error {

namespace {

// Use the X-Macro to generate an iterable list of all enum values.
#define XLA_DEBUG_ME_CONTEXT_KEY_LIST_ENTRY(name) DebugMeContextKey::k##name,
constexpr DebugMeContextKey kAllDebugMeContextKeys[] = {
    XLA_DEBUG_ME_CONTEXT_KEY_LIST(XLA_DEBUG_ME_CONTEXT_KEY_LIST_ENTRY)};
#undef XLA_DEBUG_ME_CONTEXT_KEY_LIST_ENTRY

}  // namespace

// Use the X-Macro to generate the string conversion function.
// The '#' operator string-izes the 'name' argument.
#define XLA_DEBUG_ME_CONTEXT_KEY_TO_STRING_CASE(name) \
  case DebugMeContextKey::k##name:                    \
    return #name;

std::string DebugMeContextKeyToString(DebugMeContextKey key) {
  switch (key) {
    XLA_DEBUG_ME_CONTEXT_KEY_LIST(XLA_DEBUG_ME_CONTEXT_KEY_TO_STRING_CASE)
  }

  return "Unknown DebugMeContextKey";
}
#undef XLA_DEBUG_ME_CONTEXT_KEY_TO_STRING_CASE

std::string DebugMeContextToErrorMessageString() {
  std::string error_message = "DebugMeContext:\n";

  for (DebugMeContextKey key : kAllDebugMeContextKeys) {
    const std::vector<std::string> values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(key);
    absl::StrAppend(&error_message, DebugMeContextKeyToString(key), ": ",
                    absl::StrJoin(values, "/"), "\n");
  }

  return error_message;
}

HloPassDebugMeContext::HloPassDebugMeContext(const HloPassInterface* pass)
    : tsl::DebugMeContext<DebugMeContextKey>(DebugMeContextKey::kHloPass,
                                             std::string(pass->name())) {}

}  // namespace xla::error
