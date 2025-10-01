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

#ifndef XLA_ERROR_DEBUG_ME_CONTEXT_UTIL_H_
#define XLA_ERROR_DEBUG_ME_CONTEXT_UTIL_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/debug_me_context.h"

// This file provides XLA-specific specializations and utilities for the
// thread-local debugging context system.
//
// The primary goal is to capture the XLA compiler's state (e.g., which HLO
// pass is running) to provide more insightful diagnostic and error messages.
//
// This system is built on the generic `tsl::DebugMeContext` class. For a
// detailed explanation of the underlying RAII mechanism and thread-local
// behavior, please see the comment on that class.

namespace xla {

namespace error {

// The canonical type URL for the DebugContextPayload.
// This URL is used to attach the payload to an absl::Status.
constexpr absl::string_view kDebugContextPayloadUrl =
    "types.googleapis.com/xla.errors.DebugContextPayload";

// Enumerate different types of debug context keys. These keys are used to
// identify the type of context being stored in the thread-local DebugMeContext.
enum class DebugMeContextKey : std::uint8_t {
  kCompiler,
  kHloPass,
  kHloInstruction,
};

// This function extracts all relevant context from the DebugMeContext and
// formats it in a way which is meant to be used when creating error messages in
// XLA.
std::string DebugMeContextToErrorMessageString();

// Attaches the DebugMeContextToErrorMessageString as a payload to the given
// status, if the context is not empty and the status is not OK.
void AttachDebugMeContextPayload(absl::Status& status);

}  // namespace error
}  // namespace xla

#endif  // XLA_ERROR_DEBUG_ME_CONTEXT_UTIL_H_
