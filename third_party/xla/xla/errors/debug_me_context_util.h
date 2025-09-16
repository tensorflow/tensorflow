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

#ifndef XLA_ERRORS_DEBUG_ME_CONTEXT_UTIL_H_
#define XLA_ERRORS_DEBUG_ME_CONTEXT_UTIL_H_

#include <string>

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

// Foward declaration
class HloPassInterface;

namespace error {

enum class DebugMeContextKey {
  kCompiler,
  kHloPass,
  kHloInstruction,
};

// This function extracts all relevant context from the DebugMeContext and
// formats it in a way which is meant to be used when creating error messages in
// XLA.
std::string DebugMeContextToErrorMessageString();

// This class is a specialization of the RAII DebugMeContext specifically for
// HloPasses. The details of its constructor dictate what information from the
// HloPass is stored in the context.
class HloPassDebugMeContext : public tsl::DebugMeContext<DebugMeContextKey> {
 public:
  explicit HloPassDebugMeContext(const HloPassInterface* pass);
};

}  // namespace error
}  // namespace xla

#endif  // XLA_ERRORS_DEBUG_ME_CONTEXT_UTIL_H_
