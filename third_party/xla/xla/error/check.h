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

#ifndef XLA_ERROR_CHECK_H_
#define XLA_ERROR_CHECK_H_

#include "xla/error/internal/check_impl.h"  // IWYU pragma: keep, b/300560485

// XLA_CHECK is meant to be a drop-in replacement for Abseil's CHECK in XLA. It
// 1. has the exact same API as Abseil's CHECK.
// 2. respects absl flags e.g. ABSL_MIN_LOG_LEVEL
// 3. includes all message content as that of Abseil's CHECK, but additionally
//    a. prepends an error code and appends a link to the openxla.org webpage
//    b. appends DebugMeContext information if available.
//    to the error message.
//
// Example:
//   tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
//                                                  "MyTestPass");
//   XLA_CHECK(false) << "custom message";
//
// crashes with the following error message:
//   E0012: Internal: Check failed: false custom message
//   DebugMeContext:
//     HLO Passes: MyTestPass
#define XLA_CHECK(condition) XLA_INTERNAL_CHECK_IMPL((condition), #condition)

// `XLA_QCHECK` behaves like `XLA_CHECK` but does not print a full stack trace
// and does not run registered error handlers (as `QFATAL`).  It is useful when
// the problem is definitely unrelated to program flow, e.g. when validating
// user input.
#define XLA_QCHECK(condition) XLA_INTERNAL_QCHECK_IMPL((condition), #condition)

// `XLA_DCHECK` behaves like `XLA_CHECK` in debug mode and does nothing
// otherwise (as `XLA_DCHECK`).  Unlike with `CHECK` (but as with `assert`), it
// is not safe to rely on evaluation of `condition`: when `NDEBUG` is enabled,
// DCHECK does not evaluate the condition.
#define XLA_DCHECK(condition) XLA_INTERNAL_DCHECK_IMPL((condition), #condition)

#endif  // XLA_ERROR_CHECK_H_
