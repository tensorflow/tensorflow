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

#ifndef XLA_ERROR_INTERNAL_CHECK_IMPL_H_
#define XLA_ERROR_INTERNAL_CHECK_IMPL_H_

#include <cstdlib>  // IWYU pragma: keep

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "xla/error/internal/check_helper.h"  // IWYU pragma: keep, b/300560485

namespace xla::error {

// Helper class to allow us to use the `&&` operator in the `XLA_CHECK` macros.
class Voidify final {
 public:
  // This has to be an operator with a precedence lower than << but higher than
  // ?:
  template <typename T>
  ABSL_ATTRIBUTE_COLD void operator&&(T&& message) const&& {
    // The dispatching of the completed `absl::LogEntry` to applicable
    // `absl::LogSink`s happens here.
    message.flush();
  }
};

[[noreturn]] inline void AbortQuietly() { abort(); }

}  // namespace xla::error

#define XLA_INTERNAL_CONDITION(condition) \
  switch (0)                              \
  case 0:                                 \
  default:                                \
    !(condition) ? (void)0 : ::xla::error::Voidify() &&

#ifdef ABSL_MIN_LOG_LEVEL
#define XLA_INTERNAL_CONDITION_FATAL(condition)                        \
  XLA_INTERNAL_CONDITION(                                              \
      ((condition) ? (::absl::LogSeverity::kFatal >=                   \
                              static_cast<::absl::LogSeverityAtLeast>( \
                                  ABSL_MIN_LOG_LEVEL)                  \
                          ? true                                       \
                          : (xla::error::AbortQuietly(), false))       \
                   : false))
#else  // ndef ABSL_MIN_LOG_LEVEL
#define XLA_INTERNAL_CONDITION_FATAL(condition) \
  XLA_INTERNAL_CONDITION(condition)
#endif  // ABSL_MIN_LOG_LEVEL

#ifdef ABSL_MIN_LOG_LEVEL
#define XLA_INTERNAL_CONDITION_QFATAL(condition)                       \
  XLA_INTERNAL_CONDITION(                                              \
      ((condition) ? (::absl::LogSeverity::kFatal >=                   \
                              static_cast<::absl::LogSeverityAtLeast>( \
                                  ABSL_MIN_LOG_LEVEL)                  \
                          ? true                                       \
                          : (xla::error::ExitQuietly(), false))        \
                   : false))
#else  // ndef ABSL_MIN_LOG_LEVEL
#define XLA_INTERNAL_CONDITION_QFATAL(condition) \
  XLA_INTERNAL_CONDITION(condition)
#endif  // ABSL_MIN_LOG_LEVEL

#define XLA_INTERNAL_CHECK_IMPL(condition, condition_text)       \
  XLA_INTERNAL_CONDITION_FATAL(ABSL_PREDICT_FALSE(!(condition))) \
  CheckHelper(__FILE__, __LINE__, condition_text).InternalStream()

#define XLA_INTERNAL_QCHECK_IMPL(condition, condition_text)           \
  XLA_INTERNAL_CONDITION_QFATAL(ABSL_PREDICT_FALSE(!(condition)))     \
  CheckHelper(__FILE__, __LINE__, condition_text, CheckType::kQFatal) \
      .InternalStream()

// DCHECK impl
#ifndef NDEBUG
#define XLA_INTERNAL_DCHECK_IMPL(condition, condition_text) \
  XLA_INTERNAL_CHECK_IMPL(condition, condition_text)
#else
#define XLA_INTERNAL_DCHECK_IMPL(condition, condition_text) \
  XLA_INTERNAL_CHECK_IMPL(true || (condition), "true")
#endif

#endif  // XLA_ERROR_INTERNAL_CHECK_IMPL_H_
