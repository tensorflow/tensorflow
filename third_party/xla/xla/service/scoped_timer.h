/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SCOPED_TIMER_H_
#define XLA_SERVICE_SCOPED_TIMER_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

// Creates a an xla::ScopedLoggingTimer, and an xprof::TraceMe with the given
// label.  The timer is enabled at VLOG(1) and above, and the TraceMe has level
// 2.
#define XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME(label) \
  XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME_HELPER(label, 1, __COUNTER__)

// Creates a xla::ScopedLoggingTimer, and an xprof::TraceMe with the given label
// and level.  The timer is enabled at VLOG(level) and above, and the TraceMe's
// level is level + 1.
#define XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME_LEVEL(label, level) \
  XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME_HELPER(label, level, __COUNTER__)

// Helper for implementing macros above.  Don't use directly.
//
// Forces the evaluation of "counter", which we expect is equal to __COUNTER__.
#define XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME_HELPER(label, level, counter) \
  XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME_HELPER2(label, level, counter)

// Helper for macros above.  Don't use directly.
//
// This evaluates "level" twice.  Sorry.
#define XLA_SCOPED_LOGGING_TIMER_AND_TRACE_ME_HELPER2(label, level, counter) \
  static ::xla::TimerStats XLA_TimerTraceStats##counter;                     \
  ::xla::ScopedLoggingTimerAndTraceMe                                        \
      XLA_ScopedLoggingTimerAndTraceMeInstance##counter(                     \
          label, VLOG_IS_ON(level), (level) + 1,                             \
          &XLA_TimerTraceStats##counter);

// Creates an xla::ScopedLoggingTimer, and an xprof::TraceMe with the given
// label.  The timer is enabled iff timer_enabled is true, and the TraceMe has
// level trace_level.
//
// Recommended usage of this class is via the macros above.
struct ScopedLoggingTimerAndTraceMe {
  ScopedLoggingTimerAndTraceMe(
      absl::string_view label, bool timer_enabled, int32_t trace_level,
      TimerStats *timer_stats,
      absl::SourceLocation source_location = absl::SourceLocation::current())
      : timer(label, /*enabled=*/timer_enabled, source_location.file_name(),
              source_location.line(), timer_stats),
        traceme(label, trace_level) {}

 private:
  ScopedLoggingTimer timer;
  tsl::profiler::TraceMe traceme;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCOPED_TIMER_H_
