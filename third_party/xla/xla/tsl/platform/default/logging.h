/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if defined(_WIN32)
// prevent compile error because MSVC doesn't realize in debug build that
// LOG(FATAL) finally invokes abort()
#pragma warning(disable : 4716)
#endif  // _WIN32

#ifndef XLA_TSL_PLATFORM_DEFAULT_LOGGING_H_
#define XLA_TSL_PLATFORM_DEFAULT_LOGGING_H_

// IWYU pragma: private, include "xla/tsl/platform/logging.h"
// IWYU pragma: friend third_party/tensorflow/compiler/xla/tsl/platform/logging.h

#include <atomic>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/check.h"  // IWYU pragma: export
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"

// TODO(mrry): Prevent this Windows.h #define from leaking out of our headers.
#undef ERROR

// Undef everything in case we're being mixed with some other Google library
// which already defined them itself.  Presumably all Google libraries will
// support the same syntax for these so it should not be a big deal if they
// end up using our definitions instead.
#undef LOG
#undef LOG_EVERY_N
#undef LOG_FIRST_N
#undef LOG_EVERY_POW_2
#undef LOG_EVERY_N_SEC
#undef VLOG

namespace tsl {

namespace internal {

// Emit "message" as a log message to the log for the specified
// "severity" as if it came from a LOG call at "fname:line"
void LogString(const char* fname, int line, absl::LogSeverity severity,
               const std::string& message);

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, absl::LogSeverity severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the maximum log level for VLOG statements.
  // E.g., if MaxVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int MaxVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable TF_CPP_VMODULE contains foo=3 and fname is
  // foo.cc and lvl is <= 3, this will return true. It will also return true if
  // the level is lower or equal to TF_CPP_MAX_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char* fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  absl::LogSeverity severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  template <typename T>
  void operator&(const T&) const {}
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) TF_ATTRIBUTE_COLD;
  TF_ATTRIBUTE_NORETURN ~LogMessageFatal() override;
};

// LogMessageNull supports the DVLOG macro by simply dropping any log messages.
class LogMessageNull : public std::basic_ostringstream<char> {
 public:
  LogMessageNull() = default;
  ~LogMessageNull() override {}
};

#define _TF_LOG_INFO \
  ::tsl::internal::LogMessage(__FILE__, __LINE__, absl::LogSeverity::kInfo)
#define _TF_LOG_WARNING \
  ::tsl::internal::LogMessage(__FILE__, __LINE__, absl::LogSeverity::kWarning)
#define _TF_LOG_ERROR \
  ::tsl::internal::LogMessage(__FILE__, __LINE__, absl::LogSeverity::kError)
#define _TF_LOG_FATAL ::tsl::internal::LogMessageFatal(__FILE__, __LINE__)

#define _TF_LOG_QFATAL _TF_LOG_FATAL

#ifdef NDEBUG
#define _TF_LOG_DFATAL _TF_LOG_ERROR
#else
#define _TF_LOG_DFATAL _TF_LOG_FATAL
#endif

#define LOG(severity) _TF_LOG_##severity

#ifdef IS_MOBILE_PLATFORM

// Turn VLOG off when under mobile devices for considerations of binary size.
#define VLOG_IS_ON(lvl) ((lvl) <= 0)

#else

// Otherwise, set TF_CPP_MAX_VLOG_LEVEL environment to update minimum log level
// of VLOG, or TF_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define VLOG_IS_ON(lvl)                                              \
  (([](int level, const char* fname) {                               \
    static const bool vmodule_activated =                            \
        ::tsl::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                        \
  })(lvl, __FILE__))

#endif

#define VLOG(level)                                       \
  TF_PREDICT_TRUE(!VLOG_IS_ON(level))                     \
  ? (void)0                                               \
  : ::tsl::internal::Voidifier() &                        \
          ::tsl::internal::LogMessage(__FILE__, __LINE__, \
                                      absl::LogSeverity::kInfo)

// `DVLOG` behaves like `VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing.
#ifndef NDEBUG
#define DVLOG VLOG
#else
#define DVLOG(verbose_level) \
  while (false && (verbose_level) > 0) ::tsl::internal::LogMessageNull()
#endif

class LogEveryNState {
 public:
  bool ShouldLog(int n);
  uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogFirstNState {
 public:
  bool ShouldLog(int n);
  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogEveryPow2State {
 public:
  bool ShouldLog(int ignored);
  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogEveryNSecState {
 public:
  bool ShouldLog(double seconds);
  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
  // Cycle count according to CycleClock that we should next log at.
  std::atomic<int64_t> next_log_time_cycles_{0};
};

// This macro has a lot going on!
//
// * A local static (`logging_internal_stateful_condition_state`) is
//   declared in a scope such that each `LOG_EVERY_N` (etc.) line has its own
//   state.
// * `COUNTER`, the third variable, is used to support `<< COUNTER`. It is not
//   mangled, so shadowing can be a problem, albeit more of a
//   shoot-yourself-in-the-foot one.  Don't name your variables `COUNTER`.
// * A single for loop can declare state and also test
//   `condition && state.ShouldLog()`, but there's no way to constrain it to run
//   only once (or not at all) without declaring another variable.  The outer
//   for-loop declares this variable (`do_log`).
// * Using for loops instead of if statements means there's no risk of an
//   ambiguous dangling else statement.
#define LOGGING_INTERNAL_STATEFUL_CONDITION(kind, condition, arg)   \
  for (bool logging_internal_stateful_condition_do_log(condition);  \
       logging_internal_stateful_condition_do_log;                  \
       logging_internal_stateful_condition_do_log = false)          \
    for (static ::tsl::internal::Log##kind##State                   \
             logging_internal_stateful_condition_state;             \
         logging_internal_stateful_condition_do_log &&              \
         logging_internal_stateful_condition_state.ShouldLog(arg);  \
         logging_internal_stateful_condition_do_log = false)        \
      for (const uint32_t COUNTER ABSL_ATTRIBUTE_UNUSED =           \
               logging_internal_stateful_condition_state.counter(); \
           logging_internal_stateful_condition_do_log;              \
           logging_internal_stateful_condition_do_log = false)

// An instance of `LOG_EVERY_N` increments a hidden zero-initialized counter
// every time execution passes through it and logs the specified message when
// the counter's value is a multiple of `n`, doing nothing otherwise.  Each
// instance has its own counter.  The counter's value can be logged by streaming
// the symbol `COUNTER`.  `LOG_EVERY_N` is thread-safe.
// Example:
//
//   for (const auto& user : all_users) {
//     LOG_EVERY_N(INFO, 1000) << "Processing user #" << COUNTER;
//     ProcessUser(user);
//   }
#define LOG_EVERY_N(severity, n)                       \
  LOGGING_INTERNAL_STATEFUL_CONDITION(EveryN, true, n) \
  LOG(severity)
// `LOG_FIRST_N` behaves like `LOG_EVERY_N` except that the specified message is
// logged when the counter's value is less than `n`.  `LOG_FIRST_N` is
// thread-safe.
#define LOG_FIRST_N(severity, n)                       \
  LOGGING_INTERNAL_STATEFUL_CONDITION(FirstN, true, n) \
  LOG(severity)
// `LOG_EVERY_POW_2` behaves like `LOG_EVERY_N` except that the specified
// message is logged when the counter's value is a power of 2.
// `LOG_EVERY_POW_2` is thread-safe.
#define LOG_EVERY_POW_2(severity)                         \
  LOGGING_INTERNAL_STATEFUL_CONDITION(EveryPow2, true, 0) \
  LOG(severity)
// An instance of `LOG_EVERY_N_SEC` uses a hidden state variable to log the
// specified message at most once every `n_seconds`.  A hidden counter of
// executions (whether a message is logged or not) is also maintained and can be
// logged by streaming the symbol `COUNTER`.  `LOG_EVERY_N_SEC` is thread-safe.
// Example:
//
//   LOG_EVERY_N_SEC(INFO, 2.5) << "Got " << COUNTER << " cookies so far";
#define LOG_EVERY_N_SEC(severity, n_seconds)                      \
  LOGGING_INTERNAL_STATEFUL_CONDITION(EveryNSec, true, n_seconds) \
  LOG(severity)

#define CHECK_NOTNULL(val)                          \
  ::tsl::internal::CheckNotNull(__FILE__, __LINE__, \
                                "'" #val "' Must be non NULL", (val))

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << string(exprtext);
  }
  return std::forward<T>(t);
}

absl::LogSeverityAtLeast MinLogLevelFromEnv();

int MaxVLogLevelFromEnv();

}  // namespace internal

// LogSink support adapted from absl/log/log.h
//
// `LogSink` is an interface which can be extended to intercept and process
// all log messages. LogSink implementations must be thread-safe. A single
// instance will be called from whichever thread is performing a logging
// operation.
class TFLogEntry {
 public:
  explicit TFLogEntry(absl::LogSeverity severity, absl::string_view message)
      : severity_(severity), message_(message) {}

  explicit TFLogEntry(absl::LogSeverity severity, absl::string_view fname,
                      int line, absl::string_view message)
      : severity_(severity), fname_(fname), line_(line), message_(message) {}

  absl::LogSeverity log_severity() const { return severity_; }
  std::string FName() const { return fname_; }
  int Line() const { return line_; }
  std::string ToString() const { return message_; }
  absl::string_view text_message() const { return message_; }

  // Returning similar result as `text_message` as there is no prefix in this
  // implementation.
  absl::string_view text_message_with_prefix() const { return message_; }

 private:
  const absl::LogSeverity severity_;
  const std::string fname_;
  int line_ = -1;
  const std::string message_;
};

class TFLogSink {
 public:
  virtual ~TFLogSink() = default;

  // `Send` is called synchronously during the log statement.  The logging
  // module guarantees not to call `Send` concurrently on the same log sink.
  // Implementations should be careful not to call`LOG` or `CHECK` or take
  // any locks that might be held by the `LOG` caller, to avoid deadlock.
  //
  // `e` is guaranteed to remain valid until the subsequent call to
  // `WaitTillSent` completes, so implementations may store a pointer to or
  // copy of `e` (e.g. in a thread local variable) for use in `WaitTillSent`.
  virtual void Send(const TFLogEntry& entry) = 0;

  // `WaitTillSent` blocks the calling thread (the thread that generated a log
  // message) until the sink has finished processing the log message.
  // `WaitTillSent` is called once per log message, following the call to
  // `Send`.  This may be useful when log messages are buffered or processed
  // asynchronously by an expensive log sink.
  // The default implementation returns immediately.  Like `Send`,
  // implementations should be careful not to call `LOG` or `CHECK or take any
  // locks that might be held by the `LOG` caller, to avoid deadlock.
  virtual void WaitTillSent() {}
};

// This is the default log sink. This log sink is used if there are no other
// log sinks registered. To disable the default log sink, set the
// "no_default_logger" Bazel config setting to true or define a
// NO_DEFAULT_LOGGER preprocessor symbol. This log sink will always log to
// stderr.
class TFDefaultLogSink : public TFLogSink {
 public:
  void Send(const TFLogEntry& entry) override;
};

// Add or remove a `LogSink` as a consumer of logging data.  Thread-safe.
void TFAddLogSink(TFLogSink* sink);
void TFRemoveLogSink(TFLogSink* sink);

// Get all the log sinks.  Thread-safe.
std::vector<TFLogSink*> TFGetLogSinks();

// Change verbose level of pre-defined files if envorionment
// variable `env_var` is defined. This is currently a no op.
void UpdateLogVerbosityIfDefined(const char* env_var);

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_DEFAULT_LOGGING_H_
