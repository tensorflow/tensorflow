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

#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_LOGGING_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_LOGGING_H_

// IWYU pragma: private, include "tsl/platform/logging.h"
// IWYU pragma: friend third_party/tensorflow/tsl/platform/logging.h

#include <sstream>
#include <string>
#include <vector>

#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/base/log_severity.h"
#include "absl/log/check.h"  // IWYU pragma: keep
#include "absl/log/log.h"    // IWYU pragma: keep
#include "absl/strings/string_view.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

// TODO(mrry): Prevent this Windows.h #define from leaking out of our headers.
#undef ERROR

#if defined(ABSL_LTS_RELEASE_VERSION) && ABSL_LTS_RELEASE_VERSION == 20230802
// Abseil's 2023-08-02 LTS release has a bug. It's not defining kLogDebugFatal,
// yet referencing it in the DFATAL log severity level.
// https://github.com/abseil/abseil-cpp/issues/1279 has all the details.
// As a work around we define kLogDebugFatal here. This can be removed once TSL
// was able its Abseil dependency.
namespace absl {
#ifdef NDEBUG
static constexpr absl::LogSeverity kLogDebugFatal = absl::LogSeverity::kError;
#else
static constexpr absl::LogSeverity kLogDebugFatal = absl::LogSeverity::kFatal;
#endif
}  // namespace absl
#endif

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

#ifndef VLOG_IS_ON
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
#endif

#ifndef VLOG
#define VLOG(level)                                       \
  TF_PREDICT_TRUE(!VLOG_IS_ON(level))                     \
  ? (void)0                                               \
  : ::tsl::internal::Voidifier() &                        \
          ::tsl::internal::LogMessage(__FILE__, __LINE__, \
                                      absl::LogSeverity::kInfo)
#endif

// `DVLOG` behaves like `VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing.
#ifndef DVLOG
#ifdef NDEBUG
#define DVLOG VLOG
#else
#define DVLOG(verbose_level) \
  while (false && (verbose_level) > 0) ::tsl::internal::LogMessageNull()
#endif
#endif

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

// LogSink support adapted from //base/logging.h
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

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_LOGGING_H_
