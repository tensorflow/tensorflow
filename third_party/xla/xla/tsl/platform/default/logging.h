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

#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK_GT
#undef CHECK_GE

#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LT
#undef DCHECK_LE
#undef DCHECK_GT
#undef DCHECK_GE

#undef QCHECK
#undef QCHECK_EQ
#undef QCHECK_NE
#undef QCHECK_LT
#undef QCHECK_LE
#undef QCHECK_GT
#undef QCHECK_GE

#undef PCHECK

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

// CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    CHECK(fp->Write(x) == 4)
#define CHECK(condition)              \
  if (TF_PREDICT_FALSE(!(condition))) \
  LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// CHECK* macros. It's not encouraged though.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline int16 GetReferenceableValue(int16_t t) { return t; }
inline uint16 GetReferenceableValue(uint16 t) { return t; }
inline int GetReferenceableValue(int t) { return t; }
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline int64_t GetReferenceableValue(int64_t t) { return t; }
inline uint64 GetReferenceableValue(uint64 t) { return t; }

// This formats a value for a failing CHECK_XX statement.  Ordinarily,
// it uses the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  (*os) << v;
}

// Overrides for char types provide readable values for unprintable
// characters.
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v);

#if LANG_CXX11
// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v);
#endif

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  explicit CheckOpString(string* str) : str_(str) {}
  // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
  // so there's no point in cleaning up str_.
  explicit operator bool() const { return TF_PREDICT_FALSE(str_ != nullptr); }
  string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2,
                          const char* exprtext) TF_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX
// statement.  See MakeCheckOpString for sample usage.  Other
// approaches were considered: use of a template method (e.g.,
// base::BuildCheckOpString(exprtext, base::Print<T1>, &v1,
// base::Print<T2>, &v2), however this approach has complications
// related to volatile arguments and function-pointer arguments).
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // Deletes "stream_".
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream* ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream* ForVar2();
  // Get the result (inserts the closing ")").
  string* NewString();

 private:
  std::ostringstream* stream_;
};

template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for CHECK_OP macro.
// We use the full name Check_EQ, Check_NE, etc. in case the file including
// absl/log/log.h provides its own #defines for the simpler names EQ, NE, etc.
// This happens if, for example, those are used as token names in a
// yacc grammar.
// The (int, int) overload works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
#define TF_DEFINE_CHECK_OP_IMPL(name, op)                           \
  template <typename T1, typename T2>                               \
  inline string* name##Impl(const T1& v1, const T2& v2,             \
                            const char* exprtext) {                 \
    if (TF_PREDICT_TRUE(v1 op v2))                                  \
      return NULL;                                                  \
    else                                                            \
      return ::tsl::internal::MakeCheckOpString(v1, v2, exprtext);  \
  }                                                                 \
  inline string* name##Impl(int v1, int v2, const char* exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext);                  \
  }

// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.

TF_DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
// Compilation error with CHECK_EQ(NULL, x)?
// Use CHECK(x == NULL) instead.

inline string* Check_EQImpl(int v1, size_t v2, const char* exprtext) {
  if (TF_PREDICT_FALSE(v1 < 0))
    ::tsl::internal::MakeCheckOpString(v1, v2, exprtext);

  return Check_EQImpl(size_t(v1), v2, exprtext);
}

inline string* Check_EQImpl(size_t v1, int v2, const char* exprtext) {
  return Check_EQImpl(v2, v1, exprtext);
}

TF_DEFINE_CHECK_OP_IMPL(Check_NE, !=)

inline string* Check_NEImpl(int v1, size_t v2, const char* exprtext) {
  if (v1 < 0) return NULL;

  return Check_NEImpl(size_t(v1), v2, exprtext);
}

inline string* Check_NEImpl(size_t v1, int v2, const char* exprtext) {
  return Check_NEImpl(v2, v1, exprtext);
}

TF_DEFINE_CHECK_OP_IMPL(Check_LE, <=)

inline string* Check_LEImpl(int v1, size_t v2, const char* exprtext) {
  if (v1 <= 0) return NULL;

  return Check_LEImpl(size_t(v1), v2, exprtext);
}

inline string* Check_LEImpl(size_t v1, int v2, const char* exprtext) {
  if (TF_PREDICT_FALSE(v2 < 0))
    return ::tsl::internal::MakeCheckOpString(v1, v2, exprtext);
  return Check_LEImpl(v1, size_t(v2), exprtext);
}

TF_DEFINE_CHECK_OP_IMPL(Check_LT, <)

inline string* Check_LTImpl(int v1, size_t v2, const char* exprtext) {
  if (v1 < 0) return NULL;

  return Check_LTImpl(size_t(v1), v2, exprtext);
}

inline string* Check_LTImpl(size_t v1, int v2, const char* exprtext) {
  if (v2 < 0) return ::tsl::internal::MakeCheckOpString(v1, v2, exprtext);
  return Check_LTImpl(v1, size_t(v2), exprtext);
}

// Implement GE,GT in terms of LE,LT
template <typename T1, typename T2>
inline string* Check_GEImpl(const T1& v1, const T2& v2, const char* exprtext) {
  return Check_LEImpl(v2, v1, exprtext);
}

template <typename T1, typename T2>
inline string* Check_GTImpl(const T1& v1, const T2& v2, const char* exprtext) {
  return Check_LTImpl(v2, v1, exprtext);
}

#undef TF_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define CHECK_OP_LOG(name, op, val1, val2)                                     \
  while (::tsl::internal::CheckOpString _result{::tsl::internal::name##Impl(   \
      ::tsl::internal::GetReferenceableValue(val1),                            \
      ::tsl::internal::GetReferenceableValue(val2), #val1 " " #op " " #val2)}) \
  ::tsl::internal::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// CHECK_EQ/NE/...
#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)
#define CHECK_NOTNULL(val)                          \
  ::tsl::internal::CheckNotNull(__FILE__, __LINE__, \
                                "'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// DCHECK_EQ/NE/...
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

#else

#define DCHECK(condition) \
  while (false && (condition)) LOG(FATAL)

// NDEBUG is defined, so DCHECK_EQ(x, y) and so on do nothing.
// However, we still want the compiler to parse x and y, because
// we don't want to lose potentially useful errors and warnings.
// _DCHECK_NOP is a helper, and should not be used outside of this file.
#define _TF_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0)) LOG(FATAL)

#define DCHECK_EQ(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_NE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_LE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_LT(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_GE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_GT(x, y) _TF_DCHECK_NOP(x, y)

#endif

// These are for when you don't want a CHECK failure to print a verbose
// stack trace.  The implementation of CHECK* in this file already doesn't.
#define QCHECK(condition) CHECK(condition)
#define QCHECK_EQ(x, y) CHECK_EQ(x, y)
#define QCHECK_NE(x, y) CHECK_NE(x, y)
#define QCHECK_LE(x, y) CHECK_LE(x, y)
#define QCHECK_LT(x, y) CHECK_LT(x, y)
#define QCHECK_GE(x, y) CHECK_GE(x, y)
#define QCHECK_GT(x, y) CHECK_GT(x, y)

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
