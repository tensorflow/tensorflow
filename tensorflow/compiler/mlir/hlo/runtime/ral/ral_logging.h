/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_LOGGING_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_LOGGING_H_

#include <atomic>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

namespace mlir {
namespace disc_ral {

// -------------------------------------------------------------------

// This file contains a light version of Logger implemented in
// tensorflow/core/platform/default/logging.h
//
// We re-implement it here because we do not want to rely
// on TensorFlow data structures, and hence we can easily integrate this
// part of code to targeting environments (e.g. TF, PyTorch).

// --------------------------------------------------------------------

const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace internal {

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if __has_builtin(__builtin_expect)
#define DISC_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define DISC_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define DISC_PREDICT_FALSE(x) (x)
#define DISC_PREDICT_TRUE(x) (x)
#endif

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int MinVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable DISC_CPP_VMODULE contains foo=3 and fname
  // is foo.cc and lvl is <= 3, this will return true. It will also return true
  // if the level is lower or equal to DISC_CPP_MIN_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char* fname, int level);

  // Testing only. Returns output string if severity of this logger is larger
  // than or equal to `min_log_level`.
  std::string GetFilterStringForTesting(int min_log_level);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  // clang-format off
  template <typename T>
  void operator&(const T&) const {}
  // clang-format on
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal() override;
};

}  // namespace internal

#define _DISC_LOG_INFO                                       \
  ::mlir::disc_ral::internal::LogMessage(__FILE__, __LINE__, \
                                         ::mlir::disc_ral::INFO)
#define _DISC_LOG_WARNING                                    \
  ::mlir::disc_ral::internal::LogMessage(__FILE__, __LINE__, \
                                         ::mlir::disc_ral::WARNING)
#define _DISC_LOG_ERROR                                      \
  ::mlir::disc_ral::internal::LogMessage(__FILE__, __LINE__, \
                                         ::mlir::disc_ral::ERROR)
#define _DISC_LOG_FATAL \
  ::mlir::disc_ral::internal::LogMessageFatal(__FILE__, __LINE__)

#define DISC_LOG(severity) _DISC_LOG_##severity

#define DISC_VLOG_IS_ON(lvl)                                             \
  (([](int level, const char* fname) {                                   \
    static const bool vmodule_activated =                                \
        ::mlir::disc_ral::internal::LogMessage::VmoduleActivated(fname,  \
                                                                 level); \
    return vmodule_activated;                                            \
  })(lvl, __FILE__))

#define DISC_VLOG(level)                                             \
  DISC_PREDICT_TRUE(!DISC_VLOG_IS_ON(level))                         \
  ? (void)0                                                          \
  : ::mlir::disc_ral::internal::Voidifier() &                        \
          ::mlir::disc_ral::internal::LogMessage(__FILE__, __LINE__, \
                                                 ::mlir::disc_ral::INFO)

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_LOGGING_H_
