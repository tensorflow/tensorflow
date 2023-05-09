/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_LOGGING_H_
#define TENSORFLOW_LITE_TOOLS_LOGGING_H_

// LOG and CHECK macros for tflite tooling.

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>

#ifdef _WIN32
#undef ERROR
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace tflite {
namespace logging {
// A wrapper that logs to stderr.
//
// Used for TFLITE_LOG and TFLITE_BENCHMARK_CHECK macros.
class LoggingWrapper {
 public:
  enum class LogSeverity : int {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL,
  };
  explicit LoggingWrapper(LogSeverity severity)
      : severity_(severity), should_log_(true) {}
  LoggingWrapper(LogSeverity severity, bool log)
      : severity_(severity), should_log_(log) {}
  std::stringstream& Stream() { return stream_; }
  ~LoggingWrapper() {
    if (should_log_) {
      // Also print log to logcat for android, as stderr will be hidden
      // in the app use case.
#ifdef __ANDROID__
      switch (severity_) {
        case LogSeverity::DEBUG:
          __android_log_print(ANDROID_LOG_DEBUG, "tflite", "%s",
                              stream_.str().c_str());
          break;
        case LogSeverity::INFO:
          __android_log_print(ANDROID_LOG_INFO, "tflite", "%s",
                              stream_.str().c_str());
          break;
        case LogSeverity::WARN:
          __android_log_print(ANDROID_LOG_WARN, "tflite", "%s",
                              stream_.str().c_str());
          break;
        case LogSeverity::ERROR:
          __android_log_print(ANDROID_LOG_ERROR, "tflite", "%s",
                              stream_.str().c_str());
          break;
        case LogSeverity::FATAL:
          __android_log_print(ANDROID_LOG_ERROR, "tflite", "%s",
                              stream_.str().c_str());
      }
#endif
      switch (severity_) {
        case LogSeverity::DEBUG:
#ifndef NDEBUG
          std::cout << "DEBUG: " << stream_.str() << std::endl;
#endif
          break;
        case LogSeverity::INFO:
          std::cout << "INFO: " << stream_.str() << std::endl;
          break;
        case LogSeverity::WARN:
          std::cout << "WARN: " << stream_.str() << std::endl;
          break;
        case LogSeverity::ERROR:
          std::cerr << "ERROR: " << stream_.str() << std::endl;
          break;
        case LogSeverity::FATAL:
          std::cerr << "FATAL: " << stream_.str() << std::endl;
          std::flush(std::cerr);
          std::abort();
          break;
      }
    }
  }

 private:
  std::stringstream stream_;
  LogSeverity severity_;
  bool should_log_;
};
}  // namespace logging
}  // namespace tflite

#define TFLITE_LOG(severity)                                  \
  tflite::logging::LoggingWrapper(                            \
      tflite::logging::LoggingWrapper::LogSeverity::severity) \
      .Stream()

#define TFLITE_MAY_LOG(severity, should_log)                                \
  tflite::logging::LoggingWrapper(                                          \
      tflite::logging::LoggingWrapper::LogSeverity::severity, (should_log)) \
      .Stream()

#define TFLITE_TOOLS_CHECK(condition) TFLITE_MAY_LOG(FATAL, !(condition))

#define TFLITE_TOOLS_CHECK_EQ(a, b) TFLITE_TOOLS_CHECK((a) == (b))

#endif  // TENSORFLOW_LITE_TOOLS_LOGGING_H_
