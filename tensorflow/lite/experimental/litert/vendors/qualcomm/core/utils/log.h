// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_UTILS_LOG_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_UTILS_LOG_H_

#include <cstdio>

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/common.h"

namespace qnn {

class QNNLogger {
 public:
  // Logging hook that takes variadic args.
  static void Log(LiteRtQnnLogLevel severity, const char* format, ...);

  // Set file descriptor
  static void SetLogFilePointer(FILE* fp);

  // Set log level
  static void SetLogLevel(LiteRtQnnLogLevel log_level);

 private:
  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  static FILE* log_file_pointer_;
  static LiteRtQnnLogLevel log_level_;
  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
};
}  // namespace qnn

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_VERBOSE(format, ...)                                  \
  ::qnn::QNNLogger::Log(kLogLevelVerbose, ("VERBOSE: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_INFO(format, ...) \
  ::qnn::QNNLogger::Log(kLogLevelInfo, ("INFO: [Qnn] " format), ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_WARNING(format, ...)                               \
  ::qnn::QNNLogger::Log(kLogLevelWarn, ("WARNING: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_ERROR(format, ...)                                \
  ::qnn::QNNLogger::Log(kLogLevelError, ("ERROR: [Qnn] " format), \
                        ##__VA_ARGS__);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QNN_LOG_DEBUG(format, ...)                                \
  ::qnn::QNNLogger::Log(kLogLevelDebug, ("DEBUG: [Qnn] " format), \
                        ##__VA_ARGS__);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_UTILS_LOG_H_
