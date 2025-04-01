// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <android/log.h>

#include "log.h"

namespace qnn {
namespace {

int GetPlatformSeverity(LiteRtQnnLogLevel severity) {
  switch (severity) {
    case kLogLevelError:
      return ANDROID_LOG_ERROR;
    case kLogLevelWarn:
      return ANDROID_LOG_WARN;
    case kLogLevelInfo:
      return ANDROID_LOG_INFO;
    case kLogLevelVerbose:
      return ANDROID_LOG_VERBOSE;
    default:
      return ANDROID_LOG_DEBUG;
  }
}

}  // namespace

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
FILE* QNNLogger::log_file_pointer_ = stderr;
LiteRtQnnLogLevel QNNLogger::log_level_ = kLogLevelInfo;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
void QNNLogger::SetLogFilePointer(FILE* fp) { log_file_pointer_ = fp; }
void QNNLogger::SetLogLevel(LiteRtQnnLogLevel log_level) {
  log_level_ = log_level;
}
// NOLINTNEXTLINE(cert-dcl50-cpp)
void QNNLogger::Log(LiteRtQnnLogLevel severity, const char* format, ...) {
  if (severity > log_level_) {
    return;
  }

  // Pass to LogFormatted
  va_list args;
  va_start(args, format);

  // First log to Android's explicit log(cat) API.
  va_list args_copy;
  va_copy(args_copy, args);
  __android_log_vprint(GetPlatformSeverity(severity), "qnn", format, args_copy);
  va_end(args_copy);

  // Print to file pointer.
  vfprintf(log_file_pointer_, format, args);
  fputc('\n', log_file_pointer_);

  va_end(args);
}
}  // namespace qnn
