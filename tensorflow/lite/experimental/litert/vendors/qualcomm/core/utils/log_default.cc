// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "log.h"

namespace qnn {

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
FILE* QNNLogger::log_file_pointer_ = stderr;
TfLiteQnnDelegateLogLevel QNNLogger::log_level_ = kLogLevelVerbose;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
void QNNLogger::SetLogFilePointer(FILE* fp) { log_file_pointer_ = fp; }
void QNNLogger::SetLogLevel(TfLiteQnnDelegateLogLevel log_level) {
  log_level_ = log_level;
}
// NOLINTNEXTLINE(cert-dcl50-cpp)
void QNNLogger::Log(TfLiteQnnDelegateLogLevel severity, const char* format,
                    ...) {
  if (severity > log_level_) {
    return;
  }

  // Pass to LogFormatted
  va_list args;
  va_start(args, format);

  // Print to file pointer.
  vfprintf(log_file_pointer_, format, args);
  fputc('\n', log_file_pointer_);

  va_end(args);
}
}  // namespace qnn
