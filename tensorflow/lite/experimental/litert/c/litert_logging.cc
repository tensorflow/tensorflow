// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"

#include <cstdarg>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

class LiteRtLoggerT {
 public:
  LiteRtLogSeverity GetMinSeverity() {
    return ConvertSeverity(
        tflite::logging_internal::MinimalLogger::GetMinimumLogSeverity());
  }

  void SetMinSeverity(LiteRtLogSeverity severity) {
    tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
        ConvertSeverity(severity));
  }

  void Log(LiteRtLogSeverity severity, const char* format, va_list args) {
    tflite::logging_internal::MinimalLogger::LogFormatted(
        ConvertSeverity(severity), format, args);
  }

 private:
  static tflite::LogSeverity ConvertSeverity(LiteRtLogSeverity severity) {
    return static_cast<tflite::LogSeverity>(severity);
  }

  static LiteRtLogSeverity ConvertSeverity(tflite::LogSeverity severity) {
    return static_cast<LiteRtLogSeverity>(severity);
  }
};

LiteRtStatus LiteRtCreateLogger(LiteRtLogger* logger) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *logger = new LiteRtLoggerT;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity* min_severity) {
  if (!logger || !min_severity) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *min_severity = logger->GetMinSeverity();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity min_severity) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  logger->SetMinSeverity(min_severity);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLoggerLog(LiteRtLogger logger, LiteRtLogSeverity severity,
                             const char* format, ...) {
  if (!logger || !format) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  va_list args;
  va_start(args, format);
  logger->Log(severity, format, args);
  va_end(args);
  return kLiteRtStatusOk;
}

void LiteRtDestroyLogger(LiteRtLogger logger) {
  if (logger != nullptr) {
    delete logger;
  }
}

namespace {
LiteRtLoggerT StaticLogger;
LiteRtLogger DefaultLogger = &StaticLogger;
}  // namespace

LiteRtStatus LiteRtSetDefaultLogger(LiteRtLogger logger) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  DefaultLogger = logger;
  return kLiteRtStatusOk;
}

LiteRtLogger LiteRtGetDefaultLogger() { return DefaultLogger; }
