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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_LOGGING_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_LOGGING_H_

#include <stdarg.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtLogger);

// WARNING: The values of the following enum are to be kept in sync with
// tflite::LogSeverity.
typedef enum {
  kLiteRtLogSeverityVerbose = 0,
  kLiteRtLogSeverityInfo = 1,
  kLiteRtLogSeverityWarning = 2,
  kLiteRtLogSeverityError = 3,
  kLiteRtLogSeveritySilent = 4,
} LiteRtLogSeverity;

#define LITERT_VERBOSE kLiteRtLogSeverityVerbose
#define LITERT_INFO kLiteRtLogSeverityInfo
#define LITERT_WARNING kLiteRtLogSeverityWarning
#define LITERT_ERROR kLiteRtLogSeverityError
#define LITERT_SILENT kLiteRtLogSeveritySilent

LiteRtStatus LiteRtCreateLogger(LiteRtLogger* logger);
LiteRtStatus LiteRtGetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity* min_severity);
LiteRtStatus LiteRtSetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity min_severity);
LiteRtStatus LiteRtLoggerLog(LiteRtLogger logger, LiteRtLogSeverity severity,
                             const char* format, ...);
void LiteRtDestroyLogger(LiteRtLogger logger);

LiteRtLogger LiteRtGetDefaultLogger();
LiteRtStatus LiteRtSetDefaultLogger(LiteRtLogger logger);
LiteRtStatus LiteRtDefaultLoggerLog(LiteRtLogSeverity severity,
                                    const char* format, ...);

#ifdef __cplusplus
}
#endif  // __cplusplus

#define LITERT_LOGGER_LOG_PROD(logger, severity, format, ...)                  \
  {                                                                            \
    LiteRtLogSeverity __min_severity__;                                        \
    if (LiteRtGetMinLoggerSeverity(logger, &__min_severity__) !=               \
        kLiteRtStatusOk) {                                                     \
      __min_severity__ = kLiteRtLogSeverityVerbose;                            \
    }                                                                          \
    if (severity >= __min_severity__) {                                        \
      LiteRtLoggerLog(logger, severity, "[%s:%d] " format, __FILE__, __LINE__, \
                      ##__VA_ARGS__);                                          \
    }                                                                          \
  }

#ifndef NDEBUG
#define LITERT_LOGGER_LOG LITERT_LOGGER_LOG_PROD
#else
#define LITERT_LOGGER_LOG(logger, severity, format, ...)             \
  do {                                                               \
    LITERT_LOGGER_LOG_PROD(logger, severity, format, ##__VA_ARGS__); \
  } while (false)
#endif

#define LITERT_LOG(severity, format, ...) \
  LITERT_LOGGER_LOG(LiteRtGetDefaultLogger(), severity, format, ##__VA_ARGS__);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_LOGGING_H_
