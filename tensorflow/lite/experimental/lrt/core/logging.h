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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LOGGING_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LOGGING_H_

#include <cstdarg>

namespace litert {
namespace internal {

enum class LogSeverity {
  kVerbose = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
  kSilent = 4
};

#define LITERT_VERBOSE ::litert::internal::LogSeverity::kVerbose
#define LITERT_INFO ::litert::internal::LogSeverity::kInfo
#define LITERT_WARNING ::litert::internal::LogSeverity::kWarning
#define LITERT_ERROR ::litert::internal::LogSeverity::kError
#define LITERT_SILENT ::litert::internal::LogSeverity::kSilent

class Logger {
 public:
  static void Log(LogSeverity severity, const char* format, ...);
  static LogSeverity GetMinimumSeverity();
  static LogSeverity SetMinimumSeverity(LogSeverity new_severity);
};

#define LITERT_LOG_PROD(severity, format, ...)                           \
  if (severity >= litert::internal::Logger::GetMinimumSeverity()) {      \
    litert::internal::Logger::Log(severity, "[%s:%d] " format, __FILE__, \
                                  __LINE__, ##__VA_ARGS__);              \
  }

#ifndef NDEBUG
#define LITERT_LOG LITERT_LOG_PROD
#else
#define LITERT_LOG(severity, format, ...)             \
  do {                                                \
    LITERT_LOG_PROD(severity, format, ##__VA_ARGS__); \
  } while (false)
#endif

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LOGGING_H_
