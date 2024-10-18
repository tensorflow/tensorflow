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

namespace lrt {
namespace internal {

enum class LogSeverity {
  kVerbose = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
  kSilent = 4
};

#define LRT_VERBOSE ::lrt::internal::LogSeverity::kVerbose
#define LRT_INFO ::lrt::internal::LogSeverity::kInfo
#define LRT_WARNING ::lrt::internal::LogSeverity::kWarning
#define LRT_ERROR ::lrt::internal::LogSeverity::kError
#define LRT_SILENT ::lrt::internal::LogSeverity::kSilent

class Logger {
 public:
  static void Log(LogSeverity severity, const char* format, ...);
  static LogSeverity GetMinimumSeverity();
  static LogSeverity SetMinimumSeverity(LogSeverity new_severity);
};

#define LITE_RT_LOG_PROD(severity, format, ...)                       \
  if (severity >= lrt::internal::Logger::GetMinimumSeverity()) {      \
    lrt::internal::Logger::Log(severity, "[%s:%d] " format, __FILE__, \
                               __LINE__, ##__VA_ARGS__);              \
  }

#ifndef NDEBUG
#define LITE_RT_LOG LITE_RT_LOG_PROD
#else
#define LITE_RT_LOG(severity, format, ...)             \
  do {                                                 \
    LITE_RT_LOG_PROD(severity, format, ##__VA_ARGS__); \
  } while (false)
#endif

}  // namespace internal
}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LOGGING_H_
