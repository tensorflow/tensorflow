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
#include "tensorflow/core/platform/s3/s3_logging.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>

#include <cstdarg>

namespace tensorflow {

S3LogSystem::S3LogSystem(Aws::Utils::Logging::LogLevel logLevel)
    : m_logLevel(logLevel) {}

void S3LogSystem::Log(Aws::Utils::Logging::LogLevel logLevel, const char* tag,
                      const char* formatStr, ...) {
  std::va_list args;
  va_start(args, formatStr);

  std::string s = strings::Printf(formatStr, args);
  LOG(ERROR) << s;

  va_end(args);
}

void S3LogSystem::LogStream(Aws::Utils::Logging::LogLevel logLevel,
                            const char* tag,
                            const Aws::OStringStream& message_stream) {
  LOG(ERROR) << message_stream.rdbuf()->str();
}

void S3LogSystem::LogMessage(Aws::Utils::Logging::LogLevel logLevel,
                             const std::string& message) {
  switch (logLevel) {
    case Aws::Utils::Logging::LogLevel::Info:
      LOG(INFO) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Warn:
      LOG(WARNING) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Error:
      LOG(ERROR) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Fatal:
      LOG(FATAL) << message;
      break;
    default:
      break;
  }
}
}  // namespace tensorflow
