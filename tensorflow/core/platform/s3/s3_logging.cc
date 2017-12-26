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

S3LogSystem::S3LogSystem(Aws::Utils::Logging::LogLevel log_level)
    : log_level_(log_level) {}

void S3LogSystem::Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                      const char* format, ...) {
  std::va_list args;
  va_start(args, format);

  std::string s = strings::Printf(format, args);

  va_end(args);

  LogMessage(log_level, s);
}

void S3LogSystem::LogStream(Aws::Utils::Logging::LogLevel log_level,
                            const char* tag,
                            const Aws::OStringStream& message_stream) {
  LogMessage(log_level, message_stream.rdbuf()->str().c_str());
}

void S3LogSystem::LogMessage(Aws::Utils::Logging::LogLevel log_level,
                             const std::string& message) {
  switch (log_level) {
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
      LOG(ERROR) << message;
      break;
  }
}
}  // namespace tensorflow
