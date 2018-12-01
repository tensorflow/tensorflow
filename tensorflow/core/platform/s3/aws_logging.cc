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
#include "tensorflow/core/platform/s3/aws_logging.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>

#include <cstdarg>

namespace tensorflow {

AWSLogSystem::AWSLogSystem(Aws::Utils::Logging::LogLevel log_level)
    : log_level_(log_level) {}

void AWSLogSystem::Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                       const char* format, ...) {
  std::va_list args;
  va_start(args, format);

  const string s = strings::Printf(format, args);

  va_end(args);

  LogMessage(log_level, s);
}

void AWSLogSystem::LogStream(Aws::Utils::Logging::LogLevel log_level,
                             const char* tag,
                             const Aws::OStringStream& message_stream) {
  LogMessage(log_level, message_stream.rdbuf()->str().c_str());
}

void AWSLogSystem::LogMessage(Aws::Utils::Logging::LogLevel log_level,
                              const std::string& message) {
  if (message == "Initializing Curl library") return;
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

namespace {
static const char* kAWSLoggingTag = "AWSLogging";

Aws::Utils::Logging::LogLevel ParseLogLevelFromEnv() {
  Aws::Utils::Logging::LogLevel log_level = Aws::Utils::Logging::LogLevel::Info;

  const int64_t level = 
      getenv("AWS_LOG_LEVEL") ? tensorflow::internal::LogLevelStrToInt(getenv("AWS_LOG_LEVEL")) 
                              : tensorflow::internal::MinLogLevelFromEnv();

  switch (level) {
    case INFO:
      log_level = Aws::Utils::Logging::LogLevel::Info;
      break;
    case WARNING:
      log_level = Aws::Utils::Logging::LogLevel::Warn;
      break;
    case ERROR:
      log_level = Aws::Utils::Logging::LogLevel::Error;
      break;
    case FATAL:
      log_level = Aws::Utils::Logging::LogLevel::Fatal;
      break;
    default:
      log_level = Aws::Utils::Logging::LogLevel::Info;
      break;
  }

  return log_level;
}
}  // namespace

static bool initialized = false;
static mutex s3_logging_mutex(LINKER_INITIALIZED);
void AWSLogSystem::InitializeAWSLogging() {
  std::lock_guard<mutex> s3_logging_lock(s3_logging_mutex);
  if (!initialized) {
    Aws::Utils::Logging::InitializeAWSLogging(
        Aws::MakeShared<AWSLogSystem>(kAWSLoggingTag, ParseLogLevelFromEnv()));
    initialized = true;
    return;
  }
}

void AWSLogSystem::ShutdownAWSLogging() {
  std::lock_guard<mutex> s3_logging_lock(s3_logging_mutex);
  if (initialized) {
    Aws::Utils::Logging::ShutdownAWSLogging();
    initialized = false;
    return;
  }
}

}  // namespace tensorflow
