/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/plugins/s3/aws_logging.h"

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>

#include <cstdarg>
#include <cstdio>
#include <sstream>

#include "absl/synchronization/mutex.h"
#include "tensorflow/c/logging.h"

static constexpr char kAWSLoggingTag[] = "AWSLogging";

static const std::map<const std::string, const Aws::Utils::Logging::LogLevel>
    log_levels_string_to_aws = {
        {"off", Aws::Utils::Logging::LogLevel::Off},
        {"fatal", Aws::Utils::Logging::LogLevel::Fatal},
        {"error", Aws::Utils::Logging::LogLevel::Error},
        {"warn", Aws::Utils::Logging::LogLevel::Warn},
        {"info", Aws::Utils::Logging::LogLevel::Info},
        {"debug", Aws::Utils::Logging::LogLevel::Debug},
        {"trace", Aws::Utils::Logging::LogLevel::Trace}};

static const std::map<const int, const Aws::Utils::Logging::LogLevel>
    log_levels_tf_to_aws = {{0, Aws::Utils::Logging::LogLevel::Info},
                            {1, Aws::Utils::Logging::LogLevel::Warn},
                            {2, Aws::Utils::Logging::LogLevel::Error},
                            {3, Aws::Utils::Logging::LogLevel::Fatal}};

namespace tf_s3_filesystem {

AWSLogSystem::AWSLogSystem(Aws::Utils::Logging::LogLevel log_level)
    : log_level_(log_level) {}

void AWSLogSystem::LogMessage(Aws::Utils::Logging::LogLevel log_level,
                              const std::string& message) {
  if (message == "Initializing Curl library") return;
  switch (log_level) {
    case Aws::Utils::Logging::LogLevel::Info:
      TF_Log(TF_INFO, message.c_str());
      break;
    case Aws::Utils::Logging::LogLevel::Warn:
      TF_Log(TF_WARNING, message.c_str());
      break;
    case Aws::Utils::Logging::LogLevel::Error:
      TF_Log(TF_ERROR, message.c_str());
      break;
    case Aws::Utils::Logging::LogLevel::Fatal:
      TF_Log(TF_FATAL, message.c_str());
      break;
    default:
      // this will match for DEBUG, TRACE
      TF_Log(TF_INFO, message.c_str());
      break;
  }
}

void AWSLogSystem::Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                       const char* format, ...) {
  char buffer[256];
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, 256, format, args);
  va_end(args);
  LogMessage(log_level, buffer);
}

void AWSLogSystem::LogStream(Aws::Utils::Logging::LogLevel log_level,
                             const char* tag,
                             const Aws::OStringStream& message_stream) {
  LogMessage(log_level, message_stream.rdbuf()->str().c_str());
}

void AWSLogSystem::Flush() { return; }

static Aws::Utils::Logging::LogLevel TfLogLevelToAwsLogLevel(int level) {
  // Converts TF Log Levels INFO, WARNING, ERROR and FATAL to the AWS enum
  // values for the levels
  if (log_levels_tf_to_aws.find(level) != log_levels_tf_to_aws.end()) {
    return log_levels_tf_to_aws.at(level);
  } else {
    // default to fatal
    return Aws::Utils::Logging::LogLevel::Fatal;
  }
}

static Aws::Utils::Logging::LogLevel ParseAwsLogLevelFromEnv() {
  // defaults to FATAL log level for the AWS SDK
  // this is because many normal tensorflow operations are logged as errors in
  // the AWS SDK such as checking if a file exists can log an error in AWS SDK
  // if the file does not actually exist. Another such case is when reading a
  // file till the end, TensorFlow expects to see an InvalidRange exception at
  // the end, but this would be an error in the AWS SDK. This confuses users,
  // hence the default setting.
  Aws::Utils::Logging::LogLevel log_level =
      Aws::Utils::Logging::LogLevel::Fatal;

  const char* aws_env_var_val = getenv("AWS_LOG_LEVEL");
  if (aws_env_var_val != nullptr) {
    std::string maybe_integer_str(aws_env_var_val, strlen(aws_env_var_val));
    std::istringstream ss(maybe_integer_str);
    int level;
    ss >> level;
    if (ss.fail()) {
      // wasn't a number
      // expecting a string
      std::string level_str = maybe_integer_str;
      if (log_levels_string_to_aws.find(level_str) !=
          log_levels_string_to_aws.end()) {
        log_level = log_levels_string_to_aws.at(level_str);
      }
    } else {
      // backwards compatibility
      // valid number, but this number follows the standard TensorFlow log
      // levels need to convert this to AWS SDK logging level number
      log_level = TfLogLevelToAwsLogLevel(level);
    }
  }
  return log_level;
}

static bool initialized = false;
ABSL_CONST_INIT static absl::Mutex s3_logging_mutex(absl::kConstInit);
void AWSLogSystem::InitializeAWSLogging() {
  absl::MutexLock l(&s3_logging_mutex);
  if (!initialized) {
    Aws::Utils::Logging::InitializeAWSLogging(Aws::MakeShared<AWSLogSystem>(
        kAWSLoggingTag, ParseAwsLogLevelFromEnv()));
    initialized = true;
    return;
  }
}

void AWSLogSystem::ShutdownAWSLogging() {
  absl::MutexLock l(&s3_logging_mutex);
  if (initialized) {
    Aws::Utils::Logging::ShutdownAWSLogging();
    initialized = false;
    return;
  }
}

}  // namespace tf_s3_filesystem
