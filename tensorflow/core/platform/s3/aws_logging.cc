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

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>

#include <cstdarg>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {

static const std::map<std::string, Aws::Utils::Logging::LogLevel>
    log_levels_string_to_aws = {
        {"off", Aws::Utils::Logging::LogLevel::Off},
        {"fatal", Aws::Utils::Logging::LogLevel::Fatal},
        {"error", Aws::Utils::Logging::LogLevel::Error},
        {"warn", Aws::Utils::Logging::LogLevel::Warn},
        {"info", Aws::Utils::Logging::LogLevel::Info},
        {"debug", Aws::Utils::Logging::LogLevel::Debug},
        {"trace", Aws::Utils::Logging::LogLevel::Trace}};

static const std::map<int, Aws::Utils::Logging::LogLevel> log_levels_tf_to_aws =
    {{INFO, Aws::Utils::Logging::LogLevel::Info},
     {WARNING, Aws::Utils::Logging::LogLevel::Warn},
     {ERROR, Aws::Utils::Logging::LogLevel::Error},
     {FATAL, Aws::Utils::Logging::LogLevel::Fatal}};

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
      // this will match for DEBUG, TRACE
      LOG(INFO) << message;
      break;
  }
}

void AWSLogSystem::Flush() { return; }

namespace {

Aws::Utils::Logging::LogLevel TfLogLevelToAwsLogLevel(int level) {
  // Converts TF Log Levels INFO, WARNING, ERROR and FATAL to the AWS enum
  // values for the levels
  if (log_levels_tf_to_aws.find(level) != log_levels_tf_to_aws.end()) {
    return log_levels_tf_to_aws.at(level);
  } else {
    // default to fatal
    return Aws::Utils::Logging::LogLevel::Fatal;
  }
}

static const char* kAWSLoggingTag = "AWSLogging";

Aws::Utils::Logging::LogLevel ParseAwsLogLevelFromEnv() {
  // defaults to FATAL log level for the AWS SDK
  // this is because many normal tensorflow operations are logged as errors in
  // the AWS SDK such as checking if a file exists can log an error in AWS SDK
  // if the file does not actually exist another such case is when reading a
  // file till the end, TensorFlow expects to see an InvalidRange exception at
  // the end, but this would be an error in the AWS SDK. This confuses users,
  // hence the default setting.
  Aws::Utils::Logging::LogLevel log_level =
      Aws::Utils::Logging::LogLevel::Fatal;

  const char* aws_env_var_val = getenv("AWS_LOG_LEVEL");
  if (aws_env_var_val != nullptr) {
    string maybe_integer_str(aws_env_var_val, strlen(aws_env_var_val));
    std::istringstream ss(maybe_integer_str);
    int level;
    ss >> level;
    if (ss.fail()) {
      // wasn't a number
      // expecting a string
      string level_str = maybe_integer_str;
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
}  // namespace

static bool initialized = false;
static mutex s3_logging_mutex(LINKER_INITIALIZED);
void AWSLogSystem::InitializeAWSLogging() {
  std::lock_guard<mutex> s3_logging_lock(s3_logging_mutex);
  if (!initialized) {
    Aws::Utils::Logging::InitializeAWSLogging(Aws::MakeShared<AWSLogSystem>(
        kAWSLoggingTag, ParseAwsLogLevelFromEnv()));
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
