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

#ifndef TENSORFLOW_CONTRIB_S3_S3_LOGGING_H_
#define TENSORFLOW_CONTRIB_S3_S3_LOGGING_H_

#include <atomic>
#include <string>

#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {

class AWSLogSystem : public Aws::Utils::Logging::LogSystemInterface {
 public:
  static void InitializeAWSLogging();
  static void ShutdownAWSLogging();

  explicit AWSLogSystem(Aws::Utils::Logging::LogLevel log_level);
  virtual ~AWSLogSystem() = default;

  // Gets the currently configured log level.
  virtual Aws::Utils::Logging::LogLevel GetLogLevel(void) const override {
    return log_level_;
  }

  // Set a new log level. This has the immediate effect of changing the log.
  void SetLogLevel(Aws::Utils::Logging::LogLevel log_level) {
    log_level_.store(log_level);
  }

  // Does a printf style output to ProcessFormattedStatement. Don't use this,
  // it's unsafe. See LogStream.
  // Since non-static C++ methods have an implicit this argument,
  // TF_PRINTF_ATTRIBUTE should be counted from two (vs. one).
  virtual void Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                   const char* format, ...) override TF_PRINTF_ATTRIBUTE(4, 5);

  // Writes the stream to ProcessFormattedStatement.
  virtual void LogStream(Aws::Utils::Logging::LogLevel log_level,
                         const char* tag,
                         const Aws::OStringStream& messageStream) override;

  // Flushes the buffered messages if the logger supports buffering
  virtual void Flush() override;

 private:
  void LogMessage(Aws::Utils::Logging::LogLevel log_level,
                  const string& message);
  std::atomic<Aws::Utils::Logging::LogLevel> log_level_;

  TF_DISALLOW_COPY_AND_ASSIGN(AWSLogSystem);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_S3_S3_LOGGING_H_
