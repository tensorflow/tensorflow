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

#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/logging/LogLevel.h>

#include <atomic>

namespace tensorflow {

class S3LogSystem : public Aws::Utils::Logging::LogSystemInterface {
 public:
  using Base = LogSystemInterface;

  /**
   * Initializes log system
   */
  S3LogSystem(Aws::Utils::Logging::LogLevel logLevel);
  virtual ~S3LogSystem() = default;

  /**
   * Gets the currently configured log level.
   */
  virtual Aws::Utils::Logging::LogLevel GetLogLevel(void) const override { return m_logLevel; }
  /**
   * Set a new log level. This has the immediate effect of changing the log
   * output to the new level.
   */
  void SetLogLevel(Aws::Utils::Logging::LogLevel logLevel) { m_logLevel.store(logLevel); }

  /**
   * Does a printf style output to ProcessFormattedStatement. Don't use this,
   * it's unsafe. See LogStream
   */
  virtual void Log(Aws::Utils::Logging::LogLevel logLevel, const char* tag, const char* formatStr, ...) override;

  /**
   * Writes the stream to ProcessFormattedStatement.
   */
  virtual void LogStream(Aws::Utils::Logging::LogLevel logLevel, const char* tag, const Aws::OStringStream& messageStream) override;

 private:
  std::atomic<Aws::Utils::Logging::LogLevel> m_logLevel;

  void LogMessage(Aws::Utils::Logging::LogLevel logLevel, const std::string &message);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_S3_S3_LOGGING_H_
