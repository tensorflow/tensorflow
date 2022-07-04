/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_LOGGER_H_
#define TENSORFLOW_LITE_LOGGER_H_

namespace tflite {

/// The severity level of a TFLite log message.
/// WARNING: This is an experimental API and subject to change.
enum LogSeverity {
  /// Default log severity level.
  TFLITE_LOG_VERBOSE = 0,
  /// Log routine information.
  TFLITE_LOG_INFO = 1,
  /// Log warning events that might cause problems.
  TFLITE_LOG_WARNING = 2,
  /// Log error events that are likely to cause problems.
  TFLITE_LOG_ERROR = 3,
};

/// TFLite logger specific configurations.
/// WARNING: This is an experimental API and subject to change.
class LoggerOptions {
 public:
  /// Get the minimum severity level for logging. Default is INFO in prod
  /// builds and VERBOSE in debug builds.
  /// Note: Default is always VERBOSE on Android.
  /// WARNING: This is an experimental API and subject to change.
  static LogSeverity GetMinimumLogSeverity();

  /// Set the minimum severity level for logging, returning the old severity.
  /// WARNING: This is an experimental API and subject to change.
  static LogSeverity SetMinimumLogSeverity(LogSeverity new_severity);
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_LOGGER_H_
