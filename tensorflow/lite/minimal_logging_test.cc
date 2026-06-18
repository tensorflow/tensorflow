/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/minimal_logging.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/logger.h"

namespace tflite {

TEST(MinimalLogging, Basic) {
  testing::internal::CaptureStderr();
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Foo");
  EXPECT_EQ("INFO: Foo\n", testing::internal::GetCapturedStderr());
}

TEST(MinimalLogging, BasicFormatted) {
  testing::internal::CaptureStderr();
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Foo %s %s", "Bar", "Baz");
  EXPECT_EQ("INFO: Foo Bar Baz\n", testing::internal::GetCapturedStderr());
}

TEST(MinimalLogging, Warn) {
  testing::internal::CaptureStderr();
  TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "One", "");
  EXPECT_EQ("WARNING: One\n", testing::internal::GetCapturedStderr());
}

TEST(MinimalLogging, Error) {
  testing::internal::CaptureStderr();
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Two");
  EXPECT_EQ("ERROR: Two\n", testing::internal::GetCapturedStderr());
}

TEST(MinimalLogging, UnknownSeverity) {
  testing::internal::CaptureStderr();
  LogSeverity default_log_severity = TFLITE_LOG_INFO;
#if defined(__ANDROID__) || !defined(NDEBUG)
  default_log_severity = TFLITE_LOG_VERBOSE;
#endif
  EXPECT_EQ(tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
                static_cast<LogSeverity>(-1)),
            default_log_severity);
  TFLITE_LOG_PROD(static_cast<LogSeverity>(-1), "Three");
  EXPECT_EQ("<Unknown severity>: Three\n",
            testing::internal::GetCapturedStderr());
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
      default_log_severity);
}

TEST(MinimalLogging, MinimumSeverity) {
  testing::internal::CaptureStderr();
  LogSeverity default_log_severity = TFLITE_LOG_INFO;
#if defined(__ANDROID__) || !defined(NDEBUG)
  default_log_severity = TFLITE_LOG_VERBOSE;
#endif

  EXPECT_EQ(tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
                TFLITE_LOG_WARNING),
            default_log_severity);
  TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Foo");
  TFLITE_LOG_PROD(default_log_severity, "Bar");
  EXPECT_EQ("WARNING: Foo\n", testing::internal::GetCapturedStderr());
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
      default_log_severity);
}

TEST(MinimalLogging, Once) {
  testing::internal::CaptureStderr();
  for (int i = 0; i < 10; ++i) {
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Count: %d", i);
  }
  EXPECT_EQ("INFO: Count: 0\n", testing::internal::GetCapturedStderr());
}

TEST(MinimalLogging, Debug) {
  testing::internal::CaptureStderr();
  TFLITE_LOG(TFLITE_LOG_INFO, "Foo");
  TFLITE_LOG(TFLITE_LOG_WARNING, "Bar");
  TFLITE_LOG(TFLITE_LOG_ERROR, "Baz");
#ifndef NDEBUG
  EXPECT_EQ("INFO: Foo\nWARNING: Bar\nERROR: Baz\n",
            testing::internal::GetCapturedStderr());
#else
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
#endif
}

TEST(MinimalLogging, DebugOnce) {
  testing::internal::CaptureStderr();
  for (int i = 0; i < 10; ++i) {
    TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Count: %d", i);
  }
#ifndef NDEBUG
  EXPECT_EQ("INFO: Count: 0\n", testing::internal::GetCapturedStderr());
#else
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
#endif
}

}  // namespace tflite
