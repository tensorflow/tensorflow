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
#include "tensorflow/lite/stderr_reporter.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

namespace {

void CheckWritesToStderr(ErrorReporter *error_reporter) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  testing::internal::CaptureStderr();
#endif

  // Run the code under test.
  TF_LITE_REPORT_ERROR(error_reporter, "Test: %d", 42);

#ifndef TF_LITE_STRIP_ERROR_STRINGS
  EXPECT_EQ("ERROR: Test: 42\n", testing::internal::GetCapturedStderr());
#endif
}

TEST(StderrReporterTest, DefaultErrorReporter_WritesToStderr) {
  CheckWritesToStderr(DefaultErrorReporter());
}

TEST(StderrReporterTest, StderrReporter_WritesToStderr) {
  StderrReporter stderr_reporter;
  CheckWritesToStderr(&stderr_reporter);
}

}  // namespace

}  // namespace tflite
