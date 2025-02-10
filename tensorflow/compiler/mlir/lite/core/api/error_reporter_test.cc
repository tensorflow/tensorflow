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

#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

#include <cstdio>

#include <gtest/gtest.h>

namespace tflite {

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() { buffer_[0] = 0; }
  int Report(const char* format, va_list args) override {
    vsnprintf(buffer_, kBufferSize, format, args);
    return 0;
  }
  char* GetBuffer() { return buffer_; }

 private:
  static constexpr int kBufferSize = 256;
  char buffer_[kBufferSize];
};

TEST(ErrorReporter, TestReport) {
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;
  reporter->Report("Error: %d", 23);
  EXPECT_EQ(0, strcmp(mock_reporter.GetBuffer(), "Error: 23"));
}

TEST(ErrorReporter, TestReportMacro) {
  MockErrorReporter mock_reporter;
  // Only define the reporter if it's used, to avoid warnings.
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  ErrorReporter* reporter = &mock_reporter;
#endif  // TFLITE_STRIP_ERROR_STRINGS

  TF_LITE_REPORT_ERROR(reporter, "Error: %d", 23);

#ifndef TF_LITE_STRIP_ERROR_STRINGS
  EXPECT_EQ(0, strcmp(mock_reporter.GetBuffer(), "Error: 23"));
#else   // TF_LITE_STRIP_ERROR_STRINGS
  EXPECT_EQ(0, strcmp(mock_reporter.GetBuffer(), ""));
#endif  // TF_LITE_STRIP_ERROR_STRINGS
}

}  // namespace tflite
