/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/testing/kernel_test/diff_analyzer.h"

#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/io/path.h"

namespace tflite {
namespace testing {

namespace {

TEST(DiffAnalyzerTest, ZeroDiff) {
  DiffAnalyzer diff_analyzer;
  string filename = "third_party/tensorflow/lite/testdata/test_input.csv";
  ASSERT_EQ(diff_analyzer.ReadFiles(filename, filename), kTfLiteOk);

  string output_file =
      tensorflow::io::JoinPath(FLAGS_test_tmpdir + "diff_report.csv");
  ASSERT_EQ(diff_analyzer.WriteReport(output_file), kTfLiteOk);

  std::string content;
  std::ifstream file(output_file);
  std::getline(file, content);
  std::getline(file, content);
  ASSERT_EQ(content, "0,0");
}

}  // namespace

}  // namespace testing
}  // namespace tflite
