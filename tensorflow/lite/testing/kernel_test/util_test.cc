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
#include "tensorflow/lite/testing/kernel_test/util.h"

#include <fstream>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace kernel_test {
namespace {

TEST(UtilTest, SimpleE2ETest) {
  TestOptions options;
  options.tflite_model = "tensorflow/lite/testdata/add.bin";
  options.read_input_from_file =
      "tensorflow/lite/testing/kernel_test/testdata/test_input.csv";
  options.dump_output_to_file = ::testing::TempDir() + "/test_out.csv";
  options.kernel_type = "REFERENCE";
  std::unique_ptr<TestRunner> runner(new TfLiteDriver(
      TfLiteDriver::DelegateType::kNone, /*reference_kernel=*/true));
  RunKernelTest(options, runner.get());
  std::string expected = "x:3";
  for (int i = 0; i < 1 * 8 * 8 * 3 - 1; i++) {
    expected.append(",3");
  }
  std::string content;
  std::ifstream file(options.dump_output_to_file);
  std::getline(file, content);
  EXPECT_EQ(content, expected);
}

}  // namespace
}  // namespace kernel_test
}  // namespace testing
}  // namespace tflite
