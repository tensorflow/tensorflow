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
#include "tensorflow/lite/testing/kernel_test/input_generator.h"

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {

namespace {

TEST(InputGeneratorTest, LoadModel) {
  InputGenerator input_generator;
  ASSERT_EQ(input_generator.LoadModel(
                "tensorflow/lite/testdata/multi_add.bin"),
            kTfLiteOk);
}

TEST(InputGeneratorTest, ReadWriteSimpleFile) {
  InputGenerator input_generator;
  ASSERT_EQ(
      input_generator.ReadInputsFromFile("tensorflow/lite/testing/"
                                         "kernel_test/testdata/test_input.csv"),
      kTfLiteOk);

  std::string content = "1";
  for (int i = 0; i < 1 * 8 * 8 * 3 - 1; i++) {
    content.append(",1");
  }
  std::vector<std::pair<string, string>> inputs = {{"a", content}};
  ASSERT_EQ(input_generator.GetInputs(), inputs);

  auto output_filename = ::testing::TempDir() + "/out.csv";
  ASSERT_EQ(input_generator.WriteInputsToFile(output_filename), kTfLiteOk);

  std::ifstream in(output_filename);
  std::string out;
  std::getline(in, out, '\n');
  std::string expected_out = "a:";
  expected_out.append(content);
  ASSERT_EQ(out, expected_out);
}

TEST(InputGeneratorTest, GenerateUniformInput) {
  InputGenerator input_generator;
  ASSERT_EQ(input_generator.LoadModel(
                "tensorflow/lite/testdata/multi_add.bin"),
            kTfLiteOk);
  input_generator.GenerateInput("UNIFORM");
  auto inputs = input_generator.GetInputs();
  ASSERT_EQ(inputs.size(), 4);
}

TEST(InputGeneratorTest, GenerateGaussianInput) {
  InputGenerator input_generator;
  ASSERT_EQ(input_generator.LoadModel(
                "tensorflow/lite/testdata/multi_add.bin"),
            kTfLiteOk);
  input_generator.GenerateInput("GAUSSIAN");
  auto inputs = input_generator.GetInputs();
  ASSERT_EQ(inputs.size(), 4);
}

}  // namespace
}  // namespace testing
}  // namespace tflite
