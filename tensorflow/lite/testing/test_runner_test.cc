/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/testing/test_runner.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

class ConcreteTestRunner : public TestRunner {
 public:
  void LoadModel(const string& bin_file_path) override {}
  void AllocateTensors() override {}
  bool CheckFloatSizes(size_t bytes, size_t values) {
    return CheckSizes<float>(bytes, values);
  }
  void LoadModel(const string& bin_file_path,
                 const string& signature) override {}
  void ReshapeTensor(const string& name, const string& csv_values) override {}
  void ResetTensor(const std::string& name) override {}
  string ReadOutput(const string& name) override { return ""; }
  void Invoke(const std::vector<std::pair<string, string>>& inputs) override {}
  bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes)
      override {
    return true;
  }
  std::vector<string> GetOutputNames() override { return {}; }

 private:
  std::vector<int> ids_;
};

TEST(TestRunner, ModelPath) {
  ConcreteTestRunner runner;
  EXPECT_EQ(runner.GetFullPath("test.bin"), "test.bin");
  runner.SetModelBaseDir("/tmp");
  EXPECT_EQ(runner.GetFullPath("test.bin"), "/tmp/test.bin");
}

TEST(TestRunner, InvocationId) {
  ConcreteTestRunner runner;
  EXPECT_EQ(runner.GetInvocationId(), "");
  runner.SetInvocationId("X");
  EXPECT_EQ(runner.GetInvocationId(), "X");
}

TEST(TestRunner, Invalidation) {
  ConcreteTestRunner runner;
  EXPECT_TRUE(runner.IsValid());
  EXPECT_EQ(runner.GetErrorMessage(), "");
  runner.Invalidate("Some Error");
  EXPECT_FALSE(runner.IsValid());
  EXPECT_EQ(runner.GetErrorMessage(), "Some Error");
}

TEST(TestRunner, OverallSuccess) {
  ConcreteTestRunner runner;
  EXPECT_TRUE(runner.GetOverallSuccess());
  runner.SetOverallSuccess(false);
  EXPECT_FALSE(runner.GetOverallSuccess());
}

TEST(TestRunner, CheckSizes) {
  ConcreteTestRunner runner;
  EXPECT_TRUE(runner.CheckFloatSizes(16, 4));
  EXPECT_FALSE(runner.CheckFloatSizes(16, 2));
  EXPECT_EQ(runner.GetErrorMessage(),
            "Expected '4' elements for a tensor, but only got '2'");
}

}  // namespace
}  // namespace testing
}  // namespace tflite
