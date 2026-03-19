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

#include "tensorflow/lite/tools/gen_op_registration.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::ElementsAreArray;

namespace tflite {

class GenOpRegistrationTest : public ::testing::Test {
 protected:
  GenOpRegistrationTest() {}

  void ReadOps(const string& model_path) {
    auto model = FlatBufferModel::BuildFromFile(model_path.data());
    if (model) {
      ReadOpsFromModel(model->GetModel(), &builtin_ops_, &custom_ops_);
    }
  }

  std::map<string, std::pair<int, int>> builtin_ops_;
  std::map<string, std::pair<int, int>> custom_ops_;
};

TEST_F(GenOpRegistrationTest, TestNonExistentFiles) {
  ReadOps("/tmp/tflite_model_1234");
  EXPECT_EQ(builtin_ops_.size(), 0);
  EXPECT_EQ(custom_ops_.size(), 0);
}

TEST_F(GenOpRegistrationTest, TestModels) {
  ReadOps("tensorflow/lite/testdata/test_model.bin");
  RegisteredOpMap builtin_expected{{"CONV_2D", {1, 1}}};
  RegisteredOpMap custom_expected{{"testing_op", {1, 1}}};
  EXPECT_THAT(builtin_ops_, ElementsAreArray(builtin_expected));
  EXPECT_THAT(custom_ops_, ElementsAreArray(custom_expected));
}

TEST_F(GenOpRegistrationTest, TestVersionedModels) {
  ReadOps("tensorflow/lite/testdata/test_model_versioned_ops.bin");
  RegisteredOpMap builtin_expected{{"CONV_2D", {3, 3}}};
  RegisteredOpMap custom_expected{{"testing_op", {2, 2}}};
  EXPECT_THAT(builtin_ops_, ElementsAreArray(builtin_expected));
  EXPECT_THAT(custom_ops_, ElementsAreArray(custom_expected));
}

TEST_F(GenOpRegistrationTest, TestBothModels) {
  ReadOps("tensorflow/lite/testdata/test_model.bin");
  ReadOps("tensorflow/lite/testdata/test_model_versioned_ops.bin");
  RegisteredOpMap builtin_expected{{"CONV_2D", {1, 3}}};
  RegisteredOpMap custom_expected{{"testing_op", {1, 2}}};
  EXPECT_THAT(builtin_ops_, ElementsAreArray(builtin_expected));
  EXPECT_THAT(custom_ops_, ElementsAreArray(custom_expected));
}

TEST_F(GenOpRegistrationTest, TestEmptyModels) {
  ReadOps("tensorflow/lite/testdata/empty_model.bin");
  EXPECT_EQ(builtin_ops_.size(), 0);
  EXPECT_EQ(custom_ops_.size(), 0);
}

TEST_F(GenOpRegistrationTest, TestZeroSubgraphs) {
  ReadOps("tensorflow/lite/testdata/0_subgraphs.bin");
  EXPECT_EQ(builtin_ops_.size(), 0);
  EXPECT_EQ(custom_ops_.size(), 0);
}

TEST_F(GenOpRegistrationTest, TestBrokenMmap) {
  ReadOps("tensorflow/lite/testdata/test_model_broken.bin");
  EXPECT_EQ(builtin_ops_.size(), 0);
  EXPECT_EQ(custom_ops_.size(), 0);
}

TEST_F(GenOpRegistrationTest, TestNormalizeCustomOpName) {
  std::vector<std::pair<string, string>> testcase = {
      {"CustomOp", "CUSTOM_OP"},
      {"a", "A"},
      {"custom_op", "CUSTOM_OP"},
      {"customop", "CUSTOMOP"},
  };

  for (const auto& test : testcase) {
    EXPECT_EQ(NormalizeCustomOpName(test.first), test.second);
  }
}
}  // namespace tflite
