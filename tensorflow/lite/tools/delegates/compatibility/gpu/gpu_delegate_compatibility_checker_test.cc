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

#include "tensorflow/lite/tools/delegates/compatibility/gpu/gpu_delegate_compatibility_checker.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace tools {

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok());
#endif

namespace {

class AddOpModel : public SingleOpModel {
 public:
  AddOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output, ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    // Builds interpreter.
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

}  // namespace

TEST(GpuDelegateCompatibilityCheckerTest, CheckOnlineMode) {
  const std::string& full_path =
      tensorflow::GetDataDependencyFilepath("tensorflow/lite/testdata/add.bin");
  auto fb_model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(fb_model);

  proto::CompatibilityResult compatibility_result;

  GpuDelegateCompatibilityChecker gpu_dcc;
  // Online mode is not supported by GPU DCC
  EXPECT_THAT(gpu_dcc.checkModelCompatibilityOnline(fb_model.get(),
                                                    &compatibility_result),
              testing::status::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(GpuDelegateCompatibilityCheckerTest, CompatibleModelOfflineMode) {
  const std::string& full_path =
      tensorflow::GetDataDependencyFilepath("tensorflow/lite/testdata/add.bin");
  auto fb_model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(fb_model);

  proto::CompatibilityResult compatibility_result;

  GpuDelegateCompatibilityChecker gpu_dcc;
  EXPECT_OK(gpu_dcc.checkModelCompatibilityOffline(fb_model.get(),
                                                   &compatibility_result));
  for (auto op_compatibility_result :
       compatibility_result.compatibility_results()) {
    EXPECT_TRUE(op_compatibility_result.is_supported());
  }
  EXPECT_EQ(compatibility_result.compatibility_results_size(), 2);
}

TEST(GpuDelegateCompatibilityCheckerTest, IncompatibleModelOfflineMode) {
  const std::string& full_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/lite/testdata/conv3d_huge_im2col.bin");

  auto fb_model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(fb_model);

  proto::CompatibilityResult compatibility_result;

  GpuDelegateCompatibilityChecker gpu_dcc;
  EXPECT_OK(gpu_dcc.checkModelCompatibilityOffline(fb_model.get(),
                                                   &compatibility_result));
  for (auto op_compatibility_result :
       compatibility_result.compatibility_results()) {
    EXPECT_FALSE(op_compatibility_result.is_supported());
  }
  EXPECT_EQ(compatibility_result.compatibility_results_size(), 1);
}

}  // namespace tools
}  // namespace tflite
