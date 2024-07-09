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

#include "tensorflow/lite/tools/delegates/compatibility/nnapi/nnapi_delegate_compatibility_checker.h"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"

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

// Class to test the NNAPI delegate compatibility checker (DCC)
class NnapiDccTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override { compatibility_result_.Clear(); }

  NnapiDelegateCompatibilityChecker nnapi_dcc_;
  proto::CompatibilityResult compatibility_result_;
};

}  // namespace

TEST_F(NnapiDccTest, ValidRuntimeFeatureLevel) {
  std::unordered_map dcc_configs = nnapi_dcc_.getDccConfigurations();
  EXPECT_EQ(dcc_configs["nnapi-runtime_feature_level"], "8");
  EXPECT_OK(nnapi_dcc_.setDccConfigurations(dcc_configs));

  dcc_configs["nnapi-runtime_feature_level"] = "1";
  EXPECT_OK(nnapi_dcc_.setDccConfigurations(dcc_configs));

  dcc_configs["nnapi-runtime_feature_level"] = "8";
  EXPECT_OK(nnapi_dcc_.setDccConfigurations(dcc_configs));

  dcc_configs.clear();
  EXPECT_OK(nnapi_dcc_.setDccConfigurations(dcc_configs));
  EXPECT_EQ(nnapi_dcc_.getDccConfigurations()["nnapi-runtime_feature_level"],
            "8");
}

TEST_F(NnapiDccTest, InvalidRuntimeFeatureLevel) {
  std::unordered_map dcc_configs = nnapi_dcc_.getDccConfigurations();
  dcc_configs["nnapi-runtime_feature_level"] = "03";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);

  dcc_configs["nnapi-runtime_feature_level"] = "a";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);

  dcc_configs["nnapi-runtime_feature_level"] = "28123497123489123841212344516";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);

  dcc_configs["nnapi-runtime_feature_level"] = "30.0";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);

  dcc_configs["nnapi-runtime_feature_level"] = "-30";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);

  dcc_configs["nnapi-runtime_feature_level"] = "9";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);

  dcc_configs.clear();
  dcc_configs["nnapi-runtim_feature_level"] = "8";
  EXPECT_EQ(nnapi_dcc_.setDccConfigurations(dcc_configs).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(NnapiDccTest, CompatibleModelOnlineMode) {
  const std::string& full_path =
      tensorflow::GetDataDependencyFilepath("tensorflow/lite/testdata/add.bin");
  auto fb_model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(fb_model);

  auto model = fb_model->GetModel();
  EXPECT_EQ(model->subgraphs()->size(), 1);
  EXPECT_EQ(model->subgraphs()->Get(0)->operators()->size(), 2);

  EXPECT_OK(nnapi_dcc_.checkModelCompatibilityOnline(fb_model.get(),
                                                     &compatibility_result_));
  for (auto op_compatibility_result :
       compatibility_result_.compatibility_results()) {
    EXPECT_TRUE(op_compatibility_result.is_supported());
  }
  EXPECT_EQ(compatibility_result_.compatibility_results_size(), 2);
}

TEST_F(NnapiDccTest, IncompatibleModelOperation) {
  // No activation function is supported for INT32 tensor type.
  AddOpModel add_op_model(
      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {1, 2, 2, 1}},
      {TensorType_INT32, {}}, ActivationFunctionType_RELU_N1_TO_1);

  auto fb_model = tflite::FlatBufferModel::BuildFromModel(
      tflite::GetModel(add_op_model.GetModelBuffer()));
  ASSERT_TRUE(fb_model);

  EXPECT_OK(nnapi_dcc_.checkModelCompatibilityOnline(fb_model.get(),
                                                     &compatibility_result_));
  for (auto op_compatibility_result :
       compatibility_result_.compatibility_results()) {
    EXPECT_FALSE(op_compatibility_result.is_supported());
  }
  EXPECT_EQ(compatibility_result_.compatibility_results_size(), 1);
}

TEST_F(NnapiDccTest, IncompatibleModelFeatureLevel) {
  // INT32 input tensor type is not supported if runtime feature level < 4.
  AddOpModel add_op_model({TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {}}, ActivationFunctionType_NONE);

  auto fb_model = tflite::FlatBufferModel::BuildFromModel(
      tflite::GetModel(add_op_model.GetModelBuffer()));
  ASSERT_TRUE(fb_model);

  auto nnapi_configs = nnapi_dcc_.getDccConfigurations();
  nnapi_configs["nnapi-runtime_feature_level"] = "2";
  EXPECT_OK(nnapi_dcc_.setDccConfigurations(nnapi_configs));
  EXPECT_OK(nnapi_dcc_.checkModelCompatibilityOnline(fb_model.get(),
                                                     &compatibility_result_));
  for (auto op_compatibility_result :
       compatibility_result_.compatibility_results()) {
    EXPECT_FALSE(op_compatibility_result.is_supported());
  }
  EXPECT_EQ(compatibility_result_.compatibility_results_size(), 1);
}

}  // namespace tools
}  // namespace tflite
