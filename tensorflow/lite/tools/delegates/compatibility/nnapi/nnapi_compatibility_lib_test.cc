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

#include "tensorflow/lite/tools/delegates/compatibility/nnapi/nnapi_compatibility_lib.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace tools {

namespace {

class AddOpModel : public SingleOpModel {
 public:
  AddOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output, ActivationFunctionType activation_type,
             CompatibilityCheckerDelegate* checker_delegate) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    SetDelegate(checker_delegate);
    // Builds interpreter and applies delegate.
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

}  // namespace

TEST(NnapiDelegateCompabilityTest, InvalidInput) {
  EXPECT_EQ(CheckCompatibility(nullptr, 0, nullptr, nullptr), kTfLiteError);
}

TEST(NnapiDelegateCompabilityTest, CompatibleModel) {
  CompatibilityCheckerDelegate checker_delegate(
      tflite::delegate::nnapi::kMinSdkVersionForNNAPI13);
  AddOpModel add_op_model(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE, &checker_delegate);
  EXPECT_EQ(checker_delegate.GetSupportedNodes().size(), 1);
  EXPECT_EQ(checker_delegate.GetFailuresByNode().size(), 0);
}

TEST(NnapiDelegateCompabilityTest, IncompatibleModel) {
  CompatibilityCheckerDelegate checker_delegate(
      tflite::delegate::nnapi::kMinSdkVersionForNNAPI13);
  // No activation function is supported for INT32 tensor type.
  AddOpModel add_op_model(
      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {1, 2, 2, 1}},
      {TensorType_INT32, {}}, ActivationFunctionType_RELU_N1_TO_1,
      &checker_delegate);
  EXPECT_EQ(checker_delegate.GetSupportedNodes().size(), 0);
  EXPECT_EQ(checker_delegate.GetFailuresByNode().size(), 1);
}

}  // namespace tools
}  // namespace tflite
