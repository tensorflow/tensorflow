/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class ModelType { kMax, kMin };

class MinMaxOpModel : public SingleOpModel {
 public:
  MinMaxOpModel(ModelType model_type, const TensorData& input1,
                const TensorData& input2, const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    model_type_ = model_type;

    switch (model_type_) {
      case ModelType::kMax:
        SetBuiltinOp(BuiltinOperator_STABLEHLO_MAXIMUM, BuiltinOptions_NONE, 0);
        break;
      case ModelType::kMin:
        SetBuiltinOp(BuiltinOperator_STABLEHLO_MINIMUM, BuiltinOptions_NONE, 0);
        break;
      default:
        ABSL_LOG(FATAL) << "Unknown model type.";
    }
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
  ModelType model_type_;
};

TEST(StablehloElementwise, MaxWorks) {
  MinMaxOpModel model(ModelType::kMax, {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {1.2, 2.5, -1.2, 1});
  model.PopulateTensor<float>(model.input2(), {0.1, 3, 2, 0.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(), {1.2, 3.0, 2.0, 1.0}));
}

TEST(StablehloElementwise, MinWorks) {
  MinMaxOpModel model(ModelType::kMin, {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {1.2, 2.5, -1.2, 1});
  model.PopulateTensor<float>(model.input2(), {0.1, 3, 2, 0.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(), {0.1, 2.5, -1.2, 0.5}));
}

}  // namespace
}  // namespace tflite
