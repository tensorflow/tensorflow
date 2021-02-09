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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class LocalResponseNormOpModel : public SingleOpModel {
 public:
  LocalResponseNormOpModel(std::initializer_list<int> input_shape, int radius,
                           float bias, float alpha, float beta) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
                 BuiltinOptions_LocalResponseNormalizationOptions,
                 CreateLocalResponseNormalizationOptions(builder_, radius, bias,
                                                         alpha, beta)
                     .Union());
    BuildInterpreter({input_shape});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(LocalResponseNormOpTest, SameAsL2Norm) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/0.0,
                             /*alpha=*/1.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  // The result is every input divided by 2.
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05})));
}

TEST(LocalResponseNormOpTest, WithAlpha) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/0.0,
                             /*alpha=*/4.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  // The result is every input divided by 3.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {-0.275, 0.15, 0.175, 0.3, -0.175, 0.025})));
}

TEST(LocalResponseNormOpTest, WithBias) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/9.0,
                             /*alpha=*/4.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  // The result is every input divided by 5.
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-0.22, 0.12, 0.14, 0.24, -0.14, 0.02})));
}

TEST(LocalResponseNormOpTest, SmallRadius) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/2, /*bias=*/9.0,
                             /*alpha=*/4.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {-0.264926, 0.125109, 0.140112, 0.267261, -0.161788, 0.0244266})));
}

}  // namespace
}  // namespace tflite
