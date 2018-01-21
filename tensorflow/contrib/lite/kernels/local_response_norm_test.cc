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
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

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
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/0.0f,
                             /*alpha=*/1.0f, /*beta=*/0.5f);
  m.SetInput({-1.1f, 0.6f, 0.7f, 1.2f, -0.7f, 0.1f});
  m.Invoke();
  // The result is every input divided by 2.
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-0.55f, 0.3f, 0.35f, 0.6f, -0.35f, 0.05f})));
}

TEST(LocalResponseNormOpTest, WithAlpha) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/0.0f,
                             /*alpha=*/4.0f, /*beta=*/0.5f);
  m.SetInput({-1.1f, 0.6f, 0.7f, 1.2f, -0.7f, 0.1f});
  m.Invoke();
  // The result is every input divided by 3.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {-0.275f, 0.15f, 0.175f, 0.3f, -0.175f, 0.025f})));
}

TEST(LocalResponseNormOpTest, WithBias) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/9.0f,
                             /*alpha=*/4.0f, /*beta=*/0.5f);
  m.SetInput({-1.1f, 0.6f, 0.7f, 1.2f, -0.7f, 0.1f});
  m.Invoke();
  // The result is every input divided by 5.
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-0.22f, 0.12f, 0.14f, 0.24f, -0.14f, 0.02f})));
}

TEST(LocalResponseNormOpTest, SmallRadius) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/2, /*bias=*/9.0f,
                             /*alpha=*/4.0f, /*beta=*/0.5f);
  m.SetInput({-1.1f, 0.6f, 0.7f, 1.2f, -0.7f, 0.1f});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {-0.264926f, 0.125109f, 0.140112f, 0.267261f, -0.161788f, 0.0244266f})));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
