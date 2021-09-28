/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class RoundOpModel : public SingleOpModel {
 public:
  RoundOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ROUND, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 private:
  int input_;
  int output_;
};

template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return kQuantizedStep;
}

TEST(RoundOpTest, Float32SingleDim) {
  RoundOpModel model({TensorType_FLOAT32, {6}}, {TensorType_FLOAT32, {6}});
  model.PopulateTensor<float>(model.input(), {8.5, 0.0, 3.5, 4.2, -3.5, -4.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({8, 0, 4, 4, -4, -4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
}

TEST(RoundOpTest, Float32MultiDims) {
  RoundOpModel model({TensorType_FLOAT32, {2, 1, 1, 6}},
                     {TensorType_FLOAT32, {2, 1, 1, 6}});
  model.PopulateTensor<float>(
      model.input(), {0.0001, 8.0001, 0.9999, 9.9999, 0.5, -0.0001, -8.0001,
                      -0.9999, -9.9999, -0.5, -2.5, 1.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0, 8, 1, 10, 0, 0, -8, -1, -10, -0, -2, 2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 6}));
}

TEST(RoundOpTest, Int8MultiDims) {
  RoundOpModel model({TensorType_INT8, {2, 1, 1, 8}, -5.0f, 6.0f},
                     {TensorType_INT8, {2, 1, 1, 8}, 0, 0, 1.0f});
  model.template QuantizeAndPopulate<int8_t>(
      model.input(), {-4.2, -3.5, -4.5, 1, 1, 3, 5, 1, 4, 1, 0.5, 0.77, 0.0001,
                      0.9999, -0.0001, -0.5});
  model.Invoke();
  EXPECT_THAT(
      model.template GetDequantizedOutput<int8_t>(),
      ElementsAreArray({-4, -4, -5, 1, 1, 3, 5, 1, 4, 1, 1, 1, 0, 1, 0, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 8}));
}

TEST(RoundOpTest, Int16MultiDims) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<int16_t>::max() /
      static_cast<float>(std::numeric_limits<int16_t>::max() + 1);
  RoundOpModel model({TensorType_INT16, {2, 1, 1, 8}, 5.0f * kMin, 5.0f * kMax},
                     {TensorType_INT16, {2, 1, 1, 8}, 0, 0, 1.0f, 0});
  model.template QuantizeAndPopulate<int16_t>(
      model.input(), {-4.2, -3.5, -4.5, 1, 1, 3, 5, 1, 4, 1, 0.5, 0.77, 0.0001,
                      0.9999, -0.0001, -0.5});
  model.Invoke();
  EXPECT_THAT(
      model.template GetDequantizedOutput<int16_t>(),
      ElementsAreArray({-4, -4, -5, 1, 1, 3, 5, 1, 4, 1, 1, 1, 0, 1, 0, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 8}));
}

}  // namespace
}  // namespace tflite
