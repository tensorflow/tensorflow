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
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class FakeQuantOpModel : public SingleOpModel {
 public:
  FakeQuantOpModel(const TensorData& input, const TensorType& output, float min,
                   float max, int num_bits) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_FAKE_QUANT, BuiltinOptions_FakeQuantOptions,
                 CreateFakeQuantOptions(builder_, min, max, num_bits).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <class T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
};

TEST(FakeQuantOpTest, FloatPositiveRange8Test) {
  std::initializer_list<float> data = {0.0,  1.0,       0.25,
                                       0.50, 0.4444444, 0.00001};
  FakeQuantOpModel m({TensorType_FLOAT32, {3, 1, 2}}, TensorType_FLOAT32, 0.0f,
                     1.0f, 8);
  m.SetInput<float>(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({0, 1, 0.25098, 0.498039, 0.443137, 0})));
}

TEST(FakeQuantOpTest, FloatNegativeRange8Test) {
  std::initializer_list<float> data = {0.0,  -0.9,      0.25,
                                       0.50, 0.4444444, -0.00001};
  FakeQuantOpModel m({TensorType_FLOAT32, {3, 1, 2}}, TensorType_FLOAT32, -0.9f,
                     0.9f, 8);
  m.SetInput<float>(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {0, -0.896471, 0.247059, 0.501176, 0.444706, 0})));
}

TEST(FakeQuantOpTest, FloatPositiveRange16Test) {
  std::initializer_list<float> data = {0.0,  1.0,       0.25,
                                       0.50, 0.4444444, 0.00001};
  FakeQuantOpModel m({TensorType_FLOAT32, {3, 1, 2}}, TensorType_FLOAT32, 0.0f,
                     1.0f, 16);
  m.SetInput<float>(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {0, 1, 0.250004, 0.500008, 0.44445, 1.5259e-05})));
}

TEST(FakeQuantOpTest, FloatNegativeRange16Test) {
  std::initializer_list<float> data = {0.0,  -0.9,      0.25,
                                       0.50, 0.4444444, -0.00001};
  FakeQuantOpModel m({TensorType_FLOAT32, {3, 1, 2}}, TensorType_FLOAT32, -0.9f,
                     0.9f, 16);
  m.SetInput<float>(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {0, -0.900014, 0.249998, 0.499995, 0.444431, 0})));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
