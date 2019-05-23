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
using ::testing::IsEmpty;

class BaseSqueezeOpModel : public SingleOpModel {
 public:
  BaseSqueezeOpModel(const TensorData& input, const TensorData& output,
                     std::initializer_list<int> axis) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_SQUEEZE, BuiltinOptions_SqueezeOptions,
        CreateSqueezeOptions(builder_, builder_.CreateVector<int>(axis))
            .Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

 protected:
  int input_;
  int output_;
};

class FloatSqueezeOpModel : public BaseSqueezeOpModel {
 public:
  using BaseSqueezeOpModel::BaseSqueezeOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

TEST(FloatSqueezeOpTest, SqueezeAll) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  FloatSqueezeOpModel m({TensorType_FLOAT32, {1, 24, 1}},
                        {TensorType_FLOAT32, {24}}, {});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                        9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}));
}

TEST(FloatSqueezeOpTest, SqueezeSelectedAxis) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  FloatSqueezeOpModel m({TensorType_FLOAT32, {1, 24, 1}},
                        {TensorType_FLOAT32, {24}}, {2});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                        9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}));
}

TEST(FloatSqueezeOpTest, SqueezeNegativeAxis) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  FloatSqueezeOpModel m({TensorType_FLOAT32, {1, 24, 1}},
                        {TensorType_FLOAT32, {24}}, {-1, 0});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                        9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}));
}

TEST(FloatSqueezeOpTest, SqueezeAllDims) {
  std::initializer_list<float> data = {3.85};
  FloatSqueezeOpModel m({TensorType_FLOAT32, {1, 1, 1, 1, 1, 1, 1}},
                        {TensorType_FLOAT32, {1}}, {});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.85}));
}

}  // namespace
}  // namespace tflite
