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
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseMeanOpModel : public SingleOpModel {
 public:
  void SetAxis(std::initializer_list<int> data) { PopulateTensor(axis_, data); }

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
  int axis_;
  int output_;
};

// Model for the tests case where axis is a const tensor.
class MeanOpConstModel : public BaseMeanOpModel {
 public:
  MeanOpConstModel(const TensorData& input, const TensorData& output,
                   std::initializer_list<int> axis_shape,
                   std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MEAN, BuiltinOptions_MeanOptions,
                 CreateMeanOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class MeanOpDynamicModel : public BaseMeanOpModel {
 public:
  MeanOpDynamicModel(const TensorData& input, const TensorData& output,
                     const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MEAN, BuiltinOptions_MeanOptions,
                 CreateMeanOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

TEST(ConstMeanOpTest, NotKeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({12, 13})));
}

TEST(ConstMeanOpTest, KeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                     {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({10.5, 12.5, 14.5})));
}

TEST(DynamicMeanOpTest, NotKeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                       false);
  std::initializer_list<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({12, 13})));
}

TEST(DynamicMeanOpTest, KeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}},
                       true);
  std::initializer_list<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({10.5, 12.5, 14.5})));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
