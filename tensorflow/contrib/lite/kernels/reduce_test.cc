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

class BaseOpModel : public SingleOpModel {
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

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  int Input() { return input_; }

 protected:
  int input_;
  int axis_;
  int output_;
};

// Model for the tests case where axis is a const tensor.
class MeanOpConstModel : public BaseOpModel {
 public:
  MeanOpConstModel(const TensorData& input, const TensorData& output,
                   std::initializer_list<int> axis_shape,
                   std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MEAN, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class MeanOpDynamicModel : public BaseOpModel {
 public:
  MeanOpDynamicModel(const TensorData& input, const TensorData& output,
                     const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MEAN, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a const tensor.
class SumOpConstModel : public BaseOpModel {
 public:
  SumOpConstModel(const TensorData& input, const TensorData& output,
                  std::initializer_list<int> axis_shape,
                  std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUM, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class SumOpDynamicModel : public BaseOpModel {
 public:
  SumOpDynamicModel(const TensorData& input, const TensorData& output,
                    const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUM, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// for quantized Add, the error shouldn't exceed step
float GetTolerance(int min, int max) { return (max - min) / 255.0; }

// Tests for reduce_mean
TEST(ConstFloatMeanOpTest, NotKeepDims) {
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

TEST(ConstFloatMeanOpTest, KeepDims) {
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

TEST(DynamicFloatMeanOpTest, NotKeepDims) {
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

TEST(DynamicFloatMeanOpTest, KeepDims) {
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

TEST(DynamicFloatMeanOpTest, Scale) {
  std::initializer_list<float> data = {9.527};
  MeanOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                       {TensorType_INT32, {1}}, true);
  std::initializer_list<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}


TEST(ConstUint8MeanOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::initializer_list<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                     {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                            {0.4, 0.4}, kQuantizedTolerance)));
}

TEST(ConstUint8MeanOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::initializer_list<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                     {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({0.3, 0.35, 0.55}, kQuantizedTolerance)));
}

TEST(DynamicUint8MeanOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::initializer_list<float> data = {1.3, -4.8, -3.6, 0.24};
  MeanOpDynamicModel m({TensorType_UINT8, {2, 2}, -5.0, 2.0},
                       {TensorType_UINT8, {2}, -5.0, 2.0},
                       {TensorType_INT32, {1}}, false);
  std::initializer_list<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({-1.75, -1.68}, kQuantizedTolerance)));
}

TEST(DynamicUint8MeanOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::initializer_list<float> data = {11.14, -0.14, 7.423, 0.879};
  MeanOpDynamicModel m({TensorType_UINT8, {2, 2}, -10.0, 12.0},
                       {TensorType_UINT8, {2}, -10.0, 12.0},
                       {TensorType_INT32, {1}}, true);
  std::initializer_list<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({9.2815, 0.3695}, kQuantizedTolerance)));
}

// Tests for reduce_sum

TEST(ConstFloatSumOpTest, NotKeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                    {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({144, 156})));
}

TEST(ConstFloatSumOpTest, KeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({84, 100, 116})));
}

TEST(DynamicFloatSumOpTest, NotKeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                      false);
  std::initializer_list<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({144, 156})));
}

TEST(DynamicFloatSumOpTest, KeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}}, true);
  std::initializer_list<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({84, 100, 116})));
}

TEST(DynamicFloatSumOpTest, Scale) {
  std::initializer_list<float> data = {9.527};
  SumOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                      {TensorType_INT32, {1}}, true);
  std::initializer_list<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

TEST(ConstUint8SumOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::initializer_list<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  SumOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(
                  ArrayFloatNear({-0.823529, -0.815686}, kQuantizedTolerance)));
}

TEST(ConstUint8SumOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::initializer_list<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  SumOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.407843, -0.313726, 0.0941177},
                                              kQuantizedTolerance)));
}

TEST(DynamicUint8SumOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::initializer_list<float> data = {1.3, -4.8, -3.6, 0.24};
  SumOpDynamicModel m({TensorType_UINT8, {2, 2}, -5.0, 2.0},
                      {TensorType_UINT8, {2}, -5.0, 2.0},
                      {TensorType_INT32, {1}}, false);
  std::initializer_list<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(
                  ArrayFloatNear({1.48235, 1.64706}, kQuantizedTolerance)));
}

TEST(DynamicUint8SumOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::initializer_list<float> data = {11.14, -0.14, 7.423, 0.879};
  SumOpDynamicModel m({TensorType_UINT8, {2, 2}, -10.0, 12.0},
                      {TensorType_UINT8, {2}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::initializer_list<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({6.47059, 10.698}, kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
