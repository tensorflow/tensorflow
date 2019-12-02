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

class BaseOpModel : public SingleOpModel {
 public:
  void SetAxis(const std::vector<int>& data) { PopulateTensor(axis_, data); }

  template <class T>
  void SetInput(std::vector<T> data) {
    PopulateTensor(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
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

// Model for the tests case where axis is a const tensor.
class ProdOpConstModel : public BaseOpModel {
 public:
  ProdOpConstModel(const TensorData& input, const TensorData& output,
                   std::initializer_list<int> axis_shape,
                   std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_PROD, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class ProdOpDynamicModel : public BaseOpModel {
 public:
  ProdOpDynamicModel(const TensorData& input, const TensorData& output,
                     const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_PROD, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a const tensor.
class MaxOpConstModel : public BaseOpModel {
 public:
  MaxOpConstModel(const TensorData& input, const TensorData& output,
                  std::initializer_list<int> axis_shape,
                  std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_MAX, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class MaxOpDynamicModel : public BaseOpModel {
 public:
  MaxOpDynamicModel(const TensorData& input, const TensorData& output,
                    const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_MAX, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a const tensor.
class MinOpConstModel : public BaseOpModel {
 public:
  MinOpConstModel(const TensorData& input, const TensorData& output,
                  std::initializer_list<int> axis_shape,
                  std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_MIN, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class MinOpDynamicModel : public BaseOpModel {
 public:
  MinOpDynamicModel(const TensorData& input, const TensorData& output,
                    const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_MIN, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a const tensor.
class AnyOpConstModel : public BaseOpModel {
 public:
  AnyOpConstModel(const TensorData& input, const TensorData& output,
                  std::initializer_list<int> axis_shape,
                  std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_ANY, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
class AnyOpDynamicModel : public BaseOpModel {
 public:
  AnyOpDynamicModel(const TensorData& input, const TensorData& output,
                    const TensorData& axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_REDUCE_ANY, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// for quantized Add, the error shouldn't exceed step
float GetTolerance(int min, int max) { return (max - min) / 255.0; }

// Tests for reduce_mean
TEST(ConstFloatMeanOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({12, 13})));
}

TEST(ConstFloatMeanOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                     {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({10.5, 12.5, 14.5})));
}
// Uses a set of reduction conditions that trigger the specialized 4D version
// of Mean.
TEST(ConstFloatMeanOpTest, KeepDims4DMean) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpConstModel m({TensorType_FLOAT32, {2, 2, 3, 2}},
                     {TensorType_FLOAT32, {3}}, {2}, {1, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({6, 7, 18, 19})));
}

TEST(ConstFloatMeanOpTest, KeepDims4DMeanUInt8) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2,
                             0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({TensorType_UINT8, {1, 2, 2, 3}, -1.0, 1.0},
                     {TensorType_UINT8, {2}, -1.0, 1.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 3}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({0.25098, 0.25098, 0.25098},
                                              kQuantizedTolerance)));
}

TEST(ConstFloatMeanOpTest, KeepDims4DMeanLargeDepthUInt8) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {
      0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.4, 0.2, 0.2,
      0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2,
      0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1,
      0.1, 0.1, 0.1, 0.4, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.3, 0.7, 0.7,
      0.1, 0.1, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({TensorType_UINT8, {1, 2, 2, 18}, -1.0, 1.0},
                     {TensorType_UINT8, {2}, -1.0, 1.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 18}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5, 0.55, 0.25, 0.35, 0.45, 0.5, 0.25, 0.3, 0.2, 0.2, 0.1,
                   0.15, 0.35, 0.3, 0.15, 0.2, 0.6, 0.65},
                  kQuantizedTolerance)));
}

TEST(ConstFloatMeanOpTest, KeepDims4DMeanQuantized) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2,
                             0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({TensorType_UINT8, {1, 2, 3, 2}, 0.0, 1.0},
                     {TensorType_UINT8, {3}, -5.0, 5.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({0.235294, 0.313726}, kQuantizedTolerance)));
}

TEST(ConstFloatMeanOpTest, Scalar) {
  std::vector<float> data = {3.27};
  MeanOpConstModel m({TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}}, {},
                     {0}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({3.27})));
}

TEST(DynamicFloatMeanOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                       false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({12, 13})));
}

TEST(DynamicFloatMeanOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}},
                       true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({10.5, 12.5, 14.5})));
}

TEST(DynamicFloatMeanOpTest, Scale) {
  std::vector<float> data = {9.527};
  MeanOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                       {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}


TEST(ConstUint8MeanOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                     {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.4, 0.4}, kQuantizedTolerance)));
}

TEST(ConstUint8MeanOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                     {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.3, 0.35, 0.55}, kQuantizedTolerance)));
}

TEST(ConstInt8MeanOpTest, NonSpecialAxisSameScale) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {105.0, 71.0, 233.0, 92.0, 227.0, 11.0, 14.0, 43.0};
  MeanOpConstModel m({TensorType_INT8, {1, 1, 2, 4}, 0.0, 255.0},
                     {TensorType_INT8, {1, 2, 4}, 0.0, 255.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(data, kQuantizedTolerance)));
}

TEST(ConstInt8MeanOpTest, NonSpecialAxisNonSameScale) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  MeanOpConstModel m({TensorType_INT8, {1, 1, 2, 4}, -1.0, 1.0},
                     {TensorType_INT8, {1, 2}, -5.0, 5.0}, {2}, {1, 3}, false);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.25, 0.65}, kQuantizedTolerance)));
}

TEST(ConstInt8MeanOpTest, QuantizedSameScale) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1,
                             0.1, 0.1, 0.1, 0.4, 0.2, 0.2, 0.2, 0.9, 0.9,
                             0.9, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.3,
                             0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({TensorType_INT8, {1, 2, 2, 9}, -1.0, 1.0},
                     {TensorType_INT8, {2}, -1.0, 1.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 9}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.35, 0.325, 0.2, 0.35, 0.375, 0.325, 0.225, 0.45, 0.425},
                  kQuantizedTolerance)));
}

TEST(ConstInt8MeanOpTest, QuantizedDifferentScale) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1,
                             0.1, 0.1, 0.1, 0.4, 0.2, 0.2, 0.2, 0.9, 0.9,
                             0.9, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.3,
                             0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({TensorType_INT8, {1, 2, 2, 9}, -1.0, 1.0},
                     {TensorType_INT8, {2}, -4.0, 4.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 9}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.35, 0.325, 0.2, 0.35, 0.375, 0.325, 0.225, 0.45, 0.425},
                  kQuantizedTolerance)));
}

TEST(DynamicUint8MeanOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MeanOpDynamicModel m({TensorType_UINT8, {2, 2}, -5.0, 2.0},
                       {TensorType_UINT8, {2}, -5.0, 2.0},
                       {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({-1.75, -1.68}, kQuantizedTolerance)));
}

TEST(DynamicUint8MeanOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MeanOpDynamicModel m({TensorType_UINT8, {2, 2}, -10.0, 12.0},
                       {TensorType_UINT8, {2}, -10.0, 12.0},
                       {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({9.2815, 0.3695}, kQuantizedTolerance)));
}

TEST(DynamicUint8MeanOpTest, QuantizedScalar) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {0.643};
  MeanOpDynamicModel m({TensorType_UINT8, {}, 0.0, 1.0},
                       {TensorType_UINT8, {}, -10.0, 12.0},
                       {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({0.643}, kQuantizedTolerance)));
}

TEST(ConstUint8MeanOpTest, QuantizedKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {3, 2}, 0.0, 1.0},
                     {TensorType_UINT8, {3}, -5.0, 5.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.3, 0.35, 0.55}, kQuantizedTolerance)));
}

// Tests for reduce_sum

TEST(ConstFloatSumOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                    {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({144, 156})));
}

TEST(ConstFloatSumOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({84, 100, 116})));
}

TEST(DynamicFloatSumOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                      false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({144, 156})));
}

TEST(ConstFloatSumOpTest, Scalar) {
  std::vector<float> data = {17.};
  SumOpConstModel m({TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}}, {}, {0},
                    false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({17.})));
}

TEST(DynamicFloatSumOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SumOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}}, true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({84, 100, 116})));
}

TEST(DynamicFloatSumOpTest, Scale) {
  std::vector<float> data = {9.527};
  SumOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

TEST(ConstUint8SumOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  SumOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({-0.823529, -0.815686}, kQuantizedTolerance)));
}

TEST(ConstUint8SumOpTest, NotKeepDimsRescaling) {
  float kQuantizedTolerance = GetTolerance(0.0, 2.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  SumOpConstModel m({TensorType_UINT8, {1, 3, 2}, 0.0, 1.0},
                    {TensorType_UINT8, {2}, 0.0, 2.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({1.2, 1.2}, kQuantizedTolerance)));
}

TEST(ConstUint8SumOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  SumOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({-0.407843, -0.313726, 0.0941177},
                                              kQuantizedTolerance)));
}

TEST(DynamicUint8SumOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  SumOpDynamicModel m({TensorType_UINT8, {2, 2}, -5.0, 2.0},
                      {TensorType_UINT8, {2}, -5.0, 2.0},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({1.48235, 1.64706}, kQuantizedTolerance)));
}

TEST(DynamicUint8SumOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  SumOpDynamicModel m({TensorType_UINT8, {2, 2}, -10.0, 12.0},
                      {TensorType_UINT8, {2}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({6.47059, 10.698}, kQuantizedTolerance)));
}

TEST(ConstInt8SumOpTest, Rescale) {
  const std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.3};
  SumOpConstModel m({TensorType_INT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_INT8, {2}, -5.0, 5.0}, {1}, {1}, false);
  // Expect the sum to be 0.4 + 0.3 + 0.5 = 1.2 and 0.2 + 0.4 + 0.3 = 0.9.
  const std::vector<float> expected_sum = {1.2, 0.9};
  const float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(expected_sum, kQuantizedTolerance)));
}

// Tests for reduce_prod

TEST(ConstFloatProdOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({3.162341376e+11, 1.9619905536e+12})));
}

TEST(ConstFloatProdOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                     {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  ArrayFloatNear({7.74592e+06, 1.197504e+08, 6.6889152e+08})));
}

TEST(DynamicFloatProdOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                       false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({3.16234143225e+11, 1.9619905536e+12})));
}

TEST(DynamicFloatProdOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}},
                       true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  ArrayFloatNear({7.74592e+06, 1.197504e+08, 6.6889152e+08})));
}

TEST(DynamicFloatProdOpTest, Scale) {
  std::vector<float> data = {9.527};
  ProdOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                       {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

// Tests for reduce_max

TEST(ConstFloatMaxOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MaxOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                    {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({23, 24})));
}

TEST(ConstFloatMaxOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MaxOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({20, 22, 24})));
}

TEST(DynamicFloatMaxOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MaxOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                      false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({23, 24})));
}

TEST(DynamicFloatMaxOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MaxOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}}, true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({20, 22, 24})));
}

TEST(DynamicFloatMaxOpTest, Scale) {
  std::vector<float> data = {9.527};
  MaxOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

TEST(ConstUint8MaxOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MaxOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({0.501961, 0.603922}, kQuantizedTolerance)));
}

TEST(ConstInt8MaxOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MaxOpConstModel m({TensorType_INT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_INT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({0.501961, 0.603922}, kQuantizedTolerance)));
}

TEST(ConstUint8MaxOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MaxOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({0.4, 0.4, 0.603922}, kQuantizedTolerance)));
}

TEST(ConstInt8MaxOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MaxOpConstModel m({TensorType_INT8, {3, 2}, -1.0, 1.0},
                    {TensorType_INT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({0.4, 0.4, 0.603922}, kQuantizedTolerance)));
}

TEST(DynamicUint8MaxOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MaxOpDynamicModel m({TensorType_UINT8, {2, 2}, -5.0, 2.0},
                      {TensorType_UINT8, {2}, -5.0, 2.0},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({1.2902, 0.247059}, kQuantizedTolerance)));
}

TEST(DynamicInt8MaxOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MaxOpDynamicModel m({TensorType_INT8, {2, 2}, -5.0, 2.0},
                      {TensorType_INT8, {2}, -5.0, 2.0},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({1.2902, 0.247059}, kQuantizedTolerance)));
}

TEST(DynamicUint8MaxOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MaxOpDynamicModel m({TensorType_UINT8, {2, 2}, -10.0, 12.0},
                      {TensorType_UINT8, {2}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({11.1294, 0.862745}, kQuantizedTolerance)));
}

TEST(DynamicInt8MaxOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MaxOpDynamicModel m({TensorType_INT8, {2, 2}, -10.0, 12.0},
                      {TensorType_INT8, {2}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({11.1294, 0.862745}, kQuantizedTolerance)));
}

TEST(DynamicUint8MaxOpTest, Scalar) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14};
  MaxOpDynamicModel m({TensorType_UINT8, {}, -10.0, 12.0},
                      {TensorType_UINT8, {}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({11.1294}, kQuantizedTolerance)));
}

TEST(DynamicInt8MaxOpTest, Scalar) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14};
  MaxOpDynamicModel m({TensorType_INT8, {}, -10.0, 12.0},
                      {TensorType_INT8, {}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({11.1294}, kQuantizedTolerance)));
}

// Tests for reduce_min

TEST(ConstFloatMinOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MinOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                    {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({1, 2})));
}

TEST(ConstFloatMinOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MinOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 3, 5})));
}

TEST(DynamicFloatMinOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MinOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {2}}, {TensorType_INT32, {4}},
                      false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({1, 2})));
}

TEST(DynamicFloatMinOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MinOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                      {TensorType_FLOAT32, {3}}, {TensorType_INT32, {2}}, true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 3, 5})));
}

TEST(DynamicFloatMinOpTest, Scalar) {
  std::vector<float> data = {9.527};
  MinOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

TEST(ConstUint8MinOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MinOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.294117, 0.2}, kQuantizedTolerance)));
}

TEST(ConstInt8MinOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MinOpConstModel m({TensorType_INT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_INT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.294117, 0.2}, kQuantizedTolerance)));
}

TEST(ConstUint8MinOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MinOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.2, 0.3, 0.5}, kQuantizedTolerance)));
}

TEST(ConstInt8MinOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MinOpConstModel m({TensorType_INT8, {3, 2}, -1.0, 1.0},
                    {TensorType_INT8, {3}, -1.0, 1.0}, {1}, {1}, true);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.2, 0.3, 0.5}, kQuantizedTolerance)));
}

TEST(DynamicUint8MinOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MinOpDynamicModel m({TensorType_UINT8, {2, 2}, -5.0, 2.0},
                      {TensorType_UINT8, {2}, -5.0, 2.0},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({-4.807843, -3.6}, kQuantizedTolerance)));
}

TEST(DynamicInt8MinOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-5.0, 2.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MinOpDynamicModel m({TensorType_INT8, {2, 2}, -5.0, 2.0},
                      {TensorType_INT8, {2}, -5.0, 2.0},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({-4.807843, -3.6}, kQuantizedTolerance)));
}

TEST(DynamicUint8MinOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MinOpDynamicModel m({TensorType_UINT8, {2, 2}, -10.0, 12.0},
                      {TensorType_UINT8, {2}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({7.427451, -0.164706}, kQuantizedTolerance)));
}

TEST(DynamicInt8MinOpTest, KeepDims) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MinOpDynamicModel m({TensorType_INT8, {2, 2}, -10.0, 12.0},
                      {TensorType_INT8, {2}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({7.427451, -0.164706}, kQuantizedTolerance)));
}

TEST(DynamicUint8MinOpTest, Scalar) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14};
  MinOpDynamicModel m({TensorType_UINT8, {}, -10.0, 12.0},
                      {TensorType_UINT8, {}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({11.1294}, kQuantizedTolerance)));
}

TEST(DynamicInt8MinOpTest, Scalar) {
  float kQuantizedTolerance = GetTolerance(-10.0, 12.0);
  std::vector<float> data = {11.14};
  MinOpDynamicModel m({TensorType_INT8, {}, -10.0, 12.0},
                      {TensorType_INT8, {}, -10.0, 12.0},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({11.1294}, kQuantizedTolerance)));
}

// Tests for reduce_any

TEST(ConstAnyOpTest, NotKeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpConstModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2}}, {4},
                    {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false, true}));
}

TEST(ConstAnyOpTest, KeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpConstModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {3}}, {2},
                    {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, false, true}));
}

TEST(DynamicAnyOpTest, NotKeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpDynamicModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2}},
                      {TensorType_INT32, {4}}, false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false, true}));
}

TEST(DynamicAnyOpTest, KeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpDynamicModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {3}},
                      {TensorType_INT32, {2}}, true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, false, true}));
}

TEST(DynamicAnyOpTest, Scalar) {
  std::vector<bool> data = {false};
  AnyOpDynamicModel m({TensorType_BOOL, {1}}, {TensorType_BOOL, {1}},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false}));
}

}  // namespace
}  // namespace tflite
