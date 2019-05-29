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

using ::testing::ElementsAre;

template <typename T>
class FloorDivModel : public SingleOpModel {
 public:
  FloorDivModel(const TensorData& input1, const TensorData& input2,
                const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_FLOOR_DIV, BuiltinOptions_FloorDivOptions,
                 CreateFloorDivOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int output_;
};

template <typename T>
class QuantizedFloorDivOpModel : public SingleOpModel {
 public:
  QuantizedFloorDivOpModel(const TensorData& input1, const TensorData& input2,
                           const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_FLOOR_DIV, BuiltinOptions_FloorDivOptions,
                 CreateFloorDivOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(this->output_),
                         GetScale(this->output_), GetZeroPoint(this->output_));
  }

 private:
  int input1_;
  int input2_;
  int output_;
};

TEST(FloorDivModel, Simple) {
  FloorDivModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, 3, 4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(5, 4, 3, 0));
}

TEST(FloorDivModel, NegativeValue) {
  FloorDivModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, -3, -4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(5, -5, 3, -2));
}

TEST(FloorDivModel, BroadcastFloorDiv) {
  FloorDivModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {-3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(-4, 3, 3, -3));
}

TEST(FloorDivModel, SimpleFloat) {
  FloorDivModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10.05, 9.09, 11.9, 3.01});
  model.PopulateTensor<float>(model.input2(), {2.05, 2.03, 3.03, 4.03});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(4.0, 4.0, 3.0, 0.0));
}

TEST(FloorDivModel, NegativeValueFloat) {
  FloorDivModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10.03, -9.9, -11.0, 7.0});
  model.PopulateTensor<float>(model.input2(), {2.0, 2.3, -3.0, -4.1});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(5.0, -5.0, 3.0, -2.0));
}

TEST(FloorDivModel, BroadcastFloorDivFloat) {
  FloorDivModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10.03, -9.9, -11.0, 7.0});
  model.PopulateTensor<float>(model.input2(), {-3.3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(-4.0, 2.0, 3.0, -3.0));
}

// for quantized Floor Div, the error shouldn't exceed 2*step
float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  float kQuantizedTolerance = 2.0 * kQuantizedStep;
  return kQuantizedTolerance;
}

TEST(QuantizedFloorDivOpTest, NoBroadCastInt8) {
  float kQuantizedTolerance = GetTolerance(-8.0, 8.0);
  QuantizedFloorDivOpModel<int8_t> m({TensorType_INT8, {1, 2, 2, 1}, -8.0, 8.0},
                                     {TensorType_INT8, {1, 2, 2, 1}, -8.0, 8.0},
                                     {TensorType_INT8, {}, -8.0, 8.0});
  m.QuantizeAndPopulate<int8_t>(m.input1(), {4.0, 5.0, 4.0, 7.5});
  m.QuantizeAndPopulate<int8_t>(m.input2(), {3.0, 2.0, 1.0, 2.0});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({1.0, 2.0, 4.0, 3.0}, kQuantizedTolerance)));
}

TEST(QuantizedFloorDivOpTest, NoBroadCastUInt8) {
  float kQuantizedTolerance = GetTolerance(-8.0, 8.0);
  QuantizedFloorDivOpModel<uint8_t> m(
      {TensorType_UINT8, {1, 2, 2, 1}, -8.0, 8.0},
      {TensorType_UINT8, {1, 2, 2, 1}, -8.0, 8.0},
      {TensorType_UINT8, {}, -8.0, 8.0});
  m.QuantizeAndPopulate<uint8_t>(m.input1(), {-4.0, 5.0, 7.5, 7.5});
  m.QuantizeAndPopulate<uint8_t>(m.input2(), {3.0, 2.0, 4.0, 2.0});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({-2.0, 2.0, 1.0, 3.0}, kQuantizedTolerance)));
}

TEST(QuantizedFloorDivOpTest, BroadCastInt8) {
  float kQuantizedTolerance = GetTolerance(-8.0, 8.0);
  QuantizedFloorDivOpModel<int8_t> m({TensorType_INT8, {6}, -8.0, 8.0},
                                     {TensorType_INT8, {1}, -8.0, 8.0},
                                     {TensorType_INT8, {}, -8.0, 8.0});
  m.QuantizeAndPopulate<int8_t>(m.input1(), {4.0, 3.2, 2.7, 1.8, 5.0, 6.3});
  m.QuantizeAndPopulate<int8_t>(m.input2(), {1.3});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3.0, 2.0, 2.0, 1.0, 3.0, 4.0},
                                              kQuantizedTolerance)));
}

TEST(QuantizedFloorDivOpTest, BroadCastUInt8) {
  float kQuantizedTolerance = GetTolerance(-8.0, 8.0);
  QuantizedFloorDivOpModel<uint8_t> m({TensorType_UINT8, {6}, -8.0, 8.0},
                                      {TensorType_UINT8, {1}, -8.0, 8.0},
                                      {TensorType_UINT8, {}, -8.0, 8.0});
  m.QuantizeAndPopulate<uint8_t>(m.input1(), {-3.2, -4.0, 2.7, 1.8, 5.0, 6.3});
  m.QuantizeAndPopulate<uint8_t>(m.input2(), {1.2});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({-3.0, -4.0, 2.0, 1.0, 4.0, 5.0},
                                              kQuantizedTolerance)));
}

}  // namespace

}  // namespace tflite
