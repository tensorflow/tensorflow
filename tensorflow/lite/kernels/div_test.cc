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
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseDivOpModel : public SingleOpModel {
 public:
  BaseDivOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_DIV, BuiltinOptions_DivOptions,
                 CreateDivOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

class QuantizedDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }
};

// For quantized Div, the error shouldn't exceed (2*step + step^2).
inline float GetTolerance(int min, int max) {
  const float kQuantizedStep = (max - min) / 255.0f;
  const float kQuantizedTolerance =
      2.0f * kQuantizedStep + kQuantizedStep * kQuantizedStep;
  return kQuantizedTolerance;
}

TEST(FloatDivOpTest, NoActivationInplaceInput0) {
  FloatDivOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  const int kInplaceInputTensorIdx = 0;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.4, 1.0, 0.8, 1.6})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(FloatDivOpTest, NoActivationInplaceInput1) {
  FloatDivOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  const int kInplaceInputTensorIdx = 1;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.4, 1.0, 0.8, 1.6})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(FloatDivOpTest, NoActivation) {
  FloatDivOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.4, 1.0, 0.8, 1.6})));
}

TEST(FloatDivOpTest, ActivationRELU_N1_TO_1) {
  FloatDivOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, -1.5, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 1.0, 0.8, 1.0})));
}

TEST(FloatDivOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.6, 0.5, -1.1, -0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-20.0, 1.0, 0.5, 1.6, -1.0, 20.0})))
        << "With shape number " << i;
  }
}

TEST(FloatDivOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {8}, {2, 4}, {2, 1, 4}, {1, 2, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(),
                            {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123, -0.32, 0.54});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23, -3.2, 5.4})))
        << "With shape number " << i;
  }
}

TEST(FloatDivOpTest, WithBroadcast5D) {
  std::vector<std::vector<int>> test_shapes = {{1, 2, 1, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(),
                            {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123, -0.32, 0.54});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23, -3.2, 5.4})))
        << "With shape number " << i;
  }
}

TEST(IntegerDivOpTest, NoActivation) {
  IntegerDivOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -15, 8});
  m.PopulateTensor<int32_t>(m.input2(), {5, -2, -3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, -1, 5, 1}));
}

TEST(IntegerDivOpTest, ActivationRELU_N1_TO_1) {
  IntegerDivOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -12, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, -15, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 1, 0, 1}));
}

TEST(IntegerDivOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerDivOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 3, 8, 11, -20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 6, 5, -11, -1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 1, 0, 1, -1, 20}))
        << "With shape number " << i;
  }
}

TEST(IntegerDivOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {8}, {2, 4}, {2, 1, 4}, {1, 4, 1, 2}, {1, 2, 1, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerDivOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 21, 7, 8, 11, -123, -42, -48});
    m.PopulateTensor<int32_t>(m.input2(), {3});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({-6, 7, 2, 2, 3, -41, -14, -16}))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedNoActivation() {
  const float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  QuantizedDivOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                        {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                        {tensor_type, {}, -1.0, 1.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<integer_dtype>(m.input1(), {-0.8, -0.2, 0.3, 0.7});
  m.QuantizeAndPopulate<integer_dtype>(m.input2(), {-0.8, 0.4, 0.8, 1.0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({1.0, -0.5, 0.375, 0.7},
                                              kQuantizedTolerance)));
}

TEST(QuantizedDivOpTest, QuantizedNoActivationUInt8) {
  QuantizedNoActivation<TensorType_UINT8, uint8_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedActivationRELU_N1_TO_1() {
  const float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  const std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                                   {-0.5, 0.2, 0.6, 0.3}};
  const std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                                   {0.6, 0.5, -0.8, 0.5}};
  const std::vector<std::vector<float>> results = {{-1.0, 0.5, 1.0, -0.875},
                                                   {-0.833, 0.4, -0.75, 0.6}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedDivOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedDivOpTest, QuantizedActivationRELU_N1_TO_1UInt8) {
  QuantizedActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  const float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedDivOpModel m({tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 1.7, 0.9, 0.4, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {1.3, 0.3, 1.1, 0.4, -1.1, 1.9});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(
            {-1.538, 0.667, 1.545, 2.25, -0.364, 1.053}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedDivOpTest, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedWithBroadcast() {
  const float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  const std::vector<std::vector<int>> test_shapes = {
      {8}, {2, 4}, {2, 1, 4}, {1, 4, 1, 2}, {1, 2, 1, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedDivOpModel m(
        {tensor_type, test_shapes[i], -3.0, 3.0}, {tensor_type, {}, -3.0, 3.0},
        {tensor_type, {}, -3.0, 3.0}, ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(
        m.input1(), {-2.0, 0.2, 0.7, 0.8, -0.5, 1.1, -1.3, 1.2});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.7});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.857, 0.286, 1.0, 1.143, -0.714, 1.571, -1.857, 1.714},
                    kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedDivOpTest, QuantizedWithBroadcastUInt8) {
  QuantizedWithBroadcast<TensorType_UINT8, uint8_t>();
}

}  // namespace
}  // namespace tflite
