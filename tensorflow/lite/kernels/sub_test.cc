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

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseSubOpModel : public SingleOpModel {
 public:
  BaseSubOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union());
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

class QuantizedSubOpModel : public BaseSubOpModel {
 public:
  QuantizedSubOpModel(TensorData input1, TensorData input2, TensorData output,
                      ActivationFunctionType activation_type)
      : BaseSubOpModel(SymmetricInt16Scaling(std::move(input1)),
                       SymmetricInt16Scaling(std::move(input2)),
                       SymmetricInt16Scaling(std::move(output)),
                       activation_type) {}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

 private:
  TensorData SymmetricInt16Scaling(TensorData tensor) {
    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }

    return tensor;
  }
};

// for quantized Sub, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return 2.0 * kQuantizedStep;
}

TEST(FloatSubOpModel, FirstInputZero) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  FloatSubOpModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input2(), {0.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TEST(FloatSubOpModel, SecondInputZero) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  FloatSubOpModel m({TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {0}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {0.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TEST(FloatSubOpModel, NoActivationInplaceInput0) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  const int kInplaceInputTensorIdx = 0;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(FloatSubOpModel, NoActivationInplaceInput1) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  const int kInplaceInputTensorIdx = 1;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(FloatSubOpModel, NoActivation) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3})));
}

TEST(FloatSubOpModel, ActivationRELU_N1_TO_1) {
  FloatSubOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 0.0, 1.0, -0.3})));
}

TEST(FloatSubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8, -1.1, 0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3, 0.0, 1.9})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast5D) {
  std::vector<std::vector<int>> test_shapes = {{1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(IntegerSubOpModel, NoActivation) {
  IntegerSubOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-21, 0, 4, 3}));
}

TEST(IntegerSubOpModel, ActivationRELU_N1_TO_1) {
  IntegerSubOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-1, 0, 1, 1}));
}

TEST(IntegerSubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-21, 0, 4, 3, 0, 19}))
        << "With shape number " << i;
  }
}

TEST(IntegerSubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<int32_t>(),
                ElementsAreArray(ArrayFloatNear({-21, 1, 6, 7, 10, 19})))
        << "With shape number " << i;
  }
}

TEST(Int64SubOpModel, NoActivation) {
  IntegerSubOpModel m({TensorType_INT64, {1, 2, 2, 1}},
                      {TensorType_INT64, {1, 2, 2, 1}}, {TensorType_INT64, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({-21, 0, 4, 3}));
}

TEST(Int64SubOpModel, ActivationRELU_N1_TO_1) {
  IntegerSubOpModel m({TensorType_INT64, {1, 2, 2, 1}},
                      {TensorType_INT64, {1, 2, 2, 1}}, {TensorType_INT64, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({-1, 0, 1, 1}));
}

TEST(Int64SubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT64, test_shapes[i]},
                        {TensorType_INT64, test_shapes[i]},
                        {TensorType_INT64, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({-21, 0, 4, 3, 0, 19}))
        << "With shape number " << i;
  }
}

TEST(Int64SubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT64, test_shapes[i]},
                        {TensorType_INT64, {}},  // always a scalar
                        {TensorType_INT64, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int64_t>(m.input2(), {1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<int64_t>(),
                ElementsAreArray(ArrayFloatNear({-21, 1, 6, 7, 10, 19})))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-0.9, 0.9);
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.2, 0.2, 0.4, 0.7}, {-0.01, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.2}, {0.6, 0.4, -0.18, 0.5}};
  std::vector<std::vector<float>> results = {{-0.5, -0.2, 0.0, 0.3},
                                             {-0.8, -0.2, -0.1, 0.9},
                                             {-0.61, -0.2, 0.88, -0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, {1, 2, 2, 1}, -0.7, 0.7},
                          {tensor_type, {1, 2, 2, 1}, -0.6, 0.6},
                          {tensor_type, {}, -0.9, 0.9},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationGenericInt16) {
  QuantizedTestsNoActivation<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsActivationRELU_N1_TO_1() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::vector<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                             {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, {1, 2, 2, 1}, -0.9, 0.9},
                          {tensor_type, {1, 2, 2, 1}, -0.9, 0.9},
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
TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1UInt8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1Int8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1Int16) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-2.1, 2.1);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, test_shapes[i], -2.0, 2.0},
                          {tensor_type, test_shapes[i], -1.1, 1.1},
                          {tensor_type, {}, -2.1, 2.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.1, -0.1, 0.4, 0.3, 0.0, 1.9}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesInt8) {
  QuantizedVariousInputShapes<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesInt16) {
  QuantizedVariousInputShapes<TensorType_INT16, int16_t>();
}

TEST(QuantizedSubOpModel, QuantizedLargeInputShapesInt16) {
  // This test is to cover large shape, which is more than 16 to test
  // AVX2 kernel with batch 16.
  const float kQuantizedTolerance = GetTolerance<int16_t>(-2.1, 2.1);
  const std::vector<int> test_shape = {18};
  QuantizedSubOpModel m({TensorType_INT16, test_shape, -2.0, 2.0},
                        {TensorType_INT16, test_shape, -1.1, 1.1},
                        {TensorType_INT16, {}, -2.1, 2.1},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(
      m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0, -2.0, 0.2, 0.7, 0.8, 1.1, 2.0,
                   -2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
  m.QuantizeAndPopulate<int16_t>(
      m.input2(), {0.1, 0.3, 0.3, 0.5, 1.1, 0.1, 0.1, 0.3, 0.3, 0.5, 1.1, 0.1,
                   0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {-2.1, -0.1, 0.4, 0.3, 0.0, 1.9, -2.1, -0.1, 0.4, 0.3, 0.0,
                   1.9, -2.1, -0.1, 0.4, 0.3, 0.0, 1.9},
                  kQuantizedTolerance)));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedWithBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-2.7, 2.7);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m(
        {tensor_type, test_shapes[i], -2.0, 2.0}, {tensor_type, {}, -0.7, 0.7},
        {tensor_type, {}, -2.7, 2.7}, ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.7});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.7, -0.5, 0.0, 0.1, 0.4, 1.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastUInt8) {
  QuantizedWithBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastInt8) {
  QuantizedWithBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastInt16) {
  QuantizedWithBroadcast<TensorType_INT16, int16_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.1, 1.1);
  std::vector<std::vector<float>> inputs1 = {
      {0.7, 0.6, 0.6, 0.5}, {-0.2, 0.6, 0.9, -0.1}, {-0.2, 0.6, -0.3, 0.8}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, 0.8, 0.5}};
  std::vector<std::vector<float>> results = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, -1.1, 0.3}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {1, 2, 2, 1}, -0.8, 0.8},
                          {TensorType_INT16, {}, -1.1, 1.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<int16_t>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsReluActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::vector<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                             {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<int16_t>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationBroadcastInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.1, 1.1);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, test_shapes[i], -0.9, 0.9},
                          {TensorType_INT16, {}, -0.2, 0.2},
                          {TensorType_INT16, {}, -1.1, 1.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(),
                                   {-0.9, -0.7, -0.3, 0.0, 0.3, 0.5});
    m.QuantizeAndPopulate<int16_t>(m.input2(), {0.2});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {-1.1, -0.9, -0.5, -0.2, 0.1, 0.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsReluActivationBroadcastInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, test_shapes[i], -0.9, 0.9},
                          {TensorType_INT16, {}, -0.2, 0.2},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<int16_t>(m.input1(),
                                   {-0.9, -0.7, -0.3, 0.0, 0.3, 0.5});
    m.QuantizeAndPopulate<int16_t>(m.input2(), {0.2});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {-1.0, -0.9, -0.5, -0.2, 0.1, 0.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

constexpr int kDim1 = 2;
constexpr int kDim2 = 3;
constexpr int kDim3 = 4;
constexpr int kDim4 = 5;
constexpr int kDim5 = 6;
constexpr int kDim6 = 7;

constexpr int kMaxBroadcastDim = 6;

void TestFloatBroadcast(std::vector<int> input1_shape,
                        std::vector<int> input2_shape) {
  std::array<int, kMaxBroadcastDim> input1_dims;
  std::array<int, kMaxBroadcastDim> input2_dims;
  std::array<int, kMaxBroadcastDim> output_dims;
  std::array<int, kMaxBroadcastDim> input1_strides;
  std::array<int, kMaxBroadcastDim> input2_strides;
  std::array<int, kMaxBroadcastDim> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < kMaxBroadcastDim; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = kMaxBroadcastDim; i != 0; i--) {
    input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
    input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
    output_strides[i - 1] = output_stride;
    input1_stride *= input1_dims[i - 1];
    input2_stride *= input2_dims[i - 1];
    output_stride *= output_dims[i - 1];
  }
  const int num_input1_elements = std::accumulate(
      input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
  const int num_input2_elements = std::accumulate(
      input2_dims.begin(), input2_dims.end(), 1, std::multiplies<int>());
  const int num_output_elements = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  std::vector<float> input1(num_input1_elements);
  std::vector<float> input2(num_input2_elements);
  std::vector<float> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });

  // Compute reference results.
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              output_ref[i * output_strides[0] + j * output_strides[1] +
                         k * output_strides[2] + l * output_strides[3] +
                         m * output_strides[4] + n * output_strides[5]] =
                  input1[i * input1_strides[0] + j * input1_strides[1] +
                         k * input1_strides[2] + l * input1_strides[3] +
                         m * input1_strides[4] + n * input1_strides[5]] -
                  input2[i * input2_strides[0] + j * input2_strides[1] +
                         k * input2_strides[2] + l * input2_strides[3] +
                         m * input2_strides[4] + n * input2_strides[5]];
            }
          }
        }
      }
    }
  }

  FloatSubOpModel m({TensorType_FLOAT32, input1_shape},
                    {TensorType_FLOAT32, input2_shape},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::ContainerEq(output_ref));
}

template <typename IntegerType>
void TestIntegerBroadcast(std::vector<int> input1_shape,
                          std::vector<int> input2_shape) {
  std::array<int, kMaxBroadcastDim> input1_dims;
  std::array<int, kMaxBroadcastDim> input2_dims;
  std::array<int, kMaxBroadcastDim> output_dims;
  std::array<int, kMaxBroadcastDim> input1_strides;
  std::array<int, kMaxBroadcastDim> input2_strides;
  std::array<int, kMaxBroadcastDim> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < kMaxBroadcastDim; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = kMaxBroadcastDim; i != 0; i--) {
    input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
    input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
    output_strides[i - 1] = output_stride;
    input1_stride *= input1_dims[i - 1];
    input2_stride *= input2_dims[i - 1];
    output_stride *= output_dims[i - 1];
  }
  const int num_input1_elements = std::accumulate(
      input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
  const int num_input2_elements = std::accumulate(
      input2_dims.begin(), input2_dims.end(), 1, std::multiplies<int>());
  const int num_output_elements = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  std::vector<IntegerType> input1(num_input1_elements);
  std::vector<IntegerType> input2(num_input2_elements);
  std::vector<IntegerType> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<IntegerType> dist(0, 256);

  std::generate(input1.begin(), input1.end(), [&]() { return dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return dist(rng); });

  // Compute reference results.
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              output_ref[i * output_strides[0] + j * output_strides[1] +
                         k * output_strides[2] + l * output_strides[3] +
                         m * output_strides[4] + n * output_strides[5]] =
                  input1[i * input1_strides[0] + j * input1_strides[1] +
                         k * input1_strides[2] + l * input1_strides[3] +
                         m * input1_strides[4] + n * input1_strides[5]] -
                  input2[i * input2_strides[0] + j * input2_strides[1] +
                         k * input2_strides[2] + l * input2_strides[3] +
                         m * input2_strides[4] + n * input2_strides[5]];
            }
          }
        }
      }
    }
  }

  IntegerSubOpModel m({GetTensorType<IntegerType>(), input1_shape},
                      {GetTensorType<IntegerType>(), input2_shape},
                      {GetTensorType<IntegerType>(), {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<IntegerType>(m.input1(), input1);
  m.PopulateTensor<IntegerType>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<IntegerType>(), testing::ContainerEq(output_ref));
}

// To improve automatic test sharding (via shard_count in the BUILD file),
// we need to ensure that each individual test case runs in a reasonable time,
// otherwise we end up being limited by the performance of the longest shard.
// Since TestFloat32MultiDimBroadcast has 2^12 iterations, it takes a
// long time (over 30 seconds) to execute all iterations -- too long for a
// single shard.  So we split it into a few "subshards" and have a separate
// TYPED_TEST macro invocation for each subshard.

void TestFloat32MultiDimBroadcast(int selected_subshard, int subshard_count) {
  int iteration = 0;
  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << kMaxBroadcastDim);
       bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << kMaxBroadcastDim);
         bm2++) {
      if (iteration++ % subshard_count != selected_subshard) {
        continue;  // This iteration of the loop is not part of this subshard.
      }
      const bool input1_broadcast_dim1 = bm1 & (static_cast<uint32_t>(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (static_cast<uint32_t>(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (static_cast<uint32_t>(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (static_cast<uint32_t>(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (static_cast<uint32_t>(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (static_cast<uint32_t>(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (static_cast<uint32_t>(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (static_cast<uint32_t>(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (static_cast<uint32_t>(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (static_cast<uint32_t>(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (static_cast<uint32_t>(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (static_cast<uint32_t>(1) << 5);
      const int input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const int input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const int input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const int input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const int input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const int input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const int input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const int input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const int input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const int input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const int input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const int input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      std::vector<int> input1_full_shape{input1_dim1, input1_dim2, input1_dim3,
                                         input1_dim4, input1_dim5, input1_dim6};
      std::vector<int> input2_full_shape{input2_dim1, input2_dim2, input2_dim3,
                                         input2_dim4, input2_dim5, input2_dim6};
      for (int input1_dims = 1; input1_dims <= kMaxBroadcastDim;
           ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= kMaxBroadcastDim;
             ++input2_dims) {
          std::vector<int> input1_shape(input1_dims), input2_shape(input2_dims);
          std::copy(input1_full_shape.end() - input1_dims,
                    input1_full_shape.end(), input1_shape.data());
          std::copy(input2_full_shape.end() - input2_dims,
                    input2_full_shape.end(), input2_shape.data());
          TestFloatBroadcast(input1_shape, input2_shape);
        }
      }
    }
  }
}

// Should match the number of TEST or TYPED_TEST invoations for each of
// Float32MultiDimBroadcastSubshard*,
// IntegerMultiDimBroadcastSubshard*,
// Int8QuantizedMultiDimBroadcastSubshard*, and
// Uint8QuantizedMultiDimBroadcastSubshard* below.
constexpr int kMultiDimBroadcastSubshardCount = 10;

TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard0) {
  TestFloat32MultiDimBroadcast(0, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard1) {
  TestFloat32MultiDimBroadcast(1, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard2) {
  TestFloat32MultiDimBroadcast(2, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard3) {
  TestFloat32MultiDimBroadcast(3, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard4) {
  TestFloat32MultiDimBroadcast(4, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard5) {
  TestFloat32MultiDimBroadcast(5, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard6) {
  TestFloat32MultiDimBroadcast(6, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard7) {
  TestFloat32MultiDimBroadcast(7, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard8) {
  TestFloat32MultiDimBroadcast(8, kMultiDimBroadcastSubshardCount);
}
TEST(FloatSubOpModel, Float32MultiDimBroadcastSubshard9) {
  TestFloat32MultiDimBroadcast(9, kMultiDimBroadcastSubshardCount);
}

template <typename T>
class IntegerSubOpTest : public ::testing::Test {};

using Int32Or64Types = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(IntegerSubOpTest, Int32Or64Types);

// To improve automatic test sharding (via shard_count in the BUILD file),
// we need to ensure that each individual test case runs in a reasonable time,
// otherwise we end up being limited by the performance of the longest shard.
// Since TestIntegerMultiDimBroadcast has 2^12 iterations, it takes a
// long time (over 30 seconds) to execute all iterations -- too long for a
// single shard.  So we split it into a few "subshards" and have a separate
// TYPED_TEST macro invocation for each subshard.

template <class TypeParam>
void TestIntegerMultiDimBroadcast(int selected_subshard, int subshard_count) {
  ASSERT_LT(selected_subshard, subshard_count);
  int iteration = 0;
  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << kMaxBroadcastDim);
       bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << kMaxBroadcastDim);
         bm2++) {
      if (iteration++ % subshard_count != selected_subshard) {
        continue;  // This iteration of the loop is not part of this subshard.
      }
      const bool input1_broadcast_dim1 = bm1 & (static_cast<uint32_t>(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (static_cast<uint32_t>(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (static_cast<uint32_t>(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (static_cast<uint32_t>(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (static_cast<uint32_t>(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (static_cast<uint32_t>(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (static_cast<uint32_t>(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (static_cast<uint32_t>(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (static_cast<uint32_t>(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (static_cast<uint32_t>(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (static_cast<uint32_t>(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (static_cast<uint32_t>(1) << 5);
      const int input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const int input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const int input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const int input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const int input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const int input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const int input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const int input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const int input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const int input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const int input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const int input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      std::vector<int> input1_full_shape{input1_dim1, input1_dim2, input1_dim3,
                                         input1_dim4, input1_dim5, input1_dim6};
      std::vector<int> input2_full_shape{input2_dim1, input2_dim2, input2_dim3,
                                         input2_dim4, input2_dim5, input2_dim6};
      for (int input1_dims = 1; input1_dims <= kMaxBroadcastDim;
           ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= kMaxBroadcastDim;
             ++input2_dims) {
          std::vector<int> input1_shape(input1_dims), input2_shape(input2_dims);
          std::copy(input1_full_shape.end() - input1_dims,
                    input1_full_shape.end(), input1_shape.data());
          std::copy(input2_full_shape.end() - input2_dims,
                    input2_full_shape.end(), input2_shape.data());
          TestIntegerBroadcast<TypeParam>(input1_shape, input2_shape);
        }
      }
    }
  }
}

TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard0) {
  TestIntegerMultiDimBroadcast<TypeParam>(0, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard1) {
  TestIntegerMultiDimBroadcast<TypeParam>(1, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard2) {
  TestIntegerMultiDimBroadcast<TypeParam>(2, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard3) {
  TestIntegerMultiDimBroadcast<TypeParam>(3, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard4) {
  TestIntegerMultiDimBroadcast<TypeParam>(4, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard5) {
  TestIntegerMultiDimBroadcast<TypeParam>(5, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard6) {
  TestIntegerMultiDimBroadcast<TypeParam>(6, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard7) {
  TestIntegerMultiDimBroadcast<TypeParam>(7, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard8) {
  TestIntegerMultiDimBroadcast<TypeParam>(8, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerSubOpTest, IntegerMultiDimBroadcastSubshard9) {
  TestIntegerMultiDimBroadcast<TypeParam>(9, kMultiDimBroadcastSubshardCount);
}

template <typename QuantizedType>
void TestQuantizedBroadcast(std::vector<int> input1_shape,
                            std::vector<int> input2_shape) {
  std::array<int, kMaxBroadcastDim> input1_dims;
  std::array<int, kMaxBroadcastDim> input2_dims;
  std::array<int, kMaxBroadcastDim> output_dims;
  std::array<int, kMaxBroadcastDim> input1_strides;
  std::array<int, kMaxBroadcastDim> input2_strides;
  std::array<int, kMaxBroadcastDim> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < kMaxBroadcastDim; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = kMaxBroadcastDim; i != 0; i--) {
    input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
    input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
    output_strides[i - 1] = output_stride;
    input1_stride *= input1_dims[i - 1];
    input2_stride *= input2_dims[i - 1];
    output_stride *= output_dims[i - 1];
  }
  const int num_input1_elements = std::accumulate(
      input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
  const int num_input2_elements = std::accumulate(
      input2_dims.begin(), input2_dims.end(), 1, std::multiplies<int>());
  const int num_output_elements = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  std::vector<float> input1(num_input1_elements);
  std::vector<float> input2(num_input2_elements);
  std::vector<float> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::generate(input1.begin(), input1.end(), [&]() { return dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return dist(rng); });

  QuantizedSubOpModel m(
      {GetTensorType<QuantizedType>(), input1_shape, -0.5f, 0.5f},
      {GetTensorType<QuantizedType>(), input2_shape, -0.5f, 0.5f},
      {GetTensorType<QuantizedType>(), {}, -0.5f, 0.5f},
      ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<QuantizedType>(m.input1(), input1);
  m.QuantizeAndPopulate<QuantizedType>(m.input2(), input2);
  // Compute reference results.
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              float x = input1[i * input1_strides[0] + j * input1_strides[1] +
                               k * input1_strides[2] + l * input1_strides[3] +
                               m * input1_strides[4] + n * input1_strides[5]];
              float y = input2[i * input2_strides[0] + j * input2_strides[1] +
                               k * input2_strides[2] + l * input2_strides[3] +
                               m * input2_strides[4] + n * input2_strides[5]];
              output_ref[i * output_strides[0] + j * output_strides[1] +
                         k * output_strides[2] + l * output_strides[3] +
                         m * output_strides[4] + n * output_strides[5]] = x - y;
            }
          }
        }
      }
    }
  }

  for (float& output_value : output_ref) {
    output_value = std::max<float>(output_value, -1.0f);
    output_value = std::min<float>(output_value, 1.0f);
  }

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<float> output = m.GetDequantizedOutput<QuantizedType>();
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              const size_t index =
                  i * output_strides[0] + j * output_strides[1] +
                  k * output_strides[2] + l * output_strides[3] +
                  m * output_strides[4] + n * output_strides[5];
              EXPECT_NEAR(output[index], output_ref[index], 0.6f)
                  << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k
                  << ", " << l << ", " << m << ", " << n << ")";
            }
          }
        }
      }
    }
  }
}

template <typename T>
class QuantizedSubOpTest : public ::testing::Test {};

using Int8OrUInt8OrInt16Types = ::testing::Types<int8_t, uint8_t, int16_t>;
TYPED_TEST_SUITE(QuantizedSubOpTest, Int8OrUInt8OrInt16Types);

// To improve automatic test sharding (via shard_count in the BUILD file),
// we need to ensure that each individual test case runs in a reasonable time,
// otherwise we end up being limited by the performance of the longest shard.
// Since TestQuantizedMultiDimBroadcast has 2^12 iterations, it takes a
// long time (over 30 seconds) to execute all iterations -- too long for a
// single shard.  So we split it into a few "subshards" and have a separate
// TEST macro invocation for each subshard.

template <class T>
void TestQuantizedMultiDimBroadcast(int selected_subshard, int subshard_count) {
  ASSERT_LT(selected_subshard, subshard_count);
  int iteration = 0;
  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << kMaxBroadcastDim);
       bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << kMaxBroadcastDim);
         bm2++) {
      if (iteration++ % subshard_count != selected_subshard) {
        continue;  // This iteration of the loop is not part of this subshard.
      }
      const bool input1_broadcast_dim1 = bm1 & (static_cast<uint32_t>(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (static_cast<uint32_t>(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (static_cast<uint32_t>(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (static_cast<uint32_t>(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (static_cast<uint32_t>(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (static_cast<uint32_t>(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (static_cast<uint32_t>(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (static_cast<uint32_t>(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (static_cast<uint32_t>(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (static_cast<uint32_t>(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (static_cast<uint32_t>(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (static_cast<uint32_t>(1) << 5);
      const int input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const int input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const int input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const int input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const int input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const int input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const int input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const int input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const int input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const int input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const int input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const int input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      std::vector<int> input1_full_shape{input1_dim1, input1_dim2, input1_dim3,
                                         input1_dim4, input1_dim5, input1_dim6};
      std::vector<int> input2_full_shape{input2_dim1, input2_dim2, input2_dim3,
                                         input2_dim4, input2_dim5, input2_dim6};
      for (int input1_dims = 1; input1_dims <= kMaxBroadcastDim;
           ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= kMaxBroadcastDim;
             ++input2_dims) {
          std::vector<int> input1_shape(input1_dims), input2_shape(input2_dims);
          std::copy(input1_full_shape.end() - input1_dims,
                    input1_full_shape.end(), input1_shape.data());
          std::copy(input2_full_shape.end() - input2_dims,
                    input2_full_shape.end(), input2_shape.data());
          TestQuantizedBroadcast<T>(input1_shape, input2_shape);
        }
      }
    }
  }
}

TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard0) {
  TestQuantizedMultiDimBroadcast<int8_t>(0, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard1) {
  TestQuantizedMultiDimBroadcast<int8_t>(1, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard2) {
  TestQuantizedMultiDimBroadcast<int8_t>(2, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard3) {
  TestQuantizedMultiDimBroadcast<int8_t>(3, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard4) {
  TestQuantizedMultiDimBroadcast<int8_t>(4, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard5) {
  TestQuantizedMultiDimBroadcast<int8_t>(5, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard6) {
  TestQuantizedMultiDimBroadcast<int8_t>(6, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard7) {
  TestQuantizedMultiDimBroadcast<int8_t>(7, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard8) {
  TestQuantizedMultiDimBroadcast<int8_t>(8, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Int8QuantizedMultiDimBroadcastSubshard9) {
  TestQuantizedMultiDimBroadcast<int8_t>(9, kMultiDimBroadcastSubshardCount);
}

TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard0) {
  TestQuantizedMultiDimBroadcast<uint8_t>(0, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard1) {
  TestQuantizedMultiDimBroadcast<uint8_t>(1, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard2) {
  TestQuantizedMultiDimBroadcast<uint8_t>(2, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard3) {
  TestQuantizedMultiDimBroadcast<uint8_t>(3, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard4) {
  TestQuantizedMultiDimBroadcast<uint8_t>(4, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard5) {
  TestQuantizedMultiDimBroadcast<uint8_t>(5, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard6) {
  TestQuantizedMultiDimBroadcast<uint8_t>(6, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard7) {
  TestQuantizedMultiDimBroadcast<uint8_t>(7, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard8) {
  TestQuantizedMultiDimBroadcast<uint8_t>(8, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedSubOpModel, Uint8QuantizedMultiDimBroadcastSubshard9) {
  TestQuantizedMultiDimBroadcast<uint8_t>(9, kMultiDimBroadcastSubshardCount);
}

}  // namespace
}  // namespace tflite
