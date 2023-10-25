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

#include <algorithm>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ElementWiseOpBaseModel : public SingleOpModel {
 public:
  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

class ElementWiseOpIntModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpIntModel(BuiltinOperator op,
                        std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_INT32);
    output_ = AddOutput(TensorType_INT32);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

class ElementWiseOpFloatModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpFloatModel(BuiltinOperator op,
                          std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

class ElementWiseOpQuantizedModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpQuantizedModel(BuiltinOperator op, TensorData input_tensor_data,
                              TensorData output_tensor_data) {
    input_ = AddInput(SymmetricInt16Scaling(input_tensor_data));
    output_ = AddOutput(SymmetricInt16Scaling(output_tensor_data));
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_tensor_data.shape});
  }

  template <typename T>
  void AsymmetricQuantizeAndPopulate(int index,
                                     const std::vector<float>& data) {
    std::vector<int8_t> q(data.size());
    float scaling_factor;
    int zero_point;
    tensor_utils::AsymmetricQuantizeFloats(data.data(), data.size(), q.data(),
                                           &scaling_factor, &zero_point);
    PopulateTensor<T>(index, /*offset=*/0, reinterpret_cast<T*>(q.data()),
                      reinterpret_cast<T*>(q.data() + q.size()));
  }

  template <typename T>
  std::vector<float> ExtractDequantVector(int index) {
    auto vec = ExtractVector<T>(index);
    TfLiteTensor* t = interpreter_->tensor(index);
    auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    float scaling_factor = affine_quantization->scale->data[0];
    int zero_point = affine_quantization->zero_point->data[0];
    std::vector<float> output;
    for (const auto& v : vec) {
      output.push_back((static_cast<T>(v) - zero_point) * scaling_factor);
    }
    return output;
  }

 private:
  TensorData& SymmetricInt16Scaling(TensorData& tensor) {
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

class ElementWiseOpBoolModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpBoolModel(BuiltinOperator op,
                         std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_BOOL);
    output_ = AddOutput(TensorType_BOOL);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

// A LUT of 256 values is used in the int8 case. For the int16 case a 513 LUT is
// used but as the last value is only used for interpolation we only have 512
// quantized steps.
template <typename T>
inline float GetLUTTolerance(float input_min, float input_max, float output_min,
                             float output_max) {
  static_assert(
      std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value,
      "T must be an int8_t or int16_t.");

  const float range_sum = (input_max - input_min) + (output_max - output_min);
  if (std::is_same<T, int8_t>::value) {
    return range_sum / 256.0f;
  } else {
    return range_sum / 512.0f;
  }
}

template <typename T>
float GetQuantizationStep(float min, float max) {
  const float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                              std::numeric_limits<T>::min());
  return kQuantizedStep;
}

TEST(ElementWise, Sin) {
  ElementWiseOpFloatModel m(BuiltinOperator_SIN, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 3.1415926, -3.1415926, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0.84147})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Cos) {
  ElementWiseOpFloatModel m(BuiltinOperator_COS, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 3.1415926, -3.1415926, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, -1, -1, 0.54030})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Log) {
  ElementWiseOpFloatModel m(BuiltinOperator_LOG, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 3.1415926, 1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1.14473, 0, 0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, LogInt8) {
  const float input_min = 0.0f;
  const float input_max = 13.2f;
  const float output_min = -2.3026f;
  const float output_max = 2.5802f;

  const float kQuantizedTolerance =
      GetLUTTolerance<int8_t>(input_min, input_max, output_min, output_max);

  ElementWiseOpQuantizedModel m(
      BuiltinOperator_LOG,
      {TensorType_INT8, {1, 2, 2, 2}, input_min, input_max},
      {TensorType_INT8, {}, output_min, output_max});
  m.QuantizeAndPopulate<int8_t>(
      m.input(), {0.1f, 0.5f, 1.0f, 1.15f, 2.3f, 5.01f, 11.0f, 13.2f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractDequantVector<int8_t>(m.output()),
      ElementsAreArray(ArrayFloatNear({-2.3026f, -0.6931f, 0.0f, 0.1398f,
                                       0.8329f, 1.6114, 2.3979f, 2.5802f},
                                      kQuantizedTolerance)));
}

TEST(ElementWise, LogInt16) {
  const float input_min = -13.2f;
  const float input_max = 13.2f;
  const float output_min = -2.5802f;
  const float output_max = 2.5802f;

  const float kQuantizedTolerance =
      GetLUTTolerance<int16_t>(input_min, input_max, output_min, output_max);

  ElementWiseOpQuantizedModel m(
      BuiltinOperator_LOG,
      {TensorType_INT16, {1, 2, 2, 2}, input_min, input_max},
      {TensorType_INT16, {}, output_min, output_max});
  m.QuantizeAndPopulate<int16_t>(
      m.input(), {0.1f, 0.5f, 1.0f, 1.15f, 2.3f, 5.01f, 11.0f, 13.2f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractDequantVector<int16_t>(m.output()),
      ElementsAreArray(ArrayFloatNear({-2.3026f, -0.6931f, 0.0f, 0.1398f,
                                       0.8329f, 1.6114, 2.3979f, 2.5802f},
                                      kQuantizedTolerance)));
}

TEST(ElementWise, Abs) {
  ElementWiseOpFloatModel m(BuiltinOperator_ABS, {1, 2, 4, 1});
  m.PopulateTensor<float>(m.input(), {
                                         0.f, -6.2f, 2.f, 4.f,  //
                                         3.f, -2.f, 10.f, 1.f,  //
                                     });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray({
                                                      0.f, 6.2f, 2.f, 4.f,  //
                                                      3.f, 2.f, 10.f, 1.f,  //
                                                  }));
}

TEST(ElementWise, AbsInt32) {
  ElementWiseOpIntModel m(BuiltinOperator_ABS, {1, 2, 4, 1});
  m.PopulateTensor<int32_t>(m.input(), {
                                           0, -6, 2, 4,   //
                                           3, -2, 10, 1,  //
                                       });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()), ElementsAreArray({
                                                        0, 6, 2, 4,   //
                                                        3, 2, 10, 1,  //
                                                    }));
}

TEST(ElementWise, AbsInt8) {
  std::vector<float> data = {15., 46., 78., -142., -1., -17., -49., 113.};
  std::vector<float> abs_data(data.size());
  for (int i = 0; i < abs_data.size(); i++) {
    abs_data[i] = std::abs(data[i]);
  }
  const auto minmax = std::minmax_element(data.begin(), data.end());
  const float abs_max = std::max(std::abs(*minmax.first), *minmax.second);
  const float kInputScale = (*minmax.second - *minmax.first) / 255.0;
  const float kOutputScale = abs_max / 255.0;
  const int input_zero_point = 127 - *minmax.second;
  const int output_zero_point = -128;
  ElementWiseOpQuantizedModel m(
      BuiltinOperator_ABS,
      {TensorType_INT8,
       {1, 8},
       *minmax.first,
       *minmax.second,
       kInputScale,
       input_zero_point,
       true,
       {kInputScale},
       {input_zero_point}},
      {TensorType_INT8, {1, 8}, 0, abs_max, kOutputScale, output_zero_point});
  m.AsymmetricQuantizeAndPopulate<int8_t>(m.input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractDequantVector<int8_t>(m.output()),
              ElementsAreArray(ArrayFloatNear(abs_data, kInputScale)));
}

TEST(ElementWise, AbsSameScaleInt8) {
  std::vector<float> data = {15., 46., 78., -142., -1., -17., -49., 113.};
  std::vector<float> abs_data(data.size());
  for (int i = 0; i < abs_data.size(); i++) {
    abs_data[i] = std::abs(data[i]);
  }
  const auto minmax = std::minmax_element(data.begin(), data.end());
  const float abs_max = std::max(std::abs(*minmax.first), *minmax.second);
  const float kInputScale = (*minmax.second - *minmax.first) / 255.0;
  const int input_zero_point = 127 - *minmax.second;
  ElementWiseOpQuantizedModel m(
      BuiltinOperator_ABS,
      {TensorType_INT8,
       {1, 8},
       *minmax.first,
       *minmax.second,
       kInputScale,
       input_zero_point,
       true,
       {kInputScale},
       {input_zero_point}},
      {TensorType_INT8, {1, 8}, 0, abs_max, kInputScale, input_zero_point});
  m.AsymmetricQuantizeAndPopulate<int8_t>(m.input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractDequantVector<int8_t>(m.output()),
              ElementsAreArray(ArrayFloatNear(abs_data, kInputScale)));
}

TEST(ElementWise, AbsInt16) {
  const float kQuantizedTolerance = GetQuantizationStep<int16_t>(-150, 150);
  std::vector<float> data = {15., 46., 78., -142., -1., -17., -49., 113.};
  std::vector<float> abs_data(data.size());
  for (int i = 0; i < abs_data.size(); i++) {
    abs_data[i] = std::abs(data[i]);
  }
  ElementWiseOpQuantizedModel m(BuiltinOperator_ABS,
                                {TensorType_INT16, {1, 8}, -142, 142},
                                {TensorType_INT16, {1, 8}, -150, 150});
  m.QuantizeAndPopulate<int16_t>(m.input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractDequantVector<int16_t>(m.output()),
              ElementsAreArray(ArrayFloatNear(abs_data, kQuantizedTolerance)));
}

TEST(ElementWise, Sqrt) {
  ElementWiseOpFloatModel m(BuiltinOperator_SQRT, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 1, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1, 1.41421, 2})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Rsqrt) {
  ElementWiseOpFloatModel m(BuiltinOperator_RSQRT, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 2, 4, 9});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, 0.7071, 0.5, 0.33333})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, RsqrtInt8) {
  std::vector<float> data = {15., 46., 78., 142., 1., 17., 49., 113.};
  std::vector<float> rsqrt_data(data.size());
  for (int i = 0; i < rsqrt_data.size(); i++) {
    rsqrt_data[i] = 1.f / std::sqrt(data[i]);
  }
  float kInputScale = 142.0 / 255.0;
  float kOutputScale = 1.0 / 255.0;
  int32_t zero_point = -128;
  ElementWiseOpQuantizedModel m(BuiltinOperator_RSQRT,
                                {TensorType_INT8,
                                 {1, 8},
                                 0,
                                 142.0,
                                 kInputScale,
                                 zero_point,
                                 true,
                                 {kInputScale},
                                 {zero_point}},
                                {TensorType_INT8,
                                 {1, 8},
                                 0,
                                 1.0,
                                 kOutputScale,
                                 zero_point,
                                 true,
                                 {kOutputScale},
                                 {zero_point}});
  m.QuantizeAndPopulate<int8_t>(m.input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractDequantVector<int8_t>(m.output()),
              ElementsAreArray(ArrayFloatNear(rsqrt_data, kInputScale)));
}

TEST(ElementWise, RsqrtCloseTo0Int8) {
  std::vector<float> data = {15., 46., 78., 142., 0.1, 1., 49., 113.};
  std::vector<float> rsqrt_data(data.size());
  for (int i = 0; i < rsqrt_data.size(); i++) {
    rsqrt_data[i] = 1.f / std::sqrt(data[i]);
  }
  float kInputScale = 142.0 / 255.0;
  float kOutputScale = 3.16 / 255.0;
  int32_t zero_point = -128;
  ElementWiseOpQuantizedModel m(BuiltinOperator_RSQRT,
                                {TensorType_INT8,
                                 {1, 8},
                                 0,
                                 142.0,
                                 kInputScale,
                                 zero_point,
                                 true,
                                 {kInputScale},
                                 {zero_point}},
                                {TensorType_INT8,
                                 {1, 8},
                                 0,
                                 3.16,
                                 kOutputScale,
                                 zero_point,
                                 true,
                                 {kOutputScale},
                                 {zero_point}});
  m.QuantizeAndPopulate<int8_t>(m.input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractDequantVector<int8_t>(m.output()),
              ElementsAreArray(ArrayFloatNear(rsqrt_data, kInputScale)));
}

TEST(ElementWise, RsqrtNanInt8) {
  std::vector<float> data = {15., 46., 78., 142., 1., 17., -49., 113.};
  std::vector<float> rsqrt_data(data.size());
  for (int i = 0; i < rsqrt_data.size(); i++) {
    rsqrt_data[i] = 1.f / std::sqrt(data[i]);
  }
  float kInputScale = 142.0 / 127.0;
  float kOutputScale = 1.0 / 255.0;
  int32_t input_zero_point = 0;
  int32_t output_zero_point = -128;
  ElementWiseOpQuantizedModel m(BuiltinOperator_RSQRT,
                                {TensorType_INT8,
                                 {1, 8},
                                 0,
                                 142.0,
                                 kInputScale,
                                 input_zero_point,
                                 true,
                                 {kInputScale},
                                 {input_zero_point}},
                                {TensorType_INT8,
                                 {1, 8},
                                 0,
                                 1.0,
                                 kOutputScale,
                                 output_zero_point,
                                 true,
                                 {kOutputScale},
                                 {output_zero_point}});
  m.QuantizeAndPopulate<int8_t>(m.input(), data);
  EXPECT_THAT(m.Invoke(), kTfLiteError);
}

TEST(ElementWise, RsqrtInt16) {
  const float input_min = -0.8f;
  const float input_max = 0.8f;

  const float output_min = -2.4f;
  const float output_max = 2.4f;

  const float kQuantizedTolerance =
      GetLUTTolerance<int16_t>(input_min, input_max, output_min, output_max);

  ElementWiseOpQuantizedModel m(BuiltinOperator_RSQRT,
                                {TensorType_INT16, {1, 1, 4, 1}, -10, 10},
                                {TensorType_INT16, {1, 1, 4, 1}, -10, 10});
  m.QuantizeAndPopulate<int16_t>(m.input(), {1, 0.1, 4, 9});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractDequantVector<int16_t>(m.output()),
      ElementsAreArray(ArrayFloatNear({1.00009, 3.19407, 0.500198, 0.333262},
                                      kQuantizedTolerance)));
}

TEST(ElementWise, RsqrtNanInt16) {
  const float input_min = -0.8f;
  const float input_max = 0.8f;

  const float output_min = -2.4f;
  const float output_max = 2.4f;

  const float kQuantizedTolerance =
      GetLUTTolerance<int16_t>(input_min, input_max, output_min, output_max);

  ElementWiseOpQuantizedModel m(BuiltinOperator_RSQRT,
                                {TensorType_INT16, {1, 1, 4, 1}, -10, 10},
                                {TensorType_INT16, {1, 1, 4, 1}, -10, 10});
  m.QuantizeAndPopulate<int16_t>(m.input(), {-1, 0, -4, -9});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractDequantVector<int16_t>(m.output()),
              ElementsAreArray(
                  ArrayFloatNear({10, 9.82452, 10, 10}, kQuantizedTolerance)));
}

TEST(ElementWise, Square) {
  ElementWiseOpFloatModel m(BuiltinOperator_SQUARE, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 2, 0.5, -3.0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, 4.0, 0.25, 9.0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, LogicalNot) {
  ElementWiseOpBoolModel m(BuiltinOperator_LOGICAL_NOT, {1, 1, 4, 1});
  m.PopulateTensor<bool>(m.input(), {true, false, true, false});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<bool>(m.output()),
              ElementsAreArray({false, true, false, true}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

}  // namespace
}  // namespace tflite
