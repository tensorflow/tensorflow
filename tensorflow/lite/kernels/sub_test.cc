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

  void Resize(const std::vector<int>& input1_shape,
              const std::vector<int>& input2_shape) {
    interpreter_->ResizeInputTensor(input1_, input1_shape);
    interpreter_->ResizeInputTensor(input2_, input2_shape);
    AllocateTensors();
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

template <typename T>
class SubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
};

template <typename T>
class FloatSubTest : public ::testing::Test {};

using FloatSubTestTypes = ::testing::Types<float, half, Eigen::bfloat16>;
TYPED_TEST_SUITE(FloatSubTest, FloatSubTestTypes);

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

TYPED_TEST(FloatSubTest, FirstInputZero) {
  using T = TypeParam;
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  SubOpModel<T> m({GetTensorType<T>(), {0}}, {GetTensorType<T>(), {}},
                  {GetTensorType<T>(), {}}, ActivationFunctionType_NONE);
  m.template PopulateTensor<T>(m.input2(), {0.1});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TYPED_TEST(FloatSubTest, SecondInputZero) {
  using T = TypeParam;
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  SubOpModel<T> m({GetTensorType<T>(), {}}, {GetTensorType<T>(), {0}},
                  {GetTensorType<T>(), {}}, ActivationFunctionType_NONE);
  m.template PopulateTensor<T>(m.input1(), {0.1});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TYPED_TEST(FloatSubTest, NoActivationInplaceInput0) {
  using T = TypeParam;
  SubOpModel<T> m({GetTensorType<T>(), {1, 2, 2, 1}},
                  {GetTensorType<T>(), {1, 2, 2, 1}}, {GetTensorType<T>(), {}},
                  ActivationFunctionType_NONE);
  const int kInplaceInputTensorIdx = 0;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.template PopulateTensor<T>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-2.1, 0.0, 1.4, -0.3},
                  static_cast<float>(NumericLimits<T>::epsilon()) * 10)));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TYPED_TEST(FloatSubTest, NoActivationInplaceInput1) {
  using T = TypeParam;
  SubOpModel<T> m({GetTensorType<T>(), {1, 2, 2, 1}},
                  {GetTensorType<T>(), {1, 2, 2, 1}}, {GetTensorType<T>(), {}},
                  ActivationFunctionType_NONE);
  const int kInplaceInputTensorIdx = 1;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.template PopulateTensor<T>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-2.1, 0.0, 1.4, -0.3},
                  static_cast<float>(NumericLimits<T>::epsilon()) * 10)));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TYPED_TEST(FloatSubTest, NoActivation) {
  using T = TypeParam;
  SubOpModel<T> m({GetTensorType<T>(), {1, 2, 2, 1}},
                  {GetTensorType<T>(), {1, 2, 2, 1}}, {GetTensorType<T>(), {}},
                  ActivationFunctionType_NONE);
  m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.template PopulateTensor<T>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-2.1, 0.0, 1.4, -0.3},
                  static_cast<float>(NumericLimits<T>::epsilon()) * 10)));
}

TYPED_TEST(FloatSubTest, ActivationRELU_N1_TO_1) {
  using T = TypeParam;
  SubOpModel<T> m({GetTensorType<T>(), {1, 2, 2, 1}},
                  {GetTensorType<T>(), {1, 2, 2, 1}}, {GetTensorType<T>(), {}},
                  ActivationFunctionType_RELU_N1_TO_1);
  m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.template PopulateTensor<T>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-1.0, 0.0, 1.0, -0.3},
                  static_cast<float>(NumericLimits<T>::epsilon()) * 10)));
}

TYPED_TEST(FloatSubTest, VariousInputShapes) {
  using T = TypeParam;
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    SubOpModel<T> m({GetTensorType<T>(), test_shapes[i]},
                    {GetTensorType<T>(), test_shapes[i]},
                    {GetTensorType<T>(), {}}, ActivationFunctionType_NONE);
    m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.template PopulateTensor<T>(m.input2(), {0.1, 0.2, 0.3, 0.8, -1.1, 0.1});
    TFLITE_INVOKE_AND_CHECK(T, &m);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.1, 0.0, 1.4, -0.3, 0.0, 1.9},
                    static_cast<float>(NumericLimits<T>::epsilon()) * 10)))
        << "With shape number " << i;
  }
}

TYPED_TEST(FloatSubTest, WithBroadcast) {
  using T = TypeParam;
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    SubOpModel<T> m({GetTensorType<T>(), test_shapes[i]},
                    {GetTensorType<T>(), {}},  // always a scalar
                    {GetTensorType<T>(), {}}, ActivationFunctionType_NONE);
    m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.template PopulateTensor<T>(m.input2(), {0.5});
    TFLITE_INVOKE_AND_CHECK(T, &m);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.5, -0.3, 1.2, 0.0, -1.6, 1.5},
                    static_cast<float>(NumericLimits<T>::epsilon()) * 10)))
        << "With shape number " << i;
  }
}

TYPED_TEST(FloatSubTest, WithBroadcast5D) {
  using T = TypeParam;
  const std::vector<std::vector<int>> test_shapes = {{1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    SubOpModel<T> m({GetTensorType<T>(), test_shapes[i]},
                    {GetTensorType<T>(), {}},  // always a scalar
                    {GetTensorType<T>(), {}}, ActivationFunctionType_NONE);
    m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.template PopulateTensor<T>(m.input2(), {0.5});
    TFLITE_INVOKE_AND_CHECK(T, &m);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.5, -0.3, 1.2, 0.0, -1.6, 1.5},
                    static_cast<float>(NumericLimits<T>::epsilon()) * 10)))
        << "With shape number " << i;
  }
}

TYPED_TEST(FloatSubTest, WithBroadcastRankSeven) {
  using T = TypeParam;
  const std::vector<int> input1_shape = {2, 1, 2, 1, 2, 1, 2};
  const std::vector<int> input2_shape = {1, 2, 1, 2, 1, 2, 1};
  const std::vector<int> output_shape = {2, 2, 2, 2, 2, 2, 2};
  SubOpModel<T> m({GetTensorType<T>(), input1_shape},
                  {GetTensorType<T>(), input2_shape}, {GetTensorType<T>(), {}},
                  ActivationFunctionType_NONE);

  std::vector<float> input1(16);
  std::vector<float> input2(8);
  std::iota(input1.begin(), input1.end(), 1.0f);
  std::iota(input2.begin(), input2.end(), 0.25f);
  m.template PopulateTensor<T>(m.input1(), ToVector<T>(input1));
  m.template PopulateTensor<T>(m.input2(), ToVector<T>(input2));
  TFLITE_INVOKE_AND_CHECK(T, &m);

  auto strides_for = [](const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  };
  const std::vector<int> input1_strides = strides_for(input1_shape);
  const std::vector<int> input2_strides = strides_for(input2_shape);
  const std::vector<int> output_strides = strides_for(output_shape);
  std::vector<float> expected(128);
  for (int output_index = 0; output_index < expected.size(); ++output_index) {
    int remaining_index = output_index;
    int input1_index = 0;
    int input2_index = 0;
    for (int dim = 0; dim < output_shape.size(); ++dim) {
      const int coordinate = remaining_index / output_strides[dim];
      remaining_index %= output_strides[dim];
      if (input1_shape[dim] != 1) {
        input1_index += coordinate * input1_strides[dim];
      }
      if (input2_shape[dim] != 1) {
        input2_index += coordinate * input2_strides[dim];
      }
    }
    expected[output_index] = input1[input1_index] - input2[input2_index];
  }

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          expected, static_cast<float>(NumericLimits<T>::epsilon()) * 10)));
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

template <typename T>
void TestFloatBroadcast(SubOpModel<T>& m, const std::vector<int>& input1_shape,
                        const std::vector<int>& input2_shape) {
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
  std::vector<T> input1(num_input1_elements);
  std::vector<T> input2(num_input2_elements);
  std::vector<T> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  std::generate(input1.begin(), input1.end(),
                [&]() { return static_cast<T>(f32dist(rng)); });
  std::generate(input2.begin(), input2.end(),
                [&]() { return static_cast<T>(f32dist(rng)); });

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
                  static_cast<T>(
                      static_cast<float>(
                          input1[i * input1_strides[0] + j * input1_strides[1] +
                                 k * input1_strides[2] + l * input1_strides[3] +
                                 m * input1_strides[4] +
                                 n * input1_strides[5]]) -
                      static_cast<float>(
                          input2[i * input2_strides[0] + j * input2_strides[1] +
                                 k * input2_strides[2] + l * input2_strides[3] +
                                 m * input2_strides[4] +
                                 n * input2_strides[5]]));
            }
          }
        }
      }
    }
  }

  m.Resize(input1_shape, input2_shape);
  m.template PopulateTensor<T>(m.input1(), input1);
  m.template PopulateTensor<T>(m.input2(), input2);
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          output_ref, static_cast<float>(NumericLimits<T>::epsilon()) * 10,
          /*fp16_max_abs_err=*/1e-3)));
}

template <typename IntegerType>
void TestIntegerBroadcast(IntegerSubOpModel& m,
                          const std::vector<int>& input1_shape,
                          const std::vector<int>& input2_shape) {
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

  m.Resize(input1_shape, input2_shape);
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
// single shard.  So we split it into a few \"subshards\" and have a separate
// TYPED_TEST macro invocation for each subshard.

template <typename T>
void RunFloatMultiDimBroadcastTest(int d1, int d2) {
  const int dims_constants[] = {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6};
  std::vector<int> initial_shape1(d1, 1);
  std::vector<int> initial_shape2(d2, 1);
  SubOpModel<T> m({GetTensorType<T>(), initial_shape1},
                  {GetTensorType<T>(), initial_shape2},
                  {GetTensorType<T>(), {}}, ActivationFunctionType_NONE);

  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << d1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << d2); bm2++) {
      std::vector<int> input1_shape(d1);
      std::vector<int> input2_shape(d2);
      for (int i = 0; i < d1; ++i) {
        bool broadcast = bm1 & (1 << i);
        input1_shape[i] = broadcast ? 1 : dims_constants[6 - d1 + i];
      }
      for (int i = 0; i < d2; ++i) {
        bool broadcast = bm2 & (1 << i);
        input2_shape[i] = broadcast ? 1 : dims_constants[6 - d2 + i];
      }
      TestFloatBroadcast<T>(m, input1_shape, input2_shape);
      if (testing::Test::IsSkipped()) {
        return;
      }
    }
  }
}

#define INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, d2) \
  TYPED_TEST(FloatSubTest, MultiDimBroadcast_##d1##_##d2) {    \
    RunFloatMultiDimBroadcastTest<TypeParam>(d1, d2);          \
  }

#define INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(d1) \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, 1)       \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, 2)       \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, 3)       \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, 4)       \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, 5)       \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST(d1, 6)

#define INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TESTS() \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(1)    \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(2)    \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(3)    \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(4)    \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(5)    \
  INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TEST_D2(6)

INSTANTIATE_FLOAT_SUB_MULTI_DIM_BROADCAST_TESTS()

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

template <typename TypeParam>
void RunIntegerMultiDimBroadcastTest(int d1, int d2) {
  const int dims_constants[] = {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6};
  std::vector<int> initial_shape1(d1, 1);
  std::vector<int> initial_shape2(d2, 1);
  IntegerSubOpModel m({GetTensorType<TypeParam>(), initial_shape1},
                      {GetTensorType<TypeParam>(), initial_shape2},
                      {GetTensorType<TypeParam>(), {}},
                      ActivationFunctionType_NONE);

  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << d1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << d2); bm2++) {
      std::vector<int> input1_shape(d1);
      std::vector<int> input2_shape(d2);
      for (int i = 0; i < d1; ++i) {
        bool broadcast = bm1 & (1 << i);
        input1_shape[i] = broadcast ? 1 : dims_constants[6 - d1 + i];
      }
      for (int i = 0; i < d2; ++i) {
        bool broadcast = bm2 & (1 << i);
        input2_shape[i] = broadcast ? 1 : dims_constants[6 - d2 + i];
      }
      TestIntegerBroadcast<TypeParam>(m, input1_shape, input2_shape);
      if (testing::Test::IsSkipped()) {
        return;
      }
    }
  }
}

#define INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, d2) \
  TYPED_TEST(IntegerSubOpTest, MultiDimBroadcast_##d1##_##d2) {  \
    RunIntegerMultiDimBroadcastTest<TypeParam>(d1, d2);          \
  }

#define INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(d1) \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, 1)       \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, 2)       \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, 3)       \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, 4)       \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, 5)       \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST(d1, 6)

#define INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TESTS() \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(1)    \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(2)    \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(3)    \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(4)    \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(5)    \
  INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TEST_D2(6)

INSTANTIATE_INTEGER_SUB_MULTI_DIM_BROADCAST_TESTS()

template <typename QuantizedType>
void TestQuantizedBroadcast(QuantizedSubOpModel& m,
                            const std::vector<int>& input1_shape,
                            const std::vector<int>& input2_shape) {
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

  m.Resize(input1_shape, input2_shape);
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

template <typename T>
void RunQuantizedMultiDimBroadcastTest(int d1, int d2) {
  const int dims_constants[] = {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6};
  std::vector<int> initial_shape1(d1, 1);
  std::vector<int> initial_shape2(d2, 1);
  QuantizedSubOpModel m({GetTensorType<T>(), initial_shape1, -0.5f, 0.5f},
                        {GetTensorType<T>(), initial_shape2, -0.5f, 0.5f},
                        {GetTensorType<T>(), {}, -0.5f, 0.5f},
                        ActivationFunctionType_NONE);

  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << d1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << d2); bm2++) {
      std::vector<int> input1_shape(d1);
      std::vector<int> input2_shape(d2);
      for (int i = 0; i < d1; ++i) {
        bool broadcast = bm1 & (1 << i);
        input1_shape[i] = broadcast ? 1 : dims_constants[6 - d1 + i];
      }
      for (int i = 0; i < d2; ++i) {
        bool broadcast = bm2 & (1 << i);
        input2_shape[i] = broadcast ? 1 : dims_constants[6 - d2 + i];
      }
      TestQuantizedBroadcast<T>(m, input1_shape, input2_shape);
      if (testing::Test::IsSkipped()) {
        return;
      }
    }
  }
}

#define INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, \
                                                           d2)              \
  TEST(QuantizedSubOpModel,                                                 \
       TypeName##QuantizedMultiDimBroadcast_##d1##_##d2) {                  \
    RunQuantizedMultiDimBroadcastTest<T>(d1, d2);                           \
  }

#define INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, d1) \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, 1)       \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, 2)       \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, 3)       \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, 4)       \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, 5)       \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST(T, TypeName, d1, 6)

#define INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TESTS(T, TypeName) \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, 1)  \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, 2)  \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, 3)  \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, 4)  \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, 5)  \
  INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TEST_D2(T, TypeName, 6)

INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TESTS(int8_t, Int8)
INSTANTIATE_QUANTIZED_SUB_MULTI_DIM_BROADCAST_TESTS(uint8_t, Uint8)

}  // namespace
}  // namespace tflite
