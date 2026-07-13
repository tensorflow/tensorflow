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

class BaseSquaredDifferenceOpModel : public SingleOpModel {
 public:
  BaseSquaredDifferenceOpModel(const TensorData& input1,
                               const TensorData& input2,
                               const TensorData& output, bool allocate = true) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SQUARED_DIFFERENCE,
                 BuiltinOptions_SquaredDifferenceOptions,
                 CreateSquaredDifferenceOptions(builder_).Union());
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

template <typename T>
class SquaredDifferenceOpModel : public BaseSquaredDifferenceOpModel {
 public:
  using BaseSquaredDifferenceOpModel::BaseSquaredDifferenceOpModel;

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
};

template <typename T>
class FloatSquaredDifferenceTest : public ::testing::Test {};

using FloatSquaredDifferenceTestTypes =
    ::testing::Types<float, half, Eigen::bfloat16>;
TYPED_TEST_SUITE(FloatSquaredDifferenceTest, FloatSquaredDifferenceTestTypes);

class IntegerSquaredDifferenceOpModel : public BaseSquaredDifferenceOpModel {
 public:
  using BaseSquaredDifferenceOpModel::BaseSquaredDifferenceOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  // Allow at most off-by-2.
  return kQuantizedStep * 2;
}

class QuantizedSquaredDifferenceOpModel : public BaseSquaredDifferenceOpModel {
 public:
  using BaseSquaredDifferenceOpModel::BaseSquaredDifferenceOpModel;

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }
};

TYPED_TEST(FloatSquaredDifferenceTest, FloatType_SameShape) {
  using T = TypeParam;
  SquaredDifferenceOpModel<T> m({GetTensorType<T>(), {1, 2, 2, 1}},
                                {GetTensorType<T>(), {1, 2, 2, 1}},
                                {GetTensorType<T>(), {}}, /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &m);
  m.template PopulateTensor<T>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.template PopulateTensor<T>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  TFLITE_INVOKE_AND_CHECK(T, &m);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.49, 0.0, 0.09, 0.09}, NumericLimits<T>::epsilon() * 10)));
}

TYPED_TEST(FloatSquaredDifferenceTest, FloatType_VariousInputShapes) {
  using T = TypeParam;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    SquaredDifferenceOpModel<T> m({GetTensorType<T>(), test_shapes[i]},
                                  {GetTensorType<T>(), test_shapes[i]},
                                  {GetTensorType<T>(), {}}, /*allocate=*/false);
    TFLITE_ALLOCATE_AND_CHECK(T, &m);
    m.template PopulateTensor<T>(m.input1(), {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0});
    m.template PopulateTensor<T>(m.input2(), {1.0, 0.2, 0.6, 0.4, -1.0, -0.0});
    TFLITE_INVOKE_AND_CHECK(T, &m);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                   {9.0, 0.0, 0.09, 0.16, 4.41, 4.0},
                                   NumericLimits<T>::epsilon() * 10)))
        << "With shape number " << i;
  }
}

TYPED_TEST(FloatSquaredDifferenceTest, FloatType_WithBroadcast) {
  using T = TypeParam;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    SquaredDifferenceOpModel<T> m({GetTensorType<T>(), test_shapes[i]},
                                  {GetTensorType<T>(), {}},  // always a scalar
                                  {GetTensorType<T>(), {}}, /*allocate=*/false);
    TFLITE_ALLOCATE_AND_CHECK(T, &m);
    m.template PopulateTensor<T>(m.input1(), {-0.2, 0.2, 0.5, 0.8, 0.11, 1.1});
    m.template PopulateTensor<T>(m.input2(), {0.1});
    TFLITE_INVOKE_AND_CHECK(T, &m);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                   {0.09, 0.01, 0.16, 0.49, 0.0001, 1.0},
                                   NumericLimits<T>::epsilon() * 10)))
        << "With shape number " << i;
  }
}

TEST(IntegerSquaredDifferenceOpTest, IntegerType_SameShape) {
  IntegerSquaredDifferenceOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                                    {TensorType_INT32, {1, 2, 2, 1}},
                                    {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -15, 8});
  m.PopulateTensor<int32_t>(m.input2(), {5, -2, -3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({49, 16, 144, 9}));
}

TEST(IntegerSquaredDifferenceOpTest, IntegerType_VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSquaredDifferenceOpModel m({TensorType_INT32, test_shapes[i]},
                                      {TensorType_INT32, test_shapes[i]},
                                      {TensorType_INT32, {}});
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 3, 8, 11, -20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 6, 5, -5, -20});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({441, 0, 9, 9, 256, 0}))
        << "With shape number " << i;
  }
}

TEST(IntegerSquaredDifferenceOpTest, IntegerType_WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSquaredDifferenceOpModel m(
        {TensorType_INT32, test_shapes[i]},
        {TensorType_INT32, {}},  // always a scalar
        {TensorType_INT32, {}});
    m.PopulateTensor<int32_t>(m.input1(), {-20, 10, 7, 3, 1, 13});
    m.PopulateTensor<int32_t>(m.input2(), {3});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({529, 49, 16, 0, 4, 100}))
        << "With shape number " << i;
  }
}

TEST(QuantizedSquaredDifferenceOpTest, Quantized_SameShape) {
  float kQuantizedTolerance = GetTolerance(0, 1);
  QuantizedSquaredDifferenceOpModel m(
      {TensorType_INT8, {1, 2, 2, 1}, -1.2, 0.8},
      {TensorType_INT8, {1, 2, 2, 1}, -1.5, 0.5},
      {TensorType_INT8, {}, 0.0, 0.5});
  m.QuantizeAndPopulate<int8_t>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.QuantizeAndPopulate<int8_t>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({0.49, 0.0, 0.09, 0.09},
                                              kQuantizedTolerance)));
}

TEST(QuantizedSquaredDifferenceOpTest, Quantized_VariousInputShapes) {
  float kQuantizedTolerance = GetTolerance(0, 9);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSquaredDifferenceOpModel m(
        {TensorType_INT8, test_shapes[i], -2.0, 1.7},
        {TensorType_INT8, test_shapes[i], -1.0, 1.0},
        {TensorType_INT8, {}, 0.0, 9.0});
    m.QuantizeAndPopulate<int8_t>(m.input1(), {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0});
    m.QuantizeAndPopulate<int8_t>(m.input2(), {1.0, 0.2, 0.6, 0.4, -1.0, -0.0});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {9.0, 0.0, 0.09, 0.16, 4.41, 4.0}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSquaredDifferenceOpTest, Quantized_WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  float kQuantizedTolerance = GetTolerance(0, 1);
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSquaredDifferenceOpModel m(
        {TensorType_INT8, test_shapes[i], -0.2, 1.1},
        {TensorType_INT8, {}, 0.0, 0.1}, {TensorType_INT8, {}, 0.0, 1.0});
    m.QuantizeAndPopulate<int8_t>(m.input1(), {-0.2, 0.2, 0.5, 0.8, 0.11, 1.1});
    m.QuantizeAndPopulate<int8_t>(m.input2(), {0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<int8_t>(),
        ElementsAreArray(ArrayFloatNear({0.09, 0.01, 0.16, 0.49, 0.0001, 1.0},
                                        kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

}  // namespace
}  // namespace tflite
