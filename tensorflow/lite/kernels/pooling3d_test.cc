/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

using ::testing::ElementsAreArray;

template <typename T>
class BasePoolingOpModel : public SingleOpModel {
 public:
  BasePoolingOpModel(BuiltinOperator pool_operator, TensorData input,
                     int filter_d, int filter_h, int filter_w,
                     TensorData output, Padding padding = Padding_VALID,
                     int stride_d = 2, int stride_h = 2, int stride_w = 2) {
    if (input.type == TensorType_FLOAT32) {
      // Clear quantization params.
      input.min = input.max = 0.f;
      output.min = output.max = 0.f;
    }
    input_ = AddInput(input);
    output_ = AddOutput(output);

    CHECK(pool_operator == BuiltinOperator_AVERAGE_POOL_3D ||
          pool_operator == BuiltinOperator_MAX_POOL_3D);
    SetBuiltinOp(pool_operator, BuiltinOptions_Pool3DOptions,
                 CreatePool3DOptions(builder_, padding, stride_d, stride_w,
                                     stride_h, filter_d, filter_w, filter_h)
                     .Union());

    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  std::vector<float> GetOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
};

template <>
void BasePoolingOpModel<float>::SetInput(const std::vector<float>& data) {
  PopulateTensor(input_, data);
}

template <>
std::vector<float> BasePoolingOpModel<float>::GetOutput() {
  return ExtractVector<float>(output_);
}

template <typename T>
float GetTolerance(float min, float max) {
  if (std::is_floating_point<T>::value) {
    return 0.0f;
  } else {
    const float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                                std::numeric_limits<T>::min());
    return kQuantizedStep;
  }
}

#if GTEST_HAS_DEATH_TEST
TEST(AveragePoolingOpTest, InvalidDimSize) {
  EXPECT_DEATH(
      BasePoolingOpModel<float> m(BuiltinOperator_AVERAGE_POOL_3D,
                                  /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                                  /*filter_d=*/2,
                                  /*filter_h=*/2, /*filter_w=*/2,
                                  /*output=*/{TensorType_FLOAT32, {}},
                                  /*padding=*/Padding_VALID, /*stride_d=*/1,
                                  /*stride_h=*/1, /*stride_w=*/1),
      "NumDimensions.input. != 5 .4 != 5.");
}

TEST(AveragePoolingOpTest, ZeroStride) {
  EXPECT_DEATH(BasePoolingOpModel<float> m(
                   BuiltinOperator_AVERAGE_POOL_3D,
                   /*input=*/{TensorType_FLOAT32, {1, 2, 2, 4, 1}},
                   /*filter_d=*/2,
                   /*filter_h=*/2, /*filter_w=*/2,
                   /*output=*/{TensorType_FLOAT32, {}},
                   /*padding=*/Padding_VALID, /*stride_d=*/0,
                   /*stride_h=*/0, /*stride_w=*/0),
               "Cannot allocate tensors");
}
#endif

template <typename T>
class AveragePoolingOpTest : public ::testing::Test {};

template <typename T>
class MaxPoolingOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, int8_t, int16_t>;
TYPED_TEST_SUITE(AveragePoolingOpTest, DataTypes);
TYPED_TEST_SUITE(MaxPoolingOpTest, DataTypes);

TYPED_TEST(AveragePoolingOpTest, AveragePool) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_AVERAGE_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({3.125, 4.25}, kTolerance)));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolFilterH1) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_AVERAGE_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/1, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 5.75}, kTolerance)));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolPaddingSameStride1) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_AVERAGE_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax},
      Padding_SAME,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {2.875, 4.125, 4.5, 4.5, 3.0, 3.25, 3.25, 3.5,
                                  2.5, 4.0, 5.75, 5.5, 2.5, 2.0, 3.0, 4.0},
                                 kTolerance)));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolPaddingValidStride1) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_AVERAGE_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax},
      Padding_VALID,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {2.875, 4.125, 4.5}, kTolerance)));
}

TYPED_TEST(MaxPoolingOpTest, MaxPool) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_MAX_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({6.0, 10.0}, kTolerance)));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolFilterH1) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_MAX_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/1, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10}, kTolerance)));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolPaddingSameStride1) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_MAX_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax},
      Padding_SAME,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {6, 10, 10, 7, 5, 5, 4, 4, 3, 10, 10, 7, 3, 2, 4, 4}, kTolerance)));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolPaddingValidStride1) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TypeParam>::max() /
      static_cast<float>(std::numeric_limits<TypeParam>::max() + 1);
  const float kTolerance = GetTolerance<TypeParam>(-15.9375, 15.9375);
  BasePoolingOpModel<TypeParam> m(
      BuiltinOperator_MAX_POOL_3D,
      /*input=*/
      {GetTensorType<TypeParam>(),
       {1, 2, 2, 4, 1},
       15.9375 * kMin,
       15.9375 * kMax},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/
      {GetTensorType<TypeParam>(), {}, 15.9375 * kMin, 15.9375 * kMax},
      Padding_VALID,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({6.0, 10.0, 10.0}, kTolerance)));
}

}  // namespace tflite
