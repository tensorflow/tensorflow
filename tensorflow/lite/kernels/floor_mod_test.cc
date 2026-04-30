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
#include "tensorflow/lite/kernels/floor_mod_test_common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

TEST(FloorModModel, Simple) {
  FloorModModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, 3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(FloorModModel, NegativeValue) {
  FloorModModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, -3, -4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, -2, -1));
}

TEST(FloorModModel, BroadcastFloorMod) {
  FloorModModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(-2, 0, -2, -2));
}

TEST(FloorModModel, Int64WithBroadcast) {
  FloorModModel<int64_t> model({TensorType_INT64, {1, 2, 2, 1}},
                               {TensorType_INT64, {1}}, {TensorType_INT64, {}});
  std::vector<int64_t> data1 = {10, -9, -11, (1LL << 34) + 9};
  model.PopulateTensor<int64_t>(model.input1(), data1);
  model.PopulateTensor<int64_t>(model.input2(), {-(1LL << 33)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(-8589934582, -9, -11, -8589934583));
}

template <typename T>
class FloatFloorModTest : public ::testing::Test {};

using FloatFloorModTestTypes = ::testing::Types<float, half, Eigen::bfloat16>;
TYPED_TEST_SUITE(FloatFloorModTest, FloatFloorModTestTypes);

TYPED_TEST(FloatFloorModTest, Simple) {
  using T = TypeParam;
  FloorModModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                         {GetTensorType<T>(), {1, 2, 2, 1}},
                         {GetTensorType<T>(), {}}, /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  model.template PopulateTensor<T>(model.input1(), {10, 9, 11, 3});
  model.template PopulateTensor<T>(model.input2(), {2, 2, 3, 4});
  TFLITE_INVOKE_AND_CHECK(T, &model);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({0, 1, 2, 3}, NumericLimits<T>::epsilon())));
}

TYPED_TEST(FloatFloorModTest, NegativeValue) {
  using T = TypeParam;
  FloorModModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                         {GetTensorType<T>(), {1, 2, 2, 1}},
                         {GetTensorType<T>(), {}}, /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  model.template PopulateTensor<T>(model.input1(), {10, -9, -11, 7});
  model.template PopulateTensor<T>(model.input2(), {2, 2, -3, -4});
  TFLITE_INVOKE_AND_CHECK(T, &model);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({0, 1, -2, -1}, NumericLimits<T>::epsilon())));
}

TYPED_TEST(FloatFloorModTest, BroadcastFloorMod) {
  using T = TypeParam;
  FloorModModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                         {GetTensorType<T>(), {1}}, {GetTensorType<T>(), {}},
                         /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  model.template PopulateTensor<T>(model.input1(), {10, -9, -11, 7});
  model.template PopulateTensor<T>(model.input2(), {-3});
  TFLITE_INVOKE_AND_CHECK(T, &model);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2, 0, -2, -2},
                                              NumericLimits<T>::epsilon())));
}

TEST(FloorModModel, SimpleInt16) {
  FloorModModel<int16_t> model({TensorType_INT16, {1, 2, 2, 1}},
                               {TensorType_INT16, {1, 2, 2, 1}},
                               {TensorType_INT16, {}});
  model.PopulateTensor<int16_t>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<int16_t>(model.input2(), {2, 2, 3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(FloorModModel, NegativeValueInt16) {
  FloorModModel<int16_t> model({TensorType_INT16, {1, 2, 2, 1}},
                               {TensorType_INT16, {1, 2, 2, 1}},
                               {TensorType_INT16, {}});
  model.PopulateTensor<int16_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int16_t>(model.input2(), {2, 2, -3, -4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, -2, -1));
}

TEST(FloorModModel, BroadcastFloorModInt16) {
  FloorModModel<int16_t> model({TensorType_INT16, {1, 2, 2, 1}},
                               {TensorType_INT16, {1}}, {TensorType_INT16, {}});
  model.PopulateTensor<int16_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int16_t>(model.input2(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(-2, 0, -2, -2));
}

}  // namespace
}  // namespace tflite
