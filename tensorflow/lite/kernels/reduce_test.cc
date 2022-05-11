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

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/reduce_test_common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

using MeanOpConstModel = BaseConstOpModel<BuiltinOperator_MEAN, true>;
using MeanOpDynamicModel = BaseDynamicOpModel<BuiltinOperator_MEAN>;

using SumOpConstModel = BaseConstOpModel<BuiltinOperator_SUM>;
using SumOpDynamicModel = BaseDynamicOpModel<BuiltinOperator_SUM>;

using ProdOpConstModel = BaseConstOpModel<BuiltinOperator_REDUCE_PROD, true>;
using ProdOpDynamicModel =
    BaseDynamicOpModel<BuiltinOperator_REDUCE_PROD, true>;

using MaxOpConstModel = BaseConstOpModel<BuiltinOperator_REDUCE_MAX>;
using MaxOpDynamicModel = BaseDynamicOpModel<BuiltinOperator_REDUCE_MAX>;

using MinOpConstModel = BaseConstOpModel<BuiltinOperator_REDUCE_MIN>;
using MinOpDynamicModel = BaseDynamicOpModel<BuiltinOperator_REDUCE_MIN>;

using AnyOpConstModel = BaseConstOpModel<BuiltinOperator_REDUCE_ANY>;
using AnyOpDynamicModel = BaseDynamicOpModel<BuiltinOperator_REDUCE_ANY>;

using AllOpConstModel = BaseConstOpModel<BuiltinOperator_REDUCE_ALL>;
using AllOpDynamicModel = BaseDynamicOpModel<BuiltinOperator_REDUCE_ALL>;

// for quantized Add, the error shouldn't exceed step
template <typename integer_type = int8_t>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      (max - min) / (std::numeric_limits<integer_type>::max() -
                     std::numeric_limits<integer_type>::min());
  return kQuantizedStep;
}

using BoolReductions = ::testing::Types<AnyOpConstModel, AllOpConstModel>;
using NonBoolReductions =
    ::testing::Types<MeanOpConstModel, SumOpConstModel, ProdOpConstModel,
                     MaxOpConstModel, MinOpConstModel>;
using DynamicBoolReductions =
    ::testing::Types<AnyOpDynamicModel, AllOpDynamicModel>;
using DynamicNonBoolReductions =
    ::testing::Types<MeanOpDynamicModel, SumOpDynamicModel, ProdOpDynamicModel,
                     MaxOpDynamicModel, MinOpDynamicModel>;

template <typename T>
class ReductionIsCopyTest : public testing::Test {};
template <typename T>
class ReductionIsCopyTestBool : public testing::Test {};
template <typename T>
class DynamicReductionIsCopyTest : public testing::Test {};
template <typename T>
class DynamicReductionIsCopyTestBool : public testing::Test {};

TYPED_TEST_SUITE(ReductionIsCopyTest, NonBoolReductions);
TYPED_TEST_SUITE(ReductionIsCopyTestBool, BoolReductions);
TYPED_TEST_SUITE(DynamicReductionIsCopyTest, DynamicNonBoolReductions);
TYPED_TEST_SUITE(DynamicReductionIsCopyTestBool, DynamicBoolReductions);

// Test reductions which are copies.
TYPED_TEST(ReductionIsCopyTest, ReduceIsCopy) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  TypeParam m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {4, 3, 2}},
              {0}, {}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3, 2}));
  EXPECT_THAT(m.template GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(data)));
}

TYPED_TEST(ReductionIsCopyTestBool, ReduceIsCopyBool) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  TypeParam m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2, 3, 2}}, {0},
              {}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.template GetOutput<bool>(), ElementsAreArray(data));
}

TYPED_TEST(DynamicReductionIsCopyTest, ReduceIsCopy) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  TypeParam m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {4, 3, 2}},
              {TensorType_INT32, {0}}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3, 2}));
  EXPECT_THAT(m.template GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(data)));
}

TYPED_TEST(DynamicReductionIsCopyTestBool, ReduceIsCopy) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  TypeParam m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2, 3, 2}},
              {TensorType_INT32, {0}}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.template GetOutput<bool>(), ElementsAreArray(data));
}

// Tests for reduce_mean
TEST(ConstFloatMeanOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({10.5, 12.5, 14.5})));
}

TEST(ConstFloatMeanOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  MeanOpConstModel m({TensorType_FLOAT32, {4, 0, 2}}, {TensorType_FLOAT32, {3}},
                     {2}, {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({3.27})));
}

TEST(ConstFloatMeanOpTest, ScalarAxis) {
  std::vector<float> data = {4., 2.};
  MeanOpConstModel m({TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {1}}, {},
                     {0}, true);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({3.})));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({12, 13})));
}

TEST(DynamicFloatMeanOpTest, ReduceOnLastDimNotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MeanOpDynamicModel m({TensorType_FLOAT32, {4, 3, 2}},
                       {TensorType_FLOAT32, {2}}, {TensorType_INT32, {1}},
                       false);
  std::vector<int> axis = {2};
  m.SetAxis(axis);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5,
                                       15.5, 17.5, 19.5, 21.5, 23.5})));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

TEST(ConstUint8MeanOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                     {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.3, 0.35, 0.55}, kQuantizedTolerance)));
}

TEST(ConstUint8MeanOpTest, Rounding) {
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_UINT8, {3, 2}, -1.0, 1.0},
                     {TensorType_UINT8, {3}, -1.1, 1.1}, {1}, {1}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({163, 168, 192}));
}

TEST(ConstInt8MeanOpTest, Rounding) {
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MeanOpConstModel m({TensorType_INT8, {3, 2}, -1.0, 1.0},
                     {TensorType_INT8, {3}, -1.1, 1.1}, {1}, {1}, true);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({34, 39, 63}));
}

template <typename integer_type, TensorType tensor_dtype>
void MeanOpConstModelTest() {
  float kQuantizedTolerance = GetTolerance<integer_type>(-255.0, 255.0);
  std::vector<float> data = {105.0, 71.0, 233.0, 92.0, 227.0, 11.0, 14.0, 43.0};
  MeanOpConstModel m({tensor_dtype, {1, 1, 2, 4}, -255.0, 255.0},
                     {tensor_dtype, {1, 2, 4}, -255, 255.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<integer_type>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear(data, kQuantizedTolerance)));
}

class ConstMeanOpTestSameScale : public ::testing::Test {};

TEST_F(ConstMeanOpTestSameScale, NonSpecialAxisSameScaleInt8) {
  MeanOpConstModelTest<int8_t, TensorType_INT8>();
}

TEST_F(ConstMeanOpTestSameScale, NonSpecialAxisSameScaleInt16) {
  MeanOpConstModelTest<int16_t, TensorType_INT16>();
}

template <typename integer_type, TensorType tensor_dtype>
void ConstMeanOpTestNonSameScale() {
  float kQuantizedTolerance = GetTolerance<integer_type>(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  MeanOpConstModel m({tensor_dtype, {1, 1, 2, 4}, -1.0, 1.0},
                     {tensor_dtype, {1, 2}, -5.0, 5.0}, {2}, {1, 3}, false);
  m.QuantizeAndPopulate<integer_type>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_type>(),
      ElementsAreArray(ArrayFloatNear({0.25, 0.65}, kQuantizedTolerance)));
}

class ConstMeanOpTestNonSameScale : public ::testing::Test {};

TEST_F(ConstMeanOpTestNonSameScale, NonSpecialAxisNonSameScaleInt8) {
  MeanOpConstModelTest<int8_t, TensorType_INT8>();
}

TEST_F(ConstMeanOpTestNonSameScale, NonSpecialAxisNonSameScaleInt16) {
  MeanOpConstModelTest<int16_t, TensorType_INT16>();
}

template <typename integer_type, TensorType tensor_dtype>
void MeanOpTestQuantizedSameScale() {
  float kQuantizedTolerance = GetTolerance<integer_type>(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1,
                             0.1, 0.1, 0.1, 0.4, 0.2, 0.2, 0.2, 0.9, 0.9,
                             0.9, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.3,
                             0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({tensor_dtype, {1, 2, 2, 9}, -1.0, 1.0},
                     {tensor_dtype, {2}, -1.0, 1.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<integer_type>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 9}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.35, 0.325, 0.2, 0.35, 0.375, 0.325, 0.225, 0.45, 0.425},
                  kQuantizedTolerance)));
}

class MeanOpTestQuantizedSameScale : public ::testing::Test {};

TEST_F(MeanOpTestQuantizedSameScale, QuantizedSameScaleInt8) {
  MeanOpConstModelTest<int8_t, TensorType_INT8>();
}

TEST_F(MeanOpTestQuantizedSameScale, QuantizedSameScaleInt16) {
  MeanOpConstModelTest<int16_t, TensorType_INT16>();
}

template <typename integer_type, TensorType tensor_dtype>
void MeanOpTestQuantizedDifferentScale() {
  float kQuantizedTolerance = GetTolerance<integer_type>(-5.0, 5.0);
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1,
                             0.1, 0.1, 0.1, 0.4, 0.2, 0.2, 0.2, 0.9, 0.9,
                             0.9, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.3,
                             0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({tensor_dtype, {1, 2, 2, 9}, -1.0, 1.0},
                     {tensor_dtype, {2}, -4.0, 4.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<integer_type>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 9}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.35, 0.325, 0.2, 0.35, 0.375, 0.325, 0.225, 0.45, 0.425},
                  kQuantizedTolerance)));
}

class MeanOpTestQuantizedDifferentScale : public ::testing::Test {};

TEST_F(MeanOpTestQuantizedDifferentScale, QuantizedDifferentScaleInt8) {
  MeanOpConstModelTest<int8_t, TensorType_INT8>();
}

TEST_F(MeanOpTestQuantizedDifferentScale, QuantizedDifferentScaleInt16) {
  MeanOpConstModelTest<int16_t, TensorType_INT16>();
}

TEST(ConstFloatMeanOpTest, KeepDims4DMeanLargeDepthInt8) {
  float kQuantizedTolerance = GetTolerance(-5.0, 5.0);
  std::vector<float> data = {
      0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.4, 0.2, 0.2,
      0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2,
      0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1,
      0.1, 0.1, 0.1, 0.4, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.3, 0.7, 0.7,
      0.1, 0.1, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  MeanOpConstModel m({TensorType_INT8, {1, 2, 2, 18}, -1.0, 1.0},
                     {TensorType_INT8, {2}, -1.0, 1.0}, {2}, {1, 2}, true);
  m.QuantizeAndPopulate<int8_t>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 18}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5, 0.55, 0.25, 0.35, 0.45, 0.5, 0.25, 0.3, 0.2, 0.2, 0.1,
                   0.15, 0.35, 0.3, 0.15, 0.2, 0.6, 0.65},
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({84, 100, 116})));
}

TEST(ConstFloatSumOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  SumOpConstModel m({TensorType_FLOAT32, {4, 0, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({144, 156})));
}

TEST(ConstFloatSumOpTest, Scalar) {
  std::vector<float> data = {17.};
  SumOpConstModel m({TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}}, {}, {0},
                    false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({17.})));
}

TEST(ConstFloatSumOpTest, ScalarAxis) {
  std::vector<float> data = {17., 21., 4.};
  SumOpConstModel m({TensorType_FLOAT32, {3}}, {TensorType_FLOAT32, {1}}, {},
                    {0}, true);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({42.})));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

TEST(ConstUint8SumOpTest, NotKeepDims) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  SumOpConstModel m({TensorType_UINT8, {1, 3, 2}, -1.0, 1.0},
                    {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {1}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(expected_sum, kQuantizedTolerance)));
}

// Tests for reduce_prod

TEST(ConstFloatProdOpTest, NotKeepDimsLarge) {
  const std::vector<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({3.162341376e+11, 1.9619905536e+12})));
}

template <TensorType tensor_type, typename integer_dtype>
void ConstIntProdOpTestNotKeepDimsLarge() {
  const float input_min = (tensor_type == TensorType_INT16) ? -24.0 : 0.0;
  const float input_max = 24.0;
  const float output_min =
      (tensor_type == TensorType_INT16) ? -1.9619905536e+12 : 0.0;
  const float output_max = 1.9619905536e+12;

  const std::vector<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpConstModel m({tensor_type, {4, 3, 2}, input_min, input_max},
                     {tensor_type, {2}, output_min, output_max}, {4},
                     {1, 0, -3, -3}, false);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const int reduced_axis_size = 12;
  const float kQuantizedStep =
      GetTolerance<integer_dtype>(output_min, output_max);
  const float kQuantizedTolerance = reduced_axis_size * 2 * kQuantizedStep;
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(
                  {3.162341376e+11, 1.9619905536e+12}, kQuantizedTolerance)));
}

TEST(ConstInt8ProdOpTest, NotKeepDimsLarge) {
  ConstIntProdOpTestNotKeepDimsLarge<TensorType_INT8, int8_t>();
}

TEST(ConstInt16ProdOpTest, NotKeepDimsLarge) {
  ConstIntProdOpTestNotKeepDimsLarge<TensorType_INT16, int16_t>();
}

TEST(ConstFloatProdOpTest, NotKeepDimsSmall) {
  const std::vector<float> data = {
      -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
      -0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3};
  ProdOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({-0.0062270208, 0.0062270208})));
}

template <TensorType tensor_type, typename integer_dtype>
void ConstIntProdOpTestNotKeepDimsSmall() {
  const std::vector<float> data = {
      -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
      -0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3};
  ProdOpConstModel m({tensor_type, {4, 3, 2}, -1.3, 1.3},
                     {tensor_type, {2}, -0.0062270208, 0.0062270208}, {4},
                     {1, 0, -3, -3}, false);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const int reduced_axis_size = 12;
  const float kQuantizedStep =
      GetTolerance<integer_dtype>(-0.0062270208, 0.0062270208);
  const float kQuantizedTolerance = reduced_axis_size * 2 * kQuantizedStep;
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({-0.0062270208, 0.0062270208},
                                              kQuantizedTolerance)));
}

TEST(ConstInt8ProdOpTest, NotKeepDimsSmall) {
  ConstIntProdOpTestNotKeepDimsSmall<TensorType_INT8, int8_t>();
}

TEST(ConstInt16ProdOpTest, NotKeepDimsSmall) {
  ConstIntProdOpTestNotKeepDimsSmall<TensorType_INT16, int16_t>();
}

TEST(ConstFloatProdOpTest, KeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                     {2}, {0, 2}, true);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  ArrayFloatNear({7.74592e+06, 1.197504e+08, 6.6889152e+08})));
}

TEST(ConstFloatProdOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ProdOpConstModel m({TensorType_FLOAT32, {4, 0, 2}}, {TensorType_FLOAT32, {3}},
                     {2}, {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
}

TEST(ConstInt8ProdOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ProdOpConstModel m({TensorType_INT8, {4, 0, 2}, 0.0, 1.0},
                     {TensorType_INT8, {3}, 0.0, 1.0}, {2}, {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  ArrayFloatNear({7.74592e+06, 1.197504e+08, 6.6889152e+08})));
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicIntProdOpTestKeepDims() {
  const float input_min = (tensor_type == TensorType_INT16) ? -24.0 : 0.0;
  const float input_max = 24.0;
  const float output_min =
      (tensor_type == TensorType_INT16) ? -6.6889152e+08 : 0.0;
  const float output_max = 6.6889152e+08;

  const std::vector<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  ProdOpDynamicModel m({tensor_type, {4, 3, 2}, input_min, input_max},
                       {tensor_type, {3}, output_min, output_max},
                       {TensorType_INT32, {2}}, true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const int reduced_axis_size = 8;
  const float kQuantizedStep =
      GetTolerance<integer_dtype>(output_min, output_max);
  const float kQuantizedTolerance = reduced_axis_size * 2 * kQuantizedStep;
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear(
          {7.74592e+06, 1.197504e+08, 6.6889152e+08}, kQuantizedTolerance)));
}

TEST(DynamicInt8ProdOpTest, KeepDims) {
  DynamicIntProdOpTestKeepDims<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16ProdOpTest, KeepDims) {
  DynamicIntProdOpTestKeepDims<TensorType_INT16, int16_t>();
}

TEST(DynamicFloatProdOpTest, Scale) {
  std::vector<float> data = {9.527};
  ProdOpDynamicModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {1}},
                       {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({20, 22, 24})));
}

TEST(ConstFloatMaxOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  MaxOpConstModel m({TensorType_FLOAT32, {4, 0, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

template <TensorType tensor_type, typename integer_dtype>
void ConstMaxOpTestNotKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MaxOpConstModel m({tensor_type, {1, 3, 2}, 1.0f * kMin, 1.0f * kMax},
                    {tensor_type, {2}, 1.0f * kMin, 1.0f * kMax}, {1}, {1},
                    false);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({0.5, 0.6}, kQuantizedTolerance)));
}

TEST(ConstUint8MaxOpTest, NotKeepDims) {
  ConstMaxOpTestNotKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(ConstInt8MaxOpTest, NotKeepDims) {
  ConstMaxOpTestNotKeepDims<TensorType_INT8, int8_t>();
}

TEST(ConstInt16MaxOpTest, NotKeepDims) {
  ConstMaxOpTestNotKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void ConstMaxOpTestKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MaxOpConstModel m({tensor_type, {3, 2}, 1.0f * kMin, 1.0f * kMax},
                    {tensor_type, {3}, 1.0f * kMin, 1.0f * kMax}, {1}, {1},
                    true);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({0.4, 0.4, 0.6}, kQuantizedTolerance)));
}

TEST(ConstUint8MaxOpTest, KeepDims) {
  ConstMaxOpTestKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(ConstInt8MaxOpTest, KeepDims) {
  ConstMaxOpTestKeepDims<TensorType_INT8, int8_t>();
}

TEST(ConstInt16MaxOpTest, KeepDims) {
  ConstMaxOpTestKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicMaxOpTestNotKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-5.0, 5.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MaxOpDynamicModel m({tensor_type, {2, 2}, 5.0f * kMin, 5.0f * kMax},
                      {tensor_type, {2}, 5.0f * kMin, 5.0f * kMax},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({1.3, 0.24}, kQuantizedTolerance)));
}

TEST(DynamicUint8MaxOpTest, NotKeepDims) {
  DynamicMaxOpTestNotKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(DynamicInt8MaxOpTest, NotKeepDims) {
  DynamicMaxOpTestNotKeepDims<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16MaxOpTest, NotKeepDims) {
  DynamicMaxOpTestNotKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicMaxOpTestKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-12.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MaxOpDynamicModel m({tensor_type, {2, 2}, 12.0f * kMin, 12.0f * kMax},
                      {tensor_type, {2}, 12.0f * kMin, 12.0f * kMax},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({11.14, 0.879}, kQuantizedTolerance)));
}

TEST(DynamicUint8MaxOpTest, KeepDims) {
  DynamicMaxOpTestKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(DynamicInt8MaxOpTest, KeepDims) {
  DynamicMaxOpTestKeepDims<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16MaxOpTest, KeepDims) {
  DynamicMaxOpTestKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicMaxOpTestScalar() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-12.0, 12.0);
  std::vector<float> data = {11.14};
  MaxOpDynamicModel m({tensor_type, {}, 12.0f * kMin, 12.0f * kMax},
                      {tensor_type, {}, 12.0f * kMin, 12.0f * kMax},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({11.14}, kQuantizedTolerance)));
}

TEST(DynamicUint8MaxOpTest, Scalar) {
  DynamicMaxOpTestScalar<TensorType_UINT8, uint8_t>();
}

TEST(DynamicInt8MaxOpTest, Scalar) {
  DynamicMaxOpTestScalar<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16MaxOpTest, Scalar) {
  DynamicMaxOpTestScalar<TensorType_INT16, int16_t>();
}

// Tests for reduce_min

TEST(ConstFloatMinOpTest, NotKeepDims) {
  std::vector<float> data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                             17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  MinOpConstModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                    {4}, {1, 0, -3, -3}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 3, 5})));
}

TEST(ConstFloatMinOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  MinOpConstModel m({TensorType_FLOAT32, {4, 0, 2}}, {TensorType_FLOAT32, {3}},
                    {2}, {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({9.527})));
}

template <TensorType tensor_type, typename integer_dtype>
void ConstMinOpTestNotKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MinOpConstModel m({tensor_type, {1, 3, 2}, 1.0f * kMin, 1.0f * kMax},
                    {tensor_type, {2}, 1.0f * kMin, 1.0f * kMax}, {1}, {1},
                    false);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({0.3, 0.2}, kQuantizedTolerance)));
}

TEST(ConstUint8MinOpTest, NotKeepDims) {
  ConstMinOpTestNotKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(ConstInt8MinOpTest, NotKeepDims) {
  ConstMinOpTestNotKeepDims<TensorType_INT8, int8_t>();
}

TEST(ConstInt16MinOpTest, NotKeepDims) {
  ConstMinOpTestNotKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void ConstMinOpTestKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  MinOpConstModel m({tensor_type, {3, 2}, 1.0f * kMin, 1.0f * kMax},
                    {tensor_type, {3}, 1.0f * kMin, 1.0f * kMax}, {1}, {1},
                    true);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({0.2, 0.3, 0.5}, kQuantizedTolerance)));
}

TEST(ConstUint8MinOpTest, KeepDims) {
  ConstMinOpTestKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(ConstInt8MinOpTest, KeepDims) {
  ConstMinOpTestKeepDims<TensorType_INT8, int8_t>();
}

TEST(ConstInt16MinOpTest, KeepDims) {
  ConstMinOpTestKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicMinOpTestNotKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-5.0, 5.0);
  std::vector<float> data = {1.3, -4.8, -3.6, 0.24};
  MinOpDynamicModel m({tensor_type, {2, 2}, 5.0f * kMin, 5.0f * kMax},
                      {tensor_type, {2}, 5.0f * kMin, 5.0f * kMax},
                      {TensorType_INT32, {1}}, false);
  std::vector<int> axis = {1};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({-4.8, -3.6}, kQuantizedTolerance)));
}

TEST(DynamicUint8MinOpTest, NotKeepDims) {
  DynamicMinOpTestNotKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(DynamicInt8MinOpTest, NotKeepDims) {
  DynamicMinOpTestNotKeepDims<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16MinOpTest, NotKeepDims) {
  DynamicMinOpTestNotKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicMinOpTestKeepDims() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-12.0, 12.0);
  std::vector<float> data = {11.14, -0.14, 7.423, 0.879};
  MinOpDynamicModel m({tensor_type, {2, 2}, 12.0f * kMin, 12.0f * kMax},
                      {tensor_type, {2}, 12.0f * kMin, 12.0f * kMax},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({7.423, -0.14}, kQuantizedTolerance)));
}

TEST(DynamicUint8MinOpTest, KeepDims) {
  DynamicMinOpTestKeepDims<TensorType_UINT8, uint8_t>();
}

TEST(DynamicInt8MinOpTest, KeepDims) {
  DynamicMinOpTestKeepDims<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16MinOpTest, KeepDims) {
  DynamicMinOpTestKeepDims<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void DynamicMinOpTestScalar() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-12.0, 12.0);
  std::vector<float> data = {11.14};
  MinOpDynamicModel m({tensor_type, {}, 12.0f * kMin, 12.0f * kMax},
                      {tensor_type, {}, 12.0f * kMin, 12.0f * kMax},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.QuantizeAndPopulate<integer_dtype>(m.Input(), data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({11.14}, kQuantizedTolerance)));
}

TEST(DynamicUint8MinOpTest, Scalar) {
  DynamicMinOpTestScalar<TensorType_UINT8, uint8_t>();
}

TEST(DynamicInt8MinOpTest, Scalar) {
  DynamicMinOpTestScalar<TensorType_INT8, int8_t>();
}

TEST(DynamicInt16MinOpTest, Scalar) {
  DynamicMinOpTestScalar<TensorType_INT16, int16_t>();
}

// Tests for reduce_any

TEST(ConstAnyOpTest, NotKeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpConstModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2}}, {4},
                    {1, 0, -3, -3}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false, true}));
}

TEST(ConstAnyOpTest, KeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpConstModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {3}}, {2},
                    {0, 2}, true);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, false, true}));
}

TEST(ConstAnyOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  AnyOpConstModel m({TensorType_BOOL, {2, 0, 2}}, {TensorType_BOOL, {3}}, {2},
                    {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
}

TEST(DynamicAnyOpTest, NotKeepDims) {
  std::vector<bool> data = {false, false, false, false, false, false,
                            false, true,  false, false, false, true};
  AnyOpDynamicModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2}},
                      {TensorType_INT32, {4}}, false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false}));
}

// Tests for reduce_all

TEST(ConstAllOpTest, NotKeepDims) {
  std::vector<bool> data = {true, true, true, true, true, false,
                            true, true, true, true, true, true};
  AllOpConstModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2}}, {4},
                    {1, 0, -3, -3}, false);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, false}));
}

TEST(ConstAllOpTest, KeepDims) {
  std::vector<bool> data = {true, true, true, true, true, false,
                            true, true, true, true, true, true};
  AllOpConstModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {3}}, {2},
                    {0, 2}, true);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, true, false}));
}

TEST(ConstAllOpTest, ZeroInputDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  AllOpConstModel m({TensorType_BOOL, {2, 0, 2}}, {TensorType_BOOL, {3}}, {2},
                    {0, 2}, true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 1}));
}

TEST(DynamicAllOpTest, NotKeepDims) {
  std::vector<bool> data = {true, true, true, true, true, false,
                            true, true, true, true, true, true};
  AllOpDynamicModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {2}},
                      {TensorType_INT32, {4}}, false);
  std::vector<int> axis = {1, 0, -3, -3};
  m.SetAxis(axis);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, false}));
}

TEST(DynamicAllOpTest, KeepDims) {
  std::vector<bool> data = {true, true, true, true, true, false,
                            true, true, true, true, true, true};
  AllOpDynamicModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_BOOL, {3}},
                      {TensorType_INT32, {2}}, true);
  std::vector<int> axis = {0, 2};
  m.SetAxis(axis);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, true, false}));
}

TEST(DynamicAllOpTest, Scalar) {
  std::vector<bool> data = {false};
  AllOpDynamicModel m({TensorType_BOOL, {1}}, {TensorType_BOOL, {1}},
                      {TensorType_INT32, {1}}, true);
  std::vector<int> axis = {0};
  m.SetAxis(axis);
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false}));
}

TEST(ConstInt32MinOpTest, EmptyInputButScalarOutput) {
  MinOpConstModel m({TensorType_INT32, {0}}, {TensorType_INT32, {}}, {1}, {0},
                    false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({2147483647}));
}

TEST(ConstInt32MaxOpTest, EmptyInputButScalarOutput) {
  MaxOpConstModel m({TensorType_INT32, {0}}, {TensorType_INT32, {}}, {1}, {0},
                    false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-2147483648}));
}

TEST(ConstInt32ProdOpTest, EmptyInputButScalarOutput) {
  ProdOpConstModel m({TensorType_INT32, {0}}, {TensorType_INT32, {}}, {1}, {0},
                     false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({1}));
}

TEST(ConstInt32SumOpTest, EmptyInputButScalarOutput) {
  SumOpConstModel m({TensorType_INT32, {0}}, {TensorType_INT32, {}}, {1}, {0},
                    false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({0}));
}

TEST(ConstInt32MeanOpTest, EmptyInputButScalarOutput) {
  MeanOpConstModel m({TensorType_INT32, {0}}, {TensorType_INT32, {}}, {1}, {0},
                     false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({std::numeric_limits<int32_t>::quiet_NaN()}));
}

TEST(ConstFloatMinOpTest, EmptyInputButScalarOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  MinOpConstModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}}, {1},
                    {0}, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({std::numeric_limits<float>::max()})));
}

TEST(ConstFloatMaxOpTest, EmptyInputButScalarOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  MaxOpConstModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}}, {1},
                    {0}, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({-std::numeric_limits<float>::max()})));
}

TEST(ConstFloatProdOpTest, EmptyInputButScalarOutput) {
  ProdOpConstModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}}, {1},
                     {0}, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({1.0})));
}

TEST(ConstFloatSumOpTest, EmptyInputButScalarOutput) {
  SumOpConstModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}}, {1},
                    {0}, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({0.0})));
}

TEST(ConstFloatMeanOpTest, EmptyInputButScalarOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  MeanOpConstModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}}, {1},
                     {0}, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  auto output_data = m.GetOutput<float>();
  EXPECT_TRUE(std::all_of(output_data.begin(), output_data.end(),
                          [](float value) { return std::isnan(value); }));
}

TEST(ConstAllOpTest, EmptyInputButScalarOutputKeepDim) {
  AllOpConstModel m({TensorType_BOOL, {0}}, {TensorType_BOOL, {}}, {1}, {0},
                    true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true}));
}

TEST(ConstAnyOpTest, EmptyInputButScalarOutputKeepDim) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  AnyOpConstModel m({TensorType_BOOL, {0}}, {TensorType_BOOL, {}}, {1}, {0},
                    true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({false}));
}

}  // namespace
}  // namespace tflite
