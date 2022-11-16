/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Test;

class BaseWhereOpModel : public SingleOpModel {
 public:
  BaseWhereOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_WHERE, BuiltinOptions_WhereOptions,
                 CreateWhereOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

 protected:
  int input_;
  int output_;
};

class IntegerWhereOpModel : public BaseWhereOpModel {
 public:
  using BaseWhereOpModel::BaseWhereOpModel;

  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
};

template <typename T1>
class ConstInputWhereOpModel : public SingleOpModel {
 public:
  ConstInputWhereOpModel(T1 constant_values, const TensorData& output) {
    input_ = AddConstInput(GetTensorType<T1>(), {constant_values}, {});
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_WHERE, BuiltinOptions_WhereOptions,
                 CreateWhereOptions(builder_).Union());
    BuildInterpreter({{}});
  }

  int input() { return input_; }
  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }

 protected:
  int input_;
  int output_;
};
// Utils which returns TensorType from primitive type.
// Currently Where op supports only float, bool.
template <typename T>
TensorType GetTfLiteType();

template <>
TensorType GetTfLiteType<bool>() {
  return TensorType_BOOL;
}

template <>
TensorType GetTfLiteType<float>() {
  return TensorType_FLOAT32;
}

template <>
TensorType GetTfLiteType<int8_t>() {
  return TensorType_INT8;
}

template <>
TensorType GetTfLiteType<uint8_t>() {
  return TensorType_UINT8;
}

template <>
TensorType GetTfLiteType<int32_t>() {
  return TensorType_INT32;
}

template <>
TensorType GetTfLiteType<uint32_t>() {
  return TensorType_UINT32;
}

template <>
TensorType GetTfLiteType<int64_t>() {
  return TensorType_INT64;
}

// Helper function which creates std::vector from boolean type array 'data'
// but with different type. The returned value will be in type 'T' and
// matches the true/false criteria of where op.
template <typename T>
std::vector<T> GetCompatibleData(const std::initializer_list<bool>& data) {
  std::vector<T> result;
  for (auto item : data)
    if (item)
      result.push_back(T(1));
    else
      result.push_back(T(0));
  return result;
}

// Typed test so we can run the same set of tests with different data types.
template <typename T>
class WhereOpTest : public Test {
 public:
  using List = std::list<T>;
  static T shared_;
  T value_;
};

using MyTypes =
    ::testing::Types<bool, float, int32_t, uint32_t, int64_t, int8_t, uint8_t>;
TYPED_TEST_SUITE(WhereOpTest, MyTypes);

TYPED_TEST(WhereOpTest, ScalarValueFail) {
  ConstInputWhereOpModel<bool> m(false, {TensorType_INT64, {}});
  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

TYPED_TEST(WhereOpTest, SelectFromVectorNoResult) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({false, false, false}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput().size(), 0);
}

TYPED_TEST(WhereOpTest, SelectFromVector) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, false, true}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromMatrixNoResult) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({false, false, false,  //
                                               false, false, false,  //
                                               false, false, false}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput().size(), 0);
}

TYPED_TEST(WhereOpTest, SelectFromMatrix1) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 1}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, false, true}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               2, 0}));
}

TYPED_TEST(WhereOpTest, SelectFromMatrix2) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, true, false,   //
                                               true, false, false,  //
                                               true, false, true}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               0, 1,  //
                                               1, 0,  //
                                               2, 0,  //
                                               2, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromMatrix3) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 5}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(),
      GetCompatibleData<TypeParam>({true, false, false, true, true,   //
                                    false, true, true, false, false,  //
                                    true, false, true, false, false}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               0, 3,  //
                                               0, 4,  //
                                               1, 1,  //
                                               1, 2,  //
                                               2, 0,  //
                                               2, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromRank3TensorNoResult) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 2, 2}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({false, false, false, false,  //
                                               false, false, false, false}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput().size(), 0);
}

TYPED_TEST(WhereOpTest, SelectFromRank3Tensor1) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 1, 3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, false, true,  //
                                               false, false, true}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 2,  //
                                               1, 0, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromRank3Tensor2) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 2, 2}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, true, false, true,  //
                                               false, false, true, true}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 1,  //
                                               0, 1, 1,  //
                                               1, 1, 0,  //
                                               1, 1, 1}));
}

TYPED_TEST(WhereOpTest, SelectFromRank3Tensor3) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 3, 2}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(),
      GetCompatibleData<TypeParam>({true, true, false, true, false, false,  //
                                    false, false, true, false, true, true}));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 1,  //
                                               0, 1, 1,  //
                                               1, 1, 0,  //
                                               1, 2, 0,  //
                                               1, 2, 1}));
}

}  // namespace
}  // namespace tflite
