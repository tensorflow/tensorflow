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

#include <initializer_list>
#include <sstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

constexpr int kAxisIsATensor = -1000;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

class SplitOpModel : public SingleOpModel {
 public:
  SplitOpModel(const TensorData& input, int num_splits,
               int axis = kAxisIsATensor) {
    if (axis == kAxisIsATensor) {
      axis_ = AddInput({TensorType_INT32, {1}});
    } else {
      axis_ = AddConstInput(TensorType_INT32, {axis}, {1});
    }
    input_ = AddInput(input);
    for (int i = 0; i < num_splits; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_SPLIT, BuiltinOptions_SplitOptions,
                 CreateSplitOptions(builder_, num_splits).Union());
    if (axis == kAxisIsATensor) {
      BuildInterpreter({GetShape(axis_), GetShape(input_)});
    } else {
      BuildInterpreter({{}, GetShape(input_)});
    }
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }
  void SetAxis(int axis) { PopulateTensor(axis_, {axis}); }

  template <typename T>
  std::vector<T> GetOutput(int i) {
    return ExtractVector<T>(outputs_[i]);
  }
  std::vector<int> GetOutputShape(int i) { return GetTensorShape(outputs_[i]); }

 private:
  int input_;
  int axis_;
  std::vector<int> outputs_;
};

template <typename T>
void Check(TestType test_type, int axis, int num_splits,
           std::initializer_list<int> input_shape,
           std::initializer_list<int> output_shape,
           const std::initializer_list<T>& input_data,
           const std::vector<std::initializer_list<T>>& output_data,
           const TensorType& type) {
  auto debug = [&](int i) {
    std::stringstream ss;
    ss << "for output tensor " << i << " axis=" << axis
       << " and num_splits=" << num_splits;
    return ss.str();
  };
  if (test_type == TestType::kDynamic) {
    SplitOpModel m({type, input_shape}, num_splits);
    m.SetInput(input_data);
    m.SetAxis(axis);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    for (int i = 0; i < num_splits; ++i) {
      EXPECT_THAT(m.GetOutput<T>(i), ElementsAreArray(output_data[i]))
          << debug(i);
      EXPECT_THAT(m.GetOutputShape(i), ElementsAreArray(output_shape))
          << debug(i);
    }
  } else {
    SplitOpModel const_m({type, input_shape}, num_splits, axis);
    const_m.SetInput(input_data);
    ASSERT_EQ(const_m.Invoke(), kTfLiteOk);
    for (int i = 0; i < num_splits; ++i) {
      EXPECT_THAT(const_m.GetOutput<T>(i), ElementsAreArray(output_data[i]))
          << debug(i);
      EXPECT_THAT(const_m.GetOutputShape(i), ElementsAreArray(output_shape))
          << debug(i);
    }
  }
}

template <typename T>
class SplitOpTest : public ::testing::Test {
 public:
  static std::vector<TestType> range_;
};

template <>
std::vector<TestType> SplitOpTest<TestType>::range_{TestType::kConst,
                                                    TestType::kDynamic};

using DataTypes = ::testing::Types<float, int8_t, int16_t>;
TYPED_TEST_SUITE(SplitOpTest, DataTypes);

TYPED_TEST(SplitOpTest, FourDimensional) {
  for (TestType test_type : SplitOpTest<TestType>::range_) {
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 5, 6, 7, 8},
                         {9, 10, 11, 12, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 9, 10, 11, 12},
                         {5, 6, 7, 8, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 5, 6, 9, 10, 13, 14},
                         {3, 4, 7, 8, 11, 12, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 3, 5, 7, 9, 11, 13, 15},
                         {2, 4, 6, 8, 10, 12, 14, 16},
                     },
                     GetTensorType<TypeParam>());
  }
}

TYPED_TEST(SplitOpTest, FourDimensionalInt8) {
  for (TestType test_type : SplitOpTest<TestType>::range_) {
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 5, 6, 7, 8},
                         {9, 10, 11, 12, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 9, 10, 11, 12},
                         {5, 6, 7, 8, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 5, 6, 9, 10, 13, 14},
                         {3, 4, 7, 8, 11, 12, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 3, 5, 7, 9, 11, 13, 15},
                         {2, 4, 6, 8, 10, 12, 14, 16},
                     },
                     GetTensorType<TypeParam>());
  }
}

TYPED_TEST(SplitOpTest, FourDimensionalInt32) {
  for (TestType test_type : SplitOpTest<TestType>::range_) {
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 5, 6, 7, 8},
                         {9, 10, 11, 12, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 9, 10, 11, 12},
                         {5, 6, 7, 8, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 5, 6, 9, 10, 13, 14},
                         {3, 4, 7, 8, 11, 12, 15, 16},
                     },
                     GetTensorType<TypeParam>());
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 3, 5, 7, 9, 11, 13, 15},
                         {2, 4, 6, 8, 10, 12, 14, 16},
                     },
                     GetTensorType<TypeParam>());
  }
}

TYPED_TEST(SplitOpTest, OneDimensional) {
  for (TestType test_type : SplitOpTest<TestType>::range_) {
    Check<TypeParam>(
        /*axis_as_tensor*/ test_type,
        /*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
        {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}, GetTensorType<TypeParam>());
  }
}

TYPED_TEST(SplitOpTest, NegativeAxis) {
  for (TestType test_type : SplitOpTest<TestType>::range_) {
    Check<TypeParam>(/*axis_as_tensor*/ test_type,
                     /*axis=*/-4, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {
                         {1, 2, 3, 4, 5, 6, 7, 8},
                         {9, 10, 11, 12, 13, 14, 15, 16},
                     },
                     GetTensorType<TypeParam>());
  }
}

}  // namespace
}  // namespace tflite
