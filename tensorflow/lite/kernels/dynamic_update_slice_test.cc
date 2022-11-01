/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class DynamicUpdateSliceOpModel : public SingleOpModel {
 public:
  DynamicUpdateSliceOpModel(const TensorData& operand, const TensorData& update,
                            const TensorData& start_indices) {
    input_ = AddInput(operand);
    update_ = AddInput(update);
    start_indices_ = AddInput(start_indices);
    output_ = AddOutput(operand.type);
    SetBuiltinOp(BuiltinOperator_DYNAMIC_UPDATE_SLICE,
                 BuiltinOptions_DynamicUpdateSliceOptions,
                 CreateDynamicUpdateSliceOptions(builder_).Union());
    BuildInterpreter(
        {GetShape(input_), GetShape(update_), GetShape(start_indices_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetUpdate(std::initializer_list<T> data) {
    PopulateTensor<T>(update_, data);
  }

  void SetStringInput(std::initializer_list<string> data) {
    PopulateStringTensor(input_, data);
  }

  template <typename T>
  void SetStartIndices(std::initializer_list<T> data) {
    PopulateTensor<T>(start_indices_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<string> GetStringOutput() {
    return ExtractVector<string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int update_;
  int start_indices_;
  int output_;
};

TEST(DynamicUpdateSliceOpTest, SimpleTestF32) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,   //
                                               4, -1, 6,  //
                                               7, -2, 9})));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI1) {
  DynamicUpdateSliceOpModel m({TensorType_BOOL, {3, 3}},
                              {TensorType_BOOL, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<bool>({true, true, true,  //
                    true, true, true,  //
                    true, true, true});
  m.SetUpdate<bool>({false, false});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, true, true,   //
                                                     true, false, true,  //
                                                     true, false, true}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI8) {
  DynamicUpdateSliceOpModel m({TensorType_INT8, {3, 3}},
                              {TensorType_INT8, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int8_t>({1, 2, 3,  //
                      4, 5, 6,  //
                      7, 8, 9});
  m.SetUpdate<int8_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({1, 2, 3,   //
                                                       4, -1, 6,  //
                                                       7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI32) {
  DynamicUpdateSliceOpModel m({TensorType_INT32, {3, 3}},
                              {TensorType_INT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int32_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetUpdate<int32_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({1, 2, 3,   //
                                                        4, -1, 6,  //
                                                        7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, ZeroSizeTestI32) {
  DynamicUpdateSliceOpModel m({TensorType_INT32, {3, 3}},
                              {TensorType_INT32, {2, 0}},
                              {TensorType_INT32, {2}});
  m.SetInput<int32_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({1, 2, 3,  //
                                                        4, 5, 6,  //
                                                        7, 8, 9}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI64) {
  DynamicUpdateSliceOpModel m({TensorType_INT64, {3, 3}},
                              {TensorType_INT64, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int64_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetUpdate<int64_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({1, 2, 3,   //
                                                        4, -1, 6,  //
                                                        7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, BoundaryTest) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 2}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2,  //
                      -3, -4});
  m.SetStartIndices<int32_t>({2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,    //
                                               4, -1, -2,  //
                                               7, -3, -4})));
}

TEST(DynamicUpdateSliceOpTest, UpdateShapeTooLargeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      DynamicUpdateSliceOpModel({TensorType_FLOAT32, {3, 3}},
                                {TensorType_FLOAT32, {4, 2}},
                                {TensorType_INT32, {2}}),
      "SizeOfDimension\\(update, i\\) <= SizeOfDimension\\(operand, "
      "i\\) was not true.");
}

}  // namespace
}  // namespace tflite
