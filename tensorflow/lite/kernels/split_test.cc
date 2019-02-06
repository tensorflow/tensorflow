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
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

constexpr int kAxisIsATensor = -1000;

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
void Check(int axis, int num_splits, std::initializer_list<int> input_shape,
           std::initializer_list<int> output_shape,
           const std::initializer_list<T>& input_data,
           const std::vector<std::initializer_list<T>>& output_data,
           const TensorType& type = TensorType_FLOAT32) {
  auto debug = [&](int i) {
    std::stringstream ss;
    ss << "for output tensor " << i << " axis=" << axis
       << " and num_splits=" << num_splits;
    return ss.str();
  };
  SplitOpModel m({type, input_shape}, num_splits);
  m.SetInput(input_data);
  m.SetAxis(axis);
  m.Invoke();
  for (int i = 0; i < num_splits; ++i) {
    EXPECT_THAT(m.GetOutput<T>(i), ElementsAreArray(output_data[i]))
        << debug(i);
    EXPECT_THAT(m.GetOutputShape(i), ElementsAreArray(output_shape))
        << debug(i);
  }

  SplitOpModel const_m({type, input_shape}, num_splits, axis);
  const_m.SetInput(input_data);
  const_m.Invoke();
  for (int i = 0; i < num_splits; ++i) {
    EXPECT_THAT(const_m.GetOutput<T>(i), ElementsAreArray(output_data[i]))
        << debug(i);
    EXPECT_THAT(const_m.GetOutputShape(i), ElementsAreArray(output_shape))
        << debug(i);
  }
}

TEST(SplitOpTest, FourDimensional) {
  Check<float>(/*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
               {
                   {1, 2, 3, 4, 5, 6, 7, 8},
                   {9, 10, 11, 12, 13, 14, 15, 16},
               });
  Check<float>(/*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
               {
                   {1, 2, 3, 4, 9, 10, 11, 12},
                   {5, 6, 7, 8, 13, 14, 15, 16},
               });
  Check<float>(/*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
               {
                   {1, 2, 5, 6, 9, 10, 13, 14},
                   {3, 4, 7, 8, 11, 12, 15, 16},
               });
  Check<float>(/*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
               {
                   {1, 3, 5, 7, 9, 11, 13, 15},
                   {2, 4, 6, 8, 10, 12, 14, 16},
               });
}

TEST(SplitOpTest, FourDimensionalInt8) {
  Check<int8_t>(/*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                {
                    {1, 2, 3, 4, 5, 6, 7, 8},
                    {9, 10, 11, 12, 13, 14, 15, 16},
                },
                TensorType_INT8);
  Check<int8_t>(/*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                {
                    {1, 2, 3, 4, 9, 10, 11, 12},
                    {5, 6, 7, 8, 13, 14, 15, 16},
                },
                TensorType_INT8);
  Check<int8_t>(/*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                {
                    {1, 2, 5, 6, 9, 10, 13, 14},
                    {3, 4, 7, 8, 11, 12, 15, 16},
                },
                TensorType_INT8);
  Check<int8_t>(/*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                {
                    {1, 3, 5, 7, 9, 11, 13, 15},
                    {2, 4, 6, 8, 10, 12, 14, 16},
                },
                TensorType_INT8);
}

TEST(SplitOpTest, OneDimensional) {
  Check<float>(/*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
               {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TEST(SplitOpTest, NegativeAxis) {
  Check<float>(/*axis=*/-4, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
               {
                   {1, 2, 3, 4, 5, 6, 7, 8},
                   {9, 10, 11, 12, 13, 14, 15, 16},
               });
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
