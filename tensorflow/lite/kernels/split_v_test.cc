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
#include <initializer_list>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

constexpr int kAxisIsATensor = -1000;

class SplitVOpModel : public SingleOpModel {
 public:
  SplitVOpModel(const TensorData& input, const TensorData& size_splits,
                int num_splits, int axis) {
    input_ = AddInput(input);
    size_splits_ = AddInput(size_splits);
    if (axis == kAxisIsATensor) {
      axis_ = AddInput({TensorType_INT32, {1}});
    } else {
      axis_ = AddConstInput(TensorType_INT32, {axis}, {1});
    }
    for (int i = 0; i < num_splits; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_SPLIT_V, BuiltinOptions_SplitVOptions,
                 CreateSplitVOptions(builder_, num_splits).Union());
    if (axis == kAxisIsATensor) {
      BuildInterpreter(
          {GetShape(input_), GetShape(size_splits_), GetShape(axis_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(size_splits_), {}});
    }
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetSizeSplits(std::initializer_list<int> data) {
    PopulateTensor(size_splits_, data);
  }
  void SetAxis(int axis) { PopulateTensor(axis_, {axis}); }

  std::vector<float> GetOutput(int i) {
    return ExtractVector<float>(outputs_[i]);
  }
  std::vector<int> GetOutputShape(int i) { return GetTensorShape(outputs_[i]); }

 private:
  int input_;
  int size_splits_;
  int axis_;
  std::vector<int> outputs_;
};

// TODO(ruic): Add tests to test quantized values. b/119638735
using TensorValues = std::initializer_list<float>;

void Check(int axis, std::initializer_list<int> input_shape,
           std::initializer_list<int> size_splits_shape,
           std::vector<std::initializer_list<int>> output_shapes,
           const TensorValues& input_data,
           const std::initializer_list<int>& size_splits_data,
           const std::vector<TensorValues>& output_data) {
  int num_splits = size_splits_data.size();
  SplitVOpModel m({TensorType_FLOAT32, input_shape},
                  {TensorType_INT32, size_splits_shape}, num_splits,
                  kAxisIsATensor);
  m.SetInput(input_data);
  m.SetSizeSplits(size_splits_data);
  m.SetAxis(axis);
  m.Invoke();
  for (int i = 0; i < num_splits; ++i) {
    EXPECT_THAT(m.GetOutput(i), ElementsAreArray(output_data[i]));
    EXPECT_THAT(m.GetOutputShape(i), ElementsAreArray(output_shapes[i]));
  }

  SplitVOpModel const_m({TensorType_FLOAT32, input_shape},
                        {TensorType_INT32, size_splits_shape}, num_splits,
                        axis);
  const_m.SetInput(input_data);
  const_m.SetSizeSplits(size_splits_data);
  const_m.Invoke();
  for (int i = 0; i < num_splits; ++i) {
    EXPECT_THAT(const_m.GetOutput(i), ElementsAreArray(output_data[i]));
    EXPECT_THAT(const_m.GetOutputShape(i), ElementsAreArray(output_shapes[i]));
  }
}

TEST(SplitVOpTest, TwoDimensional) {
  // Input shape: {4, 3}
  // size_splits: {1, 1, 3}
  // axis: 0
  // We should have 3 outpus with shapes respectively:
  //  output 0 : {1, 3}
  //  output 1 : {1, 3}
  //  output 1 : {2, 3}
  Check(/*axis=*/0, {4, 3}, {3}, {{1, 3}, {1, 3}, {2, 3}},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 1, 2},
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9, 10, 11, 12}});
}

TEST(SplitVOpTest, FourDimensional) {
  Check(/*axis=*/0, {2, 2, 2, 2}, {2}, {{1, 2, 2, 2}, {1, 2, 2, 2}},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 1},
        {
            {1, 2, 3, 4, 5, 6, 7, 8},
            {9, 10, 11, 12, 13, 14, 15, 16},
        });
  Check(/*axis=*/1, {2, 2, 2, 2}, {2}, {{2, 1, 2, 2}, {2, 1, 2, 2}},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, -1},
        {
            {1, 2, 3, 4, 9, 10, 11, 12},
            {5, 6, 7, 8, 13, 14, 15, 16},
        });
  Check(/*axis=*/2, {2, 2, 2, 2}, {2}, {{2, 2, 1, 2}, {2, 2, 1, 2}},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 1},
        {
            {1, 2, 5, 6, 9, 10, 13, 14},
            {3, 4, 7, 8, 11, 12, 15, 16},
        });
  Check(/*axis=*/3, {2, 2, 2, 2}, {2}, {{2, 2, 2, 1}, {2, 2, 2, 1}},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 1},
        {
            {1, 3, 5, 7, 9, 11, 13, 15},
            {2, 4, 6, 8, 10, 12, 14, 16},
        });
}

TEST(SplitVOpTest, OneDimensional) {
  Check(/*axis=*/0, {8}, {8}, {{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}},
        {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 1, 1},
        {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TEST(SplitVOpTest, OneDimensional2) {
  Check(/*axis=*/0, {8}, {8}, {{1}, {1}, {1}, {1}, {1}, {1}, {2}, {0}},
        {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 2, -1},
        {{1}, {2}, {3}, {4}, {5}, {6}, {7, 8}, {}});
}

TEST(SplitVOpTest, NegativeAxis) {
  Check(/*axis=*/-4, {2, 2, 2, 2}, {2}, {{1, 2, 2, 2}, {1, 2, 2, 2}},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 1},
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
