/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <initializer_list>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class SplitOpModel : public SingleOpModelWithHexagon {
 public:
  explicit SplitOpModel(const TensorData& input, const TensorData& output,
                        int num_splits, int axis) {
    axis_ = AddConstInput(TensorType_INT32, {axis}, {1});
    input_ = AddInput(input);
    for (int i = 0; i < num_splits; ++i) {
      outputs_.push_back(AddOutput(output));
    }
    SetBuiltinOp(BuiltinOperator_SPLIT, BuiltinOptions_SplitOptions,
                 CreateSplitOptions(builder_, num_splits).Union());
    BuildInterpreter({{}, GetShape(input_)});
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput(int idx) {
    return Dequantize<T>(ExtractVector<T>(outputs_[idx]),
                         GetScale(outputs_[idx]), GetZeroPoint(outputs_[idx]));
  }

  std::vector<int> GetOutputShape(int i) { return GetTensorShape(outputs_[i]); }

 private:
  int input_;
  int axis_;
  std::vector<int> outputs_;
};

template <typename integer_type, TensorType tensor_dtype>
void CheckSplitBehavior(
    int axis, int num_splits, std::initializer_list<int> input_shape,
    std::initializer_list<int> output_shape,
    const std::initializer_list<float>& input_data,
    const std::vector<std::initializer_list<float>>& output_data) {
  auto debug = [&](int i) {
    std::stringstream ss;
    ss << "for output tensor " << i << " axis=" << axis
       << " and num_splits=" << num_splits;
    return ss.str();
  };

  const float kMin = std::min({0.0f, std::min(input_data)});
  const float kMax = std::max(input_data);
  SplitOpModel const_m({tensor_dtype, input_shape, kMin, kMax},
                       {tensor_dtype, output_shape, kMin, kMax}, num_splits,
                       axis);
  const_m.SetInput<integer_type>(input_data);
  const_m.ApplyDelegateAndInvoke();
  for (int i = 0; i < num_splits; ++i) {
    EXPECT_THAT(
        const_m.GetDequantizedOutput<integer_type>(i),
        ElementsAreArray(ArrayFloatNear(output_data[i], /*max_abs_error=*/0.1)))
        << debug(i);
    EXPECT_THAT(const_m.GetOutputShape(i), ElementsAreArray(output_shape))
        << debug(i);
  }
}

template <typename integer_type, TensorType tensor_dtype>
void CheckFourDimSplitImpl() {
  CheckSplitBehavior<integer_type, tensor_dtype>(
      /*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {
          {1, 2, 3, 4, 5, 6, 7, 8},
          {9, 10, 11, 12, 13, 14, 15, 16},
      });
  CheckSplitBehavior<integer_type, tensor_dtype>(
      /*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {
          {1, 2, 3, 4, 9, 10, 11, 12},
          {5, 6, 7, 8, 13, 14, 15, 16},
      });
  CheckSplitBehavior<integer_type, tensor_dtype>(
      /*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {
          {1, 2, 5, 6, 9, 10, 13, 14},
          {3, 4, 7, 8, 11, 12, 15, 16},
      });
  CheckSplitBehavior<integer_type, tensor_dtype>(
      /*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {
          {1, 3, 5, 7, 9, 11, 13, 15},
          {2, 4, 6, 8, 10, 12, 14, 16},
      });
}

TEST(SplitOpModel, CheckFourDimSplitImpl_UInt8) {
  CheckSplitBehavior<uint8_t, TensorType_UINT8>(
      /*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
      {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TEST(SplitOpModel, CheckFourDimSplitImpl_Int8) {
  CheckSplitBehavior<int8_t, TensorType_INT8>(
      /*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
      {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TEST(SplitOpModel, CheckOneDimensionalSplit_UInt8) {
  CheckSplitBehavior<uint8_t, TensorType_UINT8>(
      /*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
      {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TEST(SplitOpModel, CheckOneDimensionalSplit_Int8) {
  CheckSplitBehavior<int8_t, TensorType_INT8>(
      /*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
      {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TEST(SplitOpModel, CheckNegativeOneAxisSplit_UInt8) {
  CheckSplitBehavior<uint8_t, TensorType_UINT8>(
      /*axis=*/-1, /*num_splits=*/2, {2, 2, 2}, {2, 2, 1},
      {1, 2, 3, 4, 5, 6, 7, 8},
      {
          {1, 3, 5, 7},
          {2, 4, 6, 8},
      });
}

TEST(SplitOpModel, CheckNegativeAxisSplit_UInt8) {
  CheckSplitBehavior<uint8_t, TensorType_UINT8>(
      /*axis=*/-4, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {
          {1, 2, 3, 4, 5, 6, 7, 8},
          {9, 10, 11, 12, 13, 14, 15, 16},
      });
}

TEST(SplitOpModel, CheckNegativeAxisSplit_Int8) {
  CheckSplitBehavior<int8_t, TensorType_INT8>(
      /*axis=*/-4, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {
          {1, 2, 3, 4, 5, 6, 7, 8},
          {9, 10, 11, 12, 13, 14, 15, 16},
      });
}

}  // namespace tflite
