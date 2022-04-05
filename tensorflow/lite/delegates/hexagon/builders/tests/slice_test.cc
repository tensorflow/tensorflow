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
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename index_type>
class SliceOpModel : public SingleOpModelWithHexagon {
 public:
  SliceOpModel(const TensorData& input, const TensorData& output,
               const TensorData& begin, const TensorData& size,
               std::initializer_list<index_type> begin_data,
               std::initializer_list<index_type> size_data) {
    input_ = AddInput(input);
    begin_ = AddConstInput(begin, begin_data);
    size_ = AddConstInput(size, size_data);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SLICE, BuiltinOptions_SliceOptions,
                 CreateSliceOptions(builder_).Union());
    BuildInterpreter({GetShape(input_), GetShape(begin_), GetShape(size_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int size_;
  int output_;
};

TEST(SliceOpTest, Input_1D_Uint8) {
  SliceOpModel<int> m(/*input=*/{TensorType_UINT8, {4}, -10, 10},
                      /*output=*/{TensorType_UINT8, {2}, -10, 10},
                      {TensorType_INT32, {1}}, {TensorType_INT32, {1}}, {1},
                      {2});
  m.SetInput<uint8_t>({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({2, 3}, 0.1)));
}

TEST(SliceOpTest, Input_2D_Uint8) {
  SliceOpModel<int> m(
      /*input=*/{TensorType_UINT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_UINT8, {1, 2}, -10, 10}, {TensorType_INT32, {2}},
      {TensorType_INT32, {2}}, {1, 0}, {1, 2});
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, SizeInt64_Uint8) {
  SliceOpModel<int64_t> m(/*input=*/{TensorType_UINT8, {4, 1, 1, 1}, -10, 10},
                          /*output=*/{TensorType_UINT8, {3, 1, 1, 1}, -10, 10},
                          {TensorType_INT64, {4}}, {TensorType_INT64, {4}},
                          {1, 0, 0, 0}, {3, 1, 1, 1});
  m.SetInput<uint8_t>({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, SizeMinus1) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_UINT8, {3, 2, 3, 1}, -10, 10},
      /*output=*/{TensorType_UINT8, {2, 1, 3, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 0, 0, 0},
      {2, 1, -1, 1});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis1) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_UINT8, {3, 3, 2, 1}, -10, 10},
      /*output=*/{TensorType_UINT8, {2, 2, 1, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 1, 0, 0},
      {2, -1, 1, 1});
  m.SetInput<uint8_t>({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis2) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_UINT8, {3, 2, 3, 1}, -10, 10},
      /*output=*/{TensorType_UINT8, {2, 1, 2, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 0, 1, 0},
      {2, 1, -1, 1});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis2_Int8) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_INT8, {3, 2, 3, 1}, -10, 10},
      /*output=*/{TensorType_INT8, {2, 1, 2, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 0, 1, 0},
      {2, 1, -1, 1});
  m.SetInput<int8_t>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

}  // namespace tflite
