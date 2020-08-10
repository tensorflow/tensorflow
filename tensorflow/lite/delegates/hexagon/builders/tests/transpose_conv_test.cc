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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename InputType>
class QuantizedTransposeConvOpModel : public SingleOpModelWithHexagon {
 public:
  QuantizedTransposeConvOpModel(std::initializer_list<int> output_shape_data,
                                const TensorData& filter,
                                std::initializer_list<InputType> filter_data,
                                const TensorData& input,
                                const TensorData& output, Padding padding,
                                int stride_w, int stride_h) {
    // Just to be confusing, transpose_conv has an _input_ named "output_shape"
    // that sets the shape of the output tensor of the op :). It must always be
    // an int32 1D four element tensor.
    output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {4});
    filter_ = AddConstInput(filter, filter_data);
    input_ = AddInput(input);

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, padding, stride_w, stride_h)
            .Union());
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)});
  }

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<InputType>(input_, data);
  }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<InputType>(ExtractVector<InputType>(output_),
                                 GetScale(output_), GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

TEST(QuantizedTransposeConvOpModel, SimpleTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  auto model = QuantizedTransposeConvOpModel<uint8_t>(
      {1, 4, 4, 1}, {TensorType_UINT8, {1, 3, 3, 1}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({28, 64, 84, 76, 100, 192, 236, 200, 208,
                                       372, 416, 332, 264, 448, 484, 364},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(QuantizedTransposeConvOpModel, PaddingValidTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  // 18}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137, 139,
                                                141, 143, 145, 147, 149, 151,
                                                153, 155, 157, 159, 161, 163};
  auto model = QuantizedTransposeConvOpModel<uint8_t>(
      {1, 6, 6, 1}, {TensorType_UINT8, {1, 3, 3, 2}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 2}, -63.5, 64},
      {TensorType_UINT8, {}, -4064, 4096}, Padding_VALID, 1, 1);
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0,    32,   64,   96,   128,  96,   64,   192,  416,
                   576,  544,  352,  224,  672,  1344, 1696, 1440, 864,
                   608,  1504, 2720, 3072, 2432, 1440, 864,  1984, 3360,
                   3648, 2752, 1536, 704,  1536, 2528, 2720, 2016, 1088},
                  1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 6, 1}));
}

TEST(QuantizedTransposeConvOpModel, TwoFiltersTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  // 18}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137, 139,
                                                141, 143, 145, 147, 149, 151,
                                                153, 155, 157, 159, 161, 163};
  auto model = QuantizedTransposeConvOpModel<uint8_t>(
      {1, 4, 4, 1}, {TensorType_UINT8, {1, 3, 3, 2}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 2}, -63.5, 64},
      {TensorType_UINT8, {}, -4064, 4096}, Padding_SAME, 1, 1);
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {192, 416, 576, 544, 672, 1344, 1696, 1440, 1504, 2720, 3072,
                   2432, 1984, 3360, 3648, 2752},
                  1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(QuantizedTransposeConvOpModel,
     SimpleTestQuantizedPerChannelSingleChannel) {
  const std::initializer_list<int8_t> filter_data = {14, 28, 42,  56, 71,
                                                     85, 99, 113, 127};
  auto model = QuantizedTransposeConvOpModel<int8_t>(
      {1, 4, 4, 1},
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      filter_data, {TensorType_INT8, {1, 4, 4, 1}, 0, 0, 16.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 2, -128}, Padding_SAME, 1, 1);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({28, 62, 82, 76, 98, 192, 236, 198, 206,
                                       372, 416, 330, 262, 446, 486, 366},
                                      1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(QuantizedTransposeConvOpModel, TestQuantizedPerChannelMultiChannel) {
  const std::initializer_list<int8_t> filter_data = {
      7,  22, 37, 52, 67, 82, 97, 112, 127,
      14, 28, 42, 56, 71, 85, 99, 113, 127};
  auto model = QuantizedTransposeConvOpModel<int8_t>(
      {1, 5, 5, 2},
      {TensorType_INT8,
       {2, 3, 3, 1},
       0,
       0,
       0,
       0,
       true,
       {17.0 / 127, 18.0 / 127},
       {0, 0},
       0},
      filter_data, {TensorType_INT8, {1, 2, 2, 1}, 0, 0, 4.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 1, -128}, Padding_VALID, 2, 2);
  model.SetInput({1, 2, 3, 4});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {1,  2,  3,  4,  7,  10, 6,  8,  10, 12, 7,   8,   9,  10, 25, 28, 18,
           20, 22, 24, 16, 20, 24, 28, 62, 72, 42, 48,  54,  60, 21, 24, 27, 30,
           61, 68, 36, 40, 44, 48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72},
          1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

}  // namespace tflite
