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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class ResizeBilinearOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ResizeBilinearOpModel(const TensorData& input,
                                 std::initializer_list<int> size_data,
                                 const TensorData& output) {
    input_ = AddInput(input);
    size_ = AddConstInput(TensorType_INT32, size_data, {2});
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_RESIZE_BILINEAR,
                 BuiltinOptions_ResizeBilinearOptions,
                 CreateResizeBilinearOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

  int input() { return input_; }

 private:
  int input_;
  int size_;
  int output_;
};

TEST(ResizeBilinearOpTest, HorizontalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 1, 2, 1}, -2.0, 10}, {1, 3},
                          {TensorType_UINT8, {}, -2.0, 10});
  m.SetQuantizedInput({3, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6}, /*max_abs_error=*/1)));
}

TEST(ResizeBilinearOpTest, VerticalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 2, 1, 1}, -2.0, 20}, {3, 1},
                          {TensorType_UINT8, {}, -2.0, 20});
  m.SetQuantizedInput({3, 9});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9}, /*max_abs_error=*/1)));
}

TEST(ResizeBilinearOpTest, ThreeDimensionalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 2, 2, 2}, -2, 30}, {3, 3},
                          {TensorType_UINT8, {}, -2.0, 30.0});
  m.SetQuantizedInput({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                            {
                                                3, 4, 5, 8, 6, 10,       //
                                                7, 9, 10, 12, 11, 14,    //
                                                10, 12, 12, 14, 14, 16,  //
                                            },
                                            /*max_abs_error=*/1)));
}

TEST(ResizeBilinearOpTest, TwoDimensionalResizeWithTwoBatchesUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {2, 2, 2, 1}, -2, 30}, {3, 3},
                          {TensorType_UINT8, {}, -2.0, 30.0});
  m.SetQuantizedInput({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                            {
                                                3, 5, 6,     //
                                                7, 9, 10,    //
                                                9, 11, 12,   //
                                                4, 8, 10,    //
                                                9, 12, 14,   //
                                                12, 14, 16,  //
                                            },
                                            /*max_abs_error=*/1)));
}

}  // namespace tflite
