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

class ResizeOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ResizeOpModel(BuiltinOperator op_type, const TensorData& input,
                         std::initializer_list<int> size_data,
                         const TensorData& output, bool align_corners = false,
                         bool half_pixel_centers = false) {
    input_ = AddInput(input);
    size_ = AddConstInput(TensorType_INT32, size_data, {2});
    output_ = AddOutput(output);
    if (op_type == BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
      SetBuiltinOp(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                   BuiltinOptions_ResizeNearestNeighborOptions,
                   CreateResizeNearestNeighborOptions(
                       builder_, /*align_corners*/ align_corners,
                       /*half_pixel_centers*/ half_pixel_centers)
                       .Union());
    } else {
      SetBuiltinOp(op_type, BuiltinOptions_ResizeBilinearOptions,
                   CreateResizeBilinearOptions(
                       builder_, /**align_corners**/ align_corners,
                       /**half_pixel_centers**/ half_pixel_centers)
                       .Union());
    }
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

  template <typename T>
  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  int input() { return input_; }

 private:
  int input_;
  int size_;
  int output_;
};

TEST(ResizeOpModel, HorizontalResizeBiliear_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {1, 1, 2, 1}, -2.0, 10}, {1, 3},
                  {TensorType_UINT8, {}, -2.0, 10});
  m.SetQuantizedInput<uint8_t>({3, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6}, /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, HorizontalResizeNearestNeighbor_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_INT8, {1, 1, 2, 1}, -2.0, 10}, {1, 3},
                  {TensorType_INT8, {}, -2.0, 10});
  m.SetQuantizedInput<int8_t>({3, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3.01176, 3.01176, 6.02353},
                                              /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, VerticalResizeBiliear_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_INT8, {1, 2, 1, 1}, -2.0, 20}, {3, 1},
                  {TensorType_INT8, {}, -2.0, 20});
  m.SetQuantizedInput<int8_t>({3, 9});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9}, /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, VerticalResizeNearestNeighbor_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {1, 2, 1, 1}, -2.0, 20}, {3, 1},
                  {TensorType_UINT8, {}, -2.0, 20});
  m.SetQuantizedInput<uint8_t>({3, 9});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3.01961, 3.01961, 8.97255},
                                              /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeBiliear_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {1, 2, 2, 2}, -2, 30}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<uint8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      3, 4, 5, 8, 6, 10,       //
                      7, 9, 10, 12, 11, 14,    //
                      10, 12, 12, 14, 14, 16,  //
                  },
                  /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeNearestNeighbor_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_INT8, {1, 2, 2, 2}, -2, 30}, {3, 3},
                  {TensorType_INT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<int8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      3.01177, 4.01569, 3.01177, 4.01569, 6.02353, 10.0392,  //
                      3.01177, 4.01569, 3.01177, 4.01569, 6.02353, 10.0392,  //
                      10.0392, 12.0471, 10.0392, 12.0471, 14.0549, 16.0627,  //
                  },
                  /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, TwoDimensionalResizeBilinearWithTwoBatches_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_INT8, {2, 2, 2, 1}, -2, 30}, {3, 3},
                  {TensorType_INT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<int8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
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

TEST(ResizeOpModel, TwoDimensionalResizeNNWithTwoBatches_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {2, 2, 2, 1}, -2, 30}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<uint8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      3.01177, 3.01177, 6.02353,  //
                      3.01177, 3.01177, 6.02353,  //
                      9.03529, 9.03529, 12.0471,  //
                      4.01569, 4.01569, 10.0392,  //
                      4.01569, 4.01569, 10.0392,  //
                      12.0471, 12.0471, 16.0627,  //
                  },
                  /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, TwoDimResizeBilinearWithTwoBatches_HalfPixelCenters_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {2, 2, 2, 1}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ false,
                  /**half_pixel_centers**/ true);
  m.SetQuantizedInput<uint8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({2, 4, 6,    //
                                               6, 7, 9,    //
                                               9, 10, 12,  //
                                               4, 7, 10,   //
                                               8, 10, 13,  //
                                               12, 14, 16},
                                              /*max_abs_error=*/2)));
}

TEST(ResizeOpModel, TwoDimResizeBilinearWithTwoBatches_AlignCorners_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {2, 2, 2, 1}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ true,
                  /**half_pixel_centers**/ false);
  m.SetQuantizedInput<uint8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6,    //
                                               7, 9, 10,   //
                                               9, 11, 12,  //
                                               4, 8, 10,   //
                                               9, 12, 13,  //
                                               12, 15, 16},
                                              /*max_abs_error=*/2)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeNN_AlignCorners_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {1, 2, 2, 2}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ true);
  m.SetQuantizedInput<uint8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 4, 6, 10, 6, 10,      //
                                               10, 12, 14, 16, 14, 16,  //
                                               10, 12, 14, 16, 14, 16},
                                              /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeNN_HalfPixelCenters_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {1, 2, 2, 2}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ false,
                  /**half_pixel_centers**/ true);
  m.SetQuantizedInput<uint8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 4, 6, 10, 6, 10,      //
                                               10, 12, 14, 16, 14, 16,  //
                                               10, 12, 14, 16, 14, 16},
                                              /*max_abs_error=*/1)));
}

}  // namespace tflite
