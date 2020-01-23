/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
using uint8 = std::uint8_t;

enum class TestType {
  CONST = 0,
  DYNAMIC = 1,
};

class ResizeBilinearOpModel : public SingleOpModel {
 public:
  explicit ResizeBilinearOpModel(const TensorData& input,
                                 std::initializer_list<int> size_data,
                                 TestType test_type,
                                 bool half_pixel_centers = false) {
    bool const_size = (test_type == TestType::CONST);

    input_ = AddInput(input);
    if (const_size) {
      size_ = AddConstInput(TensorType_INT32, size_data, {2});
    } else {
      size_ = AddInput({TensorType_INT32, {2}});
    }
    output_ = AddOutput(input.type);
    SetBuiltinOp(
        BuiltinOperator_RESIZE_BILINEAR, BuiltinOptions_ResizeBilinearOptions,
        CreateResizeBilinearOptions(builder_, /**align_corners**/ false,
                                    /**half_pixel_centers**/ half_pixel_centers)
            .Union());
    if (const_size) {
      BuildInterpreter({GetShape(input_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(size_)});
      PopulateTensor(size_, size_data);
    }
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int size_;
  int output_;
};

class ResizeBilinearOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(ResizeBilinearOpTest, HorizontalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 1, 2, 1}}, {1, 3},
                          GetParam());
  m.SetInput<float>({3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6})));
}

TEST_P(ResizeBilinearOpTest, HorizontalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 1, 2, 1}}, {1, 3}, GetParam());
  m.SetInput<uint8>({3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6})));
}

TEST_P(ResizeBilinearOpTest, HorizontalResizeInt8) {
  ResizeBilinearOpModel m({TensorType_INT8, {1, 1, 2, 1}}, {1, 3}, GetParam());
  m.SetInput<int8_t>({3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6})));
}

TEST_P(ResizeBilinearOpTest, VerticalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 1, 1}}, {3, 1},
                          GetParam());
  m.SetInput<float>({3, 9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9})));
}

TEST_P(ResizeBilinearOpTest, VerticalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 2, 1, 1}}, {3, 1}, GetParam());
  m.SetInput<uint8>({3, 9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9})));
}

TEST_P(ResizeBilinearOpTest, VerticalResizeInt8) {
  ResizeBilinearOpModel m({TensorType_INT8, {1, 2, 1, 1}}, {3, 1}, GetParam());
  m.SetInput<int8_t>({3, 9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9})));
}

TEST_P(ResizeBilinearOpTest, TwoDimensionalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {3, 3},
                          GetParam());
  m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 5, 6,    //
                                        7, 9, 10,   //
                                        9, 11, 12,  //
                                    })));
}

TEST_P(ResizeBilinearOpTest, TwoDimensionalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 2, 2, 1}}, {3, 3}, GetParam());
  m.SetInput<uint8>({
      3, 6,  //
      9, 12  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 5, 6,    //
                                        7, 9, 10,   //
                                        9, 11, 12,  //
                                    })));
}

TEST_P(ResizeBilinearOpTest, TwoDimensionalResizeInt8) {
  ResizeBilinearOpModel m({TensorType_INT8, {1, 2, 2, 1}}, {3, 3}, GetParam());
  m.SetInput<int8_t>({
      3, 6,  //
      9, 12  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 5, 6,    //
                                         7, 9, 10,   //
                                         9, 11, 12,  //
                                     })));
}

TEST_P(ResizeBilinearOpTest, TwoDimensionalResizeWithTwoBatches) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3},
                          GetParam());
  m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 5, 6,     //
                                        7, 9, 10,    //
                                        9, 11, 12,   //
                                        4, 8, 10,    //
                                        8, 12, 14,   //
                                        10, 14, 16,  //
                                    })));
}

TEST_P(ResizeBilinearOpTest,
       TwoDimensionalResizeWithTwoBatches_HalfPixelCenters) {
  // TODO(b/147696142): Update when NNAPI delegate can support TF2 behavior.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ResizeBilinearOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3},
                          GetParam(), /**half_pixel_centers**/ true);
  m.SetInput<float>({
      1, 2,  //
      3, 4,  //
      1, 2,  //
      3, 4   //
  });
  m.Invoke();
  // clang-format off
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
    1, 1.5, 2,  //
    2, 2.5, 3,  //
    3, 3.5, 4,  //
    1, 1.5, 2,  //
    2, 2.5, 3,  //
    3, 3.5, 4,  //
  })));
  // clang-format on
}

TEST_P(ResizeBilinearOpTest, ThreeDimensionalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 2, 2}}, {3, 3},
                          GetParam());
  m.SetInput<float>({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 5, 8, 6, 10,      //
                                        7, 8, 9, 12, 10, 14,    //
                                        9, 10, 11, 14, 12, 16,  //
                                    })));
}

TEST_P(ResizeBilinearOpTest, TwoDimensionalResizeWithTwoBatchesUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {2, 2, 2, 1}}, {3, 3}, GetParam());
  m.SetInput<uint8>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear(
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

TEST_P(ResizeBilinearOpTest,
       TwoDimensionalResizeWithTwoBatchesUInt8_HalfPixelCenters) {
  // TODO(b/147696142): Update when NNAPI delegate can support TF2 behavior.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ResizeBilinearOpModel m({TensorType_UINT8, {2, 2, 2, 1}}, {3, 3}, GetParam(),
                          /**half_pixel_centers**/ true);
  m.SetInput<uint8>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear(
                                        {
                                            2, 4, 6,     //
                                            6, 7, 9,     //
                                            9, 10, 12,   //
                                            4, 7, 10,    //
                                            8, 10, 13,   //
                                            12, 14, 16,  //
                                        },
                                        /*max_abs_error=*/1)));
}

TEST_P(ResizeBilinearOpTest, TwoDimensionalResizeWithTwoBatchesInt8) {
  ResizeBilinearOpModel m({TensorType_INT8, {2, 2, 2, 1}}, {3, 3}, GetParam());
  m.SetInput<int8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                         {
                                             3, 5, 6,     //
                                             7, 9, 10,    //
                                             9, 11, 12,   //
                                             4, 8, 10,    //
                                             9, 12, 13,   //
                                             12, 14, 16,  //
                                         },
                                         /*max_abs_error=*/1)));
}

TEST_P(ResizeBilinearOpTest, ThreeDimensionalResizeUInt8) {
  ResizeBilinearOpModel m({TensorType_UINT8, {1, 2, 2, 2}}, {3, 3}, GetParam());
  m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear(
                                        {
                                            3, 4, 5, 8, 6, 10,       //
                                            7, 9, 10, 12, 11, 14,    //
                                            10, 12, 12, 14, 14, 16,  //
                                        },
                                        /*max_abs_error=*/1)));
}

TEST_P(ResizeBilinearOpTest, ThreeDimensionalResizeInt8) {
  ResizeBilinearOpModel m({TensorType_INT8, {1, 2, 2, 2}}, {3, 3}, GetParam());
  m.SetInput<int8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                         {
                                             3, 4, 5, 8, 6, 10,       //
                                             7, 9, 10, 12, 11, 13,    //
                                             10, 12, 12, 14, 14, 16,  //
                                         },
                                         /*max_abs_error=*/1)));
}

INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpTest, ResizeBilinearOpTest,
                         testing::Values(TestType::CONST, TestType::DYNAMIC));

}  // namespace
}  // namespace tflite
