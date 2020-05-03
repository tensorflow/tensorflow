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
  kConst = 0,
  kDynamic = 1,
};

class ResizeNearestNeighborOpModel : public SingleOpModel {
 public:
  explicit ResizeNearestNeighborOpModel(const TensorData& input,
                                        std::initializer_list<int> size_data,
                                        TestType test_type) {
    bool const_size = (test_type == TestType::kConst);

    input_ = AddInput(input);
    if (const_size) {
      size_ = AddConstInput(TensorType_INT32, size_data, {2});
    } else {
      size_ = AddInput({TensorType_INT32, {2}});
    }
    output_ = AddOutput(input.type);
    SetBuiltinOp(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                 BuiltinOptions_ResizeNearestNeighborOptions,
                 CreateResizeNearestNeighborOptions(builder_).Union());
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

class ResizeNearestNeighborOpTest : public ::testing::TestWithParam<TestType> {
};

TEST_P(ResizeNearestNeighborOpTest, HorizontalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<float>({3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, HorizontalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<uint8>({3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, HorizontalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<int8_t>({-3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({-3, -3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<float>({3, 9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<uint8>({3, 9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<int8_t>({3, -9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 3, -9})));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,   //
                                        3, 3, 6,   //
                                        9, 9, 12,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<uint8>({
      3, 6,  //
      9, 12  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,   //
                                        3, 3, 6,   //
                                        9, 9, 12,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<int8_t>({
      3, -6,  //
      9, 12   //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 3, -6,  //
                                         3, 3, -6,  //
                                         9, 9, 12,  //
                                     })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatches) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        10, 10, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<float>({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 3, 4, 6, 10,     //
                                        3, 4, 3, 4, 6, 10,     //
                                        9, 10, 9, 10, 12, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<uint8>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        12, 12, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<int8_t>({
      3, 6,    //
      9, -12,  //
      -4, 10,  //
      12, 16   //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 3, 6,     //
                                         3, 3, 6,     //
                                         9, 9, -12,   //
                                         -4, -4, 10,  //
                                         -4, -4, 10,  //
                                         12, 12, 16,  //
                                     })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 3, 4, 6, 10,       //
                                        3, 4, 3, 4, 6, 10,       //
                                        10, 12, 10, 12, 14, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<int8_t>({
      3, 4, -6, 10,     //
      10, 12, -14, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 4, 3, 4, -6, 10,       //
                                         3, 4, 3, 4, -6, 10,       //
                                         10, 12, 10, 12, -14, 16,  //
                                     })));
}
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpTest,
                         ResizeNearestNeighborOpTest,
                         testing::Values(TestType::kConst, TestType::kDynamic));

}  // namespace
}  // namespace tflite
