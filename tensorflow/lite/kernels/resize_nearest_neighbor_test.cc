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

class ResizeNearestNeighborOpModel : public SingleOpModel {
 public:
  explicit ResizeNearestNeighborOpModel(
      const TensorData& input, std::initializer_list<int> size_data = {}) {
    bool const_size = size_data.size() != 0;
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
    }
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }
  void SetSize(std::initializer_list<int> data) { PopulateTensor(size_, data); }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int size_;
  int output_;
};

TEST(ResizeNearestNeighborOpTest, HorizontalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 1, 2, 1}}, {});
  m.SetInput<float>({3, 6});
  m.SetSize({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));

  ResizeNearestNeighborOpModel const_m({TensorType_FLOAT32, {1, 1, 2, 1}},
                                       {1, 3});
  const_m.SetInput<float>({3, 6});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));
}

TEST(ResizeNearestNeighborOpTest, HorizontalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 1, 2, 1}}, {});
  m.SetInput<uint8>({3, 6});
  m.SetSize({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));

  ResizeNearestNeighborOpModel const_m({TensorType_UINT8, {1, 1, 2, 1}},
                                       {1, 3});
  const_m.SetInput<uint8>({3, 6});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));
}

TEST(ResizeNearestNeighborOpTest, HorizontalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 1, 2, 1}}, {});
  m.SetInput<int8_t>({-3, 6});
  m.SetSize({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({-3, -3, 6})));

  ResizeNearestNeighborOpModel const_m({TensorType_INT8, {1, 1, 2, 1}}, {1, 3});
  const_m.SetInput<int8_t>({-3, 6});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({-3, -3, 6})));
}

TEST(ResizeNearestNeighborOpTest, VerticalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 1, 1}}, {});
  m.SetInput<float>({3, 9});
  m.SetSize({3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));

  ResizeNearestNeighborOpModel const_m({TensorType_FLOAT32, {1, 2, 1, 1}},
                                       {3, 1});
  const_m.SetInput<float>({3, 9});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));
}

TEST(ResizeNearestNeighborOpTest, VerticalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 1, 1}}, {});
  m.SetInput<uint8>({3, 9});
  m.SetSize({3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));

  ResizeNearestNeighborOpModel const_m({TensorType_UINT8, {1, 2, 1, 1}},
                                       {3, 1});
  const_m.SetInput<uint8>({3, 9});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));
}

TEST(ResizeNearestNeighborOpTest, VerticalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 1, 1}}, {});
  m.SetInput<int8_t>({3, -9});
  m.SetSize({3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 3, -9})));

  ResizeNearestNeighborOpModel const_m({TensorType_INT8, {1, 2, 1, 1}}, {3, 1});
  const_m.SetInput<int8_t>({3, -9});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 3, -9})));
}

TEST(ResizeNearestNeighborOpTest, TwoDimensionalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {});
  m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,   //
                                        3, 3, 6,   //
                                        9, 9, 12,  //
                                    })));

  ResizeNearestNeighborOpModel const_m({TensorType_FLOAT32, {1, 2, 2, 1}},
                                       {3, 3});
  const_m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                              3, 3, 6,   //
                                              3, 3, 6,   //
                                              9, 9, 12,  //
                                          })));
}

TEST(ResizeNearestNeighborOpTest, TwoDimensionalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 1}}, {});
  m.SetInput<uint8>({
      3, 6,  //
      9, 12  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,   //
                                        3, 3, 6,   //
                                        9, 9, 12,  //
                                    })));

  ResizeNearestNeighborOpModel const_m({TensorType_UINT8, {1, 2, 2, 1}},
                                       {3, 3});
  const_m.SetInput<uint8>({
      3, 6,  //
      9, 12  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                              3, 3, 6,   //
                                              3, 3, 6,   //
                                              9, 9, 12,  //
                                          })));
}

TEST(ResizeNearestNeighborOpTest, TwoDimensionalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 2, 1}}, {});
  m.SetInput<int8_t>({
      3, -6,  //
      9, 12   //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 3, -6,  //
                                         3, 3, -6,  //
                                         9, 9, 12,  //
                                     })));

  ResizeNearestNeighborOpModel const_m({TensorType_INT8, {1, 2, 2, 1}}, {3, 3});
  const_m.SetInput<int8_t>({
      3, -6,  //
      9, 12   //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                               3, 3, -6,  //
                                               3, 3, -6,  //
                                               9, 9, 12,  //
                                           })));
}

TEST(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatches) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {});
  m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        10, 10, 16,  //
                                    })));

  ResizeNearestNeighborOpModel const_m({TensorType_FLOAT32, {2, 2, 2, 1}},
                                       {3, 3});
  const_m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                              3, 3, 6,     //
                                              3, 3, 6,     //
                                              9, 9, 12,    //
                                              4, 4, 10,    //
                                              4, 4, 10,    //
                                              10, 10, 16,  //
                                          })));
}

TEST(ResizeNearestNeighborOpTest, ThreeDimensionalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 2, 2}}, {});
  m.SetInput<float>({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 3, 4, 6, 10,     //
                                        3, 4, 3, 4, 6, 10,     //
                                        9, 10, 9, 10, 12, 16,  //
                                    })));

  ResizeNearestNeighborOpModel const_m({TensorType_FLOAT32, {1, 2, 2, 2}},
                                       {3, 3});
  const_m.SetInput<float>({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                              3, 4, 3, 4, 6, 10,     //
                                              3, 4, 3, 4, 6, 10,     //
                                              9, 10, 9, 10, 12, 16,  //
                                          })));
}

TEST(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {2, 2, 2, 1}}, {});
  m.SetInput<uint8>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        12, 12, 16,  //
                                    })));

  ResizeNearestNeighborOpModel const_m({TensorType_UINT8, {2, 2, 2, 1}},
                                       {3, 3});
  const_m.SetInput<uint8>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                              3, 3, 6,     //
                                              3, 3, 6,     //
                                              9, 9, 12,    //
                                              4, 4, 10,    //
                                              4, 4, 10,    //
                                              12, 12, 16,  //
                                          })));
}

TEST(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {2, 2, 2, 1}}, {});
  m.SetInput<int8_t>({
      3, 6,    //
      9, -12,  //
      -4, 10,  //
      12, 16   //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 3, 6,     //
                                         3, 3, 6,     //
                                         9, 9, -12,   //
                                         -4, -4, 10,  //
                                         -4, -4, 10,  //
                                         12, 12, 16,  //
                                     })));

  ResizeNearestNeighborOpModel const_m({TensorType_INT8, {2, 2, 2, 1}}, {3, 3});
  const_m.SetInput<int8_t>({
      3, 6,    //
      9, -12,  //
      -4, 10,  //
      12, 16   //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                               3, 3, 6,     //
                                               3, 3, 6,     //
                                               9, 9, -12,   //
                                               -4, -4, 10,  //
                                               -4, -4, 10,  //
                                               12, 12, 16,  //
                                           })));
}

TEST(ResizeNearestNeighborOpTest, ThreeDimensionalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 2}}, {});
  m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 3, 4, 6, 10,       //
                                        3, 4, 3, 4, 6, 10,       //
                                        10, 12, 10, 12, 14, 16,  //
                                    })));

  ResizeNearestNeighborOpModel const_m({TensorType_UINT8, {1, 2, 2, 2}},
                                       {3, 3});
  const_m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                              3, 4, 3, 4, 6, 10,       //
                                              3, 4, 3, 4, 6, 10,       //
                                              10, 12, 10, 12, 14, 16,  //
                                          })));
}

TEST(ResizeNearestNeighborOpTest, ThreeDimensionalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 2, 2}}, {});
  m.SetInput<int8_t>({
      3, 4, -6, 10,     //
      10, 12, -14, 16,  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 4, 3, 4, -6, 10,       //
                                         3, 4, 3, 4, -6, 10,       //
                                         10, 12, 10, 12, -14, 16,  //
                                     })));

  ResizeNearestNeighborOpModel const_m({TensorType_INT8, {1, 2, 2, 2}}, {3, 3});
  const_m.SetInput<int8_t>({
      3, 4, -6, 10,     //
      10, 12, -14, 16,  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                               3, 4, 3, 4, -6, 10,       //
                                               3, 4, 3, 4, -6, 10,       //
                                               10, 12, 10, 12, -14, 16,  //
                                           })));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
