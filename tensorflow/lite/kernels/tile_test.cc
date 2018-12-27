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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
class TileOpModel : public SingleOpModel {
 public:
  TileOpModel(std::initializer_list<int> input_shape, TensorType input_type,
              TensorType multiply_type) {
    input_ = AddInput(input_type);
    multipliers_ = AddInput(TensorType_INT32);
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions, 0);
    BuildInterpreter({input_shape, {static_cast<int>(input_shape.size())}});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  void SetMultipliers(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(multipliers_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int multipliers_;
  int output_;
};

TEST(TileTest, Float32Vector) {
  TileOpModel m({3}, TensorType_FLOAT32, TensorType_INT32);
  m.SetInput<float>({1.f, 2.f, 3.f});
  m.SetMultipliers({2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({1.f, 2.f, 3.f, 1.f, 2.f, 3.f}));
}

TEST(TileTest, Float32Matrix) {
  TileOpModel m({2, 3}, TensorType_FLOAT32, TensorType_INT32);
  m.SetInput<float>({
      11.f,
      12.f,
      13.f,
      21.f,
      22.f,
      23.f,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({
                                        11.f,
                                        12.f,
                                        13.f,
                                        21.f,
                                        22.f,
                                        23.f,
                                        11.f,
                                        12.f,
                                        13.f,
                                        21.f,
                                        22.f,
                                        23.f,
                                    }));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
}

TEST(TileTest, Float32HighDimension) {
  TileOpModel m({1, 2, 3}, TensorType_FLOAT32, TensorType_INT32);
  m.SetInput<float>({
      11.f,
      12.f,
      13.f,
      21.f,
      22.f,
      23.f,
  });
  m.SetMultipliers({2, 3, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray({11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f,
                        21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f,
                        11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f,
                        21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 3}));
}

TEST(TileTest, Uint8Matrix) {
  TileOpModel m({2, 3}, TensorType_UINT8, TensorType_INT32);
  m.SetInput<uint8_t>({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                      }));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
}

TEST(TileTest, Int32Matrix) {
  TileOpModel m({2, 3}, TensorType_INT32, TensorType_INT32);
  m.SetInput<int32_t>({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                      }));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
}

TEST(TileTest, BooleanMatrix) {
  TileOpModel m({2, 3}, TensorType_BOOL, TensorType_INT32);
  m.SetInput<bool>({true, false, false, true, true, false});
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<bool>(),
              ElementsAreArray({
                  true, false, false, true, true, false,  // first tiletrue,
                  true, false, false, true, true, false   // second tile
              }));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
}

TEST(TileTest, Int64Matrix) {
  TileOpModel m({2, 3}, TensorType_INT64, TensorType_INT32);
  m.SetInput<int64_t>({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                      }));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
}

TEST(TileTest, Int64Matrix64Multipliers) {
  TileOpModel m({2, 3}, TensorType_INT64, TensorType_INT64);
  m.SetInput<int64_t>({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                          11,
                                          12,
                                          13,
                                          21,
                                          22,
                                          23,
                                      }));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 3}));
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
