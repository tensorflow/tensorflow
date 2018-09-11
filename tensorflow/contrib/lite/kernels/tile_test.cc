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
#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

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

  void SetInputFloat(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetInputUInt8(std::initializer_list<uint8_t> data) {
    PopulateTensor<uint8_t>(input_, data);
  }

  void SetInputInt32(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(input_, data);
  }

  void SetInputInt64(std::initializer_list<int64_t> data) {
    PopulateTensor<int64_t>(input_, data);
  }

  void SetMultipliers(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(multipliers_, data);
  }

  std::vector<float> GetOutputFloat() { return ExtractVector<float>(output_); }

  std::vector<uint8_t> GetOutputUInt8() { return ExtractVector<uint8_t>(output_); }

  std::vector<int32_t> GetOutputInt32() { return ExtractVector<int32_t>(output_); }

  std::vector<int64_t> GetOutputInt64() {
    return ExtractVector<int64_t>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int multipliers_;
  int output_;
};

TEST(TileTest, Float32Vector) {
  TileOpModel m({3}, TensorType_FLOAT32, TensorType_INT32);
  m.SetInputFloat({1.f, 2.f, 3.f});
  m.SetMultipliers({2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(),
              ElementsAreArray({1.f, 2.f, 3.f, 1.f, 2.f, 3.f}));
}

TEST(TileTest, Float32Matrix) {
  TileOpModel m({2, 3}, TensorType_FLOAT32, TensorType_INT32);
  m.SetInputFloat({
      11.f,
      12.f,
      13.f,
      21.f,
      22.f,
      23.f,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(), ElementsAreArray({
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
  m.SetInputFloat({
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
      m.GetOutputFloat(),
      ElementsAreArray({11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f,
                        21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f,
                        11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f,
                        21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 3}));
}

TEST(TileTest, Uint8Matrix) {
  TileOpModel m({2, 3}, TensorType_UINT8, TensorType_INT32);
  m.SetInputUInt8({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputUInt8(), ElementsAreArray({
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
  m.SetInputInt32({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputInt32(), ElementsAreArray({
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

TEST(TileTest, Int64Matrix) {
  TileOpModel m({2, 3}, TensorType_INT64, TensorType_INT32);
  m.SetInputInt64({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputInt64(), ElementsAreArray({
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
  m.SetInputInt64({
      11,
      12,
      13,
      21,
      22,
      23,
  });
  m.SetMultipliers({2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputInt64(), ElementsAreArray({
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
