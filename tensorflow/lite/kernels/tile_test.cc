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

template <typename T>
void Check(std::initializer_list<int> input_shape,
           std::initializer_list<T> input_data,
           std::initializer_list<int> multipliers_data,
           std::initializer_list<int> exp_output_shape,
           std::initializer_list<T> exp_output_data, TensorType input_type,
           TensorType multiply_type) {
  TileOpModel m(input_shape, input_type, multiply_type);
  m.SetInput(input_data);
  m.SetMultipliers(multipliers_data);
  m.Invoke();

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
  EXPECT_THAT(m.GetOutput<T>(), ElementsAreArray(exp_output_data));
}

TEST(TileTest, Float32Vector) {
  Check<float>(/*input_shape=*/{3}, /*input_data=*/{1.0, 2.0, 3.0},
               /*multipliers_data=*/{2}, /*exp_output_shape=*/{6},
               /*exp_output_data=*/{1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
               /*input_type=*/TensorType_FLOAT32,
               /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, Float32Matrix) {
  Check<float>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*input_type=*/TensorType_FLOAT32,
      /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, Float32HighDimension) {
  Check<float>(
      /*input_shape=*/{1, 2, 3},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multipliers_data=*/{2, 3, 1}, /*exp_output_shape=*/{2, 6, 3},
      /*exp_output_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f,
                           13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f,
                           22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f,
                           11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f,
                           13.f, 21.f, 22.f, 23.f},
      /*input_type=*/TensorType_FLOAT32,
      /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, Uint8Matrix) {
  Check<uint8_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_UINT8,
      /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, Int32Matrix) {
  Check<int32_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT32,
      /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, BooleanMatrix) {
  Check<bool>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{true, false, false, true, true, false},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {true, false, false, true, true, false, true, false, false, true, true,
       false},
      /*input_type=*/TensorType_BOOL,
      /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, Int64Matrix) {
  Check<int64_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT64,
      /*multiply_type=*/TensorType_INT32);
}

TEST(TileTest, Int64Matrix64Multipliers) {
  Check<int64_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT64,
      /*multiply_type=*/TensorType_INT64);
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
