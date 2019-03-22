/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

class BaseSqueezeOpModel : public SingleOpModel {
 public:
  BaseSqueezeOpModel(const TensorData& input, const TensorData& output,
                     std::initializer_list<int> axis) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_SQUEEZE, BuiltinOptions_SqueezeOptions,
        CreateSqueezeOptions(builder_, builder_.CreateVector<int>(axis))
            .Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

 protected:
  int input_;
  int output_;
};

class SqueezeOpModel : public BaseSqueezeOpModel {
 public:
  using BaseSqueezeOpModel::BaseSqueezeOpModel;

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

template <typename T>
void Check(std::initializer_list<int> input_shape,
           std::initializer_list<int> output_shape,
           std::initializer_list<int> exp_output_shape,
           std::initializer_list<int> axis,
           const std::initializer_list<T>& input_data,
           const TensorType& type = TensorType_FLOAT32) {
  SqueezeOpModel m({type, input_shape}, {type, output_shape}, axis);
  m.SetInput(input_data);
  m.Invoke();

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
  EXPECT_THAT(m.GetOutput<T>(), ElementsAreArray(input_data));
}

TEST(SqueezeOpTest, SqueezeAll) {
  Check<float>(
      {1, 24, 1}, {24}, {24}, {},
      {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0});
  Check<int8_t>({1, 24, 1}, {24}, {24}, {},
                {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                TensorType_INT8);
  Check<int32_t>({1, 24, 1}, {24}, {24}, {},
                 {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                 TensorType_INT32);
}

TEST(SqueezeOpTest, SqueezeSelectedAxis) {
  Check<float>(
      {1, 24, 1}, {24}, {1, 24}, {2},
      {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0});
  Check<int8_t>({1, 24, 1}, {24}, {1, 24}, {2},
                {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                TensorType_INT8);
  Check<int32_t>({1, 24, 1}, {24}, {1, 24}, {2},
                 {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                 TensorType_INT32);
}

TEST(SqueezeOpTest, SqueezeNegativeAxis) {
  Check<float>(
      {1, 24, 1}, {24}, {24}, {-1, 0},
      {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0});
  Check<int8_t>({1, 24, 1}, {24}, {24}, {-1, 0},
                {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                TensorType_INT8);
  Check<int32_t>({1, 24, 1}, {24}, {24}, {-1, 0},
                 {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                 TensorType_INT32);
}

TEST(SqueezeOpTest, SqueezeAllDims) {
  Check<float>({1, 1, 1, 1, 1, 1, 1}, {1}, {}, {}, {3.85});
  Check<int8_t>({1, 1, 1, 1, 1, 1, 1}, {1}, {}, {}, {3}, TensorType_INT8);
  Check<int32_t>({1, 1, 1, 1, 1, 1, 1}, {1}, {}, {}, {3}, TensorType_INT32);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
