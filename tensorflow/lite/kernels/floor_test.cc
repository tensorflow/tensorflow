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

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class FloorOpModel : public SingleOpModel {
 public:
  FloorOpModel(std::initializer_list<int> input_shape, TensorType input_type) {
    input_ = AddInput(input_type);
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_FLOOR, BuiltinOptions_NONE, 0);
    BuildInterpreter({
        input_shape,
    });
  }

  int input() { return input_; }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(FloorOpTest, SingleDim) {
  FloorOpModel model({2}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {8.5, 0.0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray({8, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(FloorOpTest, MultiDims) {
  FloorOpModel model({2, 1, 1, 5}, TensorType_FLOAT32);
  std::vector<float> input;
  if (AllowFp16PrecisionForFp32()) {
    input = {
        0.01, 8.01, 0.99, 9.99, 0.5, -0.01, -8.01, -0.99, -9.99, -0.5,
    };
  } else {
    input = {
        0.0001,  8.0001,  0.9999,  9.9999,  0.5,
        -0.0001, -8.0001, -0.9999, -9.9999, -0.5,
    };
  }
  model.PopulateTensor<float>(model.input(), input);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<float>(),
              ElementsAreArray({0, 8, 0, 9, 0, -1, -9, -1, -10, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 5}));
}

TEST(FloorOpTest, SingleDimFloat16) {
  FloorOpModel model({2}, TensorType_FLOAT16);
  model.PopulateTensor<>(model.input(), {Eigen::half(8.5), Eigen::half(0.0)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<Eigen::half>(), ElementsAreArray({8, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(FloorOpTest, MultiDimsFloat16) {
  FloorOpModel model({2, 1, 1, 5}, TensorType_FLOAT16);
  model.PopulateTensor<Eigen::half>(model.input(), {
                                                       Eigen::half(0.75),
                                                       Eigen::half(8.25),
                                                       Eigen::half(0.49),
                                                       Eigen::half(9.99),
                                                       Eigen::half(0.5),
                                                       Eigen::half(-0.25),
                                                       Eigen::half(-8.75),
                                                       Eigen::half(-0.99),
                                                       Eigen::half(-9.49),
                                                       Eigen::half(-0.5),
                                                   });
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<Eigen::half>(),
              ElementsAreArray({0, 8, 0, 9, 0, -1, -9, -1, -10, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 5}));
}

TEST(FloorOpTest, SingleDimBFloat16) {
  FloorOpModel model({2}, TensorType_BFLOAT16);
  model.PopulateTensor<>(model.input(),
                         {Eigen::bfloat16(8.5), Eigen::bfloat16(0.0)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<Eigen::bfloat16>(), ElementsAreArray({8, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(FloorOpTest, MultiDimsBFloat16) {
  FloorOpModel model({2, 1, 1, 5}, TensorType_BFLOAT16);
  model.PopulateTensor<Eigen::bfloat16>(model.input(),
                                        {
                                            Eigen::bfloat16(1.75),
                                            Eigen::bfloat16(8.5),
                                            Eigen::bfloat16(1.49),
                                            Eigen::bfloat16(9.01),
                                            Eigen::bfloat16(1.5),
                                            Eigen::bfloat16(-1.25),
                                            Eigen::bfloat16(-8.99),
                                            Eigen::bfloat16(-1.99),
                                            Eigen::bfloat16(-9.5),
                                            Eigen::bfloat16(-1.5),
                                        });
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<Eigen::bfloat16>(),
              ElementsAreArray({1, 8, 1, 9, 1, -2, -9, -2, -10, -2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 5}));
}

}  // namespace
}  // namespace tflite
