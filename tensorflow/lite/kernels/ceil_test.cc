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

class CeilOpModel : public SingleOpModel {
 public:
  CeilOpModel(std::initializer_list<int> input_shape, TensorType input_type) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_CEIL, BuiltinOptions_NONE, 0);
    BuildInterpreter({
        input_shape,
    });
  }

  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(CeilOpTest, SingleDim) {
  CeilOpModel model({2}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {8.5, 0.0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({9, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(CeilOpTest, MultiDims) {
  CeilOpModel model({2, 1, 1, 5}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {
                                                 0.0001,
                                                 8.0001,
                                                 0.9999,
                                                 9.9999,
                                                 0.5,
                                                 -0.0001,
                                                 -8.0001,
                                                 -0.9999,
                                                 -9.9999,
                                                 -0.5,
                                             });
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 9, 1, 10, 1, 0, -8, 0, -9, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 5}));
}

}  // namespace
}  // namespace tflite
