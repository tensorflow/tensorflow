
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

class ExpandDimsOpModel : public SingleOpModel {
 public:
  ExpandDimsOpModel(std::initializer_list<int> input_shape,
                    TensorType input_type) {
    input_ = AddInput(input_type);
    axis_ = AddInput(TensorType_INT32);
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_EXPAND_DIMS, BuiltinOptions_ExpandDimsOptions,
                 0);
    BuildInterpreter({input_shape, {1}});
  }
  void SetInputFloat(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }
  void SetAxis(int axis) { PopulateTensor<int32_t>(axis_, {axis}); }
  std::vector<float> GetValuesFloat() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int axis_;
  int output_;
};

TEST(ExpandDimsOpTest, DifferentAxis) {
  ExpandDimsOpModel m({2, 2}, TensorType_FLOAT32);
  std::initializer_list<float> values = {-1.f, 1.f, -2.f, 2.f};
  m.SetInputFloat(values);
  m.SetAxis(0);
  m.Invoke();
  EXPECT_THAT(m.GetValuesFloat(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2}));

  m.SetAxis(1);
  m.Invoke();
  EXPECT_THAT(m.GetValuesFloat(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2}));

  m.SetAxis(2);
  m.Invoke();
  EXPECT_THAT(m.GetValuesFloat(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1}));

  m.SetAxis(-1);
  m.Invoke();
  EXPECT_THAT(m.GetValuesFloat(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1}));
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
