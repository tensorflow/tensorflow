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
#include <math.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ExpOpModel : public SingleOpModel {
 public:
  ExpOpModel(const TensorData& input, const TensorType& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_EXP, BuiltinOptions_ExpOptions,
                 CreateExpOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <class T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
};

TEST(ExpOpTest, FloatTest) {
  std::initializer_list<float> data = {0.0f,    1.0f,  -1.0f, 100.0f,
                                       -100.0f, 0.01f, -0.01f};
  ExpOpModel m({TensorType_FLOAT32, {1, 1, 7}}, TensorType_FLOAT32);
  m.SetInput<float>(data);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 7}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear(
          {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f), std::exp(100.0f),
           std::exp(-100.0f), std::exp(0.01f), std::exp(-0.01f)})));
}

}  // namespace
}  // namespace tflite
