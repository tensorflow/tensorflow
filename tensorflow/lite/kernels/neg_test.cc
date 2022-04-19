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
#include <stdint.h>

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

class NegOpModel : public SingleOpModel {
 public:
  NegOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_NEG, BuiltinOptions_NegOptions,
                 CreateNegOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <class T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int output_;
};

TEST(NegOpModel, NegFloat) {
  NegOpModel m({TensorType_FLOAT32, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.SetInput<float>({-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({2.0f, 1.0f, 0.f, -1.0f, -2.0f, -3.0f}));
}

TEST(NegOpModel, NegInt32) {
  NegOpModel m({TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.SetInput<int32_t>({-2, -1, 0, 1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({2, 1, 0, -1, -2, -3}));
}

TEST(NegOpModel, NegInt64) {
  NegOpModel m({TensorType_INT64, {2, 3}}, {TensorType_INT64, {2, 3}});
  m.SetInput<int64_t>({-2, -1, 0, 1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({2, 1, 0, -1, -2, -3}));
}

}  // namespace
}  // namespace tflite
