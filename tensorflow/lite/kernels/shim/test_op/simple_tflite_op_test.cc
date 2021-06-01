/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/test_op/simple_tflite_op.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {

class SimpleOpModel : public SingleOpModel {
 public:
  // Builds the op model and feeds in inputs, ready to invoke.
  SimpleOpModel(const std::string& input, const int output2_size) {
    // Define inputs.
    const int input_idx = AddInput(tflite::TensorType_STRING);
    // Define outputs.
    output_1_idx_ = AddOutput(tflite::TensorType_INT32);
    output_2_idx_ = AddOutput(tflite::TensorType_FLOAT32);
    output_3_idx_ = AddOutput(tflite::TensorType_INT32);

    // Build the interpreter.
    flexbuffers::Builder builder;
    builder.Map([&]() { builder.Int("output2_size", output2_size); });
    builder.Finish();
    SetCustomOp(OpName_SIMPLE_OP(), builder.GetBuffer(), Register_SIMPLE_OP);
    BuildInterpreter({{}});

    // Populate inputs.
    PopulateStringTensor(input_idx, {input});
  }

  std::tuple<std::vector<int>, std::vector<float>, std::vector<int>>
  GetOutputs() {
    return {ExtractVector<int>(output_1_idx_),
            ExtractVector<float>(output_2_idx_),
            ExtractVector<int>(output_3_idx_)};
  }

  // Tensor indices
  int output_1_idx_;
  int output_2_idx_;
  int output_3_idx_;
};

TEST(SimpleOpModel, OutputSize5) {
  SimpleOpModel m(/*input=*/"abc", /*output2_size=*/5);
  m.Invoke();
  // When C++17 is available use structured binding
  std::vector<int> output_1;
  std::vector<float> output_2;
  std::vector<int> output_3;
  std::tie(output_1, output_2, output_3) = m.GetOutputs();
  EXPECT_THAT(output_1, testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(output_2, testing::ElementsAre(0, 0.5, 1.0, 1.5, 2.0));
  EXPECT_THAT(output_3, testing::ElementsAre(0, 1, 2));
}

TEST(SimpleOpModel, OutputSize3) {
  SimpleOpModel m(/*input=*/"dummy", /*output2_size=*/3);
  m.Invoke();
  // When C++17 is available use structured binding
  std::vector<int> output_1;
  std::vector<float> output_2;
  std::vector<int> output_3;
  std::tie(output_1, output_2, output_3) = m.GetOutputs();
  EXPECT_THAT(output_1, testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(output_2, testing::ElementsAre(0, 0.5, 1.0));
  EXPECT_THAT(output_3, testing::ElementsAre(0, 1, 2, 3, 4));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
