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
#include <string>

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
  SimpleOpModel(const std::vector<uint8_t>& op_options,
                const std::vector<tflite::TensorType>& input_types,
                const std::vector<std::vector<int>>& input_shapes,
                const std::string& input0,
                const std::vector<std::vector<int64_t>>& input1,
                const std::vector<tflite::TensorType>& output_types) {
    // Define inputs.
    std::vector<int> input_idx;
    for (const auto input_type : input_types) {
      input_idx.push_back(AddInput(input_type));
    }
    // Define outputs.
    for (const auto output_type : output_types) {
      output_idx_.push_back(AddOutput(output_type));
    }
    // Build the interpreter.
    SetCustomOp(OpName_SIMPLE_OP(), op_options, Register_SIMPLE_OP);
    BuildInterpreter(input_shapes);
    // Populate inputs.
    PopulateStringTensor(input_idx[0], {input0});
    for (int i = 0; i < input1.size(); ++i) {
      PopulateTensor(input_idx[1 + i], input1[i]);
    }
  }

  template <typename T>
  std::vector<T> GetOutput(const int i) {
    return ExtractVector<T>(output_idx_[i]);
  }

  std::vector<int> GetOutputShape(const int i) {
    return GetTensorShape(output_idx_[i]);
  }

 protected:
  // Tensor indices
  std::vector<int> output_idx_;
};

TEST(SimpleOpModel, OutputSize_5_N_2) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("output1_size", 5);
    builder.String("output2_suffix", "foo");
    builder.Int("N", 2);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}, {}, {2}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_STRING,
                                                 tflite::TensorType_INT64,
                                                 tflite::TensorType_INT64};
  std::vector<tflite::TensorType> output_types = {
      tflite::TensorType_INT32, tflite::TensorType_FLOAT32,
      tflite::TensorType_STRING, tflite::TensorType_INT64,
      tflite::TensorType_INT64};
  const std::string input0 = "abc";
  const std::vector<std::vector<int64_t>> input1 = {{123}, {456, 789}};
  // Run the op
  SimpleOpModel m(/*op_options=*/builder.GetBuffer(), input_types, input_shapes,
                  input0, input1, output_types);
  m.Invoke();
  // Assertions
  EXPECT_THAT(m.GetOutput<int>(0), testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(m.GetOutput<float>(1),
              testing::ElementsAre(0, 0.5, 1.0, 1.5, 2.0));
  EXPECT_THAT(m.GetOutput<std::string>(2),
              testing::ElementsAre("0", "1", "2", "foo"));
  EXPECT_THAT(m.GetOutput<int64_t>(3), testing::ElementsAre(124));
  EXPECT_THAT(m.GetOutputShape(3), testing::ElementsAre());
  EXPECT_THAT(m.GetOutput<int64_t>(4), testing::ElementsAre(457, 790));
  EXPECT_THAT(m.GetOutputShape(4), testing::ElementsAre(2));
}

TEST(SimpleOpModel, OutputSize_3_N_0) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("output1_size", 3);
    builder.String("output2_suffix", "foo");
    builder.Int("N", 0);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_STRING};
  std::vector<tflite::TensorType> output_types = {tflite::TensorType_INT32,
                                                  tflite::TensorType_FLOAT32,
                                                  tflite::TensorType_STRING};
  const std::string input0 = "abcde";
  const std::vector<std::vector<int64_t>> input1;
  // Run the op
  SimpleOpModel m(/*op_options=*/builder.GetBuffer(), input_types, input_shapes,
                  input0, input1, output_types);
  m.Invoke();
  // Assertions
  EXPECT_THAT(m.GetOutput<int>(0), testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(m.GetOutput<float>(1), testing::ElementsAre(0, 0.5, 1.0));
  EXPECT_THAT(m.GetOutput<std::string>(2),
              testing::ElementsAre("0", "1", "2", "3", "4", "foo"));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
