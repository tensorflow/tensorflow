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
#include "tensorflow/lite/kernels/shim/test_op/tmpl_tflite_op.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace shim {
namespace {

template <typename AType, typename BType>
class TmplOpModel : public SingleOpModel {
 public:
  // Builds the op model and feeds in inputs, ready to invoke.
  TmplOpModel(const std::vector<uint8_t>& op_options,
              const std::vector<tflite::TensorType>& input_types,
              const std::vector<std::vector<int>>& input_shapes,
              const std::vector<AType>& input0,
              const std::vector<BType>& input1,
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
    SetCustomOp(ops::custom::OpName_TMPL_OP(), op_options,
                ops::custom::Register_TMPL_OP);
    BuildInterpreter(input_shapes);
    // Populate inputs.
    PopulateTensor(input_idx[0], input0);
    PopulateTensor(input_idx[1], input1);
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

TEST(TmplOpModel, float_int32) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("AType", kTfLiteFloat32);
    builder.Int("BType", kTfLiteInt32);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}, {}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_FLOAT32,
                                                 tflite::TensorType_INT32};
  std::vector<tflite::TensorType> output_types = {tflite::TensorType_FLOAT32};
  const std::vector<float> input0 = {5.6f};
  const std::vector<int32_t> input1 = {3};
  // Run the op
  TmplOpModel<float, int32_t> m(
      /*op_options=*/builder.GetBuffer(), input_types, input_shapes, input0,
      input1, output_types);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Assertions
  EXPECT_THAT(m.GetOutput<float>(0), testing::ElementsAre(8.6f));
}

TEST(TmplOpModel, int32_int64) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("AType", kTfLiteInt32);
    builder.Int("BType", kTfLiteInt64);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}, {}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_INT32,
                                                 tflite::TensorType_INT64};
  std::vector<tflite::TensorType> output_types = {tflite::TensorType_FLOAT32};
  const std::vector<int32_t> input0 = {12};
  const std::vector<int64_t> input1 = {33l};
  // Run the op
  TmplOpModel<int32_t, int64_t> m(
      /*op_options=*/builder.GetBuffer(), input_types, input_shapes, input0,
      input1, output_types);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Assertions
  EXPECT_THAT(m.GetOutput<float>(0), testing::ElementsAre(45.0f));
}

TEST(TmplOpModel, int32_bool) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("AType", kTfLiteInt32);
    builder.Int("BType", kTfLiteBool);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}, {}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_INT32,
                                                 tflite::TensorType_BOOL};
  std::vector<tflite::TensorType> output_types = {tflite::TensorType_FLOAT32};
  const std::vector<int32_t> input0 = {12};
  const std::vector<bool> input1 = {true};
  // Run the op
  TmplOpModel<int32_t, bool> m(
      /*op_options=*/builder.GetBuffer(), input_types, input_shapes, input0,
      input1, output_types);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Assertions
  EXPECT_THAT(m.GetOutput<float>(0), testing::ElementsAre(13.0f));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
