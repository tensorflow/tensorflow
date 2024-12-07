// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"

namespace litert {
namespace {

static constexpr absl::string_view kTfliteFile =
    "third_party/tensorflow/lite/experimental/litert/test/testdata/"
    "simple_model.tflite";

TEST(CompiledModelTest, Basic) {
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(kTfliteFile.data(), &model),
            kLiteRtStatusOk);
  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(model, kHwAccelCpu, &compiled_model),
            kLiteRtStatusOk);

  LiteRtSubgraph subgraph;
  ASSERT_EQ(LiteRtGetModelSubgraph(model, 0, &subgraph), kLiteRtStatusOk);
  LiteRtParamIndex num_inputs;
  LiteRtTensorArray input_tensors;
  ASSERT_EQ(LiteRtGetSubgraphInputs(subgraph, &num_inputs, &input_tensors),
            kLiteRtStatusOk);
  std::vector<LiteRtTensorBufferRequirements> input_buffer_requirements;
  input_buffer_requirements.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    LiteRtTensorBufferRequirements buffer_requirements;
    ASSERT_EQ(
        LiteRtGetCompiledModelInputBufferRequirements(
            compiled_model, /*signature_index=*/0, i, &buffer_requirements),
        kLiteRtStatusOk);
    input_buffer_requirements.push_back(buffer_requirements);
  }

  LiteRtParamIndex num_outputs;
  LiteRtTensorArray output_tensors;
  ASSERT_EQ(LiteRtGetSubgraphOutputs(subgraph, &num_outputs, &output_tensors),
            kLiteRtStatusOk);
  std::vector<LiteRtTensorBufferRequirements> output_buffer_requirements;
  output_buffer_requirements.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    LiteRtTensorBufferRequirements buffer_requirements;
    ASSERT_EQ(
        LiteRtGetCompiledModelOutputBufferRequirements(
            compiled_model, /*signature_index=*/0, i, &buffer_requirements),
        kLiteRtStatusOk);
    output_buffer_requirements.push_back(buffer_requirements);
  }
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
}

}  // namespace
}  // namespace litert
