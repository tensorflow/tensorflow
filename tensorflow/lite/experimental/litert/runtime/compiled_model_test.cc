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

#include "tensorflow/lite/experimental/litert/runtime/compiled_model.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

Expected<std::vector<LiteRtTensorBuffer>> CreateInputBuffers(
    LiteRtModel& model, LiteRtCompiledModelT& compiled_model,
    absl::string_view signature_key) {
  std::vector<LiteRtTensorBuffer> input_buffers;
  auto subgraph = model->FindSubgraph(signature_key);
  auto& input_tensors = (*subgraph)->inputs;
  size_t num_inputs = input_tensors.size();
  input_buffers.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto litert_input_buffer_requirements =
        compiled_model.GetInputBufferRequirements(signature_key, i);
    if (!litert_input_buffer_requirements.HasValue()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        litert_input_buffer_requirements.Error().Message());
    }
    TensorBufferRequirements input_buffer_requirements =
        TensorBufferRequirements(*litert_input_buffer_requirements,
                                 /*owned=*/false);
    auto ranked_tensor_type = input_tensors[i]->type_detail.ranked_tensor_type;
    LiteRtTensorBufferType tensor_buffer_type =
        input_buffer_requirements.SupportedTypes()->at(0);
    LiteRtTensorBuffer input_buffer;
    if (auto status = LiteRtCreateManagedTensorBuffer(
            tensor_buffer_type, &ranked_tensor_type,
            input_buffer_requirements.BufferSize().Value(), &input_buffer);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create input tensor buffer");
    }
    input_buffers.push_back(input_buffer);
  }

  return std::move(input_buffers);
}

Expected<std::vector<LiteRtTensorBuffer>> CreateOutputBuffers(
    LiteRtModel& model, LiteRtCompiledModelT& compiled_model,
    absl::string_view signature_key) {
  std::vector<LiteRtTensorBuffer> output_buffers;

  auto subgraph = model->FindSubgraph(signature_key);
  auto& output_tensors = (*subgraph)->outputs;
  size_t num_outputs = output_tensors.size();
  output_buffers.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto litert_output_buffer_requirements =
        compiled_model.GetOutputBufferRequirements(signature_key, i);
    if (!litert_output_buffer_requirements.HasValue()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        litert_output_buffer_requirements.Error().Message());
    }
    TensorBufferRequirements output_buffer_requirements =
        TensorBufferRequirements(*litert_output_buffer_requirements,
                                 /*owned=*/false);
    auto ranked_tensor_type = output_tensors[i]->type_detail.ranked_tensor_type;
    LiteRtTensorBufferType tensor_buffer_type =
        output_buffer_requirements.SupportedTypes()->at(0);
    LiteRtTensorBuffer output_buffer;
    if (auto status = LiteRtCreateManagedTensorBuffer(
            tensor_buffer_type, &ranked_tensor_type,
            output_buffer_requirements.BufferSize().Value(), &output_buffer);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create output tensor buffer");
    }
    output_buffers.push_back(output_buffer);
  }

  return std::move(output_buffers);
}

constexpr const float kTestInput0Tensor[] = {1, 2};
constexpr const size_t kTestInput0Size =
    sizeof(kTestInput0Tensor) / sizeof(kTestInput0Tensor[0]);
constexpr const float kTestInput1Tensor[] = {10, 20};
constexpr const size_t kTestInput1Size =
    sizeof(kTestInput1Tensor) / sizeof(kTestInput1Tensor[0]);
constexpr const float kTestOutputTensor[] = {11, 22};
constexpr const size_t kTestOutputSize =
    sizeof(kTestOutputTensor) / sizeof(kTestOutputTensor[0]);

static constexpr absl::string_view kTfliteFile =
    "third_party/tensorflow/lite/experimental/litert/test/testdata/"
    "simple_model.tflite";

TEST(CompiledModelTest, Basic) {
  LiteRtModel model;
  auto status = LiteRtCreateModelFromFile(kTfliteFile.data(), &model);
  ASSERT_EQ(status, kLiteRtStatusOk);

  auto res_compiled_model = LiteRtCompiledModelT::Create(model);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel";
  auto& compiled_model = **res_compiled_model;

  auto& signatures = model->signatures;
  ASSERT_EQ(signatures.size(), 1);
  auto signature_key = signatures[0]->key;
  EXPECT_EQ(signature_key, LITERT_DEFAULT_SIGNATURE_KEY);

  auto input_buffers_res =
      CreateInputBuffers(model, compiled_model, signature_key);
  EXPECT_TRUE(input_buffers_res);
  auto input_buffers = std::move(*input_buffers_res);

  auto output_buffers_res =
      CreateOutputBuffers(model, compiled_model, signature_key);
  EXPECT_TRUE(output_buffers_res);
  auto output_buffers = std::move(*output_buffers_res);

  // Fill model inputs.
  auto input_names = signatures[0]->input_names;
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  auto& input_0_buffer = input_buffers[0];
  {
    TensorBuffer cpu_buffer(input_0_buffer, /*owned=*/false);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  auto& input_1_buffer = input_buffers[1];
  {
    TensorBuffer cpu_buffer(input_1_buffer, /*owned=*/false);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  compiled_model.Run(signature_key, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0]->output_names;
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  auto& output_buffer = output_buffers[0];
  {
    TensorBuffer cpu_buffer(output_buffer, /*owned=*/false);
    float output_buffer_data[kTestOutputSize];
    auto output_span = absl::MakeSpan(output_buffer_data, kTestOutputSize);
    auto read_success = cpu_buffer.Read<float>(output_span);

    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                     << kTestOutputTensor[i];
    }
    EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
}

}  // namespace
}  // namespace litert
