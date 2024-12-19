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
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

namespace litert {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

// Creates input buffers for the given LiteRtTensorBufferType and size.
Expected<std::vector<LiteRtTensorBuffer>> CreateInputBuffers(
    LiteRtModel& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  std::vector<LiteRtTensorBuffer> input_buffers;
  auto* subgraph = *LookupSubgraph(*model, signature_key);
  auto& input_tensors = subgraph->Inputs();
  const size_t num_inputs = subgraph->NumInputs();
  input_buffers.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const auto& ranked_tensor_type =
        input_tensors[i]->Type().second.ranked_tensor_type;
    LiteRtTensorBuffer input_buffer;
    if (auto status = LiteRtCreateManagedTensorBuffer(
            buffer_type, &ranked_tensor_type, bytes, &input_buffer);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create input tensor buffer");
    }
    input_buffers.push_back(input_buffer);
  }
  return std::move(input_buffers);
}

// Creates input buffers for the given LiteRtCompiledModelT by leveraging
// TensorBufferRequirements.
Expected<std::vector<LiteRtTensorBuffer>> CreateInputBuffers(
    LiteRtModel& model, LiteRtCompiledModelT& compiled_model,
    absl::string_view signature_key) {
  auto litert_input_buffer_requirements =
      compiled_model.GetInputBufferRequirements(signature_key, 0);
  if (!litert_input_buffer_requirements.HasValue()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      litert_input_buffer_requirements.Error().Message());
  }
  TensorBufferRequirements input_buffer_requirements =
      TensorBufferRequirements(*litert_input_buffer_requirements,
                               /*owned=*/false);
  LiteRtTensorBufferType tensor_buffer_type =
      input_buffer_requirements.SupportedTypes()->at(0);

  return CreateInputBuffers(model, signature_key, tensor_buffer_type,
                            input_buffer_requirements.BufferSize().Value());
}

// Creates output buffers for the given LiteRtTensorBufferType and size.
Expected<std::vector<LiteRtTensorBuffer>> CreateOutputBuffers(
    LiteRtModel& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  std::vector<LiteRtTensorBuffer> output_buffers;
  auto* subgraph = *LookupSubgraph(*model, signature_key);
  auto& output_tensors = subgraph->Outputs();
  size_t num_outputs = subgraph->NumOutputs();
  output_buffers.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto ranked_tensor_type =
        output_tensors[i]->Type().second.ranked_tensor_type;
    LiteRtTensorBuffer output_buffer;
    if (auto status = LiteRtCreateManagedTensorBuffer(
            buffer_type, &ranked_tensor_type, bytes, &output_buffer);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create output tensor buffer");
    }
    output_buffers.push_back(output_buffer);
  }
  return std::move(output_buffers);
}

// Creates output buffers for the given LiteRtCompiledModelT by leveraging
// TensorBufferRequirements.
Expected<std::vector<LiteRtTensorBuffer>> CreateOutputBuffers(
    LiteRtModel& model, LiteRtCompiledModelT& compiled_model,
    absl::string_view signature_key) {
  auto litert_output_buffer_requirements =
      compiled_model.GetOutputBufferRequirements(signature_key, 0);
  if (!litert_output_buffer_requirements.HasValue()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      litert_output_buffer_requirements.Error().Message());
  }
  TensorBufferRequirements output_buffer_requirements =
      TensorBufferRequirements(*litert_output_buffer_requirements,
                               /*owned=*/false);
  LiteRtTensorBufferType tensor_buffer_type =
      output_buffer_requirements.SupportedTypes()->at(0);
  return CreateOutputBuffers(model, signature_key, tensor_buffer_type,
                             output_buffer_requirements.BufferSize().Value());
}

TEST(CompiledModelTest, Basic) {
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  auto res_compiled_model =
      LiteRtCompiledModelT::Create(model, kLiteRtHwAccelatorNone);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel: "
                                  << res_compiled_model.Error().Message();
  auto& compiled_model = **res_compiled_model;

  auto signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  auto signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  auto input_buffers_res =
      CreateInputBuffers(model, compiled_model, signature_key);
  EXPECT_TRUE(input_buffers_res);
  auto input_buffers = std::move(*input_buffers_res);

  auto output_buffers_res =
      CreateOutputBuffers(model, compiled_model, signature_key);
  EXPECT_TRUE(output_buffers_res);
  auto output_buffers = std::move(*output_buffers_res);

  // Fill model inputs.
  auto& input_names = signatures[0]->InputNames();
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
  auto output_names = signatures[0]->OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     /*event=*/nullptr),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
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

TEST(CompiledModelTest, UseAhwbBuffer) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices";
#endif
  auto path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  auto res_compiled_model =
      LiteRtCompiledModelT::Create(model, kLiteRtHwAccelatorNone);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel";
  auto& compiled_model = **res_compiled_model;

  auto signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  auto signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  auto input_buffers_res =
      CreateInputBuffers(model, signature_key, kLiteRtTensorBufferTypeAhwb,
                         sizeof(float) * kTestInput0Size);
  EXPECT_TRUE(input_buffers_res);
  auto input_buffers = std::move(*input_buffers_res);

  auto output_buffers_res =
      CreateOutputBuffers(model, signature_key, kLiteRtTensorBufferTypeAhwb,
                          sizeof(float) * kTestOutputSize);
  EXPECT_TRUE(output_buffers_res);
  auto output_buffers = std::move(*output_buffers_res);

  // Fill model inputs.
  auto input_names = signatures[0]->InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  auto& input_0_buffer = input_buffers[0];
  EXPECT_EQ(input_0_buffer->buffer_type(), kLiteRtTensorBufferTypeAhwb);
  {
    TensorBuffer ahwb_buffer(input_0_buffer, /*owned=*/false);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  auto& input_1_buffer = input_buffers[1];
  {
    TensorBuffer ahwb_buffer(input_1_buffer, /*owned=*/false);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  compiled_model.Run(signature_key, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0]->OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     /*event=*/nullptr),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
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
