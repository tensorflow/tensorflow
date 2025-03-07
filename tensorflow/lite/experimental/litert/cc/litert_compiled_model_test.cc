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

#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::SizeIs;

namespace litert {
namespace {

constexpr absl::string_view kNpuFile = kGoogleTensorModelFileName;
constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";
constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

TEST(CompiledModelTest, Basic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  std::vector<absl::string_view> input_names = signatures[0].InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  std::vector<absl::string_view> output_names = signatures[0].OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(signature_index,
                                                 /*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(signature_index));

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelTest, RunWithInputOutputMap) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  std::vector<absl::string_view> input_names = signatures[0].InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  std::vector<absl::string_view> output_names = signatures[0].OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(signature_index,
                                                 /*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_buffer0,
      compiled_model.CreateInputBuffer(signature_key, "arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_buffer1,
      compiled_model.CreateInputBuffer(signature_key, "arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer output_buffer0,
      compiled_model.CreateOutputBuffer(signature_key, "tfl.add"));

  ASSERT_TRUE(input_buffer0.Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffer1.Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Create input and output map.
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map["arg0"] = std::move(input_buffer0);
  input_map["arg1"] = std::move(input_buffer1);

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["tfl.add"] = std::move(output_buffer0);

  // Execute model with input and output maps instead of buffers.
  compiled_model.Run(signature_key, input_map, output_map);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create<const float>(
                                output_map["tfl.add"]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelTest, RunWithNpuModel) {
  if (!litert::internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  // Environment setup.
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  // Create Model.

  // TODO(gcarranza): Replace internal API with C++ API or single npu tflite
  // file.
  LITERT_ASSERT_OK_AND_ASSIGN(
      BufferRef<uint8_t> model_with_byte_code,
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile)));

  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromBuffer(model_with_byte_code));
  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(model.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(model.DefaultSignatureKey()));

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(*input_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*input_buffers[1].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*output_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Run compiled model.
  compiled_model.Run(model.DefaultSignatureKey(), input_buffers,
                     output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

}  // namespace
}  // namespace litert
