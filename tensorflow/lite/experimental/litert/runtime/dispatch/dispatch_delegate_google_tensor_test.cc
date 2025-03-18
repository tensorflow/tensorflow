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

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compilation_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

#if defined(__ANDROID__)
#include "platforms/darwinn/tachyon/core/fence/fence.h"
#endif
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/darwinn/driver_shared/fence/fence_test_util.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"

using litert::testing::MakeRuntimeFromTestFileWithNpuModel;
using testing::FloatNear;
using testing::Pointwise;
using Fence = std::shared_ptr<platforms::darwinn::tachyon::Fence>;
using ::testing::ElementsAre;

namespace litert {
namespace {

constexpr absl::string_view kNpuFile = kGoogleTensorModelFileName;
constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";
constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

litert::Expected<Environment> CreateDefaultEnvironment() {
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  return litert::Environment::Create(absl::MakeConstSpan(environment_options));
}

TEST(DispatchDelegate, GoogleTensorCpuBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      testing::TflRuntime::Ptr runtime,
      MakeRuntimeFromTestFileWithNpuModel(kTfliteFile, kNpuFile));
  tflite::Interpreter& interpreter = runtime->Interpreter();

  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  internal::ExternalLiteRtBufferContext buffer_context;
  interpreter.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  LiteRtEnvironmentOptions env_options = nullptr;
  LiteRtGetEnvironmentOptions(env.Get(), &env_options);
  DispatchDelegateOptionsPtr dispatch_delegate_options =
      CreateDispatchDelegateOptionsPtr(env_options);
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           runtime->Flatbuffer().Buf().Data());
  DispatchDelegatePtr dispatch_delegate = CreateDispatchDelegatePtr(
      env_options, std::move(dispatch_delegate_options));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Get the list of signatures and check it.
  auto signature_defs = interpreter.signature_keys();
  ASSERT_EQ(signature_defs.size(), 1);

  tflite::impl::SignatureRunner* runner =
      interpreter.GetSignatureRunner(/*signature_key=*/nullptr);
  ASSERT_NE(runner, nullptr);

  EXPECT_EQ(runner->AllocateTensors(), kTfLiteOk);

  // Fill model inputs.
  ASSERT_STREQ(runner->input_names()[0], "arg0");
  TfLiteTensor* input_0_tensor = runner->input_tensor("arg0");
  ASSERT_NE(input_0_tensor, nullptr);
  float* input_0 = input_0_tensor->data.f;
  std::memcpy(input_0, kTestInput0Tensor, sizeof(kTestInput0Tensor));

  ASSERT_STREQ(runner->input_names()[1], "arg1");
  TfLiteTensor* input_1_tensor = runner->input_tensor("arg1");
  ASSERT_NE(input_1_tensor, nullptr);
  auto* input_1 = input_1_tensor->data.f;
  std::memcpy(input_1, kTestInput1Tensor, sizeof(kTestInput1Tensor));

  EXPECT_EQ(runner->Invoke(), kTfLiteOk);

  // Check model output.
  ASSERT_STREQ(runner->output_names()[0], "tfl.custom");
  auto output_tensor = runner->output_tensor("tfl.custom");
  ASSERT_NE(output_tensor, nullptr);
  auto output = absl::MakeSpan(output_tensor->data.f, kTestOutputSize);
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
  }
  EXPECT_THAT(output, Pointwise(::testing::FloatNear(1e-5), kTestOutputTensor));
}

TEST(DispatchDelegate, GoogleTensorHwBuffer) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  LITERT_ASSERT_OK_AND_ASSIGN(
      testing::TflRuntime::Ptr runtime,
      MakeRuntimeFromTestFileWithNpuModel(kTfliteFile, kNpuFile));
  tflite::Interpreter& interpreter = runtime->Interpreter();

  internal::ExternalLiteRtBufferContext buffer_context;
  interpreter.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  LiteRtEnvironmentOptions env_options = nullptr;
  LiteRtGetEnvironmentOptions(env.Get(), &env_options);

  DispatchDelegateOptionsPtr dispatch_delegate_options =
      CreateDispatchDelegateOptionsPtr(env_options);
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           runtime->Flatbuffer().Buf().Data());
  DispatchDelegatePtr dispatch_delegate = CreateDispatchDelegatePtr(
      env_options, std::move(dispatch_delegate_options));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Create and register tensor buffers for all inputs and outputs.
  std::vector<litert::TensorBuffer> input_buffers;
  for (int i = 0; i < interpreter.inputs().size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements * input_buffer_requirements,
        buffer_context.GetBufferRequirement(interpreter.input_tensor(i)));
    ASSERT_EQ(input_buffer_requirements->SupportedTypes()->at(0),
              kLiteRtTensorBufferTypeAhwb);
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer input_buffer,
        buffer_context.CreateBufferForTensor(interpreter.input_tensor(i)));
    ASSERT_TRUE(input_buffer.IsOwned());
    ASSERT_EQ(*input_buffer.BufferType(), kLiteRtTensorBufferTypeAhwb);
    LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer duplicate_buffer,
                                input_buffer.Duplicate());
    auto status = buffer_context.RegisterTensorBuffer(
        interpreter.input_tensor(i), std::move(duplicate_buffer));
    ASSERT_EQ(status, kLiteRtStatusOk);
    input_buffers.push_back(std::move(input_buffer));
  }

  std::vector<litert::TensorBuffer> output_buffers;
  for (int i = 0; i < interpreter.outputs().size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements * output_buffer_requirements,
        buffer_context.GetBufferRequirement(interpreter.output_tensor(i)));
    ASSERT_NE(output_buffer_requirements, nullptr);
    ASSERT_EQ(output_buffer_requirements->SupportedTypes()->at(0),
              kLiteRtTensorBufferTypeAhwb);
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer output_buffer,
        buffer_context.CreateBufferForTensor(interpreter.output_tensor(i)));
    ASSERT_TRUE(output_buffer.IsOwned());
    ASSERT_EQ(*output_buffer.BufferType(), kLiteRtTensorBufferTypeAhwb);
    LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer duplicate_buffer,
                                output_buffer.Duplicate());
    auto status = buffer_context.RegisterTensorBuffer(
        interpreter.output_tensor(i), std::move(duplicate_buffer));
    ASSERT_EQ(status, kLiteRtStatusOk);
    output_buffers.push_back(std::move(output_buffer));
  }

  // Get the list of signatures and check it.
  auto signature_defs = interpreter.signature_keys();
  ASSERT_EQ(signature_defs.size(), 1);

  tflite::impl::SignatureRunner* runner =
      interpreter.GetSignatureRunner(/*signature_key=*/nullptr);
  ASSERT_NE(runner, nullptr);

  EXPECT_EQ(runner->AllocateTensors(), kTfLiteOk);

  // Fill model inputs.
  ASSERT_STREQ(runner->input_names()[0], "arg0");
  auto& input_0_buffer = input_buffers[0];
  input_0_buffer.Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));

  ASSERT_STREQ(runner->input_names()[1], "arg1");
  auto& input_1_buffer = input_buffers[1];
  input_1_buffer.Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));

  EXPECT_EQ(runner->Invoke(), kTfLiteOk);

  // Check model output.
  ASSERT_STREQ(runner->output_names()[0], "tfl.custom");
  auto& output_buffer = output_buffers[0];
  float output_buffer_data[kTestOutputSize];
  auto output_span = absl::MakeSpan(output_buffer_data, kTestOutputSize);
  auto read_success = output_buffer.Read<float>(output_span);
  ASSERT_TRUE(read_success);
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                   << kTestOutputTensor[i];
  }
  EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
}

TEST(DispatchDelegate, CompiledModel) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  // Create Model and check signatures.
  LITERT_ASSERT_OK_AND_ASSIGN(
      OwningBufferRef<uint8_t> model_with_byte_code,
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile)));
  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromBuffer(model_with_byte_code));

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);
  Signature& signature = signatures.at(0);
  EXPECT_EQ(signature.Key(), Model::DefaultSignatureKey());
  size_t signature_index = 0;

  std::vector<absl::string_view> input_names = signature.InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  std::vector<absl::string_view> output_names = signature.OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.custom"));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  // Check CompiledModel buffer requirements.
  // input and output expect AHWB.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeAhwb));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeAhwb));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(signature_index,
                                                 /*output_name=*/"tfl.custom"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types, ElementsAre(kLiteRtTensorBufferTypeAhwb));

  // Create and fill input and output tensor buffers.
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

  // Execute compiled model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  float output_buffer_data[kTestOutputSize];
  absl::Span<float> output_span =
      absl::MakeSpan(output_buffer_data, kTestOutputSize);
  ASSERT_TRUE(output_buffers[0].Read(output_span));
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                   << kTestOutputTensor[i];
  }
  EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
}

TEST(DispatchDelegate, CompiledModelSharedInput) {
  auto model_with_byte_code = internal::GetModelBufWithByteCode(
      testing::GetTestFilePath("shared_input_cpu_npu.tflite"),
      testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);
  auto model = Model::CreateFromBuffer(*model_with_byte_code);
  ASSERT_TRUE(model);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif
  auto jit_compilation_options = CompilationOptions::Create();
  ASSERT_TRUE(jit_compilation_options);
  ASSERT_TRUE(jit_compilation_options->SetHardwareAccelerators(
      kLiteRtHwAcceleratorCpu));

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));
  ASSERT_TRUE(env);
  auto res_compiled_model =
      CompiledModel::Create(*env, *model, *jit_compilation_options);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel";
  auto& compiled_model = *res_compiled_model;

  size_t signature_index = 0;
  auto signature = *model->GetSignature(signature_index);
  auto input_buffers = *compiled_model.CreateInputBuffers(signature_index);
  auto output_buffers = *compiled_model.CreateOutputBuffers(signature_index);

  // Fill model inputs.
  auto input_names = signature.InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signature.OutputNames();
  EXPECT_EQ(output_names.size(), 2);
  {
    EXPECT_EQ(output_names.at(0), "tfl.add");
    float output_buffer_data[kTestOutputSize];
    auto output_span = absl::MakeSpan(output_buffer_data, kTestOutputSize);
    ASSERT_TRUE(output_buffers[0].Read(output_span));
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                     << kTestOutputTensor[i];
    }
    EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
  {
    EXPECT_EQ(output_names.at(1), "tfl.custom");
    float output_buffer_data[kTestOutputSize];
    auto output_span = absl::MakeSpan(output_buffer_data, kTestOutputSize);
    ASSERT_TRUE(output_buffers[1].Read(output_span));
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                     << kTestOutputTensor[i];
    }
    EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(DispatchDelegate, CompiledModelAsync) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "The rest of this test is specific to Android devices with  a "
         "GoogleTensor eTPU";
#endif
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  // Create Model and check signatures.
  LITERT_ASSERT_OK_AND_ASSIGN(
      OwningBufferRef<uint8_t> model_with_byte_code,
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile)));

  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromBuffer(model_with_byte_code));

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);
  Signature& signature = signatures.at(0);
  absl::string_view signature_key = signature.Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  std::vector<absl::string_view> input_names = signature.InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  std::vector<absl::string_view> output_names = signature.OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.custom"));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  // Create and fill input and output tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_0_cpu_addr_and_lock,
                              TensorBufferScopedLock::Create(input_buffers[0]));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_1_cpu_addr_and_lock,
                              TensorBufferScopedLock::Create(input_buffers[1]));

  // Attach events to input buffers.
  Fence input_fence_0 = platforms::darwinn::fence_util::CreateFence();
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_0,
      litert::Event::CreateFromSyncFenceFd(input_fence_0->GetFd(),
                                           /*owns_fd=*/false));
  input_buffers[0].SetEvent(std::move(input_event_0));

  Fence input_fence_1 = platforms::darwinn::fence_util::CreateFence();
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_1,
      litert::Event::CreateFromSyncFenceFd(input_fence_1->GetFd(),
                                           /*owns_fd=*/false));
  input_buffers[1].SetEvent(std::move(input_event_1));

  // Start the model asynchronously.
  bool async;
  compiled_model.RunAsync(signature_index, input_buffers, output_buffers,
                          async);
  ASSERT_TRUE(async);
  ASSERT_TRUE(output_buffers[0].HasEvent());

  // Set input values.
  std::memcpy(input_0_cpu_addr_and_lock.second, kTestInput0Tensor,
              sizeof(kTestInput0Tensor));
  std::memcpy(input_1_cpu_addr_and_lock.second, kTestInput1Tensor,
              sizeof(kTestInput1Tensor));

  // Signal input fences so that the inference can start.
  ASSERT_OK(input_fence_0->Signal(/*success=*/true));
  ASSERT_OK(input_fence_1->Signal(/*success=*/true));

  // Check model output.
  float output_buffer_data[kTestOutputSize];
  absl::Span<float> output_span =
      absl::MakeSpan(output_buffer_data, kTestOutputSize);
  // The next read operation will block on the output buffer's sync fence.
  ASSERT_TRUE(output_buffers[0].Read(output_span));
  // Print and confirm the output values are correct.
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                   << kTestOutputTensor[i];
  }
  EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
}

}  // namespace
}  // namespace litert
