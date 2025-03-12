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

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compilation_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"

namespace litert {
namespace {

using ::litert::testing::MakeRuntimeFromTestFileWithNpuModel;
using ::testing::FloatNear;
using ::testing::Pointwise;

static constexpr absl::string_view kNpuFile = kQualcommModelFileName;
static constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";
static constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

TEST(DispatchDelegate, QualcommCpuBuffer) {
  auto runtime = MakeRuntimeFromTestFileWithNpuModel(kTfliteFile, kNpuFile);
  ASSERT_TRUE(runtime) << "Failed to initialize tflite interpreter";
  auto& rt = **runtime;
  auto& interpreter = rt.Interpreter();

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));
  ASSERT_TRUE(env);

  litert::internal::ExternalLiteRtBufferContext buffer_context;
  interpreter.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  auto dispatch_delegate_options =
      CreateDispatchDelegateOptionsPtr(*env->Get());
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           rt.Flatbuffer().Buf().Data());
  auto dispatch_delegate = CreateDispatchDelegatePtr(
      *env->Get(), std::move(dispatch_delegate_options));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
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
  auto input_0_tensor = runner->input_tensor("arg0");
  ASSERT_NE(input_0_tensor, nullptr);
  auto* input_0 = input_0_tensor->data.f;
  std::memcpy(input_0, kTestInput0Tensor, sizeof(kTestInput0Tensor));

  ASSERT_STREQ(runner->input_names()[1], "arg1");
  auto input_1_tensor = runner->input_tensor("arg1");
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

TEST(DispatchDelegate, QualcommHwBuffer) {
  auto runtime = MakeRuntimeFromTestFileWithNpuModel(kTfliteFile, kNpuFile);
  ASSERT_TRUE(runtime) << "Failed to initialize tflite interpreter";
  auto& rt = **runtime;
  auto& interpreter = rt.Interpreter();
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));
  ASSERT_TRUE(env);

  litert::internal::ExternalLiteRtBufferContext buffer_context;
  interpreter.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  auto dispatch_delegate_options =
      CreateDispatchDelegateOptionsPtr(*env->Get());
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           rt.Flatbuffer().Buf().Data());
  auto dispatch_delegate = CreateDispatchDelegatePtr(
      *env->Get(), std::move(dispatch_delegate_options));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Create and register tensor buffers for all inputs and outputs.

  std::vector<litert::TensorBuffer> input_buffers;
  for (int i = 0; i < interpreter.inputs().size(); ++i) {
    auto input_buffer_requirements =
        buffer_context.GetBufferRequirement(interpreter.input_tensor(i));
    ASSERT_TRUE(input_buffer_requirements);
    ASSERT_EQ((*input_buffer_requirements)->SupportedTypes().Value()[0],
              kLiteRtTensorBufferTypeFastRpc);
    auto input_buffer =
        buffer_context.CreateBufferForTensor(interpreter.input_tensor(i));
    ASSERT_TRUE(input_buffer);
    ASSERT_TRUE(input_buffer->IsOwned());
    ASSERT_EQ(*input_buffer->BufferType(), kLiteRtTensorBufferTypeFastRpc);
    auto duplicate_buffer = (*input_buffer).Duplicate();
    ASSERT_TRUE(duplicate_buffer);
    auto status = buffer_context.RegisterTensorBuffer(
        interpreter.input_tensor(i), std::move(*duplicate_buffer));
    ASSERT_EQ(status, kLiteRtStatusOk);
    input_buffers.push_back(std::move(*input_buffer));
  }

  std::vector<litert::TensorBuffer> output_buffers;
  for (int i = 0; i < interpreter.outputs().size(); ++i) {
    auto output_buffer_requirements =
        buffer_context.GetBufferRequirement(interpreter.output_tensor(i));
    ASSERT_TRUE(output_buffer_requirements);
    ASSERT_EQ((*output_buffer_requirements)->SupportedTypes().Value()[0],
              kLiteRtTensorBufferTypeFastRpc);
    auto output_buffer =
        buffer_context.CreateBufferForTensor(interpreter.output_tensor(i));
    ASSERT_TRUE(output_buffer.HasValue());
    ASSERT_TRUE(output_buffer->IsOwned());
    ASSERT_EQ(*output_buffer->BufferType(), kLiteRtTensorBufferTypeFastRpc);
    auto duplicate_buffer = (*output_buffer).Duplicate();
    ASSERT_TRUE(duplicate_buffer);
    auto status = buffer_context.RegisterTensorBuffer(
        interpreter.output_tensor(i), std::move(*duplicate_buffer));
    ASSERT_EQ(status, kLiteRtStatusOk);
    output_buffers.push_back(std::move(*output_buffer));
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
  auto model_with_byte_code =
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);
  auto model = Model::CreateFromBuffer(*model_with_byte_code);
  ASSERT_TRUE(model);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
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

  auto signatures = model->GetSignatures();
  ASSERT_TRUE(signatures);
  EXPECT_EQ(signatures->size(), 1);
  auto& signature = signatures->at(0);
  auto signature_key = signature.Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  auto input_buffers_res = compiled_model.CreateInputBuffers(signature_index);
  EXPECT_TRUE(input_buffers_res);
  auto& input_buffers = *input_buffers_res;

  auto output_buffers_res = compiled_model.CreateOutputBuffers(signature_index);
  EXPECT_TRUE(output_buffers_res);
  auto& output_buffers = *output_buffers_res;

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
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.custom");
  float output_buffer_data[kTestOutputSize];
  auto output_span = absl::MakeSpan(output_buffer_data, kTestOutputSize);
  ASSERT_TRUE(output_buffers[0].Read(output_span));
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << "Result: " << output_span.at(i) << "\t"
                   << kTestOutputTensor[i];
  }
  EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5), kTestOutputTensor));
}

TEST(DispatchDelegate, QualcommSharedInput) {
  auto model_with_byte_code = internal::GetModelBufWithByteCode(
      testing::GetTestFilePath("shared_input_cpu_npu.tflite"),
      testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);
  auto model = Model::CreateFromBuffer(*model_with_byte_code);
  ASSERT_TRUE(model);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
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

}  // namespace
}  // namespace litert
