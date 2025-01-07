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

#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

constexpr const char* kCompilerPluginLibSearchPath = "/data/local/tmp";

TEST(JitCompilation, Qualcomm) {
  auto model_path = litert::testing::GetTestFilePath(kModelFileName);
  auto model = litert::Model::CreateFromFile(model_path);
  ASSERT_TRUE(model);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  constexpr const std::array<const absl::string_view, 1>
      compiler_plugin_lib_search_paths = {kCompilerPluginLibSearchPath};
  auto compiler_plugin = litert::internal::CompilerPlugin::LoadPlugin(
      compiler_plugin_lib_search_paths, "Qualcomm");
  ASSERT_TRUE(compiler_plugin);

  auto api_version = compiler_plugin->ApiVersion();
  ASSERT_TRUE(api_version);

  ABSL_LOG(INFO) << "Found compiler plugin with version " << api_version->major
                 << "." << api_version->minor << "." << api_version->patch;

  auto npu_bytecode = ApplyPlugin(*compiler_plugin, *model);
  EXPECT_TRUE(npu_bytecode);
  EXPECT_GT(npu_bytecode->Size(), 0);

  auto serialized_model = litert::internal::GetModelBufWithByteCode(
      std::move(*model->Get()), *npu_bytecode);
  EXPECT_TRUE(serialized_model);

  model = litert::Model::CreateFromBuffer(*serialized_model);

  auto flatbuffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(serialized_model->Data()),
      serialized_model->Size());

  EXPECT_TRUE(flatbuffer_model != nullptr);

  tflite::Interpreter::Ptr interpreter = nullptr;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*flatbuffer_model, resolver)(&interpreter);
  EXPECT_TRUE(interpreter != nullptr);

  EXPECT_EQ(interpreter->nodes_size(), 1);
  EXPECT_EQ(interpreter->inputs().size(), 2);
  EXPECT_EQ(interpreter->outputs().size(), 1);
  ASSERT_EQ(interpreter->execution_plan().size(), 1);

  litert::internal::ExternalLiteRtBufferContext buffer_context;
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  auto dispatch_delegate_options = litert::CreateDispatchDelegateOptionsPtr();
  LiteRtDispatchDelegateAddAllocBaseOption(
      dispatch_delegate_options.get(), flatbuffer_model->allocation()->base());
  auto dispatch_delegate =
      litert::CreateDispatchDelegatePtr(std::move(dispatch_delegate_options));

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Get the list of signatures and check it.
  auto signature_defs = interpreter->signature_keys();
  ASSERT_EQ(signature_defs.size(), 0);

  tflite::impl::SignatureRunner* runner =
      interpreter->GetSignatureRunner(/*signature_key=*/nullptr);
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
  auto output_tensor = runner->output_tensor(runner->output_names()[0]);
  ASSERT_NE(output_tensor, nullptr);
  auto* output = output_tensor->data.f;
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
  }
  for (auto i = 0; i < kTestOutputSize; ++i) {
    EXPECT_NEAR(output[i], kTestOutputTensor[i], 1e-5);
  }
}
