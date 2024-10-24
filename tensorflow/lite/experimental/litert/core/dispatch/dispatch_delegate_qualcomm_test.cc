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

#include <cstring>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

TEST(DispatchDelegate, Qualcomm) {
  auto npu_model_file_name = kQualcommModelFileName;
  auto npu_model = litert::testing::LoadBinaryFile(npu_model_file_name);
  ASSERT_TRUE(npu_model.ok());
  ABSL_LOG(INFO) << "Loaded model " << npu_model_file_name << ", "
                 << npu_model->size() << " bytes";

  auto tflite_file_name =
      litert::testing::GetTestFilePath("simple_model_npu.tflite");
  auto model = tflite::FlatBufferModel::BuildFromFile(tflite_file_name.data());
  ASSERT_NE(model, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_EQ(interpreter->nodes_size(), 1);
  EXPECT_EQ(interpreter->inputs().size(), 2);
  EXPECT_EQ(interpreter->outputs().size(), 1);
  ASSERT_EQ(interpreter->execution_plan().size(), 1);

  auto dispatch_delegate_options = litert::CreateDispatchDelegateOptionsPtr();
  ASSERT_EQ(
      LiteRtAddDispatchDelegateExecInfoOption(
          dispatch_delegate_options.get(), "npu_bytecode", npu_model->data(),
          npu_model->size(), /*function_name=*/"simple"),
      kTfLiteOk);
  auto dispatch_delegate =
      litert::CreateDispatchDelegatePtr(std::move(dispatch_delegate_options));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

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
  ASSERT_STREQ(runner->output_names()[0], "tfl.custom");
  auto output_tensor = runner->output_tensor("tfl.custom");
  ASSERT_NE(output_tensor, nullptr);
  auto* output = output_tensor->data.f;
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
  }
  for (auto i = 0; i < kTestOutputSize; ++i) {
    EXPECT_NEAR(output[i], kTestOutputTensor[i], 1e-5);
  }
}
