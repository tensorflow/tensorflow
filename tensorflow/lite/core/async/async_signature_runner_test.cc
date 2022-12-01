/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/async/async_signature_runner.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/common.h"
#include "tensorflow/lite/core/async/testing/mock_async_kernel.h"
#include "tensorflow/lite/core/async/testing/test_backend.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

namespace tflite {
namespace async {

class AsyncSignatureRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    kernel_ =
        std::make_unique<::testing::StrictMock<testing::MockAsyncKernel>>();
    backend_ = std::make_unique<testing::TestBackend>(kernel_->kernel());

    interpreter_ = std::make_unique<Interpreter>();
    interpreter_->AddTensors(2);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({1});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration* reg = ops::builtin::Register_ADD();
    void* builtin_data_1 = malloc(sizeof(int));
    interpreter_->AddNodeWithParameters({0, 0}, {1}, nullptr, 0, builtin_data_1,
                                        reg);
    signature_def_.signature_key = "serving_default";
    signature_def_.inputs["input"] = 0;
    signature_def_.outputs["output"] = 1;
    signature_def_.subgraph_index = 0;
  }

  void BuildAsyncSignatureRunner() {
    interpreter_->ModifyGraphWithDelegate(backend_->get_delegate());
    signature_runner_ = std::make_unique<AsyncSignatureRunner>(
        &signature_def_, interpreter_->subgraph(0));
  }

  int GetTensorIndex(TfLiteIoType io_type, const char* name) {
    return signature_runner_->GetTensorIndex(io_type, name);
  }

  void TearDown() override { signature_runner_.reset(); }

 protected:
  std::unique_ptr<::testing::StrictMock<testing::MockAsyncKernel>> kernel_;
  std::unique_ptr<testing::TestBackend> backend_;
  std::unique_ptr<Interpreter> interpreter_;
  internal::SignatureDef signature_def_;
  std::unique_ptr<AsyncSignatureRunner> signature_runner_;
};

TEST_F(AsyncSignatureRunnerTest, InputNameTest) {
  BuildAsyncSignatureRunner();
  EXPECT_EQ(0, GetTensorIndex(TfLiteIoType::kTfLiteIoInput, "input"));
  EXPECT_EQ(-1, GetTensorIndex(TfLiteIoType::kTfLiteIoInput, "output"));
  EXPECT_EQ(-1, GetTensorIndex(TfLiteIoType::kTfLiteIoInput, "foo"));
}

TEST_F(AsyncSignatureRunnerTest, OutputNameTest) {
  BuildAsyncSignatureRunner();
  EXPECT_EQ(1, GetTensorIndex(TfLiteIoType::kTfLiteIoOutput, "output"));
  EXPECT_EQ(-1, GetTensorIndex(TfLiteIoType::kTfLiteIoOutput, "input"));
  EXPECT_EQ(-1, GetTensorIndex(TfLiteIoType::kTfLiteIoOutput, "foo"));
}

TEST_F(AsyncSignatureRunnerTest, CreateTaskTest) {
  EXPECT_CALL(*kernel_, Finish(::testing::_, ::testing::_));

  BuildAsyncSignatureRunner();
  auto* task = signature_runner_->CreateTask();
  EXPECT_NE(nullptr, task);

  TfLiteExecutionTaskSetBuffer(task, TfLiteIoType::kTfLiteIoInput, "input", 24);
  TfLiteExecutionTaskSetBuffer(task, TfLiteIoType::kTfLiteIoOutput, "output",
                               12);
  TfLiteBufferHandle input_buffer, output_buffer;
  input_buffer = TfLiteExecutionTaskGetBufferByIndex(task, 0);
  output_buffer = TfLiteExecutionTaskGetBufferByIndex(task, 1);
  EXPECT_EQ(24, input_buffer);
  EXPECT_EQ(12, output_buffer);
  EXPECT_EQ(kTfLiteOk, signature_runner_->Finish(task));
}

}  // namespace async
}  // namespace tflite
