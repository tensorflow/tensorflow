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
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"
#include "tensorflow/lite/core/async/testing/mock_async_kernel.h"
#include "tensorflow/lite/core/async/testing/test_backend.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter_test_util.h"

using ::testing::_;
using ::testing::Return;

namespace tflite {
namespace async {

class AsyncSignatureRunnerTest : public InterpreterTest {
 protected:
  void SetUp() override {
    InitInterpreter();
    const char kSignatureKey[] = "serving_default";
    BuildSignature(kSignatureKey, {{"input", 0}}, {{"output", 1}});
  }

  void InitInterpreter() {
    kernel_ =
        std::make_unique<::testing::StrictMock<testing::MockAsyncKernel>>();
    backend_ = std::make_unique<testing::TestBackend>(kernel_->kernel());

    interpreter_ = std::make_unique<Interpreter>();
    interpreter_->AddTensors(2);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({1});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "x", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "a", {3},
                                               quant);
    TfLiteRegistration* reg = ops::builtin::Register_ADD();
    void* builtin_data_1 = malloc(sizeof(int));
    interpreter_->AddNodeWithParameters({0, 0}, {1}, nullptr, 0, builtin_data_1,
                                        reg);
    interpreter_->ModifyGraphWithDelegate(backend_->get_delegate());
  }

  int GetTensorIndex(TfLiteIoType io_type, const char* name) {
    return signature_runner_->GetTensorIndex(io_type, name);
  }

 protected:
  std::unique_ptr<::testing::StrictMock<testing::MockAsyncKernel>> kernel_;
  std::unique_ptr<testing::TestBackend> backend_;
  internal::SignatureDef signature_def_;
  AsyncSignatureRunner* signature_runner_ = nullptr;
};

TEST_F(AsyncSignatureRunnerTest, GetAsyncSignatureRunner) {
  EXPECT_EQ(nullptr, signature_runner_);
  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");
  EXPECT_NE(nullptr, signature_runner_);
  auto* signature_runner_null_key =
      interpreter_->GetAsyncSignatureRunner(nullptr);
  EXPECT_EQ(signature_runner_null_key, signature_runner_);
  EXPECT_STREQ("serving_default", signature_runner_->signature_key().c_str());

  EXPECT_EQ(nullptr, interpreter_->GetAsyncSignatureRunner("foo"));
}

TEST_F(AsyncSignatureRunnerTest, WrongSignatureKeyTest) {
  const char kSignatureKey[] = "serving_default";
  BuildSignature(interpreter_.get(), kSignatureKey, {{"input", 0}},
                 {{"output", 1}}, 1);
  signature_runner_ = interpreter_->GetAsyncSignatureRunner(nullptr);
  EXPECT_EQ(nullptr, signature_runner_);
}

TEST_F(AsyncSignatureRunnerTest, InputsTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");
  EXPECT_EQ(1, signature_runner_->input_size());
  auto* input_name = signature_runner_->input_names()[0];
  EXPECT_STREQ("input", input_name);
  EXPECT_STREQ(
      "x", TfLiteOpaqueTensorName(signature_runner_->input_tensor(input_name)));
}

TEST_F(AsyncSignatureRunnerTest, OutputsTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");
  EXPECT_EQ(1, signature_runner_->output_size());
  auto* output_name = signature_runner_->output_names()[0];
  EXPECT_STREQ("output", output_name);
  EXPECT_STREQ("a", TfLiteOpaqueTensorName(
                        signature_runner_->output_tensor(output_name)));
}

TEST_F(AsyncSignatureRunnerTest, InputNameTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");
  EXPECT_EQ(0, GetTensorIndex(kTfLiteIoTypeInput, "input"));
  EXPECT_EQ(-1, GetTensorIndex(kTfLiteIoTypeInput, "output"));
  EXPECT_EQ(-1, GetTensorIndex(kTfLiteIoTypeInput, "foo"));
}

TEST_F(AsyncSignatureRunnerTest, OutputNameTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");
  EXPECT_EQ(1, GetTensorIndex(kTfLiteIoTypeOutput, "output"));
  EXPECT_EQ(-1, GetTensorIndex(kTfLiteIoTypeOutput, "input"));
  EXPECT_EQ(-1, GetTensorIndex(kTfLiteIoTypeOutput, "foo"));
}

TEST_F(AsyncSignatureRunnerTest, CreateTaskTest) {
  EXPECT_CALL(*kernel_, Finish(::testing::_, ::testing::_));

  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");
  auto* task = signature_runner_->CreateTask();
  EXPECT_NE(nullptr, task);

  TfLiteExecutionTaskSetBuffer(task, kTfLiteIoTypeInput, "input", 24);
  TfLiteExecutionTaskSetBuffer(task, kTfLiteIoTypeOutput, "output", 12);
  TfLiteBufferHandle input_buffer, output_buffer;
  input_buffer = TfLiteExecutionTaskGetBufferByIndex(task, 0);
  output_buffer = TfLiteExecutionTaskGetBufferByIndex(task, 1);
  EXPECT_EQ(24, input_buffer);
  EXPECT_EQ(12, output_buffer);
  EXPECT_EQ(kTfLiteOk, signature_runner_->Finish(task));
}

TEST_F(AsyncSignatureRunnerTest, ReconcileTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner("serving_default");

  EXPECT_CALL(*kernel_, ReconcileRestrictions(_, _, _, _, _, _))
      .WillOnce(Return(true));
  EXPECT_CALL(*kernel_, SetAttributes(_, _, _, _)).WillOnce(Return(kTfLiteOk));

  auto* attrs = new TfLiteAttributeMap(kTfLiteAttrMapTypeBuffer);
  EXPECT_TRUE(signature_runner_->ReconcileRestrictions(
      kTfLiteIoTypeInput, "input", attrs, attrs, attrs));
  EXPECT_EQ(kTfLiteOk, signature_runner_->SetAttributes(kTfLiteIoTypeInput,
                                                        "input", attrs));

  EXPECT_FALSE(signature_runner_->ReconcileRestrictions(
      kTfLiteIoTypeInput, "foo", attrs, attrs, attrs));
  EXPECT_EQ(kTfLiteError,
            signature_runner_->SetAttributes(kTfLiteIoTypeInput, "foo", attrs));

  delete attrs;
}

class AsyncSignatureRunnerNoSignatureDefTest : public AsyncSignatureRunnerTest {
 public:
  void SetUp() override { InitInterpreter(); }
};

TEST_F(AsyncSignatureRunnerNoSignatureDefTest, GetAsyncSignatureRunner) {
  EXPECT_EQ(nullptr, signature_runner_);
  EXPECT_NE(nullptr, interpreter_->GetAsyncSignatureRunner(nullptr));

  EXPECT_EQ(nullptr, interpreter_->GetAsyncSignatureRunner("foo"));
}

TEST_F(AsyncSignatureRunnerNoSignatureDefTest, InputsTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner(nullptr);
  EXPECT_EQ(1, signature_runner_->input_size());
  EXPECT_EQ(1, signature_runner_->input_names().size());

  EXPECT_EQ(1, signature_runner_->inputs().size());
  EXPECT_NE(nullptr, signature_runner_->tensor(signature_runner_->inputs()[0]));
}

TEST_F(AsyncSignatureRunnerNoSignatureDefTest, OutputsTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner(nullptr);
  EXPECT_EQ(1, signature_runner_->output_size());
  EXPECT_EQ(1, signature_runner_->output_names().size());

  EXPECT_EQ(1, signature_runner_->outputs().size());
  EXPECT_NE(nullptr,
            signature_runner_->tensor(signature_runner_->outputs()[0]));
}

TEST_F(AsyncSignatureRunnerNoSignatureDefTest, CreateTaskTest) {
  EXPECT_CALL(*kernel_, Finish(::testing::_, ::testing::_));

  signature_runner_ = interpreter_->GetAsyncSignatureRunner(nullptr);
  auto* task = signature_runner_->CreateTask();
  EXPECT_NE(nullptr, task);

  TfLiteExecutionTaskSetBufferByIndex(task, 0, 24);
  TfLiteExecutionTaskSetBufferByIndex(task, 1, 12);
  TfLiteBufferHandle input_buffer, output_buffer;
  input_buffer = TfLiteExecutionTaskGetBufferByIndex(task, 0);
  output_buffer = TfLiteExecutionTaskGetBufferByIndex(task, 1);
  EXPECT_EQ(24, input_buffer);
  EXPECT_EQ(12, output_buffer);
  EXPECT_EQ(kTfLiteOk, signature_runner_->Finish(task));
}

TEST_F(AsyncSignatureRunnerNoSignatureDefTest, ReconcileTest) {
  signature_runner_ = interpreter_->GetAsyncSignatureRunner(nullptr);

  EXPECT_CALL(*kernel_, ReconcileRestrictions(_, _, _, _, _, _))
      .WillOnce(Return(true));
  EXPECT_CALL(*kernel_, SetAttributes(_, _, _, _)).WillOnce(Return(kTfLiteOk));

  auto* attrs = new TfLiteAttributeMap(kTfLiteAttrMapTypeBuffer);
  EXPECT_TRUE(signature_runner_->ReconcileRestrictions(0, attrs, attrs, attrs));
  EXPECT_EQ(kTfLiteOk, signature_runner_->SetAttributes(0, attrs));

  EXPECT_FALSE(
      signature_runner_->ReconcileRestrictions(42, attrs, attrs, attrs));
  EXPECT_EQ(kTfLiteError, signature_runner_->SetAttributes(42, attrs));

  delete attrs;
}

}  // namespace async
}  // namespace tflite
