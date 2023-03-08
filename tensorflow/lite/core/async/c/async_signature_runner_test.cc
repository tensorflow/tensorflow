/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/async/c/async_signature_runner.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/c/internal.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/async/testing/mock_async_kernel.h"
#include "tensorflow/lite/core/async/testing/test_backend.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/interpreter_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

using ::testing::_;
using ::testing::Return;

namespace tflite {
namespace async {

class AsyncSignatureRunnerTest : public InterpreterTest {
 protected:
  void SetUp() override {
    kernel_ =
        std::make_unique<::testing::StrictMock<testing::MockAsyncKernel>>();
    backend_ = std::make_unique<testing::TestBackend>(kernel_->kernel());

    auto interpreter = std::make_unique<Interpreter>();
    interpreter->AddTensors(2);
    interpreter->SetInputs({0});
    interpreter->SetOutputs({1});
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "x", {3},
                                              quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteFloat32, "a", {3},
                                              quant);
    TfLiteRegistration* reg = ops::builtin::Register_ADD();
    void* builtin_data_1 = malloc(sizeof(int));
    interpreter->AddNodeWithParameters({0, 0}, {1}, nullptr, 0, builtin_data_1,
                                       reg);
    const char kSignatureKey[] = "serving_default";
    BuildSignature(interpreter.get(), kSignatureKey, {{"input", 0}},
                   {{"output", 1}});
    interpreter->ModifyGraphWithDelegate(backend_->get_delegate());
    tflite_interpreter_.impl = std::move(interpreter);
    runner_ = TfLiteInterpreterGetAsyncSignatureRunner(&tflite_interpreter_,
                                                       kSignatureKey);
    ASSERT_NE(nullptr, runner_);
  }

  void TearDown() override { TfLiteAsyncSignatureRunnerDelete(runner_); }

 protected:
  TfLiteAsyncSignatureRunner* runner_ = nullptr;
  std::unique_ptr<::testing::StrictMock<testing::MockAsyncKernel>> kernel_;
  std::unique_ptr<testing::TestBackend> backend_;
  internal::SignatureDef signature_def_;
  TfLiteInterpreter tflite_interpreter_{};
};

TEST_F(AsyncSignatureRunnerTest, RegisterBufferTest) {
  EXPECT_CALL(*kernel_, RegisterBuffer(_, _, _, _, _))
      .WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, RegisterBufferSlice(_, _, _, _))
      .WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, UnregisterBuffer(_, _)).WillOnce(Return(kTfLiteOk));
  TfLiteBufferHandle handle;
  auto* attr = TfLiteAttributeMapCreate(0);
  auto* buf = TfLiteBackendBufferCreate();
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerRegisterBuffer(
                           runner_, kTfLiteIoTypeInput, buf, attr, &handle));
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerRegisterBufferSlice(
                           runner_, handle, attr, &handle));
  EXPECT_EQ(kTfLiteOk,
            TfLiteAsyncSignatureRunnerUnregisterBuffer(runner_, handle));
  TfLiteAttributeMapDelete(attr);
  TfLiteBackendBufferDelete(buf);
}

TEST_F(AsyncSignatureRunnerTest, SupportedTypesTest) {
  const char* const* buffer_types = nullptr;
  size_t num_buffer_types = 0;
  EXPECT_EQ(kTfLiteOk,
            TfLiteAsyncSignatureRunnerGetSupportedBufferTypes(
                runner_, kTfLiteIoTypeInput, &buffer_types, &num_buffer_types));
  EXPECT_EQ(1, num_buffer_types);
  EXPECT_STREQ("buffer_type", buffer_types[0]);
  const char* const* sync_types = nullptr;
  size_t num_sync_types = 0;
  EXPECT_EQ(kTfLiteOk,
            TfLiteAsyncSignatureRunnerGetSupportedSynchronizationTypes(
                runner_, kTfLiteIoTypeInput, &sync_types, &num_sync_types));
  EXPECT_EQ(1, num_sync_types);
  EXPECT_STREQ("sync_type", sync_types[0]);
}

TEST_F(AsyncSignatureRunnerTest, ReconcileTest) {
  EXPECT_CALL(*kernel_, ReconcileRestrictions(_, _, _, _, _, _))
      .WillOnce(Return(true));
  EXPECT_CALL(*kernel_, SetAttributes(_, _, _, _)).WillOnce(Return(kTfLiteOk));
  auto* attr = TfLiteAttributeMapCreate(0);
  EXPECT_TRUE(TfLiteAsyncSignatureRunnerReconcileRestrictions(
      runner_, kTfLiteIoTypeInput, "input", attr, attr, nullptr));
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerSetAttributes(
                           runner_, kTfLiteIoTypeInput, "input", attr));
  TfLiteAttributeMapDelete(attr);
}

TEST_F(AsyncSignatureRunnerTest, ExecutionTest) {
  EXPECT_CALL(*kernel_, Prepare(_, _)).WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, Eval(_, _, _)).WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, Wait(_, _)).WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, Finish(_, _)).WillOnce(Return(kTfLiteOk));

  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerPrepareBackends(runner_));

  auto* task = TfLiteAsyncSignatureRunnerCreateTask(runner_);

  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerInvokeAsync(runner_, task));
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerWait(runner_, task));
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerFinish(runner_, task));
}

TEST_F(AsyncSignatureRunnerTest, InputsTest) {
  EXPECT_EQ(1, TfLiteAsyncSignatureRunnerGetInputCount(runner_));
  EXPECT_STREQ("input", TfLiteAsyncSignatureRunnerGetInputName(runner_, 0));
  EXPECT_STREQ("x",
               TfLiteOpaqueTensorName(
                   TfLiteAsyncSignatureRunnerGetInputTensor(runner_, "input")));
}

TEST_F(AsyncSignatureRunnerTest, OutputsTest) {
  EXPECT_EQ(1, TfLiteAsyncSignatureRunnerGetOutputCount(runner_));
  EXPECT_STREQ("output", TfLiteAsyncSignatureRunnerGetOutputName(runner_, 0));
  EXPECT_STREQ(
      "a", TfLiteOpaqueTensorName(
               TfLiteAsyncSignatureRunnerGetOutputTensor(runner_, "output")));
}

}  // namespace async
}  // namespace tflite
