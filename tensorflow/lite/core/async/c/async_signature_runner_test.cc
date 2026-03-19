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
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter_test_util.h"

using ::testing::_;
using ::testing::Return;

namespace tflite {
namespace async {

class AsyncSignatureRunnerTest : public InterpreterTest,
                                 public ::testing::WithParamInterface<bool> {
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
    tflite_interpreter_.impl = std::move(interpreter);
  }

  void BuildRunner(bool has_signature) {
    auto* interpreter = tflite_interpreter_.impl.get();
    if (has_signature) {
      const char kSignatureKey[] = "serving_default";
      BuildSignature(interpreter, kSignatureKey, {{"input", 0}},
                     {{"output", 1}});
      interpreter->ModifyGraphWithDelegate(backend_->get_delegate());
      runner_ = TfLiteInterpreterGetAsyncSignatureRunner(&tflite_interpreter_,
                                                         kSignatureKey);
    } else {
      interpreter->ModifyGraphWithDelegate(backend_->get_delegate());
      runner_ = TfLiteInterpreterGetAsyncSignatureRunner(&tflite_interpreter_,
                                                         nullptr);
    }
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

INSTANTIATE_TEST_SUITE_P(AsyncSignatureRunnerTest, AsyncSignatureRunnerTest,
                         ::testing::Bool());

TEST_P(AsyncSignatureRunnerTest, RegisterBufferTest) {
  BuildRunner(GetParam());
  EXPECT_CALL(*kernel_, RegisterBuffer(_, _, _, _, _))
      .WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, RegisterBufferSlice(_, _, _, _))
      .WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel_, UnregisterBuffer(_, _)).WillOnce(Return(kTfLiteOk));
  TfLiteBufferHandle handle;
  auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeBuffer);
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

TEST_P(AsyncSignatureRunnerTest, SupportedTypesTest) {
  BuildRunner(GetParam());
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

TEST_P(AsyncSignatureRunnerTest, ReconcileTest) {
  bool has_signature = GetParam();
  BuildRunner(has_signature);
  EXPECT_CALL(*kernel_, ReconcileRestrictions(_, _, _, _, _, _))
      .WillOnce(Return(true));
  EXPECT_CALL(*kernel_, SetAttributes(_, _, _, _)).WillOnce(Return(kTfLiteOk));
  auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeBuffer);
  if (has_signature) {
    EXPECT_TRUE(TfLiteAsyncSignatureRunnerReconcileRestrictions(
        runner_, kTfLiteIoTypeInput, "input", attr, attr, nullptr));
    EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerSetAttributes(
                             runner_, kTfLiteIoTypeInput, "input", attr));
  } else {
    EXPECT_TRUE(TfLiteAsyncSignatureRunnerReconcileRestrictionsByIndex(
        runner_, 0, attr, attr, nullptr));
    EXPECT_EQ(kTfLiteOk,
              TfLiteAsyncSignatureRunnerSetAttributesByIndex(runner_, 0, attr));
  }
  TfLiteAttributeMapDelete(attr);
}

TEST_P(AsyncSignatureRunnerTest, ExecutionTest) {
  BuildRunner(GetParam());
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

TEST_P(AsyncSignatureRunnerTest, InputsTest) {
  bool has_signature = GetParam();
  BuildRunner(has_signature);
  EXPECT_EQ(1, TfLiteAsyncSignatureRunnerGetInputCount(runner_));
  if (has_signature) {
    EXPECT_STREQ("input", TfLiteAsyncSignatureRunnerGetInputName(runner_, 0));
    EXPECT_STREQ(
        "x", TfLiteOpaqueTensorName(
                 TfLiteAsyncSignatureRunnerGetInputTensor(runner_, "input")));
  } else {
    EXPECT_STREQ("x", TfLiteAsyncSignatureRunnerGetInputName(runner_, 0));
    EXPECT_STREQ("x",
                 TfLiteOpaqueTensorName(
                     TfLiteAsyncSignatureRunnerGetInputTensor(runner_, "x")));
  }
}

TEST_P(AsyncSignatureRunnerTest, OutputsTest) {
  bool has_signature = GetParam();
  BuildRunner(has_signature);
  EXPECT_EQ(1, TfLiteAsyncSignatureRunnerGetOutputCount(runner_));
  if (has_signature) {
    EXPECT_STREQ("output", TfLiteAsyncSignatureRunnerGetOutputName(runner_, 0));
    EXPECT_STREQ(
        "a", TfLiteOpaqueTensorName(
                 TfLiteAsyncSignatureRunnerGetOutputTensor(runner_, "output")));
  } else {
    EXPECT_STREQ("a", TfLiteAsyncSignatureRunnerGetOutputName(runner_, 0));
    EXPECT_STREQ("a",
                 TfLiteOpaqueTensorName(
                     TfLiteAsyncSignatureRunnerGetOutputTensor(runner_, "a")));
  }
}

TEST_P(AsyncSignatureRunnerTest, InputByIndexTest) {
  BuildRunner(GetParam());
  EXPECT_EQ(1, TfLiteAsyncSignatureRunnerGetInputCount(runner_));
  auto* indices = TfLiteAsyncSignatureRunnerInputTensorIndices(runner_);
  EXPECT_NE(nullptr, indices);
  auto indice = indices[0];
  EXPECT_STREQ("x", TfLiteOpaqueTensorName(
                        TfLiteAsyncSignatureRunnerGetTensor(runner_, indice)));
}

TEST_P(AsyncSignatureRunnerTest, OutputsByIndexTest) {
  BuildRunner(GetParam());
  EXPECT_EQ(1, TfLiteAsyncSignatureRunnerGetOutputCount(runner_));
  auto* indices = TfLiteAsyncSignatureRunnerOutputTensorIndices(runner_);
  EXPECT_NE(nullptr, indices);
  auto indice = indices[0];
  EXPECT_STREQ("a", TfLiteOpaqueTensorName(
                        TfLiteAsyncSignatureRunnerGetTensor(runner_, indice)));
}

TEST_P(AsyncSignatureRunnerTest, IndexOutOfBound) {
  BuildRunner(GetParam());
  EXPECT_EQ(nullptr, TfLiteAsyncSignatureRunnerGetTensor(runner_, 42));
}

TEST(AsyncSignatureRunnerTest, TestNoSignatures) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "third_party/tensorflow/lite/testdata/no_signatures.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  auto kernel =
      std::make_unique<::testing::StrictMock<testing::MockAsyncKernel>>();
  auto backend = std::make_unique<testing::TestBackend>(kernel->kernel());
  TfLiteInterpreterOptionsAddDelegate(options, backend->get_delegate());

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  TfLiteInterpreterOptionsDelete(options);

  int nun_signatures = TfLiteInterpreterGetSignatureCount(interpreter);
  ASSERT_EQ(nun_signatures, 0);

  ASSERT_EQ(TfLiteInterpreterGetAsyncSignatureRunner(interpreter, "foo"),
            nullptr);

  TfLiteAsyncSignatureRunner* runner =
      TfLiteInterpreterGetAsyncSignatureRunner(interpreter, nullptr);
  ASSERT_NE(runner, nullptr);

  int num_interpreter_inputs =
      TfLiteInterpreterGetInputTensorCount(interpreter);
  int num_runner_inputs = TfLiteAsyncSignatureRunnerGetInputCount(runner);
  ASSERT_EQ(num_runner_inputs, num_interpreter_inputs);

  for (int i = 0; i < num_interpreter_inputs; ++i) {
    auto* interpreter_input_tensor =
        TfLiteInterpreterGetInputTensor(interpreter, i);
    ASSERT_NE(interpreter_input_tensor, nullptr);
    auto* interpreter_input_name = TfLiteTensorName(interpreter_input_tensor);
    ASSERT_NE(interpreter_input_name, nullptr);
    auto* runner_input_name = TfLiteAsyncSignatureRunnerGetInputName(runner, i);
    ASSERT_NE(runner_input_name, nullptr);
    EXPECT_STREQ(runner_input_name, interpreter_input_name);
    auto* runner_input_tensor = TfLiteAsyncSignatureRunnerGetInputTensor(
        runner, interpreter_input_name);
    ASSERT_NE(runner_input_tensor, nullptr);
    ASSERT_EQ(runner_input_tensor, reinterpret_cast<const TfLiteOpaqueTensor*>(
                                       interpreter_input_tensor));
  }

  int num_interpreter_outputs =
      TfLiteInterpreterGetOutputTensorCount(interpreter);
  int num_runner_outputs = TfLiteAsyncSignatureRunnerGetOutputCount(runner);
  ASSERT_EQ(num_runner_outputs, num_interpreter_outputs);

  for (int i = 0; i < num_interpreter_outputs; ++i) {
    auto* interpreter_output_tensor =
        TfLiteInterpreterGetOutputTensor(interpreter, i);
    ASSERT_NE(interpreter_output_tensor, nullptr);
    auto* interpreter_output_name = TfLiteTensorName(interpreter_output_tensor);
    ASSERT_NE(interpreter_output_name, nullptr);
    auto* runner_output_name =
        TfLiteAsyncSignatureRunnerGetOutputName(runner, i);
    ASSERT_NE(runner_output_name, nullptr);
    EXPECT_STREQ(runner_output_name, interpreter_output_name);
    auto* runner_output_tensor = TfLiteAsyncSignatureRunnerGetOutputTensor(
        runner, interpreter_output_name);
    ASSERT_NE(runner_output_tensor, nullptr);
    ASSERT_EQ(runner_output_tensor, reinterpret_cast<const TfLiteOpaqueTensor*>(
                                        interpreter_output_tensor));
  }

  EXPECT_CALL(*kernel, Prepare(_, _)).WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel, Eval(_, _, _)).WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel, Wait(_, _)).WillOnce(Return(kTfLiteOk));
  EXPECT_CALL(*kernel, Finish(_, _)).WillOnce(Return(kTfLiteOk));

  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerPrepareBackends(runner));

  auto* task = TfLiteAsyncSignatureRunnerCreateTask(runner);

  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerInvokeAsync(runner, task));
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerWait(runner, task));
  EXPECT_EQ(kTfLiteOk, TfLiteAsyncSignatureRunnerFinish(runner, task));

  TfLiteAsyncSignatureRunnerDelete(runner);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

}  // namespace async
}  // namespace tflite
