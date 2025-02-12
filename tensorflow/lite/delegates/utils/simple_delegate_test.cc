/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <stdlib.h>

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace {
class TestDelegate : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ = std::make_unique<Interpreter>();
    interpreter_->AddTensors(5);
    interpreter_->SetInputs({0, 1});
    interpreter_->SetOutputs({3, 4});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration* reg = ops::builtin::Register_ADD();
    void* builtin_data_1 = malloc(sizeof(int));
    void* builtin_data_2 = malloc(sizeof(int));
    void* builtin_data_3 = malloc(sizeof(int));
    interpreter_->AddNodeWithParameters({0, 0}, {2}, nullptr, 0, builtin_data_1,
                                        reg);
    interpreter_->AddNodeWithParameters({1, 1}, {3}, nullptr, 0, builtin_data_2,
                                        reg);
    interpreter_->AddNodeWithParameters({2, 1}, {4}, nullptr, 0, builtin_data_3,
                                        reg);
  }

  void TearDown() override { interpreter_.reset(); }

 protected:
  std::unique_ptr<Interpreter> interpreter_;
};

TEST_F(TestDelegate, BasicDelegate) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();
  options.allowed_builtin_code = kTfLiteBuiltinAdd;
  auto delegate = TfLiteDummyDelegateCreateUnique(&options);
  interpreter_->ModifyGraphWithDelegate(std::move(delegate));

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  int node = interpreter_->execution_plan()[0];
  const auto* node_and_reg = interpreter_->node_and_registration(node);
  EXPECT_STREQ("DummyDelegate", node_and_reg->second.custom_name);
  EXPECT_EQ(1, node_and_reg->second.version);

  const TfLiteDelegateParams* params = static_cast<const TfLiteDelegateParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_EQ(params->nodes_to_replace->size, 3);
  EXPECT_EQ(params->nodes_to_replace->data[0], 0);
  EXPECT_EQ(params->nodes_to_replace->data[1], 1);
  EXPECT_EQ(params->nodes_to_replace->data[2], 2);

  ASSERT_EQ(params->input_tensors->size, 2);
  EXPECT_EQ(params->input_tensors->data[0], 0);
  EXPECT_EQ(params->input_tensors->data[1], 1);

  ASSERT_EQ(params->output_tensors->size, 2);
  EXPECT_EQ(params->output_tensors->data[0], 3);
  EXPECT_EQ(params->output_tensors->data[1], 4);
}

TEST_F(TestDelegate, NoNodesToDelegate) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();
  options.allowed_builtin_code = kTfLiteBuiltinSub;
  auto delegate = TfLiteDummyDelegateCreateUnique(&options);
  interpreter_->ModifyGraphWithDelegate(std::move(delegate));

  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
}

TEST_F(TestDelegate, DelegateFailedPrepare) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();
  options.allowed_builtin_code = kTfLiteBuiltinAdd;
  options.error_during_prepare = true;
  auto delegate = TfLiteDummyDelegateCreateUnique(&options);
  ASSERT_EQ(kTfLiteDelegateError,
            interpreter_->ModifyGraphWithDelegate(std::move(delegate)));
}

TEST_F(TestDelegate, DelegateFailedInvoke) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();
  options.allowed_builtin_code = kTfLiteBuiltinAdd;
  options.error_during_invoke = true;
  auto delegate = TfLiteDummyDelegateCreateUnique(&options);
  ASSERT_EQ(kTfLiteOk,
            interpreter_->ModifyGraphWithDelegate(std::move(delegate)));
  ASSERT_EQ(kTfLiteError, interpreter_->Invoke());
}

TEST_F(TestDelegate, DelegateFailedInit) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();
  options.allowed_builtin_code = kTfLiteBuiltinAdd;
  options.error_during_init = true;
  auto delegate = TfLiteDummyDelegateCreateUnique(&options);
  ASSERT_EQ(kTfLiteDelegateError,
            interpreter_->ModifyGraphWithDelegate(std::move(delegate)));
}
}  // namespace
}  // namespace tflite
