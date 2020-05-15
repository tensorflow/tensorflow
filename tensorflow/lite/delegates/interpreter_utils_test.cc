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

#include "tensorflow/lite/delegates/interpreter_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter_test.h"

namespace tflite {
namespace {

TEST_F(TestDelegate, DelegateNodeInvokeFailureFallback) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Delegation modified execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(
      delegates::InterpreterUtils::InvokeWithCPUFallback(interpreter_.get()),
      kTfLiteDelegateError);
  // Delegation removed, returning to original execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  // Check outputs.
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, TestFallbackWithMultipleDelegates) {
  // First delegate only supports node 0.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({0}, kTfLiteDelegateFlagsAllowDynamicTensors));
  // Second delegate supports nodes 1 & 2, and makes the graph immutable.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {1, 2}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/));
  // Pre-delegation execution plan should have three nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegates nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(
      delegates::InterpreterUtils::InvokeWithCPUFallback(interpreter_.get()),
      kTfLiteDelegateError);
  // All delegates should be undone.
  EXPECT_EQ(interpreter_->execution_plan().size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

}  // namespace
}  // namespace tflite
