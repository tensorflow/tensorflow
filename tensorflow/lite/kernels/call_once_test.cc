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

#include <stdint.h>

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"

namespace tflite {

using subgraph_test_util::ControlFlowOpTest;

namespace {

class CallOnceTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildCallOnceAndReadVariableSubgraph(
        &interpreter_->primary_subgraph());
    builder_->BuildAssignRandomValueToVariableSubgraph(
        interpreter_->subgraph(1));
    builder_->BuildCallOnceAndReadVariablePlusOneSubgraph(
        interpreter_->subgraph(2));

    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(interpreter_->subgraph(2)->AllocateTensors(), kTfLiteOk);
  }
};

TEST_F(CallOnceTest, TestSimple) {
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  ASSERT_EQ(output->dims->size, 1);
  ASSERT_EQ(output->dims->data[0], 1);
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(NumElements(output), 1);

  // The value of the variable must be non-zero, which will be assigned by the
  // initialization subgraph.
  EXPECT_GT(output->data.i32[0], 0);
}

TEST_F(CallOnceTest, TestInvokeMultipleTimes) {
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  ASSERT_EQ(output->dims->size, 1);
  ASSERT_EQ(output->dims->data[0], 1);
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(NumElements(output), 1);

  // The value of the variable must be non-zero, which will be assigned by the
  // initialization subgraph.
  int value = output->data.i32[0];
  EXPECT_GT(value, 0);

  for (int i = 0; i < 3; ++i) {
    // Make sure that no more random value assignment in the initialization
    // subgraph.
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
    ASSERT_EQ(output->dims->size, 1);
    ASSERT_EQ(output->dims->data[0], 1);
    ASSERT_EQ(output->type, kTfLiteInt32);
    ASSERT_EQ(NumElements(output), 1);
    ASSERT_EQ(output->data.i32[0], value);
  }
}

TEST_F(CallOnceTest, TestInvokeOnceAcrossMultipleEntryPoints) {
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  ASSERT_EQ(output->dims->size, 1);
  ASSERT_EQ(output->dims->data[0], 1);
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(NumElements(output), 1);

  // The value of the variable must be non-zero, which will be assigned by the
  // initialization subgraph.
  int value = output->data.i32[0];
  EXPECT_GT(value, 0);

  // Make sure that no more random value assignment in the initialization
  // subgraph while invoking the other subgraph, which has the CallOnce op.
  ASSERT_EQ(interpreter_->subgraph(2)->Invoke(), kTfLiteOk);
  output = interpreter_->subgraph(2)->tensor(
      interpreter_->subgraph(2)->outputs()[0]);
  ASSERT_EQ(output->dims->size, 1);
  ASSERT_EQ(output->dims->data[0], 1);
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(NumElements(output), 1);
  ASSERT_EQ(output->data.i32[0], value + 1);
}

}  // namespace
}  // namespace tflite
