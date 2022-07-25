/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

using subgraph_test_util::CheckIntTensor;
using subgraph_test_util::ControlFlowOpTest;
using subgraph_test_util::FillIntTensor;

namespace {

// A simple test that performs `ADD` if condition is true, and `MUL` otherwise.
// The computation is: `cond ? a + b : a * b`.
class SimpleIfTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildIfSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
};

TEST_F(SimpleIfTest, TestIfTrue) {
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {6, 9});
}

TEST_F(SimpleIfTest, TestIfFalse) {
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5, 14});
}

TEST_F(SimpleIfTest, TestIfTrueWithLargeInputs) {
  const size_t kNumLargeTensors = 100000;
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1],
                                  {kNumLargeTensors});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const std::vector<int> input_vector(kNumLargeTensors, 1);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), input_vector);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {9});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected(kNumLargeTensors, 10);
  CheckIntTensor(output, {kNumLargeTensors}, expected);
}

TEST_F(SimpleIfTest, TestIfFalseWithLargeInputs) {
  const size_t kNumLargeTensors = 100000;
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1],
                                  {kNumLargeTensors});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const std::vector<int> input_vector(kNumLargeTensors, 1);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), input_vector);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {0});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected(kNumLargeTensors, 0);
  CheckIntTensor(output, {kNumLargeTensors}, expected);
}

// Test IF op using subgraphs with dynamically sized outputs.
// The computation is: `cond ? a + b : pad(a, b)`.
class DynamicSubgraphIfTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildPadSubgraph(interpreter_->subgraph(2));
    builder_->BuildIfSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
};

TEST_F(DynamicSubgraphIfTest, TestIfTrue) {
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  // Even if the true branch has a static type output, the output of the
  // if op is dynamic because the other branch has dynamic output.
  EXPECT_TRUE(IsDynamicTensor(output));
  CheckIntTensor(output, {1, 2}, {6, 9});
}

TEST_F(DynamicSubgraphIfTest, TestIfFalse) {
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  // The false branch has dynamic output.
  EXPECT_TRUE(IsDynamicTensor(output));
  CheckIntTensor(output, {5}, {0, 5, 7, 0, 0});
}

}  // namespace
}  // namespace tflite
