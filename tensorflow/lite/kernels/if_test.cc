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
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"

namespace tflite {

using subgraph_test_util::CheckIntTensor;
using subgraph_test_util::CheckScalarStringTensor;
using subgraph_test_util::CheckStringTensor;
using subgraph_test_util::ControlFlowOpTest;
using subgraph_test_util::FillIntTensor;
using subgraph_test_util::FillScalarStringTensor;

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

TEST_F(SimpleIfTest, TestIfTrueWithLargeInputsTwice) {
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

  // Second invocation.
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {19});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected2(kNumLargeTensors, 20);
  CheckIntTensor(output, {kNumLargeTensors}, expected2);
}

TEST_F(SimpleIfTest, TestIfFalseWithLargeInputsTwice) {
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

  // Second invocation.
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {7});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected2(kNumLargeTensors, 7);
  CheckIntTensor(output, {kNumLargeTensors}, expected2);
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

class IfTest : public ControlFlowOpTest {};

TEST_F(IfTest, TestWithXNNPACK) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildXNNPACKSubgraph(interpreter_->subgraph(1));
  builder_->BuildXNNPACKSubgraph(interpreter_->subgraph(2));
  builder_->BuildFloatIfSubgraph(&interpreter_->primary_subgraph(), 3);

  const auto opt = TfLiteXNNPackDelegateOptionsDefault();
  TfLiteDelegate* xnnpack_delegate = TfLiteXNNPackDelegateCreate(&opt);
  interpreter_->primary_subgraph().MarkAsDelegationSkippable();
  interpreter_->subgraph(1)->MarkAsDelegationSkippable();
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(xnnpack_delegate), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  float* input0 =
      GetTensorData<float>(interpreter_->tensor(interpreter_->inputs()[1]));
  input0[0] = 1;
  float* input1 =
      GetTensorData<float>(interpreter_->tensor(interpreter_->inputs()[2]));
  input1[0] = 1;

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  float* output0_data = GetTensorData<float>(output0);
  ASSERT_EQ(output0_data[0], 4);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  float* output1_data = GetTensorData<float>(output1);
  ASSERT_EQ(output1_data[0], 4);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteXNNPackDelegateDelete(xnnpack_delegate);
}

TEST_F(IfTest, TestInputIsOutput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(1));
  builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 4);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {2});

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  CheckIntTensor(output0, {1}, {2});
  CheckIntTensor(output1, {1}, {2});
}

TEST_F(IfTest, TestInputIsOutputButDifferent) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildInputIsDifferentOutputSubgraph(interpreter_->subgraph(1));
  builder_->BuildInputIsDifferentOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestFlexOutput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildFlexOutputSubgraph(interpreter_->subgraph(1));
  builder_->BuildFlexOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2, 3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {2}, {3, 4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestCounterOnly) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildCounterOnlySubgraph(interpreter_->subgraph(1));
  builder_->BuildCounterOnlySubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 2);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestAllCases) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildAllInplaceScenariosSubgraph(interpreter_->subgraph(1));
  builder_->BuildAllInplaceScenariosSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 6);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[4], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[5], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[4]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[5]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {3});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {3});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(output2, {2}, {2, 2});
  TfLiteTensor* output3 = interpreter_->tensor(interpreter_->outputs()[3]);
  CheckIntTensor(output3, {2}, {3, 3});
  TfLiteTensor* output4 = interpreter_->tensor(interpreter_->outputs()[4]);
  CheckIntTensor(output4, {1}, {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestStaticUnconsumedOutputs) {
  for (bool dynamic_tensors : {true, false}) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(2);
    builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(1));
    builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(2));
    builder_->BuildMultiInputIfSubgraphWithUnconsumedOutput(
        &interpreter_->primary_subgraph(), 4);

    InterpreterOptions options;
    if (dynamic_tensors) {
      options.OptimizeMemoryForLargeTensors(1);
      interpreter_->ApplyOptions(&options);
    }

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    interpreter_->typed_input_tensor<bool>(0)[0] = true;
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output0, {1}, {2});
    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output1, {1}, {4});

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {2}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2, 2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    CheckIntTensor(output1, {2}, {4, 4});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    interpreter_->typed_input_tensor<bool>(0)[0] = false;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
}

// Test a body subgraph which triggers the reallocation of an inplace output
// tensor whose corresponding input has not been consumed yet. This tests that
// the input pointer has be updated.
TEST_F(IfTest, TestDynamicOpTriggersAllocationOfUnsedInput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildDynamicOpTriggersAllocationOfUnsedInputSubgraph(
      interpreter_->subgraph(1));
  builder_->BuildDynamicOpTriggersAllocationOfUnsedInputSubgraph(
      interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 4);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {3});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {2}, {4, 4});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(output2, {2}, {2, 2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestStaticInPlace) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildDeepBodySubgraph(interpreter_->subgraph(1));
  builder_->BuildDeepBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {0});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {1});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {1}, {3});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestStaticInPlaceLarge) {
  int size = 10000;
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeBodySubgraph(interpreter_->subgraph(1));
  builder_->BuildLargeBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {size}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]),
                std::vector<int>(size, 1));

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {}, {10000});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {size}, std::vector<int>(size, 6));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

// The test builds a model that produces the i-th number of
// triangular number sequence.
TEST_F(IfTest, TestTriangularNumberSequence) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(1));
  builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});

  // Check If BODY inputs are static tensors.
  auto body_subgraph = interpreter_->subgraph(2);
  TfLiteTensor* subgraph_input2 =
      body_subgraph->tensor(body_subgraph->inputs()[1]);
  EXPECT_EQ(subgraph_input2->allocation_type, kTfLiteCustom);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {2});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {1}, {3});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestTriangularNumberSequenceWithShallowCopy) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(1));
  builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  // Use 4MB inputs to test shallow copy.
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1000000});
  // Apply DynamicAllocationForLargeTensors option to enable shallow copy.
  InterpreterOptions options;
  options.OptimizeMemoryForLargeTensors(1000000);
  ASSERT_EQ(interpreter_->ApplyOptions(&options), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  const std::vector<int> input_vector(1000000, 1);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), input_vector);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  auto body_subgraph = interpreter_->subgraph(2);
  // If BODY inputs are dynamic tensors with shallow copy.
  TfLiteTensor* subgraph_input2 =
      body_subgraph->tensor(body_subgraph->inputs()[1]);
  ASSERT_EQ(subgraph_input2->allocation_type, kTfLiteCustom);

  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {2});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  const std::vector<int> expected2(1000000, 3);
  CheckIntTensor(output2, {1000000}, expected2);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestPadLoop) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(1), {1, 2});
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {5, 7});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {2});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {5}, {0, 5, 7, 0, 0});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestDynamicBodyWithSharingEarlyExit) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildDynamicIncreasingSizeSubgraph(interpreter_->subgraph(1));
  builder_->BuildDynamicIncreasingSizeSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 5);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {10000});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2, 3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {3}, {2, 3, 4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestDynamicBodyWithSharing) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);

  builder_->BuildDynamicIncreasingSizeSubgraph(interpreter_->subgraph(1));
  builder_->BuildDynamicIncreasingSizeSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 5);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1000000});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[4], {1000000});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2, 3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {3}, {2, 3, 4});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  EXPECT_EQ(output2->dims->data[0], 1000000);
  TfLiteTensor* output3 = interpreter_->tensor(interpreter_->outputs()[3]);
  EXPECT_EQ(output3->dims->data[0], 1000000);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestDynamicBodyWithSharingAndAliases) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildDynamicBodySubgraphWithAliases(interpreter_->subgraph(1));
  builder_->BuildDynamicBodySubgraphWithAliases(interpreter_->subgraph(2));
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 6);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[4], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {0});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[4]), {3});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[5]), {4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {1});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {11});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(output2, {1}, {12});
  TfLiteTensor* output3 = interpreter_->tensor(interpreter_->outputs()[4]);
  CheckIntTensor(output3, {1}, {13});
  TfLiteTensor* output4 = interpreter_->tensor(interpreter_->outputs()[4]);
  CheckIntTensor(output4, {1}, {13});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestOutputNotConsumed) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildOutputNotConsumedSubgraph(*interpreter_->subgraph(1));
  builder_->BuildOutputNotConsumedSubgraph(*interpreter_->subgraph(2));
  builder_->BuildOutputNotConsumedIfSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestPadLoopWithSharing) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargePadSubgraph(interpreter_->subgraph(1), {1, 2});
  builder_->BuildLargePadSubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 4);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {3, 4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {3});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {2}, {5, 6});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(output2, {5}, {0, 5, 6, 0, 0});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestPadLoopWithShallowCopy) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(1), {1, 2});
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildMultiInputIfSubgraph(&interpreter_->primary_subgraph(), 3);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  // Use 4MB inputs to test shallow copy.
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1000000});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  std::vector<int> input_vector(1000000, 0);
  input_vector[0] = 5;
  input_vector[1] = 7;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), input_vector);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {2});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  std::vector<int> output_vector(1000003, 0);
  output_vector[1] = 5;
  output_vector[2] = 7;
  CheckIntTensor(output2, {1000003}, output_vector);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(IfTest, TestIfLoopWithDynamicTensor) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildBodySubgraphWithDynamicTensor(interpreter_->subgraph(1));
  builder_->BuildBodySubgraphWithDynamicTensor(interpreter_->subgraph(2));
  builder_->BuildIfSubgraphWithDynamicTensor(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  FillScalarStringTensor(interpreter_->tensor(interpreter_->inputs()[1]), "A");
  FillScalarStringTensor(interpreter_->tensor(interpreter_->inputs()[2]), "A");
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* string_output1 =
      interpreter_->tensor(interpreter_->outputs()[0]);
  CheckScalarStringTensor(string_output1, "A");
  TfLiteTensor* string_output2 =
      interpreter_->tensor(interpreter_->outputs()[1]);
  CheckStringTensor(string_output2, {2}, {"A", "A"});
  TfLiteTensor* integer_output =
      interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(integer_output, {1}, {2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
