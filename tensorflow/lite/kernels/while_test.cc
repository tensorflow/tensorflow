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

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/profiling/memory_info.h"

namespace tflite {

using subgraph_test_util::CheckIntTensor;
using subgraph_test_util::CheckScalarStringTensor;
using subgraph_test_util::CheckStringTensor;
using subgraph_test_util::ControlFlowOpTest;
using subgraph_test_util::FillIntTensor;
using subgraph_test_util::FillScalarStringTensor;

namespace {

class WhileTest : public ControlFlowOpTest {};

TEST_F(WhileTest, TestWithXNNPACK) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildFloatLessCondSubgraph(interpreter_->subgraph(1), 100);
  builder_->BuildXNNPACKSubgraph(interpreter_->subgraph(2));
  builder_->BuildFloatWhileSubgraph(&interpreter_->primary_subgraph(), 2);

  const auto opt = TfLiteXNNPackDelegateOptionsDefault();
  TfLiteDelegate* xnnpack_delegate = TfLiteXNNPackDelegateCreate(&opt);
  interpreter_->primary_subgraph().MarkAsDelegationSkippable();
  interpreter_->subgraph(1)->MarkAsDelegationSkippable();
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(xnnpack_delegate), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  float* input0 =
      GetTensorData<float>(interpreter_->tensor(interpreter_->inputs()[0]));
  input0[0] = 1;
  float* input1 =
      GetTensorData<float>(interpreter_->tensor(interpreter_->inputs()[1]));
  input1[0] = 1;

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  float* output0_data = GetTensorData<float>(output0);
  ASSERT_EQ(output0_data[0], 256);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  float* output1_data = GetTensorData<float>(output1);
  ASSERT_EQ(output1_data[0], 256);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteXNNPackDelegateDelete(xnnpack_delegate);
}

TEST_F(WhileTest, TestInputIsOutput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 3);
  builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {4});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestInputIsOutputButDifferent) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 2);
  builder_->BuildInputIsDifferentOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 2);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {5});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {8});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestFlexOutput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 2);
  builder_->BuildFlexOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 2);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2, 3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {4});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {2}, {5, 6});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestCounterOnly) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 1);
  builder_->BuildCounterOnlySubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 1);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestAllCases) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 5);
  builder_->BuildAllInplaceScenariosSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 5);

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
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[4]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {4});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {5});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(output2, {6}, {2, 2, 2, 2, 2, 2});
  TfLiteTensor* output3 = interpreter_->tensor(interpreter_->outputs()[3]);
  CheckIntTensor(output3, {6}, {4, 4, 4, 4, 4, 4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestStaticUnconsumedOutputs) {
  for (bool dynamic_tensors : {true, false}) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(2);
    builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 3);
    builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(2));
    builder_->BuildMultiInputWhileSubgraphWithUnconsumedOutput(
        &interpreter_->primary_subgraph(), 3);

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
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output0, {1}, {4});
    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output1, {1}, {8});

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {2}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2, 2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    CheckIntTensor(output1, {2}, {8, 8});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
}

// Test a body subgraph which triggers the reallocation of an inplace output
// tensor whose corresponding input has not been consumed yet. This tests that
// the input pointer has be updated.
TEST_F(WhileTest, TestDynamicOpTriggersAllocationOfUnsedInput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 2, 3);
  builder_->BuildDynamicOpTriggersAllocationOfUnsedInputSubgraph(
      interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 3);

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});

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

TEST_F(WhileTest, TestStaticInPlace) {
  const std::vector<int> expected = {6, 10, 15, 21, 28};
  for (int i = 0; i < expected.size(); ++i) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(2);
    builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), i + 1);
    builder_->BuildDeepBodySubgraph(interpreter_->subgraph(2));
    builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {0});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output1, {1}, {i + 2});
    TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output2, {1}, {expected[i]});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
}

TEST_F(WhileTest, TestStaticInPlaceLarge) {
  int size = 10000;
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), 60000);
  builder_->BuildLargeBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {size}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]),
                std::vector<int>(size, 1));

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {}, {10010 * size});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {size}, std::vector<int>(size, 70014));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

// The test builds a model that produces the i-th number of
// triangular number sequence.
TEST_F(WhileTest, TestTriangularNumberSequence) {
  const std::vector<int> expected = {1, 3, 6, 10, 15, 21, 28};
  for (int i = 0; i < expected.size(); ++i) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(2);
    builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), i);
    builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
    builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});

    // Check While BODY inputs are static tensors.
    auto body_subgraph = interpreter_->subgraph(2);
    TfLiteTensor* subgraph_input2 =
        body_subgraph->tensor(body_subgraph->inputs()[1]);
    EXPECT_EQ(subgraph_input2->allocation_type, kTfLiteCustom);

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output1, {1}, {i + 1});
    TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output2, {1}, {expected[i]});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
}

TEST_F(WhileTest, TestTriangularNumberSequenceWithShallowCopy) {
  const std::vector<int> expected = {1, 3, 6, 10, 15, 21, 28};
  for (int i = 0; i < expected.size(); ++i) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(2);
    builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), i);
    builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
    builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    // Use 4MB inputs to test shallow copy.
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1000000});
    // Apply DynamicAllocationForLargeTensors option to enable shallow copy.
    InterpreterOptions options;
    options.OptimizeMemoryForLargeTensors(1000000);
    ASSERT_EQ(interpreter_->ApplyOptions(&options), kTfLiteOk);
    const size_t initial_mem_usage =
        profiling::memory::GetMemoryUsage().mem_footprint_kb;
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    // Memory usage shouldn't exceed 9MB (2 x inputs + margin).
    ASSERT_LE(profiling::memory::GetMemoryUsage().mem_footprint_kb -
                  initial_mem_usage,
              9000);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
    const std::vector<int> input_vector(1000000, 1);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]),
                  input_vector);

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    auto body_subgraph = interpreter_->subgraph(2);
    // While BODY inputs are dynamic tensors with shallow copy.
    TfLiteTensor* subgraph_input2 =
        body_subgraph->tensor(body_subgraph->inputs()[1]);
    ASSERT_EQ(subgraph_input2->allocation_type, kTfLiteCustom);

    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output1, {1}, {i + 1});
    TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
    const std::vector<int> expected2(1000000, expected[i]);
    CheckIntTensor(output2, {1000000}, expected2);
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
}

TEST_F(WhileTest, TestPadLoop) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), 4);
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {5});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {14}, {0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0});

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestDynamicBodyWithSharingEarlyExit) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 0, 4);
  builder_->BuildDynamicIncreasingSizeSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 4);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {10000});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1, 2, 3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {1});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {3}, {1, 2, 3});

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestDynamicBodyWithSharing) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);

  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 4);
  builder_->BuildDynamicIncreasingSizeSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 4);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1000000});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1000000});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1, 2, 3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {4});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {18},
                 {4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  EXPECT_EQ(output2->dims->data[0], 1000000);
  TfLiteTensor* output3 = interpreter_->tensor(interpreter_->outputs()[3]);
  EXPECT_EQ(output3->dims->data[0], 1000000);

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestDynamicBodyWithSharingAndAliases) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 0, 5);
  builder_->BuildDynamicBodySubgraphWithAliases(interpreter_->subgraph(2));
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 5);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[4], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {0});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {3});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[4]), {4});

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

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestOutputNotConsumed) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 11, 3);
  builder_->BuildOutputNotConsumedSubgraph(*interpreter_->subgraph(2));
  builder_->BuildOutputNotConsumedWhileSubgraph(
      &interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {3});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {3}, {18, 18, 18});

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestPadLoopWithSharing) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLargeLessEqualCondSubgraph(interpreter_->subgraph(1), 3, 3);
  builder_->BuildLargePadSubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildMultiInputWhileSubgraph(&interpreter_->primary_subgraph(), 3);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {3, 4});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {5});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {5}, {4, 9, 10, 4, 4});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(output2, {8}, {0, 4, 9, 10, 4, 4, 0, 0});

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestPadLoopWithShallowCopy) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), 3);
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  // Use 4MB inputs to test shallow copy.
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1000000});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  std::vector<int> input_vector(1000000, 0);
  input_vector[0] = 5;
  input_vector[1] = 7;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), input_vector);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {4});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  std::vector<int> output_vector(1000009, 0);
  output_vector[3] = 5;
  output_vector[4] = 7;
  CheckIntTensor(output2, {1000009}, output_vector);

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(WhileTest, TestWhileLoopWithDynamicTensor) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraphWithDynamicTensor(
      interpreter_->subgraph(1), 3);
  builder_->BuildBodySubgraphWithDynamicTensor(interpreter_->subgraph(2));
  builder_->BuildWhileSubgraphWithDynamicTensor(
      &interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillScalarStringTensor(interpreter_->tensor(interpreter_->inputs()[0]), "A");
  FillScalarStringTensor(interpreter_->tensor(interpreter_->inputs()[1]), "A");
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* string_output1 =
      interpreter_->tensor(interpreter_->outputs()[0]);
  CheckScalarStringTensor(string_output1, "A");
  TfLiteTensor* string_output2 =
      interpreter_->tensor(interpreter_->outputs()[1]);
  CheckStringTensor(string_output2, {4}, {"A", "A", "A", "A"});
  TfLiteTensor* integer_output =
      interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(integer_output, {1}, {4});

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
