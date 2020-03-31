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
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {

using subgraph_test_util::CheckIntTensor;
using subgraph_test_util::ControlFlowOpTest;
using subgraph_test_util::FillIntTensor;

namespace {

class WhileTest : public ControlFlowOpTest {};

// The test builds a model that produces the i-th number of
// triangular number sequence.
//
// TODO(ycling): Consider to improve this test case by adding a
// concat into the body subgraph.
TEST_F(WhileTest, TestTriangularNumberSequence) {
  const std::vector<int> expected = {1, 3, 6, 10, 15, 21, 28};
  for (int i = 0; i < expected.size(); ++i) {
    interpreter_.reset(new Interpreter);
    interpreter_->AddSubgraphs(2);
    builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), i);
    builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
    builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output1, {1}, {i + 1});
    TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output2, {1}, {expected[i]});
  }
}

TEST_F(WhileTest, TestPadLoop) {
  interpreter_.reset(new Interpreter);
  interpreter_->AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), 3);
  builder_->BuildPadLoopBodySubgraph(interpreter_->subgraph(2), {1, 2});
  builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {4});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {11}, {0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0});

  // The extra invocation serves as a regression test: There was a bug that
  // invoking a while loop with dynamic shaped body makes the interpreter
  // state uninvokable.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
