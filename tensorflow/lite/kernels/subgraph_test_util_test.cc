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

#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

namespace subgraph_test_util {

namespace {

class SubgraphBuilderTest : public ::testing::Test {
 public:
  SubgraphBuilderTest()
      : interpreter_(new Interpreter), builder_(new SubgraphBuilder) {}

  ~SubgraphBuilderTest() override {
    interpreter_.reset();
    builder_.reset();
  }

 protected:
  void TestAccumelateLoopBody(int input1, int input2, int output1,
                              int output2) {
    interpreter_.reset(new Interpreter);
    builder_->BuildAccumulateLoopBodySubgraph(
        &interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {input1});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {input2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output_tensor1 =
        interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output_tensor1, {1}, {output1});
    TfLiteTensor* output_tensor2 =
        interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output_tensor2, {1}, {output2});
  }

  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<SubgraphBuilder> builder_;
};

TEST_F(SubgraphBuilderTest, TestBuildAddSubgraph) {
  builder_->BuildAddSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1, 2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {5, 7});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {6, 9});
}

TEST_F(SubgraphBuilderTest, TestBuildMulSubgraph) {
  builder_->BuildMulSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1, 2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {5, 7});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5, 14});
}

TEST_F(SubgraphBuilderTest, TestBuildPadSubgraph) {
  builder_->BuildPadSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1, 2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {5, 7});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {5}, {0, 5, 7, 0, 0});
}

TEST_F(SubgraphBuilderTest, TestBuildLessEqualCondSubgraph) {
  builder_->BuildLessEqualCondSubgraph(&interpreter_->primary_subgraph(), 3);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {5});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {10, 10});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // Test [1, 2, 3, 4, 5] <= 3 == [true, true, true, false, false]
  // (with broadcasting).
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]),
                {1, 2, 3, 4, 5});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckBoolTensor(output, {5}, {true, true, true, false, false});
}

TEST_F(SubgraphBuilderTest, TestBuildAccumulateLoopBodySubgraph) {
  TestAccumelateLoopBody(1, 1, 2, 3);
  TestAccumelateLoopBody(2, 3, 3, 6);
  TestAccumelateLoopBody(3, 6, 4, 10);
}

TEST_F(SubgraphBuilderTest, TestBuildPadLoopBodySubgraph) {
  builder_->BuildPadLoopBodySubgraph(&interpreter_->primary_subgraph(), {1, 2});

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {5});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]),
                {0, 5, 7, 0, 0});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {2});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {8}, {0, 0, 5, 7, 0, 0, 0, 0});
}

}  // namespace
}  // namespace subgraph_test_util
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
