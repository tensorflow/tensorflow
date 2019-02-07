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

// SubGraphTestUtilTest tests the helper functions defined in this file.
TEST(SubGraphTestUtilTest, TestBuildAddSubgraph) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  BuildAddSubgraph(&interpreter->primary_subgraph());

  interpreter->ResizeInputTensor(interpreter->inputs()[0], {2});
  interpreter->ResizeInputTensor(interpreter->inputs()[1], {1, 2});
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter->tensor(interpreter->inputs()[0]), {5, 7});
  FillIntTensor(interpreter->tensor(interpreter->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {6, 9});
}

TEST(SubGraphTestUtilTest, TestBuildMulSubgraph) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  BuildMulSubgraph(&interpreter->primary_subgraph());

  interpreter->ResizeInputTensor(interpreter->inputs()[0], {2});
  interpreter->ResizeInputTensor(interpreter->inputs()[1], {1, 2});
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter->tensor(interpreter->inputs()[0]), {5, 7});
  FillIntTensor(interpreter->tensor(interpreter->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5, 14});
}

TEST(SubGraphTestUtilTest, TestBuildPadSubgraph) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  BuildPadSubgraph(&interpreter->primary_subgraph());

  interpreter->ResizeInputTensor(interpreter->inputs()[0], {2});
  interpreter->ResizeInputTensor(interpreter->inputs()[1], {1, 2});
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  FillIntTensor(interpreter->tensor(interpreter->inputs()[0]), {5, 7});
  FillIntTensor(interpreter->tensor(interpreter->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
  CheckIntTensor(output, {5}, {0, 5, 7, 0, 0});
}

}  // namespace
}  // namespace subgraph_test_util
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
