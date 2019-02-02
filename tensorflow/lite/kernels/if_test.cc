/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace ops {
namespace builtin {
// ADD and MUL are used to test simple branch.
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_MUL();
// ADD and MUL are used to test dynamic sized subgraphs.
TfLiteRegistration* Register_PAD();
}  // namespace builtin
namespace custom {
TfLiteRegistration* Register_IF();
}  // namespace custom
}  // namespace ops

namespace {

void SetupTensor(Subgraph* subgraph, int tensor_index,
                 const std::vector<int>& shape, TfLiteType type) {
  ASSERT_EQ(subgraph->SetTensorParametersReadWrite(
                tensor_index, type, "", shape.size(), shape.data(), {}, false),
            kTfLiteOk);
}

// TODO(ycling): Consider to move all the test helper functions to another
// build target (e.g. subgraph_test_util).
// Build a subgraph with an add op. Helper function for testing.
void BuildAddSubgraph(Subgraph* subgraph) {
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({2}), kTfLiteOk);

  SetupTensor(subgraph, 0, {2}, kTfLiteInt32);
  SetupTensor(subgraph, 1, {1, 2}, kTfLiteInt32);
  // Intentionally set the wrong output size for testing. This should be
  // overridden by Prepare function.
  SetupTensor(subgraph, 2, {100}, kTfLiteInt32);

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  int node_index;
  subgraph->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_ADD(),
                                  &node_index);
}

// Build a subgraph with an mul op. Helper function for testing.
void BuildMulSubgraph(Subgraph* subgraph) {
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({2}), kTfLiteOk);

  SetupTensor(subgraph, 0, {2}, kTfLiteInt32);
  SetupTensor(subgraph, 1, {1, 2}, kTfLiteInt32);
  // Intentionally set the wrong output size for testing. This should be
  // overridden by Prepare function.
  SetupTensor(subgraph, 2, {100}, kTfLiteInt32);

  TfLiteMulParams* params =
      reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  params->activation = kTfLiteActNone;
  int node_index;
  subgraph->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_MUL(),
                                  &node_index);
}

// Build a subgraph with a pad op. Helper function for testing.
void BuildPadSubgraph(Subgraph* subgraph) {
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({2}), kTfLiteOk);

  SetupTensor(subgraph, 0, {2}, kTfLiteInt32);
  SetupTensor(subgraph, 1, {1, 2}, kTfLiteInt32);
  // Intentionally set the wrong output size for testing. This should be
  // overridden by Prepare function.
  SetupTensor(subgraph, 2, {100}, kTfLiteInt32);

  TfLitePadParams* params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLitePadParams)));
  int node_index;
  subgraph->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_PAD(),
                                  &node_index);
}

void BuildIfSubgraph(Subgraph* subgraph) {
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(4, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({0, 1, 2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({3}), kTfLiteOk);

  SetupTensor(subgraph, 0, {1}, kTfLiteBool);
  SetupTensor(subgraph, 1, {2}, kTfLiteInt32);
  SetupTensor(subgraph, 2, {1, 2}, kTfLiteInt32);
  // Intentionally set the wrong output size for testing. This should be
  // overridden by Prepare function.
  SetupTensor(subgraph, 3, {100}, kTfLiteInt32);

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("then_subgraph_index", 1);
    fbb.Int("else_subgraph_index", 2);
  });
  fbb.Finish();
  const auto& buffer = fbb.GetBuffer();

  int node_index;
  subgraph->AddNodeWithParameters(
      {0, 1, 2}, {3}, reinterpret_cast<const char*>(buffer.data()),
      buffer.size(), nullptr, ::tflite::ops::custom::Register_IF(),
      &node_index);
}

void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data) {
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    tensor->data.i32[i] = data[i];
  }
}

void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data) {
  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteInt32);
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(tensor->data.i32[i], data[i]);
  }
}

// TestHelperfunctionTest tests the helper functions defined in this file.
TEST(TestHelperfunctionTest, TestBuildAddSubgraph) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  BuildAddSubgraph(&interpreter->primary_subgraph());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter->tensor(interpreter->inputs()[0]), {5, 7});
  FillIntTensor(interpreter->tensor(interpreter->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {6, 9});
}

TEST(TestHelperfunctionTest, TestBuildMulSubgraph) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  BuildMulSubgraph(&interpreter->primary_subgraph());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter->tensor(interpreter->inputs()[0]), {5, 7});
  FillIntTensor(interpreter->tensor(interpreter->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5, 14});
}

TEST(TestHelperfunctionTest, TestBuildPadSubgraph) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  BuildPadSubgraph(&interpreter->primary_subgraph());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter->tensor(interpreter->inputs()[0]), {5, 7});
  FillIntTensor(interpreter->tensor(interpreter->inputs()[1]), {1, 2});
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
  CheckIntTensor(output, {5}, {0, 5, 7, 0, 0});
}

// A simple test that performs `ADD` if condition is true, and `MUL` otherwise.
// The computation is: `cond ? a + b : a * b`.
class SimpleIfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);
    interpreter_->AddSubgraphs(2);
    BuildAddSubgraph(interpreter_->subgraph(1));
    BuildMulSubgraph(interpreter_->subgraph(2));
    BuildIfSubgraph(&interpreter_->primary_subgraph());
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
  std::unique_ptr<Interpreter> interpreter_;
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

// Test IF op using subgraphs with dynamically sized outputs.
// The computation is: `cond ? a + b : pad(a, b)`.
class DynamicSubgraphIfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);
    interpreter_->AddSubgraphs(2);
    BuildAddSubgraph(interpreter_->subgraph(1));
    BuildPadSubgraph(interpreter_->subgraph(2));
    BuildIfSubgraph(&interpreter_->primary_subgraph());
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
  std::unique_ptr<Interpreter> interpreter_;
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

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
