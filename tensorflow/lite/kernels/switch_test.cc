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
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
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

class SimpleSwitchTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {}
  void BuildGraph(TfLiteType condtype = kTfLiteBool,
                  TfLiteType inputtype = kTfLiteInt32) {
    interpreter_->AddSubgraphs(1);
    builder_->BuildSwitchSubgraph(&interpreter_->primary_subgraph(), inputtype);

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1, 2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {5, 7});
  }
};

TEST_F(SimpleSwitchTest, TestSwitchTrue) {
  BuildGraph();
  interpreter_->typed_input_tensor<bool>(1)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output, {1, 2}, {5, 7});
}

TEST_F(SimpleSwitchTest, TestSwitchFalse) {
  BuildGraph();
  interpreter_->typed_input_tensor<bool>(1)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5, 7});
}

class SimpleSwitchBoolTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {}
  void BuildGraph(TfLiteType inputtype = kTfLiteInt32) {
    interpreter_->AddSubgraphs(1);
    builder_->BuildSwitchSubgraph(&interpreter_->primary_subgraph(), inputtype);

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {5});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    tflite::subgraph_test_util::FillBoolTensor(
        interpreter_->tensor(interpreter_->inputs()[0]),
        {true, true, true, true, true});
  }
};

TEST_F(SimpleSwitchBoolTest, TestSwitchBoolTrue) {
  BuildGraph(kTfLiteBool);
  interpreter_->typed_input_tensor<bool>(1)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[1]);
  tflite::subgraph_test_util::CheckBoolTensor(output, {5},
                                              {true, true, true, true, true});
}

TEST_F(SimpleSwitchBoolTest, TestSwitchBoolFalse) {
  BuildGraph(kTfLiteBool);
  interpreter_->typed_input_tensor<bool>(1)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  tflite::subgraph_test_util::CheckBoolTensor(output, {5},
                                              {true, true, true, true, true});
}

class SimpleSwitchFloatTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {}
  void BuildGraph(TfLiteType inputtype = kTfLiteInt32) {
    interpreter_->AddSubgraphs(1);
    builder_->BuildSwitchSubgraph(&interpreter_->primary_subgraph(), inputtype);

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1, 2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    tflite::subgraph_test_util::FillFloatTensor(
        interpreter_->tensor(interpreter_->inputs()[0]), {1.0, 2.0});
  }
};

TEST_F(SimpleSwitchFloatTest, TestSwitchFloatTrue) {
  BuildGraph(kTfLiteFloat32);
  interpreter_->typed_input_tensor<bool>(1)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[1]);
  tflite::subgraph_test_util::CheckFloatTensor(output, {1, 2}, {1.0, 2.0});
}

TEST_F(SimpleSwitchFloatTest, TestSwitchFloatFalse) {
  BuildGraph(kTfLiteFloat32);
  interpreter_->typed_input_tensor<bool>(1)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  tflite::subgraph_test_util::CheckFloatTensor(output, {1, 2}, {1.0, 2.0});
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
