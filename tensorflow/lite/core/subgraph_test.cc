/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/subgraph.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"

namespace tflite {

namespace ops {
namespace builtin {
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_NEG();
}  // namespace builtin
}  // namespace ops

namespace {

TEST(RemoveUnusedInputs, NothingToRemove) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(4);
  subgraph.SetInputs({0, 1});
  subgraph.SetOutputs({3});
  TfLiteRegistration* pad_op = tflite::ops::builtin::Register_PADV2();
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  subgraph.AddNodeWithParameters({0, 1}, {2}, {}, nullptr, 0, nullptr, pad_op);
  subgraph.AddNodeWithParameters({2}, {3}, {}, nullptr, 0, nullptr, neg_op);

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({0, 1}));
}

TEST(RemoveUnusedInputs, HasUnusedInputs) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(4);
  subgraph.SetInputs({0, 1, 2});
  subgraph.SetOutputs({3});
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  subgraph.AddNodeWithParameters({2}, {3}, {}, nullptr, 0, nullptr, neg_op);

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({-1, -1, 2}));
}

TEST(RemoveUnusedInputs, BypassInputsWithoutOp) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(3);
  subgraph.SetInputs({0, 1, 2});
  subgraph.SetOutputs({0, 2});

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({0, -1, 2}));
}

}  // namespace
}  // namespace tflite
