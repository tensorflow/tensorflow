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

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

// Forward declaration for op kernels.
namespace ops {
namespace custom {

TfLiteRegistration* Register_ASSIGN_VARIABLE();
TfLiteRegistration* Register_READ_VARIABLE();

namespace random_int {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  // TODO(jaesung): Make output size be changeable depending on user's input to
  // make it generic.
  return context->ResizeTensor(context, output, outputSize);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  std::random_device rd;
  std::uniform_int_distribution<int> dist(1, 32768);
  output.data.i32[0] = dist(rd);
  return kTfLiteOk;
}

}  // namespace random_int

TfLiteRegistration* Register_RANDOM_INT() {
  static TfLiteRegistration r = {nullptr, nullptr, random_int::Prepare,
                                 random_int::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops

namespace subgraph_test_util {

namespace {

void SetupTensor(Subgraph* subgraph, int tensor_index, TfLiteType type) {
  ASSERT_EQ(subgraph->SetTensorParametersReadWrite(tensor_index, type, "", 0,
                                                   nullptr, {}, false),
            kTfLiteOk);
}

}  // namespace

SubgraphBuilder::~SubgraphBuilder() {
  for (auto buffer : buffers_) {
    free(buffer);
  }
}

void SubgraphBuilder::BuildAddSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |ADD| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, add_reg, &node_index);
}

// Build a subgraph with an mul op. Helper function for testing.
void SubgraphBuilder::BuildMulSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |MUL| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteMulParams* params =
      reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  params->activation = kTfLiteActNone;
  auto* mul_reg = ops::builtin::Register_MUL();
  mul_reg->builtin_code = kTfLiteBuiltinMul;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, mul_reg, &node_index);
}

// Build a subgraph with a pad op. Helper function for testing.
void SubgraphBuilder::BuildPadSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |PAD| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLitePadParams* params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLitePadParams)));
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, pad_reg, &node_index);
}

void SubgraphBuilder::BuildIfSubgraph(Subgraph* subgraph) {
  const int kCondInput = 0;
  const int kInput1 = 1;
  const int kInput2 = 2;
  const int kOutput = 3;
  const int kTensorCount = 4;

  // kCondInput(0) --> +----+
  // kInput1(1)  ----> | IF | --> kOutput(3)
  // kInput2(2)  ----> +----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kCondInput, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kCondInput, kTfLiteBool);
  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteIfParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters({kCondInput, kInput1, kInput2}, {kOutput}, {},
                                  nullptr, 0, params, if_reg, &node_index);
}

void SubgraphBuilder::BuildLessEqualCondSubgraph(Subgraph* subgraph, int rhs) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kConstRhs = 3;
  const int kTensorCount = 4;

  // kInput1(0) ----> +------------+
  //                  | LESS_EQUAL | --> kOutput(2)
  // kConstRhs(3) --> +------------+
  //
  // kInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantInt32Tensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kConstRhs}, {kOutput}, {}, nullptr,
                                  0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildAccumulateLoopBodySubgraph(Subgraph* subgraph) {
  const int kInputCounter = 0;
  const int kInputValue = 1;
  const int kOutputCounter = 2;
  const int kOutputValue = 3;
  const int kConstStep = 4;
  const int kTensorCount = 5;

  // kInputCounter(0) --> +-----+
  //                      | ADD | --> kOutputCounter(2)
  // kConstStep(4) -----> +-----+            |
  //                                         |
  //                                         v
  //                                      +-----+
  //                                      | ADD | --> kOutputValue(3)
  // kInputValue(1) ----------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  SetupTensor(subgraph, kInputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kInputValue, kTfLiteInt32);
  SetupTensor(subgraph, kOutputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kOutputValue, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstStep, {1}, {1});

  int node_index;
  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({0, 4}, {2}, {}, nullptr, 0, params, add_reg,
                                  &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  subgraph->AddNodeWithParameters({2, 1}, {3}, {}, nullptr, 0, params, add_reg,
                                  &node_index);
}

void SubgraphBuilder::BuildPadLoopBodySubgraph(Subgraph* subgraph,
                                               const std::vector<int> padding) {
  const int kInputCounter = 0;
  const int kInputValue = 1;
  const int kOutputCounter = 2;
  const int kOutputValue = 3;
  const int kConstStep = 4;
  const int kConstPadding = 5;
  const int kTensorCount = 6;

  // kInputCounter(0) --> +-----+
  //                      | ADD | --> kOutputCounter(2)
  // kConstStep(4) -----> +-----+
  //
  // kInputValue(1) ----> +-----+
  //                      | PAD | --> kOutputValue(3)
  // kConstPadding(5) --> +-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  SetupTensor(subgraph, kInputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kInputValue, kTfLiteInt32);
  SetupTensor(subgraph, kOutputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kOutputValue, kTfLiteInt32);

  CreateConstantInt32Tensor(subgraph, kConstStep, {1}, {1});
  ASSERT_EQ(padding.size() % 2, 0);
  int padding_dims = padding.size();
  CreateConstantInt32Tensor(subgraph, kConstPadding, {1, padding_dims},
                            padding);

  int node_index;
  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({kInputCounter, kConstStep}, {kOutputCounter},
                                  {}, nullptr, 0, add_params, add_reg,
                                  &node_index);
  TfLitePadParams* pad_params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLiteAddParams)));
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  subgraph->AddNodeWithParameters({kInputValue, kConstPadding}, {kOutputValue},
                                  {}, nullptr, 0, pad_params, pad_reg,
                                  &node_index);
}

void SubgraphBuilder::BuildWhileSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput1 = 2;
  const int kOutput2 = 3;
  const int kTensorCount = 4;

  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput1, kOutput2}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput1, kTfLiteInt32);
  SetupTensor(subgraph, kOutput2, kTfLiteInt32);

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters({0, 1}, {2, 3}, {}, nullptr, 0, params,
                                  while_reg, &node_index);
}

void SubgraphBuilder::BuildAssignRandomValueToVariableSubgraph(
    Subgraph* subgraph) {
  const int kConstResourceId = 0;
  const int kRandomValue = 1;
  const int kTensorCount = 3;

  // Construct a graph like ths:
  //   %1 = random_int()
  //   variable_assign(%0, %1)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({}), kTfLiteOk);

  SetupTensor(subgraph, kRandomValue, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstResourceId, {1}, {1024});

  int node_index;
  subgraph->AddNodeWithParameters({}, {kRandomValue}, {}, nullptr, 0, nullptr,
                                  ::tflite::ops::custom::Register_RANDOM_INT(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId, kRandomValue}, {}, {}, nullptr, 0, nullptr,
      ::tflite::ops::custom::Register_ASSIGN_VARIABLE(), &node_index);
}

void SubgraphBuilder::BuildCallOnceAndReadVariableSubgraph(Subgraph* subgraph) {
  const int kConstResourceId = 0;
  const int kOutput = 1;
  const int kTensorCount = 2;

  // Construct a graph like ths:
  //   Output: %1
  //   %1 = read_variable(%0)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kOutput, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstResourceId, {1}, {1024});

  TfLiteCallOnceParams* params = reinterpret_cast<TfLiteCallOnceParams*>(
      malloc(sizeof(TfLiteCallOnceParams)));
  params->init_subgraph_index = 1;

  int node_index;
  subgraph->AddNodeWithParameters({}, {}, {}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_CALL_ONCE(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId}, {kOutput}, {}, nullptr, 0, nullptr,
      ::tflite::ops::custom::Register_READ_VARIABLE(), &node_index);
}

void SubgraphBuilder::BuildLessEqualCondSubgraphWithDynamicTensor(
    Subgraph* subgraph, int rhs) {
  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kOutput = 3;
  const int kConstRhs = 4;
  const int kTensorCount = 5;

  // kIntegerInput(2) --> +------------+
  //                      | LESS_EQUAL | --> kOutput(3)
  //     kConstRhs(4) --> +------------+
  //
  // kStringInput1(0) --> (unused)
  // kStringInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantInt32Tensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kIntegerInput, kConstRhs}, {kOutput}, {},
                                  nullptr, 0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildBodySubgraphWithDynamicTensor(Subgraph* subgraph) {
  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kStringOutput1 = 0;  // Forwarded of the `kStringInput1` tensor.
  const int kStringOutput2 = 4;
  const int kIntegerOutput = 5;
  const int kConst = 6;
  const int kTensorCount = 7;

  // Construct a graph like this:
  //   %5 = tf.Add(%2, 1)
  //   %4 = tf.Fill(%0, %5)
  //   yield(%0, %4, %5)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kStringOutput1, kStringOutput2, kIntegerOutput}),
      kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);
  SetupTensor(subgraph, kConst, kTfLiteInt32);

  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;

  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;

  CreateConstantInt32Tensor(subgraph, kConst, {1}, {1});
  int node_index;
  subgraph->AddNodeWithParameters({kIntegerInput, kConst}, {kIntegerOutput}, {},
                                  nullptr, 0, add_params, add_reg, &node_index);

  auto* fill_reg = ops::builtin::Register_FILL();
  fill_reg->builtin_code = kTfLiteBuiltinFill;
  subgraph->AddNodeWithParameters({kIntegerOutput, kStringInput1},
                                  {kStringOutput2}, {}, nullptr, 0, nullptr,
                                  fill_reg, &node_index);
}

void SubgraphBuilder::BuildWhileSubgraphWithDynamicTensor(Subgraph* subgraph) {
  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kStringOutput1 = 3;
  const int kStringOutput2 = 4;
  const int kIntegerOutput = 5;
  const int kTensorCount = 6;

  // Create a while op with 2 string tensor and 1 integer tensor.
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kStringOutput1, kStringOutput2, kIntegerOutput}),
      kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters(
      {kStringInput1, kStringInput2, kIntegerInput},
      {kStringOutput1, kStringOutput2, kIntegerOutput}, {}, nullptr, 0, params,
      while_reg, &node_index);
}

void SubgraphBuilder::CreateConstantInt32Tensor(Subgraph* subgraph,
                                                int tensor_index,
                                                const std::vector<int>& shape,
                                                const std::vector<int>& data) {
  ASSERT_GT(shape.size(), 0);
  int num_elements = 1;
  for (int dim : shape) {
    num_elements *= dim;
  }
  ASSERT_EQ(data.size(), num_elements);
  size_t size_in_bytes = sizeof(int32_t) * num_elements;
  // Maybe aligned.
  int32_t* buffer = reinterpret_cast<int32_t*>(malloc(size_in_bytes));
  for (int i = 0; i < num_elements; ++i) {
    buffer[i] = data[i];
  }
  buffers_.push_back(buffer);
  ASSERT_EQ(subgraph->SetTensorParametersReadOnly(
                tensor_index, kTfLiteInt32, "", shape, {},
                reinterpret_cast<const char*>(buffer), size_in_bytes),
            kTfLiteOk);
}

void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data) {
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    tensor->data.i32[i] = data[i];
  }
}

void FillScalarStringTensor(TfLiteTensor* tensor, const std::string& data) {
  StringRef str_ref;
  str_ref.str = data.c_str();
  str_ref.len = data.size();
  DynamicBuffer buf;
  buf.AddString(str_ref);
  buf.WriteToTensor(tensor, /*new_shape=*/TfLiteIntArrayCreate(0));
}

void CheckScalarStringTensor(const TfLiteTensor* tensor,
                             const std::string& data) {
  ASSERT_EQ(tensor->dims->size, 0);
  ASSERT_EQ(tensor->type, kTfLiteString);
  StringRef str_ref = GetString(tensor, 0);
  EXPECT_EQ(std::string(str_ref.str, str_ref.len), data);
}

void CheckStringTensor(const TfLiteTensor* tensor,
                       const std::vector<int>& shape,
                       const std::vector<std::string>& data) {
  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteString);
  int count = GetStringCount(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    StringRef str_ref = GetString(tensor, i);
    EXPECT_EQ(std::string(str_ref.str, str_ref.len), data[i]);
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

void CheckBoolTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                     const std::vector<bool>& data) {
  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteBool);
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(tensor->data.b[i], data[i]);
  }
}

}  // namespace subgraph_test_util
}  // namespace tflite
