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

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {

namespace ops {
namespace builtin {
// ADD and MUL are used to test simple branch.
// ADD and MUL can be used to test dynamic sized subgraphs with the
// use of IF op.
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_MUL();
// PAD is used to test dynamic sized subgraphs.
TfLiteRegistration* Register_PAD();
TfLiteRegistration* Register_LESS_EQUAL();
}  // namespace builtin
namespace custom {
TfLiteRegistration* Register_IF();
TfLiteRegistration* Register_WHILE();
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
  int node_index;
  subgraph->AddNodeWithParameters(
      {kInput1, kInput2}, {kOutput}, nullptr, 0, params,
      ::tflite::ops::builtin::Register_ADD(), &node_index);
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
  int node_index;
  subgraph->AddNodeWithParameters(
      {kInput1, kInput2}, {kOutput}, nullptr, 0, params,
      ::tflite::ops::builtin::Register_MUL(), &node_index);
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
  int node_index;
  subgraph->AddNodeWithParameters(
      {kInput1, kInput2}, {kOutput}, nullptr, 0, params,
      ::tflite::ops::builtin::Register_PAD(), &node_index);
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

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("then_subgraph_index", 1);
    fbb.Int("else_subgraph_index", 2);
  });
  fbb.Finish();
  const auto& buffer = fbb.GetBuffer();

  int node_index;
  subgraph->AddNodeWithParameters(
      {kCondInput, kInput1, kInput2}, {kOutput},
      reinterpret_cast<const char*>(buffer.data()), buffer.size(), nullptr,
      ::tflite::ops::custom::Register_IF(), &node_index);
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

  CreateConstantInt32Tensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters(
      {kInput1, kConstRhs}, {kOutput}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_LESS_EQUAL(), &node_index);
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
  subgraph->AddNodeWithParameters({0, 4}, {2}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_ADD(),
                                  &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  subgraph->AddNodeWithParameters({2, 1}, {3}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_ADD(),
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
  subgraph->AddNodeWithParameters(
      {kInputCounter, kConstStep}, {kOutputCounter}, nullptr, 0, add_params,
      ::tflite::ops::builtin::Register_ADD(), &node_index);
  TfLitePadParams* pad_params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLiteAddParams)));
  subgraph->AddNodeWithParameters(
      {kInputValue, kConstPadding}, {kOutputValue}, nullptr, 0, pad_params,
      ::tflite::ops::builtin::Register_PAD(), &node_index);
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

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("cond_subgraph_index", 1);
    fbb.Int("body_subgraph_index", 2);
  });
  fbb.Finish();
  const auto& buffer = fbb.GetBuffer();

  int node_index;
  subgraph->AddNodeWithParameters(
      {0, 1}, {2, 3}, reinterpret_cast<const char*>(buffer.data()),
      buffer.size(), nullptr, ::tflite::ops::custom::Register_WHILE(),
      &node_index);
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
