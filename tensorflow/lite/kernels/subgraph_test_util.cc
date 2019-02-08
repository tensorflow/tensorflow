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
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_MUL();
// ADD and MUL are used to test dynamic sized subgraphs.
TfLiteRegistration* Register_PAD();
}  // namespace builtin
namespace custom {
TfLiteRegistration* Register_IF();
}  // namespace custom
}  // namespace ops

namespace subgraph_test_util {

void SetupTensor(Subgraph* subgraph, int tensor_index, TfLiteType type) {
  ASSERT_EQ(subgraph->SetTensorParametersReadWrite(tensor_index, type, "", 0,
                                                   nullptr, {}, false),
            kTfLiteOk);
}

void BuildAddSubgraph(Subgraph* subgraph) {
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({2}), kTfLiteOk);

  SetupTensor(subgraph, 0, kTfLiteInt32);
  SetupTensor(subgraph, 1, kTfLiteInt32);
  SetupTensor(subgraph, 2, kTfLiteInt32);

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

  SetupTensor(subgraph, 0, kTfLiteInt32);
  SetupTensor(subgraph, 1, kTfLiteInt32);
  SetupTensor(subgraph, 2, kTfLiteInt32);

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

  SetupTensor(subgraph, 0, kTfLiteInt32);
  SetupTensor(subgraph, 1, kTfLiteInt32);
  SetupTensor(subgraph, 2, kTfLiteInt32);

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

  SetupTensor(subgraph, 0, kTfLiteBool);
  SetupTensor(subgraph, 1, kTfLiteInt32);
  SetupTensor(subgraph, 2, kTfLiteInt32);
  SetupTensor(subgraph, 3, kTfLiteInt32);

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

}  // namespace subgraph_test_util
}  // namespace tflite
