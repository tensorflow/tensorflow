/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/variants/list_ops_subgraph_test_util.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::tflite::subgraph_test_util::SetupTensor;
using ::tflite::variants::detail::ListReserveOptions;
using ::tflite::variants::ops::Register_LIST_RESERVE;
using ::tflite::variants::ops::Register_LIST_STACK;

namespace tflite {

ListReserveOptions* ListOpsSubgraphBuilder::RequestReserveOptions(
    TensorType element_type) {
  list_reserve_opts_.push_back(ListReserveOptions{element_type});
  return &list_reserve_opts_.back();
}

void ListOpsSubgraphBuilder::CreateConstantInt32Tensor(
    Subgraph* subgraph, int tensor_index, absl::Span<const int> shape,
    absl::Span<const int> data) {
  const bool all_static_dimensions =
      std::all_of(shape.begin(), shape.end(), [](int i) { return i >= 0; });
  TF_LITE_ASSERT(all_static_dimensions);
  TF_LITE_ASSERT(!shape.empty());

  // tflite only supports tensors with at most rank 5
  const bool will_not_overflow =
      std::all_of(shape.begin(), shape.end(),
                  [](int i) { return i < (INT_MAX / 5); }) &&
      shape.size() <= 5;
  TF_LITE_ASSERT(will_not_overflow);

  const int num_elements =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  TF_LITE_ASSERT_EQ(num_elements, data.size());

  size_t bytes = sizeof(int32_t) * num_elements;
  int_buffers_.push_back(std::vector<int32_t>(data.begin(), data.end()));

  TfLiteStatus stat = subgraph->SetTensorParametersReadOnly(
      tensor_index, kTfLiteInt32, /*name=*/"",
      std::vector<int>(shape.begin(), shape.end()), /*quantization=*/{},
      reinterpret_cast<const char*>(int_buffers_.back().data()), bytes);
  TF_LITE_ASSERT_EQ(stat, kTfLiteOk);
}

void ListOpsSubgraphBuilder::AddConstSubgraph(Subgraph* subgraph) {
  constexpr int kLHS = 0;
  constexpr int kRHS = 1;
  constexpr int kOut = 2;
  constexpr int kTensorCount = 3;
  // kLHS(0) --> +-----------+
  //             |    ADD    | --> kOut(2)
  // kRHS(1) --> +-----------+

  int first_new_tensor_index;
  TF_LITE_ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
                    kTfLiteOk);
  TF_LITE_ASSERT_EQ(first_new_tensor_index, 0);
  TF_LITE_ASSERT_EQ(subgraph->SetOutputs({kOut}), kTfLiteOk);

  CreateConstantInt32Tensor(subgraph, kLHS, {2}, {2, 2});
  CreateConstantInt32Tensor(subgraph, kRHS, {2}, {3, 3});
  SetupTensor(subgraph, kOut, kTfLiteInt32);

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  TfLiteRegistration* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  int node_index;
  TfLiteStatus stat = subgraph->AddNodeWithParameters(
      {kLHS, kRHS}, {kOut}, {}, nullptr, 0, params, add_reg, &node_index);
  TF_LITE_ASSERT_EQ(stat, kTfLiteOk);
}

void ListOpsSubgraphBuilder::AddReserveSubgraph(Subgraph* subgraph,
                                                TensorType element_type) {
  constexpr int kElementShape = 0;
  constexpr int kNumElements = 1;
  constexpr int kListOut = 2;
  constexpr int kTensorCount = 3;
  // kElementShape(0) --> +-------------------+
  //                      |    ListReserve    | --> kListOut(2)
  // kNumElements(1)  --> +-------------------+

  int first_new_tensor_index;
  TF_LITE_ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
                    kTfLiteOk);
  TF_LITE_ASSERT_EQ(first_new_tensor_index, 0);

  TF_LITE_ASSERT_EQ(subgraph->SetOutputs({kListOut}), kTfLiteOk);
  SetupTensor(subgraph, kListOut, kTfLiteVariant);

  TF_LITE_ASSERT_EQ(subgraph->SetInputs({kElementShape, kNumElements}),
                    kTfLiteOk);
  SetupTensor(subgraph, kElementShape, kTfLiteInt32);
  SetupTensor(subgraph, kNumElements, kTfLiteInt32);

  TfLiteRegistration* reserve_reg = Register_LIST_RESERVE();
  reserve_reg->builtin_code = BuiltinOperator_CUSTOM;
  reserve_reg->custom_name = "ListReserve";

  ListReserveOptions* options = RequestReserveOptions(element_type);

  int node_index;
  TfLiteStatus stat = subgraph->AddNodeWithParameters(
      {kElementShape, kNumElements}, {kListOut}, {},
      reinterpret_cast<const char*>(options), sizeof(ListReserveOptions),
      nullptr, reserve_reg, &node_index);
  TF_LITE_ASSERT_EQ(stat, kTfLiteOk);
}

void ListOpsSubgraphBuilder::AddReserveStackSubgraph(Subgraph* subgraph) {
  constexpr int kElementShape = 0;
  constexpr int kNumElements = 1;
  constexpr int kStackShape = 2;
  constexpr int kReserveOut = 3;
  constexpr int kStackOut = 4;
  constexpr int kTensorCount = 5;
  // kElementShape(0) --> +-------------+
  //                      | ListReserve |
  // kNumElements(1)  --> +-------------+ --> kReserveOut(2)
  //                                                |
  //                                          +-----------+
  //                                          | ListStack |
  //                       kStackShape(3) --> +-----------+ --> kStackOut(4)

  int first_new_tensor_index;
  TF_LITE_ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
                    kTfLiteOk);
  TF_LITE_ASSERT_EQ(first_new_tensor_index, 0);

  TF_LITE_ASSERT_EQ(subgraph->SetOutputs({kStackOut}), kTfLiteOk);
  SetupTensor(subgraph, kStackOut, kTfLiteInt32);

  TF_LITE_ASSERT_EQ(
      subgraph->SetInputs({kElementShape, kNumElements, kStackShape}),
      kTfLiteOk);
  SetupTensor(subgraph, kElementShape, kTfLiteInt32);
  SetupTensor(subgraph, kNumElements, kTfLiteInt32);
  SetupTensor(subgraph, kStackShape, kTfLiteInt32);
  SetupTensor(subgraph, kReserveOut, kTfLiteVariant);

  TfLiteRegistration* reserve_reg = Register_LIST_RESERVE();
  reserve_reg->builtin_code = BuiltinOperator_CUSTOM;
  reserve_reg->custom_name = "ListReserve";

  ListReserveOptions* options = RequestReserveOptions(TensorType_INT32);

  TfLiteStatus reg_stat = subgraph->AddNodeWithParameters(
      {kElementShape, kNumElements}, {kReserveOut}, {},
      reinterpret_cast<const char*>(options), sizeof(ListReserveOptions),
      nullptr, reserve_reg);
  TF_LITE_ASSERT_EQ(reg_stat, kTfLiteOk);

  TfLiteRegistration* stack_reg = Register_LIST_STACK();
  stack_reg->builtin_code = BuiltinOperator_CUSTOM;
  stack_reg->custom_name = "ListStack";

  TfLiteStatus stack_stat =
      subgraph->AddNodeWithParameters({kReserveOut, kStackShape}, {kStackOut},
                                      {}, nullptr, 0, nullptr, stack_reg);
  TF_LITE_ASSERT_EQ(stack_stat, kTfLiteOk);
}

}  // namespace tflite
