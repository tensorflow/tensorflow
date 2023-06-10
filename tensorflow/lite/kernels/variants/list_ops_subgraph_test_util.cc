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
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"

namespace tflite {
using ::tflite::subgraph_test_util::SetupTensor;

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

}  // namespace tflite
