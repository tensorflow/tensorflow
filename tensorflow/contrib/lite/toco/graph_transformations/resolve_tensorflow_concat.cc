/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveTensorFlowConcat::Run(Model* model, std::size_t op_index) {
  auto concat_it = model->operators.begin() + op_index;
  const auto* tf_concat_op = concat_it->get();
  if (tf_concat_op->type != OperatorType::kTensorFlowConcat &&
      tf_concat_op->type != OperatorType::kTensorFlowConcatV2) {
    return false;
  }

  CHECK_GE(tf_concat_op->inputs.size(), 2);
  // TensorFlow Concat and ConcatV2 nodes only differ by the ordering
  // of inputs: in Concat, the concat_dim is the first input, while in
  // ConcatV2, it is the last input.
  std::size_t concat_dim_pos = 0;
  if (tf_concat_op->type == OperatorType::kTensorFlowConcatV2) {
    concat_dim_pos = tf_concat_op->inputs.size() - 1;
  }
  const string concat_dim_name = tf_concat_op->inputs[concat_dim_pos];
  std::vector<string> concat_input_names;
  for (std::size_t i = 0; i < tf_concat_op->inputs.size(); i++) {
    if (i != concat_dim_pos) {
      concat_input_names.push_back(tf_concat_op->inputs[i]);
    }
  }
  // If the concat_dim array hasn't been resolved to a constant yet,
  // we need to yield.
  const auto& concat_dim_array = model->GetArray(concat_dim_name);
  if (!concat_dim_array.buffer) {
    AddMessageF("Waiting for the concat_dim of %s to be resolved to a constant",
                LogName(*tf_concat_op));
    return false;
  }

  CHECK(concat_dim_array.data_type == ArrayDataType::kInt32);
  const auto& concat_dim_data =
      concat_dim_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(concat_dim_data.size(), 1);
  const int concat_dim = concat_dim_data[0];

  // Create the Concatenation op replacing the TensorFlowConcat op.
  auto* concatenation_op = new ConcatenationOperator;
  concatenation_op->concat_dim = concat_dim;
  concatenation_op->inputs = concat_input_names;
  concatenation_op->outputs = {tf_concat_op->outputs[0]};
  auto depth_concat_it = model->operators.emplace(concat_it, concatenation_op);
  CHECK_EQ(depth_concat_it->get(), concatenation_op);
  // Update invalidated iterator
  concat_it = depth_concat_it + 1;
  CHECK_EQ(concat_it->get(), tf_concat_op);

  // Remove the concat_dim array if it is not used by anything else.
  if (CountOpsWithInput(*model, concat_dim_name) == 1) {
    model->arrays.erase(concat_dim_name);
  }
  // Remove the TensorFlowConcat op
  model->operators.erase(concat_it);
  return true;
}

}  // namespace toco
