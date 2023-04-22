/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/propagator_debug_utils.h"

#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/graph_view.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;


const Tensor* GetTensorValueForDump(const Entry& input) {
  switch (input.state) {
    case Entry::State::NO_VALUE:
      return kEmptyTensor;
    case Entry::State::HAS_VALUE:
      return input.val.get();
    case Entry::State::HAS_CONST_TENSOR:
      return input.const_tensor;
    case Entry::State::HAS_REF_TENSOR:
      return input.ref_tensor.tensor;
  }
}

void DumpPendingNodeState(const NodeItem& node_item, const Entry* input_vector,
                          const bool show_nodes_with_no_ready_inputs) {
  const int input_base = node_item.input_start;
  if (!show_nodes_with_no_ready_inputs) {
    bool has_ready_input = false;
    for (int i = 0; i < node_item.num_inputs; ++i) {
      const Entry& input = input_vector[input_base + i];
      const Tensor* tensor = GetTensorValueForDump(input);
      if (tensor && tensor->IsInitialized()) {
        has_ready_input = true;
        break;
      }
    }
    if (!has_ready_input) {
      return;
    }
  }
  LOG(WARNING) << "    Pending Node: " << node_item.DebugString();
  for (int i = 0; i < node_item.num_inputs; ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void DumpActiveNodeState(const NodeItem& node_item, const Entry* input_vector) {
  LOG(WARNING) << "    Active Node: " << node_item.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node_item.num_inputs; ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

}  // namespace tensorflow
