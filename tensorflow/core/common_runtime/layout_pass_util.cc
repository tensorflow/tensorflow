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

#if defined(INTEL_MKL) || defined(AMD_ZENDNN)

// This file refactors common utility functions from oneDNN and ZenDNN layout
// rewrite passes.
// TODO(penporn): Make mkl_layout_pass.cc call these functions.

#include "tensorflow/core/common_runtime/layout_pass_util.h"

#include <utility>
#include <vector>

namespace tensorflow {
namespace zendnn {

inline bool ArgIsList(const OpDef::ArgDef &arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

inline int GetTensorListLength(const OpDef::ArgDef &arg, const Node *n) {
  CHECK_EQ(ArgIsList(arg), true);  // Crash ok.
  int N = 0;
  if (!arg.type_list_attr().empty()) {
    std::vector<DataType> value;
    TF_CHECK_OK(GetNodeAttr(n->def(), arg.type_list_attr(), &value));
    N = value.size();
  } else {
    TF_CHECK_OK(GetNodeAttr(n->def(), arg.number_attr(), &N));
  }
  return N;
}

bool CanOpRunOnCPUDevice(const Node *n) {
  bool result = true;
  string reason;

  const char *const kCPUDeviceSubStr = "CPU";
  const char *const kXLACPUDeviceSubStr = "XLA_CPU";

  // If Op has been specifically assigned to a non-CPU or XLA_CPU device, then
  // No.
  if (!n->assigned_device_name().empty() &&
      (!absl::StrContains(n->assigned_device_name(), kCPUDeviceSubStr) ||
       absl::StrContains(n->assigned_device_name(), kXLACPUDeviceSubStr))) {
    result = false;
    reason = "Op has been assigned a runtime device that is not CPU.";
  }
  // If user has specifically assigned this op to a non-CPU or XLA_CPU device,
  // then No.
  if (!n->def().device().empty() &&
      (!absl::StrContains(n->def().device(), kCPUDeviceSubStr) ||
       absl::StrContains(n->def().device(), kXLACPUDeviceSubStr))) {
    result = false;
    reason = "User has assigned a device that is not CPU.";
  }

  if (!result) {
    VLOG(1) << "CanOpRunOnCPUDevice: Node skipped for rewrite"
            << n->type_string() << " reason : " << reason;
  }

  return result;
}

void GetNodesProducingTFTensorList(
    const gtl::InlinedVector<std::pair<Node *, int>, 4> &inputs, int *input_idx,
    int list_length, std::vector<NodeBuilder::NodeOut> *output_nodes) {
  CHECK_LT(*input_idx, inputs.size());  // Crash ok.
  CHECK_GT(list_length, 0);             // Crash ok.
  CHECK_NOTNULL(output_nodes);          // Crash ok.
  output_nodes->reserve(list_length);

  while (list_length != 0) {
    CHECK_GT(list_length, 0);             // Crash ok.
    CHECK_LT(*input_idx, inputs.size());  // Crash ok.
    Node *n = inputs[*input_idx].first;
    int slot = inputs[*input_idx].second;
    // If input node 'n' is just producing a single tensor at
    // output slot 'slot' then we just add that single node.
    output_nodes->push_back(NodeBuilder::NodeOut(n, slot));
    (*input_idx)++;
    list_length--;
  }
}

Status CopyInputs(
    const Node *old_node,
    const gtl::InlinedVector<std::pair<Node *, int>, 4> &old_node_inputs,
    NodeBuilder *nb) {
  // Number of input slots to old node.
  // Input slots are represented by .Input() calls in REGISTER_OP.
  int old_node_input_slots = old_node->op_def().input_arg_size();
  // Actual number of inputs can be greater than or equal to number
  // of Input slots because inputs of type list could be unfolded.
  auto old_node_input_size = old_node_inputs.size();

  if (old_node->type_string() == "_FusedConv2D") {
    // [TODO zendnn-tf]
    // commit 5be9a5 updates _FusedConv2D with additional host_args in vanilla
    // tensorflow, temporarily the addtional argument is removed for Zen op
    // conversion as it is yet to support in ZenDNN.
    old_node_input_slots--;
  }

  DCHECK_GE(old_node_input_size, old_node_input_slots);

  // Copy all inputs of old node to new node.
  int iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    DCHECK_LT(iidx, old_node_input_size);
    const OpDef::ArgDef &arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int tensor_list_length = GetTensorListLength(arg, old_node);
      GetNodesProducingTFTensorList(old_node_inputs, &iidx, tensor_list_length,
                                    &new_node_inputs);
      nb->Input(new_node_inputs);
    } else {
      nb->Input(old_node_inputs[iidx].first, old_node_inputs[iidx].second);
      iidx++;
    }
  }
  return OkStatus();
}

}  // namespace zendnn
}  // namespace tensorflow

#endif  // INTEL_MKL || AMD_ZENDNN
