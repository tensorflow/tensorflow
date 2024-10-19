/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/side_effect_util.h"

#include "absl/strings/numbers.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

const char kXlaTokenInputNodesAttrName[] = "_xla_token_input_nodes";

const char kXlaTokenArgNodeName[] = "_xla_token_arg_node";

const char kXlaHasHostTransferAttrName[] = "_xla_has_host_transfer";

const char kXlaReplicaIdAttrName[] = "_xla_replica_id";

const char kXlaIsPlaceholderForTailOcAttrName[] =
    "_xla_is_placeholder_for_tail_oc";

const char kXlaOriginalOutsideCompilationNodeName[] =
    "_xla_original_oc_node_name";

absl::Status SetDeviceOrdinalAttributeForNode(Node* node, int device_ordinal) {
  if (!HasNodeAttr(node->def(), kXlaHasHostTransferAttrName)) {
    return errors::InvalidArgument("Node ", node->DebugString(),
                                   " does not have attribute ",
                                   kXlaHasHostTransferAttrName);
  }

  if (node->type_string() == "_XlaRecvAtHost" ||
      node->type_string() == "_XlaSendFromHost") {
    node->ClearAttr("device_ordinal");
    node->AddAttr("device_ordinal", device_ordinal);
  } else if (node->IsIfNode()) {
    AttrValue device_ordinal_value;
    device_ordinal_value.set_i(device_ordinal);
    for (const string& attr_name :
         std::vector<string>{"then_branch", "else_branch"}) {
      NameAttrList branch_func;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr_name, &branch_func));
      (*branch_func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
      node->ClearAttr(attr_name);
      node->AddAttr(attr_name, branch_func);
    }
  } else if (node->IsWhileNode()) {
    AttrValue device_ordinal_value;
    device_ordinal_value.set_i(device_ordinal);
    for (const string& attr_name : std::vector<string>{"cond", "body"}) {
      NameAttrList branch_func;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr_name, &branch_func));
      (*branch_func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
      node->ClearAttr(attr_name);
      node->AddAttr(attr_name, branch_func);
    }
  } else if (HasNodeAttr(node->def(), "_device_ordinal")) {
    // Function call node containing outside compilation.
    node->ClearAttr("_device_ordinal");
    node->AddAttr("_device_ordinal", device_ordinal);
  } else {
    return errors::Internal("Unknown node type to set 'device_ordinal': ",
                            node->DebugString());
  }
  return absl::OkStatus();
}

std::set<std::string> CalculateTokenInputsForOutputToken(const Graph& g) {
  std::set<std::string> results;
  Node* first_side_effecting_node_on_path = nullptr;
  ReverseDFS(g,
             [&](Node* n) {
               std::vector<string> token_input_nodes;
               if (!GetNodeAttr(n->attrs(), kXlaTokenInputNodesAttrName,
                                &token_input_nodes)
                        .ok() ||
                   token_input_nodes.empty()) {
                 return;
               }

               if (first_side_effecting_node_on_path != nullptr) {
                 return;
               }

               first_side_effecting_node_on_path = n;
               string original_node_name;
               TF_CHECK_OK(GetNodeAttr(n->def(),
                                       kXlaOriginalOutsideCompilationNodeName,
                                       &original_node_name));
               results.insert(original_node_name);
             },
             [&](Node* n) {
               if (first_side_effecting_node_on_path == n) {
                 first_side_effecting_node_on_path = nullptr;
               }
             },
             NodeComparatorName());
  return results;
}

bool HasSideEffectingNodes(const Graph& g) {
  for (Node* n : g.nodes()) {
    std::vector<string> token_input_nodes;
    if (GetNodeAttr(n->attrs(), kXlaTokenInputNodesAttrName, &token_input_nodes)
            .ok() &&
        !token_input_nodes.empty()) {
      return true;
    }
  }
  return false;
}

absl::Status ParseHostComputeCoreList(
    absl::Span<const string> list_from_attr,
    std::map<string, int>* host_compute_core) {
  for (const auto& hc_core : list_from_attr) {
    std::vector<string> parts = str_util::Split(hc_core, ":");
    if (parts.size() != 2) {
      return errors::InvalidArgument(
          "Malformed host_compute_core entry ", hc_core,
          " should be <cluster_name>:<core_number>.");
    }
    int core;
    if (!absl::numbers_internal::safe_strto32_base(parts[1], &core, 10)) {
      return errors::InvalidArgument("Malformed host_compute_core entry ",
                                     hc_core,
                                     " part after ':' should be an integer.");
    }
    if (host_compute_core->find(parts[0]) != host_compute_core->end()) {
      return errors::InvalidArgument(
          "Duplicate host_compute_core entry for cluster ", parts[0]);
    }
    (*host_compute_core)[parts[0]] = core;
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
