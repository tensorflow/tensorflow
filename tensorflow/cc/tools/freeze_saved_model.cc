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

#include "tensorflow/cc/tools/freeze_saved_model.h"

#include <iostream>
#include <queue>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

namespace {

// Gets tensor names from tensor_info and inserts them into the set of tensor
// names.
void GetTensorNamesFromTensorInfo(const TensorInfo& tensor_info,
                                  std::unordered_set<string>* tensor_names) {
  if (tensor_info.has_coo_sparse()) {
    // If the tensor is sparse we have to add all three tensors of the sparse
    // representations.
    const TensorInfo_CooSparse& coo_sparse = tensor_info.coo_sparse();
    tensor_names->insert(coo_sparse.values_tensor_name());
    tensor_names->insert(coo_sparse.indices_tensor_name());
    tensor_names->insert(coo_sparse.dense_shape_tensor_name());
  } else if (tensor_info.has_composite_tensor()) {
    for (const auto& component : tensor_info.composite_tensor().components()) {
      tensor_names->insert(component.name());
    }
  } else {
    tensor_names->insert(tensor_info.name());
  }
}

// Gets the union of all inputs and outputs of all SignatureDefs in the bundle
void GetSignatureDefsInputsAndOutputs(
    const SavedModelBundle& saved_model_bundle,
    std::unordered_set<string>* inputs, std::unordered_set<string>* outputs) {
  for (auto& sigdef_elem : saved_model_bundle.meta_graph_def.signature_def()) {
    const SignatureDef& signature_def = sigdef_elem.second;
    for (auto& input_elem : signature_def.inputs()) {
      GetTensorNamesFromTensorInfo(input_elem.second, inputs);
    }
    for (auto& output_elem : signature_def.outputs()) {
      GetTensorNamesFromTensorInfo(output_elem.second, outputs);
    }
  }
}

// Gets a map from string node name to NodeDef.
void GetNodeNameToNodeDefMap(
    GraphDef* graph_def,
    std::unordered_map<string, NodeDef*>* name_to_node_map) {
  for (size_t i = 0; i < graph_def->node_size(); i++) {
    NodeDef* node = graph_def->mutable_node(i);
    (*name_to_node_map)[node->name()] = node;
  }
}

// Strips off the tensor part of the tensor_name to get the node_name.
const string GetNodeNameFromTensorName(string tensor_name) {
  if (tensor_name[0] == '^') {
    tensor_name.erase(0, 1);
  }
  std::vector<string> tensor_name_parts = str_util::Split(tensor_name, ':');
  return tensor_name_parts[0];
}

// Gets the set of node names needed by `outputs` and the corresponding set of
// variable nodes to convert.
void GetReachableNodesAndVariables(
    GraphDef* graph_def, const std::unordered_set<string>& outputs,
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    std::unordered_set<string>* reachable_node_names,
    std::unordered_set<string>* variable_node_names) {
  const std::unordered_set<string> kVariableTypes = {"Variable", "VariableV2", "VarHandleOp"};

  std::queue<string> nodes_to_visit;
  for (const string& output_tensor_name : outputs) {
    nodes_to_visit.push(GetNodeNameFromTensorName(output_tensor_name));
  }
  // We do a traversal backwards from the outputs specified in the MetaGraphDef.
  while (!nodes_to_visit.empty()) {
    const string node_name = nodes_to_visit.front();
    nodes_to_visit.pop();
    if (reachable_node_names->find(node_name) != reachable_node_names->end()) {
      continue;
    }
    reachable_node_names->insert(node_name);

    auto node_iter = name_to_node_map.find(node_name);
    if (node_iter == name_to_node_map.end()) {
      LOG(ERROR) << "Node " << node_name << " not found in the graph!";
      continue;
    }
    NodeDef* node = node_iter->second;

    if (kVariableTypes.find(node->op()) != kVariableTypes.end()) {
      variable_node_names->insert(node->name());
    }
    for (const string& input_tensor_name : node->input()) {
      nodes_to_visit.push(GetNodeNameFromTensorName(input_tensor_name));
    }
  }
}

// Gets a map from variable name to variable value.
Status GetVariableNameToTensorMap(
    Session* session,
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    std::unordered_set<string> variable_names_set,
    std::unordered_map<string, Tensor>* variable_name_to_value_map) {
  if (variable_names_set.empty()) {
    return absl::OkStatus();
  }
  std::vector<string> variable_names;
  variable_names.reserve(variable_names_set.size());
  std::vector<string> tensor_names;
  tensor_names.reserve(variable_names_set.size());
  for (const string& node_name : variable_names_set) {
    variable_names.push_back(node_name);
    NodeDef* node_def = name_to_node_map.at(node_name);
    if (node_def->op() == "VarHandleOp") {
      tensor_names.push_back(node_name + "/Read/ReadVariableOp:0");
    } else {
      tensor_names.push_back(node_name + ":0");
    }
  }
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(
      session->Run(/* inputs */ {}, tensor_names, /* targets */ {}, &outputs));
  for (size_t i = 0; i < variable_names.size(); i++) {
    (*variable_name_to_value_map)[variable_names[i]] = outputs[i];
  }
  return absl::OkStatus();
}

// Converts a Variable NodeDef into a Constant NodeDef.
void ConvertVariableToConstant(const NodeDef& variable_node,
                               const Tensor& variable_value,
                               NodeDef* const_node) {
  const_node->set_name(variable_node.name());
  const_node->set_op("Const");
  (*const_node->mutable_attr())["dtype"] = variable_node.attr().at("dtype");
  variable_value.AsProtoTensorContent(
      (*const_node->mutable_attr())["value"].mutable_tensor());
}

// Converts a ReadVariableOp NodeDef to an Identity NodeDef.
void ConvertReadVariableOpToIdentity(const NodeDef& node,
                                     NodeDef* identity_node) {
  identity_node->set_name(node.name());
  identity_node->set_op("Identity");
  (*identity_node->mutable_attr())["T"] = node.attr().at("dtype");
  identity_node->add_input(node.input(0));
}

// Returns the name of the VarHandleOp that provides input (possibly indirectly)
// to node with node_name. A typical indirect chain of nodes (that can occur due
// to graph inlining) is the following: VarHandleOp -> Identity -> Identity ->
// ReadVariableOp.
StatusOr<string> GetVarHandleName(
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    string node_name) {
  const NodeDef* node = name_to_node_map.at(node_name);
  while (node->input_size() > 0) {
    auto parent = name_to_node_map.find(node->input(0));
    if (parent == name_to_node_map.end()) break;
    node = parent->second;
    if (node->op() != "Identity") {
      VLOG(2) << "Stopping at non-identity node " << node->op();
      break;
    }
  }
  if (node->op() != "VarHandleOp") {
    return tsl::errors::InvalidArgument(
        "Expected variable ", node_name, " to be a resource variable.");
  }
  return node->name();
}

}  // namespace

absl::Status FreezeSavedModel(SavedModelBundle* saved_model_bundle,
                              bool input_tensors_required,
                              GraphDef* frozen_graph_def) {
  const MetaGraphDef& meta_graph_def = saved_model_bundle->meta_graph_def;

  // Step 1: Get all input and output nodes of the graph, as well as the list of
  // all variable nodes.
  const GraphDef& graph_def = meta_graph_def.graph_def();
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  GetSignatureDefsInputsAndOutputs(*saved_model_bundle, &inputs, &outputs);

  if (inputs.empty() && input_tensors_required) {
    LOG(ERROR) << "No input tensors found in graph.";
    return absl::FailedPreconditionError("No input tensors found in graph.");
  }
  if (outputs.empty()) {
    LOG(ERROR) << "No output tensors found in graph.";
    return absl::FailedPreconditionError("No output tensors found in graph.");
  }

  *frozen_graph_def = meta_graph_def.graph_def();
  std::unordered_map<string, NodeDef*> name_to_node_map;
  GetNodeNameToNodeDefMap(frozen_graph_def, &name_to_node_map);

  // Step 2: Get the set of reachable nodes and the set of variable nodes.
  std::unordered_set<string> reachable_node_names;
  std::unordered_set<string> variable_node_names;
  GetReachableNodesAndVariables(frozen_graph_def, outputs, name_to_node_map,
                                &reachable_node_names, &variable_node_names);

  // Step 3: Evaluate the values of all variables that are required for the
  // graph outputs.
  std::unordered_map<string, Tensor> variable_name_to_value_map;
  TF_RETURN_IF_ERROR(GetVariableNameToTensorMap(
      saved_model_bundle->session.get(), name_to_node_map, variable_node_names,
      &variable_name_to_value_map));

  // Step 4: Add all constant nodes for the variable values.
  for (const auto& variable : variable_name_to_value_map) {
    NodeDef* node = name_to_node_map[variable.first];
    NodeDef* const_node = frozen_graph_def->add_node();
    ConvertVariableToConstant(*node, variable.second, const_node);
  }

  // Step 5: Convert ReadVariableOp nodes to Identity nodes.
  for (size_t i = 0; i < frozen_graph_def->node_size(); i++) {
    NodeDef* node = frozen_graph_def->mutable_node(i);
    if (node->op() == "ReadVariableOp") {
      NodeDef identity_node;
      ConvertReadVariableOpToIdentity(*node, &identity_node);
      *node = std::move(identity_node);
    } else if (node->op() == "VarHandleOp") {
      node->set_op("Const");
    }
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
