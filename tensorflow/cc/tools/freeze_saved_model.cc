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

#include <queue>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

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

// Gets the set of node names needed by `outputs` and the corresponding set of
// variable nodes to convert.
void GetReachableNodesAndVariables(
    GraphDef* graph_def, const std::unordered_set<string>& outputs,
    std::unordered_set<string>* reachable_node_names,
    std::unordered_set<string>* variable_node_names) {
  // TODO(suharshs): Add support for ResourceVariables.
  static const std::unordered_set<string>* kVariableTypes =
      new std::unordered_set<string>({"Variable", "VariableV2"});
  // name_to_node_map is needed to get the inputs from the NodeDef corresponding
  // the a string node name. These inputs are used when doing our backwards
  // traversal.
  std::unordered_map<string, NodeDef*> name_to_node_map;
  GetNodeNameToNodeDefMap(graph_def, &name_to_node_map);
  std::queue<string> nodes_to_visit;
  for (const string& tensor_name : outputs) {
    // We need to strip off the tensor part to get the node name.
    std::vector<string> tensor_name_parts = str_util::Split(tensor_name, ':');
    nodes_to_visit.push(tensor_name_parts[0]);
  }
  // We do a traversal backwards from the outputs specified in the MetaGraphDef.
  while (!nodes_to_visit.empty()) {
    const string node_name = nodes_to_visit.front();
    nodes_to_visit.pop();
    if (reachable_node_names->find(node_name) != reachable_node_names->end()) {
      continue;
    }
    reachable_node_names->insert(node_name);
    NodeDef* node = name_to_node_map[node_name];
    if (kVariableTypes->find(node->op()) != kVariableTypes->end()) {
      variable_node_names->insert(node->name());
    }
    for (const string& input : node->input()) {
      nodes_to_visit.push(input);
    }
  }
}

// Gets a map from variable name to variable value.
Status GetVariableNameToTensorMap(
    Session* session, std::unordered_set<string> variable_names_set,
    std::unordered_map<string, Tensor>* variable_name_to_value_map) {
  if (variable_names_set.empty()) {
    return Status::OK();
  }
  std::vector<string> variable_names;
  std::vector<string> tensor_names;
  for (const string& node_name : variable_names_set) {
    variable_names.push_back(node_name);
    // We need to run tensors, so append ":0".
    tensor_names.push_back(node_name + ":0");
  }
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(
      session->Run(/* inputs */ {}, tensor_names, /* targets */ {}, &outputs));
  for (size_t i = 0; i < variable_names.size(); i++) {
    (*variable_name_to_value_map)[variable_names[i]] = outputs[i];
  }
  return Status::OK();
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

// Freezes the subgraph of all nodes needed by `outputs`.
Status FreezeGraphDef(const SavedModelBundle& saved_model_bundle,
                      const std::unordered_set<string>& outputs,
                      GraphDef* frozen_graph_def) {
  GraphDef graph_def = saved_model_bundle.meta_graph_def.graph_def();
  // Copy versions and library as-is from original graph.
  *frozen_graph_def->mutable_versions() = graph_def.versions();
  *frozen_graph_def->mutable_library() = graph_def.library();
  // If the graph is empty there is nothing left to do.
  if (graph_def.node_size() == 0) {
    return Status::OK();
  }
  std::unordered_set<string> reachable_node_names;
  std::unordered_set<string> variable_node_names;
  GetReachableNodesAndVariables(&graph_def, outputs, &reachable_node_names,
                                &variable_node_names);
  std::unordered_map<string, Tensor> variable_to_value_map;
  TF_RETURN_IF_ERROR(
      GetVariableNameToTensorMap(saved_model_bundle.session.get(),
                                 variable_node_names, &variable_to_value_map));
  // We copy the nodes in the same order they were in the original graph_def.
  for (const NodeDef& node : graph_def.node()) {
    if (reachable_node_names.find(node.name()) == reachable_node_names.end()) {
      continue;
    }
    if (variable_node_names.find(node.name()) != variable_node_names.end()) {
      ConvertVariableToConstant(node, variable_to_value_map[node.name()],
                                frozen_graph_def->add_node());
    } else {
      // If the node isn't a variable, just copy the node as-is.
      *frozen_graph_def->add_node() = node;
    }
  }
  return Status::OK();
}

}  // namespace

Status FreezeSavedModel(const SavedModelBundle& saved_model_bundle,
                        GraphDef* frozen_graph_def,
                        std::unordered_set<string>* inputs,
                        std::unordered_set<string>* outputs) {
  GetSignatureDefsInputsAndOutputs(saved_model_bundle, inputs, outputs);
  TF_RETURN_IF_ERROR(
      FreezeGraphDef(saved_model_bundle, *outputs, frozen_graph_def));
  return Status::OK();
}

}  // namespace tensorflow
