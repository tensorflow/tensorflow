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

#ifndef TENSORFLOW_CORE_FRAMEWORK_GRAPH_TO_FUNCTIONDEF_H_
#define TENSORFLOW_CORE_FRAMEWORK_GRAPH_TO_FUNCTIONDEF_H_

#include <optional>
#include <string>
#include <vector>

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Graph to FunctionDef conversion. This code is closely modeled on the Python
// function graph_to_function_def(), which is located in
// tensorflow/python/framework/graph_to_function_def.py.
Status GraphToFunctionDef(const Graph& fn_body, const string& fn_name,
                          bool append_hash_to_fn_name,
                          bool set_stateful_from_nodes,
                          bool copy_placeholder_attrs_from_nodes,
                          const std::vector<const Node*>& body_nodes,
                          const std::vector<OutputTensor>& inputs,
                          const std::vector<OutputTensor>& outputs,
                          const std::vector<string>& output_names,
                          const std::vector<const Node*>& control_outputs,
                          const std::vector<string>& control_output_names,
                          const char* description, FunctionDef* fdef);

// Converts 'graph' to a FunctionDef 'fdef', with name 'name':
//
// (1) 'node->IsArg()' nodes converted to function inputs.
// (2) 'node->IsRetval()' nodes converted to function output.
// (3) 'control_ret' returns an optional with a control output name, that will
//     be added to the function `control_ret` map (see FunctionDef) and
//     `control_output` in Op definition (see OpDef). Control output name must
//     be unique for all control output nodes.
Status GraphToFunctionDef(
    const Graph& graph, const string& name,
    const std::function<absl::optional<string>(const Node*)>& control_ret,
    FunctionDef* fdef);

Status GraphToFunctionDef(const Graph& graph, const string& name,
                          FunctionDef* fdef);

Status GraphToFunctionDef(const Graph& graph, const string& name,
                          const std::vector<std::string>& output_names,
                          FunctionDef* fdef);

Status GraphToFunctionDef(
    std::unique_ptr<Graph> graph, const string& name,
    const std::function<std::optional<string>(const Node*)>& control_ret,
    FunctionDef* fdef);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_GRAPH_TO_FUNCTIONDEF_H_
