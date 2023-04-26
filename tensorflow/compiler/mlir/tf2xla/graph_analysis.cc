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

#include "tensorflow/compiler/mlir/tf2xla/graph_analysis.h"

#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"

namespace tensorflow {

namespace {

class MlirGenericBridgeGraphAnalyzer {
 public:
  bool HasUnsupportedFeatures(
      const Graph& graph, const FunctionLibraryDefinition* function_library) {
    return !AnalyzeGraphAndReachableFunctions(graph, function_library).ok();
  }

 private:
  enum Color { gray = 1, black };

  bool IsBasedOnConst(const Node* node,
                      absl::flat_hash_map<const Node*, Color>& visited) {
    assert(node && "Null graph node pointer");
    // Return false if loop is detected in data dependencies. This is not
    // expected in practice.
    if (visited[node] == gray) {
      LOG(WARNING) << "Loop of data dependencies is not supported.";
      return false;
    }
    if (node->type_string() == "Const") {
      return true;
    }
    // Returns false if this is a leaf node and not Const.
    if (!node->num_inputs()) {
      return false;
    }
    // Returns true if this node has been visited, which implies its children
    // all derive from Const nodes.
    if (visited[node] == black) {
      return true;
    }
    // Only return true if all its children derive from Const nodes.
    visited[node] = gray;
    for (const Node* in : node->in_nodes()) {
      if (!IsBasedOnConst(in, visited)) {
        return false;
      }
    }
    visited[node] = black;
    return true;
  }

  Status AnalyzeGraphNodes(const Graph& graph) {
    for (const Node* node : graph.nodes()) {
      if (node->type_string() == "Reshape") {
        auto output_shape = node->in_nodes().begin();
        ++output_shape;
        absl::flat_hash_map<const Node*, Color> visited;
        if (!IsBasedOnConst(*output_shape, visited)) {
          return errors::Unimplemented(
              "Reshape ops with non-constant output shape is not supported");
        }
      }
    }
    return OkStatus();
  }

  Status AnalyzeReachableFunctions(
      const GraphDef& graph_def,
      const FunctionLibraryDefinition& function_library) {
    for (const std::string& func_name :
         function_library.ReachableDefinitions(graph_def).ListFunctionNames()) {
      const FunctionDef* func_def = function_library.Find(func_name);
      // Check the function body.
      std::unique_ptr<FunctionBody> func_body;
      TF_RETURN_IF_ERROR(
          FunctionDefToBodyHelper(*func_def, AttrSlice(&func_def->attr()),
                                  &function_library, &func_body));
      TF_RETURN_IF_ERROR(AnalyzeGraphNodes(*func_body->graph));
    }
    return OkStatus();
  }

  Status AnalyzeGraphAndReachableFunctions(
      const Graph& graph, const FunctionLibraryDefinition* function_library) {
    // Analyze each node in this graph.
    TF_RETURN_IF_ERROR(AnalyzeGraphNodes(graph));
    // Check any associated functions in the graph defined in its
    // FunctionLibraryDefinition.
    GraphDef graph_def;
    graph.ToGraphDef(&graph_def);
    TF_RETURN_IF_ERROR(AnalyzeReachableFunctions(graph_def, graph.flib_def()));

    // Check any associated functions in the graph defined in a separate
    // FunctionLibraryDefinition.
    if (function_library != nullptr) {
      TF_RETURN_IF_ERROR(
          AnalyzeReachableFunctions(graph_def, *function_library));
    }

    return OkStatus();
  }
};

}  // namespace

// Analyzes whether the graph has features not guaranteed to be supported by
// the MLIR-based TF XLA bridge.
bool GraphHasFeaturesUnsupportedByMlirGenericBridge(
    const Graph& graph, const FunctionLibraryDefinition* function_library) {
  return MlirGenericBridgeGraphAnalyzer().HasUnsupportedFeatures(
      graph, function_library);
}

}  // namespace tensorflow
