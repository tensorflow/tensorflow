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

#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

const char kXlaTokenInputNodesAttrName[] = "_xla_token_input_nodes";

const char kXlaTokenArgNodeName[] = "_xla_token_arg_node";

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
               results.insert(n->name());
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

}  // namespace tensorflow
