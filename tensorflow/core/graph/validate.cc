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

#include "tensorflow/core/graph/validate.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace graph {

Status ValidateGraphDef(const GraphDef& graph_def,
                        const OpRegistryInterface& op_registry) {
  Status s;
  const int version = graph_def.versions().producer();
  for (const NodeDef& node_def : graph_def.node()) {
    // Look up the OpDef for the node_def's op name.
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(op_registry.LookUpOpDef(node_def.op(), &op_def));
    TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));
    TF_RETURN_IF_ERROR(CheckOpDeprecation(*op_def, version));
  }

  return s;
}

Status ValidateGraphDefAgainstOpRegistry(
    const GraphDef& graph_def, const OpRegistryInterface& op_registry) {
  GraphDef copy(graph_def);
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&copy, op_registry, 0));
  return ValidateGraphDef(copy, op_registry);
}

Status ValidateGraphDefAgainstOpList(const GraphDef& graph_def,
                                     const OpList& op_list) {
  OpListOpRegistry registry(&op_list);
  return ValidateGraphDefAgainstOpRegistry(graph_def, registry);
}

void GetOpListForValidation(OpList* op_list, const OpRegistry& op_registry) {
  op_registry.Export(false, op_list);
  RemoveDescriptionsFromOpList(op_list);
}

Status ValidateGraphHasNoCycle(const Graph& graph) {
  // A node is ready when all of its inputs have been visited.
  std::vector<const Node*> ready;
  std::vector<int> pending_count(graph.num_node_ids(), 0);

  for (int i = 0; i < graph.num_node_ids(); ++i) {
    const Node* n = graph.FindNodeId(i);
    if (n == nullptr) continue;
    pending_count[i] = n->in_edges().size();
    if (n->IsMerge()) {
      // While-loop cycles are legal cycles so we manually adjust the
      // pending_count to make sure that the loop is visited.
      for (const Edge* e : n->in_edges()) {
        if (!e->IsControlEdge() && e->src()->IsNextIteration()) {
          pending_count[i]--;
        }
      }
    }
    if (pending_count[i] == 0) {
      ready.push_back(n);
    }
  }

  int processed = 0;
  while (!ready.empty()) {
    const Node* node = ready.back();
    ready.pop_back();
    ++processed;

    for (const Edge* out : node->out_edges()) {
      const int output_id = out->dst()->id();
      pending_count[output_id]--;
      if (pending_count[output_id] == 0) {
        ready.push_back(out->dst());
      }
    }
  }

  if (processed < graph.num_nodes()) {
    std::vector<string> nodes_in_cycle;
    for (int i = 0; i < pending_count.size() && nodes_in_cycle.size() < 3;
         ++i) {
      if (pending_count[i] != 0) {
        nodes_in_cycle.push_back(graph.FindNodeId(i)->name());
      }
    }
    return errors::InvalidArgument(
        "Graph is invalid, contains a cycle with ",
        graph.num_nodes() - processed,
        " nodes, including: ", absl::StrJoin(nodes_in_cycle, ", "));
  }
  return Status::OK();
}

Status VerifyNoDuplicateNodeNames(const GraphDef& graph) {
  absl::flat_hash_set<absl::string_view> nodes;
  for (const auto& node : graph.node()) {
    if (nodes.contains(node.name())) {
      return errors::AlreadyExists("Node already exists: ", node.name());
    }
    nodes.insert(node.name());
  }
  return Status::OK();
}

}  // namespace graph
}  // namespace tensorflow
