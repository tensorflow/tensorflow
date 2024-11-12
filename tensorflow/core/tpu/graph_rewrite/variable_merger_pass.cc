
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/graph_rewrite/variable_merger_pass.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

namespace {

// The name of a stateful op is semantically meaningful because ops with the
// same name will share the same kernel. We therefore form new op names using a
// deterministic function (a fingerprint) of the old names.
uint64 MergedOpFingerprint(absl::Span<Node* const> ops) {
  std::vector<string> op_names;
  op_names.reserve(ops.size());
  for (const Node* node : ops) {
    op_names.push_back(node->name());
  }
  return Fingerprint64(absl::StrJoin(op_names, ","));
}

absl::Status MergeVarHandleOps(const string& device,
                               absl::Span<Node* const> nodes, Graph* graph) {
  int num_var_handles(nodes.size());
  if (num_var_handles <= 1) return absl::OkStatus();

  std::vector<string> containers(num_var_handles);
  std::vector<string> names(num_var_handles);
  DataTypeVector dtypes(num_var_handles);
  std::vector<PartialTensorShape> shapes(num_var_handles);
  for (int i = 0; i < num_var_handles; ++i) {
    TF_RETURN_IF_ERROR(
        GetNodeAttr(nodes[i]->attrs(), "container", &containers[i]));
    TF_RETURN_IF_ERROR(
        GetNodeAttr(nodes[i]->attrs(), "shared_name", &names[i]));
    TF_RETURN_IF_ERROR(GetNodeAttr(nodes[i]->attrs(), "dtype", &dtypes[i]));
    TF_RETURN_IF_ERROR(GetNodeAttr(nodes[i]->attrs(), "shape", &shapes[i]));
  }
  NodeDefBuilder builder(graph->NewName(strings::StrCat(
                             "VarHandles_", MergedOpFingerprint(nodes))),
                         "_VarHandlesOp");
  builder.Attr("N", num_var_handles);
  builder.Attr("containers", containers);
  builder.Attr("shared_names", names);
  builder.Attr("dtypes", dtypes);
  builder.Attr("shapes", shapes);
  builder.Device(device);
  NodeDef node_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&node_def));
  TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(node_def));
  node->set_assigned_device_name(device);

  graph->AddControlEdge(graph->source_node(), node);
  for (int i = 0; i < num_var_handles; ++i) {
    std::vector<std::pair<Node*, int>> consumers;
    for (const Edge* e : nodes[i]->out_edges()) {
      consumers.emplace_back(e->dst(), e->dst_input());
    }
    graph->RemoveNode(nodes[i]);
    for (const auto& t : consumers) {
      graph->AddEdge(node, t.second < 0 ? -1 : i, t.first, t.second);
    }
  }
  return absl::OkStatus();
}

absl::Status MergeReadVariableOps(Node* handle_op, Node* control_node,
                                  absl::Span<Node* const> nodes, Graph* graph) {
  int num_reads(nodes.size());
  if (num_reads <= 1) return absl::OkStatus();

  DataTypeVector dtypes(num_reads);
  for (int i = 0; i < num_reads; ++i) {
    TF_RETURN_IF_ERROR(GetNodeAttr(nodes[i]->attrs(), "dtype", &dtypes[i]));
  }
  NodeDef node_def;
  node_def.set_name(graph->NewName(
      strings::StrCat("ReadVariables_", MergedOpFingerprint(nodes))));
  node_def.set_op("_ReadVariablesOp");
  AddNodeAttr("N", num_reads, &node_def);
  AddNodeAttr("dtypes", dtypes, &node_def);
  node_def.set_device(handle_op->requested_device());
  TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(node_def));
  node->set_assigned_device_name(handle_op->assigned_device_name());
  if (control_node) graph->AddControlEdge(control_node, node);
  for (int i = 0; i < num_reads; ++i) {
    const Edge* handle_edge;
    TF_RETURN_IF_ERROR(nodes[i]->input_edge(0, &handle_edge));
    graph->AddEdge(handle_edge->src(), handle_edge->src_output(), node, i);

    std::vector<std::pair<Node*, int>> consumers;
    for (const Edge* e : nodes[i]->out_edges()) {
      consumers.emplace_back(e->dst(), e->dst_input());
    }
    graph->RemoveNode(nodes[i]);
    for (const auto& t : consumers) {
      graph->AddEdge(node, t.second < 0 ? -1 : i, t.first, t.second);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status VariableMergerPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  VLOG(1) << DumpGraphToFile("variable_merger_pass_before", *graph);

  // Find VarHandleOps that are graph roots and group them by assigned device.
  // Also find any ReadVariableOps that are consumers of those handles.
  absl::flat_hash_map<string, std::vector<Node*>> var_handle_ops_by_device;
  absl::flat_hash_set<Node*> read_variable_ops;

  for (Node* m : graph->source_node()->out_nodes()) {
    // We check that the VarHandleOp has no control edges, other than the one we
    // followed from the source node.
    if (m->type_string() == "VarHandleOp" && m->in_edges().size() == 1) {
      var_handle_ops_by_device[m->assigned_device_name()].push_back(m);
      for (Node* n : m->out_nodes()) {
        // ReadVariableOp could have control edges, we will group them by
        // merged VarHandleOp and control dependency.
        if (n->type_string() == "ReadVariableOp" && n->in_edges().size() <= 2) {
          read_variable_ops.insert(n);
        }
      }
    }
  }

  auto node_name_comparator = [](Node* a, Node* b) {
    return a->name() < b->name();
  };

  // First merge the var handle ops.
  for (auto& vh : var_handle_ops_by_device) {
    // Sort the handles by name for determinism.
    std::sort(vh.second.begin(), vh.second.end(), node_name_comparator);
    TF_RETURN_IF_ERROR(MergeVarHandleOps(vh.first, vh.second, graph));
  }

  // ReadVariableOps by a pair of <VarHandleOp, ControlDependencyNode>.
  // ControlDependencyNode could be nullptr.
  absl::flat_hash_map<std::pair<Node*, Node*>, std::vector<Node*>> read_var_ops;

  for (Node* n : read_variable_ops) {
    Node* control_node = nullptr;
    Node* var_handle_op = nullptr;
    // Each ReadVariableOp has at most one control input since we only choose
    // ReadVariableOp with at most 2 input edges.
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        control_node = e->src();
      } else {
        var_handle_op = e->src();
      }
    }
    TF_RET_CHECK(var_handle_op != nullptr);
    read_var_ops[std::pair<Node*, Node*>(var_handle_op, control_node)]
        .push_back(n);
  }

  for (auto& r : read_var_ops) {
    // Sort the reads by name for determinism.
    std::sort(r.second.begin(), r.second.end(), node_name_comparator);
    TF_RETURN_IF_ERROR(
        MergeReadVariableOps(r.first.first, r.first.second, r.second, graph));
  }

  VLOG(1) << DumpGraphToFile("variable_merger_pass_after", *graph);
  return absl::OkStatus();
}

}  // namespace tensorflow
