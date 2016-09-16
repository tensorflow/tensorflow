/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_graph_utils.h"

#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

// static
Status DebugNodeInserter::InsertNodes(
    const protobuf::RepeatedPtrField<DebugTensorWatch>& watches, Graph* graph,
    Device* device) {
  if (watches.empty()) {
    // Nothing to do: Return OK right away.
    return Status::OK();
  }

  // A map from tensor name (e.g., "node_a:0") to list of debug op names
  // (e.g., {"DebugIdentity", "DebugNanCount"})
  std::unordered_map<string, std::vector<string>> tensor_watches;
  // A map from tensor name to debug_url.
  std::unordered_map<string, std::vector<string>> tensor_watch_urls;

  // Cache the proto content for fast lookup later
  for (const DebugTensorWatch& watch : watches) {
    if (watch.output_slot() < 0) {
      // The semantics of output_slot == -1 is that the node is watched only
      // for completion, but not for output tensor values (see
      // NodeCompletionCallback in debug_gateway.h).
      continue;
    }
    if (watch.debug_ops().empty()) {
      continue;
    }

    string tensor_name =
        strings::StrCat(watch.node_name(), ":", watch.output_slot());

    std::vector<string> debug_ops;
    for (const string& debug_op : watch.debug_ops()) {
      debug_ops.push_back(debug_op);
    }

    tensor_watches[tensor_name] = debug_ops;

    std::vector<string> urls;
    for (const string& url : watch.debug_urls()) {
      urls.push_back(url);
    }
    tensor_watch_urls[tensor_name] = urls;
  }

  if (tensor_watches.empty()) {
    return Status::OK();
  }

  DeviceType device_type = DeviceType{device->device_type()};
  // 1. Record existing edges in the graph.
  std::vector<const Edge*> existing_edges;
  for (const Edge* edge : graph->edges()) {
    existing_edges.push_back(edge);
  }

  // A map from tensor names to edges to be removed
  std::unordered_map<string, std::vector<const Edge*>> edges_to_remove;
  // A map from tensor names to newly added debug nodes (maybe more than one
  // for a given tensor).
  std::unordered_map<string, std::vector<Node*>> added_debug_nodes;
  std::unordered_map<string, Node*> added_copy_nodes;

  // 2. Iterate through the edges, look for edges that match the tensor watch
  // list.
  for (const Edge* edge : existing_edges) {
    Node* src_node = edge->src();
    Node* dst_node = edge->dst();

    if (edge->IsControlEdge()) {
      continue;
    }

    const bool is_ref = IsRefType(dst_node->input_type(edge->dst_input()));
    MemoryType memory_type;
    MemoryTypeForOutput(device_type, graph, src_node, edge->src_output(),
                        &memory_type);

    const string tensor_name =
        strings::StrCat(src_node->name(), ":", edge->src_output());
    if (tensor_watches.find(tensor_name) == tensor_watches.end()) {
      // Add debug nodes only for edges with matching source node and source
      // output slot.
      continue;
    }

    if (added_copy_nodes.find(tensor_name) == added_copy_nodes.end()) {
      // It is the first time an edge with this source tensor is encountered:
      // we will:
      //   1) Mark this edge as to be removed, iff the destination node has
      //      non-Ref input
      //   2) Create a Copy node
      //   3) Add a new edge, from the source tensor to the Copy node
      //   4) Add a new edge, from the Copy node to the destination node, iff
      //      the destination node has non-Ref input
      //   5) Create all the requested debug nodes and their edges to the Copy
      //      node.
      if (!is_ref) {
        std::vector<const Edge*> node_edges_to_remove;
        node_edges_to_remove.push_back(edge);
        edges_to_remove[tensor_name] = node_edges_to_remove;
      }

      const DataType src_dt = src_node->output_type(edge->src_output());

      // Create the copy node.
      Node* copy_node;
      Status copy_s = CreateCopyNode(
          graph, device_type, memory_type == HOST_MEMORY, src_node->name(),
          edge->src_output(), src_dt, tensor_name, &copy_node);
      if (!copy_s.ok()) {
        return Status(
            error::FAILED_PRECONDITION,
            strings::StrCat("Failed to create Copy/CopyHost node for tensor ",
                            tensor_name, ", due to: ", copy_s.error_message()));
      }

      // Record the added copy node for later use.
      added_copy_nodes[tensor_name] = copy_node;

      // Add edge from watched tensor to the copy node.
      graph->AddEdge(src_node, edge->src_output(), copy_node, 0);

      // Add  edge from the copy node to the destination node, iff the
      // destination node has non-Ref input.
      if (!is_ref) {
        graph->AddEdge(copy_node, 0, dst_node, edge->dst_input());
      }

      // Create all requested debug nodes and their edges to the Copy node.
      std::vector<Node*> node_added_debug_nodes;
      for (size_t i = 0; i < tensor_watches[tensor_name].size(); ++i) {
        const string& debug_op_name = tensor_watches[tensor_name][i];

        Node* debug_node;
        Status debug_s = CreateDebugNode(
            graph, device_type, copy_node->name(), src_dt, tensor_name,
            tensor_watch_urls[tensor_name], i, debug_op_name, &debug_node);
        if (!debug_s.ok()) {
          return Status(
              error::FAILED_PRECONDITION,
              strings::StrCat("Failed to create debug node ", debug_op_name,
                              " for tensor ", tensor_name, ", due to: ",
                              debug_s.error_message()));
        }

        node_added_debug_nodes.push_back(debug_node);

        // Create edges from the Copy node to the debug node.
        graph->AddEdge(copy_node, 0, debug_node, 0);

        // Add control edges from the debug nodes to the destination node
        // to ensure that the debug nodes are executed before the destination
        // node.
        graph->AddEdge(debug_node, Graph::kControlSlot, dst_node,
                       Graph::kControlSlot);
      }
      added_debug_nodes[tensor_name] = node_added_debug_nodes;
    } else {
      // It is not the first time an edge with this source is encountered.
      // We will do the following iff the destination node has non-Ref input
      //   1) Mark the edge for removal
      //   2) Create an edge from the copy node to the destination node
      // Iff the destination has Ref-input, the edge will not change.
      // Regardless of whether the destination has Ref-inpt, we will
      //   3) Add control edges from the already-created debug node(s) for the
      //      watched tensor to the destination node.
      if (!is_ref) {
        edges_to_remove[tensor_name].push_back(edge);
        graph->AddEdge(added_copy_nodes[tensor_name], 0, dst_node,
                       edge->dst_input());
      }

      for (Node* debug_node : added_debug_nodes[tensor_name]) {
        graph->AddEdge(debug_node, Graph::kControlSlot, dst_node,
                       Graph::kControlSlot);
      }
    }
  }

  // Remove all edges marked for removal.
  for (auto it : edges_to_remove) {
    std::vector<const Edge*> edges = it.second;

    for (const Edge* edge : edges) {
      graph->RemoveEdge(edge);
    }
  }

  return Status::OK();
}

// static
const string DebugNodeInserter::GetCopyNodeName(const string& node_name,
                                                const int output_slot) {
  // For example, if the watched node is named "node1" and the output slot
  // is 0, the debug node will be called: __copy_node1_0
  return strings::StrCat("__copy_", node_name, "_", output_slot);
}

// static
const string DebugNodeInserter::GetDebugNodeName(const string& tensor_name,
                                                 const int debug_op_num,
                                                 const string& debug_op_name) {
  // For example, if the watched node is named "node1" and the debug op that
  // watches the output slot of node1 is of the type "DebugNanCount", the
  // debug node will be called: __dbg_node1_0_0_DebugNanCount.
  return strings::StrCat("__dbg_", tensor_name, "_", debug_op_num, "_",
                         debug_op_name);
}

// static
Status DebugNodeInserter::CreateCopyNode(
    Graph* graph, const DeviceType device_type, const bool is_host_memory,
    const string& src_node_name, const int src_output, const DataType src_dt,
    const string& tensor_name, Node** copy_node) {
  NodeDef node_def;
  const KernelDef* kdef;

  const string copy_op_name = is_host_memory ? "CopyHost" : "Copy";
  const string copy_node_name = GetCopyNodeName(src_node_name, src_output);

  auto builder = NodeDefBuilder(copy_node_name, copy_op_name)
                     .Input(src_node_name, src_output, src_dt);

  if (!builder.Finalize(&node_def).ok()) {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat("Failed to create node definition ", "for copy op ",
                        copy_node_name, " on watched tensor ", tensor_name));
  }
  Status s = FindKernelDef(device_type, node_def, &kdef, nullptr);

  if (!s.ok()) {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat("Failed to find kernel definition ", "for copy op ",
                        copy_node_name, " on watched tensor ", tensor_name));
  }
  if (!NodeBuilder(builder).Finalize(graph, copy_node).ok()) {
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create copy node ", copy_node_name,
                                  " on watched tensor ", tensor_name));
  }

  return Status::OK();
}

// static
Status DebugNodeInserter::CreateDebugNode(
    Graph* graph, const DeviceType device_type,
    const string& src_copy_node_name, const DataType src_dt,
    const string& tensor_name, const std::vector<string>& debug_urls,
    const int debug_op_num, const string& debug_op_name, Node** debug_node) {
  NodeDef node_def;
  const KernelDef* kdef;

  const string debug_node_name =
      GetDebugNodeName(tensor_name, debug_op_num, debug_op_name);
  auto builder = NodeDefBuilder(debug_node_name, debug_op_name)
                     .Input(src_copy_node_name, 0, src_dt)
                     .Attr("tensor_name", tensor_name)
                     .Attr("debug_urls", debug_urls);

  if (!builder.Finalize(&node_def).ok()) {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat("Failed to create node definition ", "for debug op ",
                        debug_op_name, " on watched tensor ", tensor_name));
  }
  if (!FindKernelDef(device_type, node_def, &kdef, nullptr).ok()) {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat("Failed to find kernel definition ", "for debug op ",
                        debug_op_name, " on watched tensor ", tensor_name));
  }
  if (!NodeBuilder(builder).Finalize(graph, debug_node).ok()) {
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create debug node ", debug_op_name,
                                  " on watched tensor ", tensor_name));
  }

  return Status::OK();
}

}  // namespace tensorflow
