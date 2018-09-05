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

#include "tensorflow/compiler/jit/xla_cluster_util.h"

#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

const char* const kXlaClusterAttr = "_XlaCluster";
const char* const kXlaOutsideCompilationAttr = "_XlaOutsideCompilation";

namespace {
// Returns a string describing how an edge from src to dst would
// create a cycle.
string DescribeCycle(const GraphCycles* cycles, const Graph& graph, int src,
                     int dst) {
  int32 max_path_size = graph.num_node_ids() + 1;
  std::vector<int32> path(max_path_size);
  int32 path_size = cycles->FindPath(dst, src, max_path_size, path.data());
  if (path_size == 0) {
    return "";
  }

  auto node_name = [cycles, &graph](int node_id) {
    if (!FastBoundsCheck(node_id, graph.num_node_ids())) {
      return string("(null)");
    }
    auto* node = graph.FindNodeId(node_id);
    if (node == nullptr) {
      return string("(null)");
    }
    return node->name();
  };

  string description;
  absl::StrAppend(&description, "Edge from ", node_name(src), " to ",
                  node_name(dst), " would create a cycle.\n");
  path.resize(path_size);
  for (int32 node_id : path) {
    string ascii_art;
    if (node_id == dst) {
      ascii_art = "+-> ";
    } else if (node_id != src) {
      ascii_art = "|   ";
    } else {
      ascii_art = "+-- ";
    }
    absl::StrAppend(&description, ascii_art, node_name(node_id), "\n");
  }
  return description;
}

bool AlwaysForwardsRefInput(const Node& node) { return node.IsIdentity(); }

}  // namespace

Status DeviceToDeviceType(const string& device, DeviceType* device_type) {
  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device, &parsed)) {
    return errors::Internal("Malformed assigned device '", device, "'");
  }
  *device_type = DeviceType(parsed.type);
  return Status::OK();
}

bool HasForwardedRefInput(const Node& node) {
  if (AlwaysForwardsRefInput(node)) {
    for (const Edge* incoming_edge : node.in_edges()) {
      if (incoming_edge->IsControlEdge()) {
        continue;
      }

      Node* incoming_node = incoming_edge->src();
      if (IsRefType(incoming_node->output_type(incoming_edge->src_output()))) {
        VLOG(2) << "Node " << node.def().ShortDebugString() << " has ref input "
                << incoming_node->name() << " " << incoming_node->type_string();
        return true;
      }
    }
  }
  return false;
}

Status CreateCycleDetectionGraph(const Graph* graph, GraphCycles* cycles) {
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    // We rely on the node IDs in the cycle detection graph being consecutive
    // integers starting from 0.
    CHECK_EQ(i, cycles->NewNode());
  }

  // Compute the loop structure of the graph.
  std::vector<ControlFlowInfo> control_flow_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, &control_flow_info));

  // The clustering code must avoid adding cycles to the graph to prevent
  // deadlock. However, the graph may contain loops, which would trigger the
  // cycle detection code. To handle loops, we alter the structure of the cycle
  // detection graph, disconnecting each loop from the enclosing graph.
  // Specifically, we:
  // * add a new "frame" node for each loop.
  // * replace edges to "Enter" nodes, and edges from "Exit" nodes with edges
  //   to/from the corresponding frame node. In essence, we collapse the loop
  //   into a single node for the purpose of cycle detection in the enclosing
  //   graph.
  // * the body of the loop should now be disconnected from the rest of the
  //   graph; we make it acyclic by breaking loop backedges (edges outgoing from
  //   "NextIteration" nodes.

  // Map from frame name strings to node IDs in the cycle detection graph.
  std::unordered_map<string, int> frame_nodes;

  // Get the cycle graph node ID for frame 'frame_name', or add one if none
  // exists.
  auto GetOrAddFrameNodeId = [&frame_nodes, cycles](const string& frame_name) {
    int& frame_id = frame_nodes.emplace(frame_name, -1).first->second;
    if (frame_id < 0) {
      // The emplace succeeded; we have not allocated a frame node yet.
      frame_id = cycles->NewNode();
    }
    return frame_id;
  };

  for (Edge const* edge : graph->edges()) {
    if (edge->dst()->IsEnter() || edge->src()->IsExit()) {
      const char* src_type = "pre-enter";
      const char* dst_type = "post-exit";
      int src = edge->src()->id();
      int dst = edge->dst()->id();

      if (edge->dst()->IsEnter()) {
        // Lift edges to an "Enter" node to the corresponding frame node.
        const string& frame_name =
            control_flow_info[edge->dst()->id()].frame_name;
        dst = GetOrAddFrameNodeId(frame_name);
        dst_type = "frame";
      }

      if (edge->src()->IsExit()) {
        // Lift edges from an "Exit" node to the corresponding frame node.
        const string& frame_name =
            control_flow_info[edge->src()->id()].frame_name;
        src = GetOrAddFrameNodeId(frame_name);
        src_type = "frame";
      }

      if (!cycles->InsertEdge(src, dst)) {
        return errors::Internal(
            "Cycle detected when adding ", src_type, "->", dst_type,
            " edge: ", DescribeCycle(cycles, *graph, src, dst));
      }
      // Drop the original edge.
      continue;
    }
    if (edge->src()->IsNextIteration()) {
      // Break loop back-edges.
      continue;
    }
    if (!cycles->InsertEdge(edge->src()->id(), edge->dst()->id())) {
      // This should never happen. All cycles in the graph should contain
      // a control flow operator.
      return errors::Internal(
          "Found cycle in graph without control flow operator during XLA "
          "compilation: ",
          DescribeCycle(cycles, *graph, edge->src()->id(), edge->dst()->id()));
    }
  }
  return Status::OK();
}

absl::optional<absl::string_view> GetXlaClusterForNode(const Node& node) {
  const AttrValue* attr_value = node.attrs().Find(kXlaClusterAttr);
  if (attr_value == nullptr) {
    return absl::nullopt;
  }
  Status s = AttrValueHasType(*attr_value, "string");
  if (!s.ok()) {
    return absl::nullopt;
  }
  return attr_value->s();
}

bool HasResourceInputOrOutput(const Node& node) {
  return std::find(node.input_types().begin(), node.input_types().end(),
                   DT_RESOURCE) != node.input_types().end() ||
         std::find(node.output_types().begin(), node.output_types().end(),
                   DT_RESOURCE) != node.output_types().end();
}

void RemoveFromXlaCluster(NodeDef* node_def) {
  node_def->mutable_attr()->erase(kXlaClusterAttr);
}

Status AdjustCycleDetectionGraphForResourceOps(
    const Graph* graph, const FunctionLibraryDefinition* flib_def,
    const std::function<Status(const Node&, bool*)>& resource_ops_to_ignore,
    GraphCycles* cycles) {
  std::vector<std::pair<int, int>> unsafe_deps;
  TF_RETURN_IF_ERROR(ComputeIncompatibleResourceOperationPairs(
      *graph, flib_def, resource_ops_to_ignore, &unsafe_deps));

  // An edge {P,Q} in `unsafe_deps` denotes that P and Q, both of which are
  // operations that interact with resource variables, must not be put in the
  // same cluster.  We enforce this constraint by creating a phantom node, X,
  // and adding edges P->X and X->Q.  MarkForCompilation then cannot cluster P
  // and Q together since that would create a cycle with X.

  for (std::pair<int, int> unsafe_dep : unsafe_deps) {
    int phantom_node_id = cycles->NewNode();
    CHECK(cycles->InsertEdge(unsafe_dep.first, phantom_node_id));
    CHECK(cycles->InsertEdge(phantom_node_id, unsafe_dep.second));
  }
  return Status::OK();
}

}  // namespace tensorflow
