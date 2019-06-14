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

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

const char* const kXlaClusterAttr = "_XlaCluster";
const char* const kXlaOutsideCompilationAttr = "_XlaOutsideCompilation";
const char* const kXlaCompileTimeConstantInputsAttr =
    "_XlaCompileTimeConstantInputs";

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

  auto node_name = [&graph](int node_id) {
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

xla::StatusOr<bool> CreateCycleDetectionGraph(const Graph* graph,
                                              GraphCycles* cycles) {
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
        // TODO(b/127521408): We can probably handle this situation with a more
        // sophisticated SCC based algorithm, but for now we bail out.
        VLOG(1) << "Cycle detected when adding " << src_type << "->" << dst_type
                << " edge: " << DescribeCycle(cycles, *graph, src, dst);
        return false;
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

  return true;
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

void RemoveFromXlaCluster(Node* node) { node->ClearAttr(kXlaClusterAttr); }

namespace {
struct XlaGlobalJitLevel {
  OptimizerOptions::GlobalJitLevel single_gpu;
  OptimizerOptions::GlobalJitLevel general;
};

XlaGlobalJitLevel GetXlaGlobalJitLevel(
    const GraphOptimizationPassOptions& options) {
  XlaGlobalJitLevel result;

  OptimizerOptions::GlobalJitLevel jit_level_in_session_opts =
      options.session_options->config.graph_options()
          .optimizer_options()
          .global_jit_level();
  if (jit_level_in_session_opts == OptimizerOptions::DEFAULT) {
    // To set compilation to be on by default, change the following line.
    result.single_gpu = result.general = OptimizerOptions::OFF;
  } else {
    result.single_gpu = result.general = jit_level_in_session_opts;
  }

  // If the flag tf_xla_auto_jit is a valid, non-DEFAULT setting, it overrides
  // the setting in ConfigProto.
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  if (flags->xla_auto_jit_flag.optimization_level_single_gpu !=
      OptimizerOptions::DEFAULT) {
    result.single_gpu = static_cast<OptimizerOptions::GlobalJitLevel>(
        flags->xla_auto_jit_flag.optimization_level_single_gpu);
  }
  if (flags->xla_auto_jit_flag.optimization_level_general !=
      OptimizerOptions::DEFAULT) {
    result.general = static_cast<OptimizerOptions::GlobalJitLevel>(
        flags->xla_auto_jit_flag.optimization_level_general);
  }

  return result;
}

int GetGpuNumber(const string& device_name) {
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
    return -1;
  }

  return parsed_name.type == DEVICE_GPU ? parsed_name.id : -1;
}
}  // namespace

bool IsSingleGpuGraph(const Graph& g) {
  int gpus_seen = 0;
  absl::flat_hash_set<string> devices_seen;

  for (Node* n : g.op_nodes()) {
    if (devices_seen.contains(n->assigned_device_name())) {
      continue;
    }

    int gpu_number = GetGpuNumber(n->assigned_device_name());
    if (gpu_number != -1) {
      if (++gpus_seen > 1) {
        return false;
      }
    }

    devices_seen.insert(n->assigned_device_name());
  }

  return gpus_seen == 1;
}

OptimizerOptions::GlobalJitLevel GetGlobalJitLevelForGraph(
    const GraphOptimizationPassOptions& options) {
  XlaGlobalJitLevel xla_global_jit_level = GetXlaGlobalJitLevel(options);
  if (xla_global_jit_level.single_gpu == xla_global_jit_level.general) {
    VLOG(4) << "GetGlobalJitLevelForGraph returning "
            << xla_global_jit_level.single_gpu;
    return xla_global_jit_level.single_gpu;
  }
  OptimizerOptions::GlobalJitLevel result =
      IsSingleGpuGraph(**options.graph) ? xla_global_jit_level.single_gpu
                                        : xla_global_jit_level.general;
  VLOG(4) << "GetGlobalJitLevelForGraph returning " << result;
  return result;
}

bool MayCallFunction(const Node& n, const FunctionLibraryDefinition* flib_def) {
  if (flib_def->Contains(n.type_string())) {
    return true;
  }

  // This is a conservative check: there may be nodes with a `func`
  // attribute that do not make function calls.
  return absl::c_any_of(n.def().attr(),
                        [](const std::pair<string, AttrValue>& name_attr_pair) {
                          return name_attr_pair.second.has_func();
                        });
}
bool IsShapeConsumerOp(const Node& node) {
  return node.type_string() == "Shape" || node.type_string() == "Rank" ||
         node.type_string() == "Size";
}
}  // namespace tensorflow
