/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/colocate_predecessor_trees_pass.h"

#include <optional>
#include <queue>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kClassAttr = "_class";

// Check if the node is a valid tree node. Noticed this node must not be the
// root of the tree. We find root of the tree in other place.
// For a valid tree node, it must
// 1. not a arg node
// 2. not have device attr (neither assigned device nor requested device)
// 3. not have colocation attr
// 4. must register for CPU
// 5. only have one output node
bool IsValidTreeNode(const Node& node, bool in_node_mode) {
  if (node.IsArg()) {
    return false;
  }
  if (node.has_assigned_device_name()) {
    return false;
  }
  if (!node.requested_device().empty()) {
    return false;
  }
  if (HasNodeAttr(node.def(), kClassAttr)) {
    return false;
  }
  if (!KernelDefAvailable(DeviceType(DEVICE_CPU), node.def())) {
    return false;
  }

  int num_parents_to_tree_nodes = 0;
  auto parent_nodes = in_node_mode ? node.out_nodes() : node.in_nodes();
  for (auto parent_node : parent_nodes) {
    if (in_node_mode && (parent_node->IsExit() || parent_node->IsSink()))
      continue;
    if (!in_node_mode && parent_node->IsSource()) continue;
    num_parents_to_tree_nodes++;
  }
  if (num_parents_to_tree_nodes != 1) return false;
  return true;
}

// Check if the node is potential root node. For a valid root node, it must
// 1. have requested device attr
// 2. not a arg node
// 3. must register for CPU has device type must be CPU
// 4. the output node can only be exit or sink node
bool IsPotentialRootNode(const Node& node) {
  if (node.requested_device().empty()) {
    return false;
  }
  auto device_name = node.requested_device();
  DeviceNameUtils::ParsedName parsed_device_name;
  DeviceNameUtils::ParseFullName(device_name, &parsed_device_name);
  if (parsed_device_name.type != DEVICE_CPU) {
    return false;
  }
  if (node.IsArg()) {
    return false;
  }
  if (!KernelDefAvailable(DeviceType(DEVICE_CPU), node.def())) {
    return false;
  }
  return true;
}

// Find all tree nodes for the root node. Otherwise, return false.
std::optional<absl::flat_hash_set<Node*>> FindTreeNodes(Node* potential_root) {
  absl::flat_hash_set<Node*> tree_nodes;
  tree_nodes.insert(potential_root);

  auto seek_tree_nodes = [&](bool in_node_mode) {
    std::queue<Node*> pending_nodes;
    auto nodes_to_potential_nodes =
        in_node_mode ? potential_root->in_nodes() : potential_root->out_nodes();
    for (Node* node : nodes_to_potential_nodes) {
      if (in_node_mode && node->IsSource()) continue;
      if (!in_node_mode && (node->IsSink() || node->IsExit())) continue;
      pending_nodes.push(node);
    }
    while (!pending_nodes.empty()) {
      Node* node = pending_nodes.front();
      pending_nodes.pop();
      if (tree_nodes.find(node) != tree_nodes.end()) {
        return false;
      }
      if (!IsValidTreeNode(*node, in_node_mode)) {
        return false;
      }
      tree_nodes.insert(node);
      auto nodes_to_potential_node =
          in_node_mode ? node->in_nodes() : node->out_nodes();
      for (Node* node : nodes_to_potential_node) {
        if (in_node_mode && node->IsSource()) continue;
        if (!in_node_mode && (node->IsSink() || node->IsExit())) continue;
        pending_nodes.push(node);
      }
    }
    return true;
  };

  if (!seek_tree_nodes(/*in_node_mode=*/true)) {
    return std::nullopt;
  }

  // size of tree node must larger than one which means the tree contains at
  // least one non root node.
  if (tree_nodes.size() == 1) {
    return std::nullopt;
  }

  return tree_nodes;
}

// Propagate colocation info from root node to each tree nodes.
void PropagateColocationInfo(Node* root_node,
                             absl::flat_hash_set<Node*>& tree_nodes) {
  VLOG(2) << "PropagateColocationInfo: tree root node is " << root_node->name();
  std::string colocation_prefix = "loc:@";
  std::string node_name = root_node->name();
  for (auto node : tree_nodes) {
    node->AddAttr(std::string(kClassAttr),
                  {absl::StrCat(colocation_prefix, node_name)});
  }
}

}  // namespace

Status ColocatePredecessorTreesPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (!flags::Global().enable_tf2min_ici_weight.value()) {
    return absl::OkStatus();
  }

  // find all potential node.
  if (options.graph == nullptr) {
    VLOG(1) << "No graph in colocate_predecessor_trees_pass.\n";
    return absl::OkStatus();
  }
  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << DumpGraphToFile("before_colocate_predecessor_trees", *graph,
                               options.flib_def);
  }

  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> tree_nodes_map;
  for (Node* node : graph->nodes()) {
    if (IsPotentialRootNode(*node)) {
      std::optional<absl::flat_hash_set<Node*>> nodes = FindTreeNodes(node);
      if (nodes.has_value()) {
        tree_nodes_map[node] = *std::move(nodes);
      }
    }
  }

  for (auto& [root_node, tree_nodes] : tree_nodes_map) {
    PropagateColocationInfo(root_node, tree_nodes);
  }

  if (VLOG_IS_ON(1)) {
    VLOG(1) << DumpGraphToFile("after_colocate_predecessor_trees", *graph,
                               options.flib_def);
  }

  return absl::OkStatus();
}

// TODO(b/331245915): Fix the regression issue then set flag
// enable_tf2min_ici_weight to true.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 50,
                      ColocatePredecessorTreesPass);

}  // namespace tensorflow
