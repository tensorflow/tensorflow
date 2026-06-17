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
constexpr absl::string_view kFill = "Fill";

bool IsValidFillOp(const Node& node) {
  if (node.type_string() != kFill) {
    return false;
  }
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
  return true;
}

bool IsValidIdentityNode(const Node& node) {
  if (!node.IsIdentity()) {
    return false;
  }
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

std::optional<std::string> GetColocateStringName(const Node& fill_node) {
  std::string device = "";
  std::string colocation_prefix = "loc:@";
  std::string colocation_name = "";
  for (auto output_node : fill_node.out_nodes()) {
    if (!IsValidIdentityNode(*output_node)) return std::nullopt;
    if (device.empty()) {
      device = output_node->requested_device();
      colocation_name = absl::StrCat(colocation_prefix, output_node->name());
    } else if (device != output_node->requested_device()) {
      return std::nullopt;
    }
  }
  if (colocation_name.empty()) return std::nullopt;
  return colocation_name;
}

bool AreAllInNodesQualifiedConst(const Node& node) {
  for (auto in_node : node.in_nodes()) {
    if (!in_node->IsConstant()) {
      return false;
    }
    if (in_node->IsArg()) {
      return false;
    }
    if (in_node->has_assigned_device_name()) {
      return false;
    }
    if (!in_node->requested_device().empty()) {
      return false;
    }
    if (HasNodeAttr(in_node->def(), kClassAttr)) {
      return false;
    }
    if (!KernelDefAvailable(DeviceType(DEVICE_CPU), in_node->def())) {
      return false;
    }
  }
  return true;
}

bool ShouldRunPass(const GraphOptimizationPassOptions& options) {
  if (!flags::Global().enable_tf2min_ici_weight.value()) {
    VLOG(1) << "ColocatePredecessorTreesPass is disabled.";
    return false;
  }
  VLOG(1) << "ColocatePredecessorTreesPass is enabled.";

  // find all potential node.
  if (options.graph == nullptr) {
    LOG(INFO) << "No graph in colocate_predecessor_trees_pass.\n";
    return false;
  }
  return true;
}

void LogGraphProperties(bool is_graph_changed, bool has_valid_fill_op,
                        bool has_colocation_name, bool has_qualified_const,
                        Graph* graph,
                        const GraphOptimizationPassOptions& options) {
  if (is_graph_changed) {
    VLOG(1) << "Graph is changed by ColocatePredecessorTreesPass.";
    VLOG(1) << DumpGraphToFile("graph_changed_after_colocate_predecessor_trees",
                               *graph, options.flib_def);
  } else {
    VLOG(1) << "Graph is not changed by ColocatePredecessorTreesPass.";
    VLOG(1) << "has_valid_fill_op: " << has_valid_fill_op;
    VLOG(1) << "has_colocation_name: " << has_colocation_name;
    VLOG(1) << "has_qualified_const: " << has_qualified_const;
    VLOG(1) << DumpGraphToFile(
        "graph_not_changed_after_colocate_predecessor_trees", *graph,
        options.flib_def);
  }
}

}  // namespace
absl::Status ColocatePredecessorTreesPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (!ShouldRunPass(options)) {
    return absl::OkStatus();
  }

  Graph* graph = options.graph->get();
  VLOG(1) << DumpGraphToFile("graph_before_colocate_predecessor_trees", *graph,
                             options.flib_def);

  bool is_graph_changed = false;
  bool has_valid_fill_op = false;
  bool has_colocation_name = false;
  bool has_qualified_const = false;
  for (Node* node : graph->nodes()) {
    if (!IsValidFillOp(*node)) {
      continue;
    }
    has_valid_fill_op = true;
    auto colocation_name = GetColocateStringName(*node);
    if (!colocation_name.has_value()) continue;
    has_colocation_name = true;
    if (!AreAllInNodesQualifiedConst(*node)) continue;
    has_qualified_const = true;
    is_graph_changed = true;
    node->AddAttr(std::string(kClassAttr), {*colocation_name});
    for (auto in_node : node->in_nodes()) {
      in_node->AddAttr(std::string(kClassAttr), {*colocation_name});
    }
    for (auto out_node : node->out_nodes()) {
      out_node->AddAttr(std::string(kClassAttr), {*colocation_name});
    }
  }

  LogGraphProperties(is_graph_changed, has_valid_fill_op, has_colocation_name,
                     has_qualified_const, graph, options);
  return absl::OkStatus();
}

// TODO(b/331245915): Fix the regression issue then set flag
// enable_tf2min_ici_weight to true.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 50,
                      ColocatePredecessorTreesPass);

}  // namespace tensorflow
