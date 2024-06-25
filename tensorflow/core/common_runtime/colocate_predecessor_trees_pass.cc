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

  for (Node* node : graph->nodes()) {
    if (!IsValidFillOp(*node)) {
      continue;
    }
    auto colocation_name = GetColocateStringName(*node);
    if (!colocation_name.has_value()) continue;
    if (!AreAllInNodesQualifiedConst(*node)) continue;
    node->AddAttr(std::string(kClassAttr), {*colocation_name});
    for (auto in_node : node->in_nodes()) {
      in_node->AddAttr(std::string(kClassAttr), {*colocation_name});
    }
    for (auto out_node : node->out_nodes()) {
      out_node->AddAttr(std::string(kClassAttr), {*colocation_name});
    }
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
