/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/grappler_item.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

GrapplerItem::OptimizationOptions CreateOptOptionsForEager() {
  GrapplerItem::OptimizationOptions optimization_options;
  // Tensorflow 2.0 in eager mode with automatic control dependencies will
  // prune all nodes that are not in the transitive fanin of the fetch nodes.
  // However because the function will be executed via FunctionLibraryRuntime,
  // and current function implementation does not prune stateful and dataset
  // ops, we rely on Grappler to do the correct graph pruning.
  optimization_options.allow_pruning_stateful_and_dataset_ops = true;

  optimization_options.is_eager_mode = true;

  // All the nested function calls will be executed and optimized via
  // PartitionedCallOp, there is no need to optimize functions now.
  optimization_options.optimize_function_library = false;

  return optimization_options;
}

GrapplerItem GrapplerItem::WithGraph(GraphDef&& graph_def) const {
  GrapplerItem item;
  item.id = id;
  item.feed = feed;
  item.fetch = fetch;
  item.init_ops = init_ops;
  item.keep_ops = keep_ops;
  item.expected_init_time = expected_init_time;
  item.save_op = save_op;
  item.restore_op = restore_op;
  item.save_restore_loc_tensor = save_restore_loc_tensor;
  item.queue_runners = queue_runners;
  item.devices_ = devices_;
  item.optimization_options_ = optimization_options_;
  item.graph.Swap(&graph_def);
  return item;
}

std::vector<const NodeDef*> GrapplerItem::MainOpsFanin() const {
  std::vector<const NodeDef*> fanin_nodes;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, fetch, &fanin_nodes));
  return fanin_nodes;
}

std::vector<const NodeDef*> GrapplerItem::EnqueueOpsFanin() const {
  std::vector<string> enqueue_ops;
  for (const auto& queue_runner : queue_runners) {
    for (const string& enqueue_op : queue_runner.enqueue_op_name()) {
      enqueue_ops.push_back(enqueue_op);
    }
  }
  std::vector<const NodeDef*> fanin_nodes;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, fetch, &fanin_nodes));
  return fanin_nodes;
}

std::vector<const NodeDef*> GrapplerItem::InitOpsFanin() const {
  std::vector<const NodeDef*> fanin_nodes;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, init_ops, &fanin_nodes));
  return fanin_nodes;
}

std::vector<const NodeDef*> GrapplerItem::MainVariables() const {
  std::vector<const NodeDef*> fanin;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, init_ops, &fanin));
  std::vector<const NodeDef*> vars;
  for (const NodeDef* node : fanin) {
    if (IsVariable(*node)) {
      vars.push_back(node);
    }
  }
  return vars;
}

std::unordered_set<string> GrapplerItem::NodesToPreserve() const {
  std::unordered_set<string> result;
  for (const string& f : fetch) {
    VLOG(1) << "Add fetch " << f;
    result.insert(NodeName(f));
  }
  for (const auto& f : feed) {
    VLOG(1) << "Add feed " << f.first;
    result.insert(NodeName(f.first));
  }
  for (const auto& node : init_ops) {
    result.insert(NodeName(node));
  }
  for (const auto& node : keep_ops) {
    result.insert(NodeName(node));
  }
  if (!save_op.empty()) {
    result.insert(NodeName(save_op));
  }
  if (!restore_op.empty()) {
    result.insert(NodeName(restore_op));
  }
  if (!save_restore_loc_tensor.empty()) {
    result.insert(NodeName(save_restore_loc_tensor));
  }

  for (const auto& queue_runner : queue_runners) {
    for (const string& enqueue_op : queue_runner.enqueue_op_name()) {
      result.insert(NodeName(enqueue_op));
    }
    if (!queue_runner.close_op_name().empty()) {
      result.insert(NodeName(queue_runner.close_op_name()));
    }
    if (!queue_runner.cancel_op_name().empty()) {
      result.insert(NodeName(queue_runner.cancel_op_name()));
    }
  }

  absl::optional<FunctionLibraryDefinition> fn_library;
  if (!optimization_options_.allow_pruning_stateful_and_dataset_ops) {
    fn_library.emplace(OpRegistry::Global(), graph.library());
  }
  for (const NodeDef& node : graph.node()) {
    const auto attrs = AttrSlice(&node.attr());

    // Tensorflow functions do not prune stateful or dataset-output ops from
    // the function body (see PruneFunctionBody in common_runtime/function.cc).
    if (!optimization_options_.allow_pruning_stateful_and_dataset_ops &&
        (IsStateful(node, &*fn_library) || IsDataset(node))) {
      result.insert(node.name());
    }

    // Do not remove ops with attribute _grappler_do_not_remove. This is useful
    // for debugging.
    bool do_not_remove;
    if (TryGetNodeAttr(attrs, "_grappler_do_not_remove", &do_not_remove) &&
        do_not_remove) {
      result.insert(node.name());
    }
  }

  return result;
}

const std::unordered_set<string>& GrapplerItem::devices() const {
  return devices_;
}

Status GrapplerItem::AddDevice(const string& device) {
  DeviceNameUtils::ParsedName name;

  if (!DeviceNameUtils::ParseFullName(device, &name)) {
    return errors::InvalidArgument("Invalid device name: device=", device);

  } else if (!name.has_job || !name.has_replica || !name.has_task ||
             !name.has_type || !name.has_id) {
    return errors::InvalidArgument("Not a fully defined device name: device=",
                                   device);
  }

  devices_.insert(DeviceNameUtils::ParsedNameToString(name));
  return Status::OK();
}

Status GrapplerItem::AddDevices(const GrapplerItem& other) {
  std::vector<absl::string_view> invalid_devices;
  for (const string& device : other.devices()) {
    Status added = AddDevice(device);
    if (!added.ok()) invalid_devices.emplace_back(device);
  }
  return invalid_devices.empty()
             ? Status::OK()
             : errors::InvalidArgument("Skipped invalid devices: [",
                                       absl::StrJoin(invalid_devices, ", "),
                                       "]");
}

Status GrapplerItem::InferDevicesFromGraph() {
  absl::flat_hash_set<absl::string_view> invalid_devices;
  for (const NodeDef& node : graph.node()) {
    Status added = AddDevice(node.device());
    if (!added.ok()) invalid_devices.insert(node.device());
  }
  VLOG(2) << "Inferred device set: [" << absl::StrJoin(devices_, ", ") << "]";
  return invalid_devices.empty()
             ? Status::OK()
             : errors::InvalidArgument("Skipped invalid devices: [",
                                       absl::StrJoin(invalid_devices, ", "),
                                       "]");
}

void GrapplerItem::ClearDevices() { devices_.clear(); }

const GrapplerItem::OptimizationOptions& GrapplerItem::optimization_options()
    const {
  return optimization_options_;
}

GrapplerItem::OptimizationOptions& GrapplerItem::optimization_options() {
  return optimization_options_;
}

}  // end namespace grappler
}  // end namespace tensorflow
