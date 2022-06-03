/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tensorflow {

namespace {
constexpr char kDefaultCpuDeviceName[] = "CPU:0";
}  // namespace

Status TransformGraphFunction(const std::string& func_name,
                              const FunctionDef& fdef,
                              const std::string& device_name,
                              const tensorflow::DeviceSet& device_set,
                              EagerContext* eager_ctx, bool enable_grappler,
                              std::unique_ptr<FunctionBody>* fbody,
                              std::unique_ptr<Graph> graph,
                              tfrt::ArrayRef<const tfrt::Device*> input_devices,
                              FunctionLibraryDefinition* func_lib_def) {
  const DeviceMgr* device_mgr = eager_ctx->local_device_mgr();
  if (device_mgr == nullptr)
    return errors::Internal("Cannot find device manager");
  DumpGraph("Input function graph", graph.get());

  std::vector<string> ret_node_names;
  std::vector<string> control_ret_node_names;
  // Mapping from a function body node name to the control output name.
  std::unordered_map<string, string> node_name_to_control_ret;
  std::vector<Node*> arg_nodes, ret_nodes;
  DataTypeVector ret_types;
  auto attrs = AttrSlice(&fdef.attr());
  TF_RETURN_IF_ERROR(GetGraphAndArgRets(
      func_name, attrs, &fdef, func_lib_def, &graph, &arg_nodes, &ret_nodes,
      &ret_node_names, &ret_types, &control_ret_node_names));
  for (const auto& control_ret : fdef.control_ret()) {
    node_name_to_control_ret.emplace(control_ret.second, control_ret.first);
  }
  for (Node* node : arg_nodes) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
    int64_t index = attr_value->i();
    node->set_assigned_device_name(input_devices[index]->name().str());
  }

  std::vector<string> input_device_names;
  int input_size = input_devices.size();
  input_device_names.reserve(input_size);
  for (int i = 0; i < input_size; ++i) {
    input_device_names.push_back(input_devices[i]->name().str());
  }

  std::vector<string> output_device_names;
  int output_size = fdef.signature().output_arg_size();
  output_device_names.reserve(output_size);
  for (int i = 0; i < output_size; ++i) {
    output_device_names.push_back(device_name);
  }

  // set default_device for placer.
  Device* default_device = nullptr;
  tensorflow::Status s = device_mgr->LookupDevice(device_name, &default_device);
  if (!s.ok())
    VLOG(1) << "TransformGraphFunction(): " << device_name << " is unknown."
            << " default device for placer is not set.";

  TF_RETURN_IF_ERROR(ProcessFunctionLibraryRuntime::PinArgsAndRets(
      input_device_names, output_device_names, device_set, arg_nodes, ret_nodes,
      func_lib_def,
      eager_ctx->AllowSoftPlacement() ? default_device : nullptr));
  DumpGraph("After running PinArgsAndRets", graph.get());

  ConfigProto config;
  bool control_rets_updated = false;
  TF_RETURN_IF_ERROR(FunctionOptimizationPassRegistry::Global().Run(
      device_set, config, &graph, func_lib_def, &control_ret_node_names,
      &control_rets_updated));

  if (control_rets_updated) {
    // Function graph pass may have resulted in different nodes/node names for
    // control rets.
    for (const auto& control_ret : control_ret_node_names) {
      node_name_to_control_ret.emplace(control_ret, control_ret);
    }
  } else {
    for (const auto& control_ret : fdef.control_ret()) {
      node_name_to_control_ret.emplace(control_ret.second, control_ret.first);
    }
  }
  DumpGraph("After running function optimization pass (bridge)", graph.get());

  // Run function inlining so that placer can place ops in nested functions.
  GraphOptimizationPassOptions optimization_options;
  SessionOptions session_options;
  // In TFRT we don't lower v2 control flow to v1.
  session_options.config.mutable_experimental()->set_use_tfrt(true);
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  optimization_options.session_options = &session_options;
  optimization_options.graph = &graph;
  optimization_options.flib_def = func_lib_def;
  optimization_options.device_set = &device_set;
  optimization_options.is_function_graph = true;
  optimization_options.default_function_device = default_device;
  optimization_options.function_def = &fdef;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));
  DumpGraph("After running pre placement passes", graph.get());

  // Run placer before importing GraphDef to MLIR.
  Placer placer(graph.get(), func_name, func_lib_def, &device_set,
                default_device, eager_ctx->AllowSoftPlacement(),
                /*log_device_placement=*/false);
  TF_RETURN_IF_ERROR(placer.Run());
  DumpGraph("After running placer", graph.get());

  if (enable_grappler) {
    Device* cpu_device;
    TF_RETURN_IF_ERROR(
        device_mgr->LookupDevice(kDefaultCpuDeviceName, &cpu_device));

    ConfigProto config_proto;
    config_proto.mutable_experimental()->set_use_tfrt(true);
    config_proto.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_do_function_inlining(true);
    // Do not skip grappler optimization even for small graphs.
    config_proto.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    grappler::GrapplerItem::OptimizationOptions grappler_options =
        grappler::CreateOptOptionsForEager();
    auto status = grappler::OptimizeGraph(
        std::move(ret_node_names), std::move(control_ret_node_names),
        func_lib_def, device_set, cpu_device, config_proto,
        fdef.signature().name(), grappler_options, &graph);
    if (!status.ok()) {
      LOG(WARNING) << "Ignoring multi-device function optimization failure: "
                   << status.ToString();
    }
    DumpGraph("After grappler optimization", graph.get());
  }

  // We must preserve control returns in each of the function components,
  // otherwise after function inlining we might prune side-effectful nodes.
  const auto control_ret =
      [&node_name_to_control_ret](const Node* n) -> absl::optional<string> {
    const auto it = node_name_to_control_ret.find(n->name());
    if (it != node_name_to_control_ret.end())
      return absl::make_optional<string>(it->second);
    return absl::nullopt;
  };
  FunctionDef new_func;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*graph, func_name, control_ret, &new_func));
  // Refresh `fbody`.
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(new_func, AttrSlice(), func_lib_def, fbody));
  return OkStatus();
}
}  // namespace tensorflow
