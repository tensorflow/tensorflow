/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/optimize_function_graph_utils.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {
Status ValidateNoListArguments(
    const protobuf::RepeatedPtrField<OpDef::ArgDef>& args, const char* arg_type,
    const string& function_name) {
  for (const OpDef::ArgDef& arg : args) {
    if (!arg.number_attr().empty() || !arg.type_list_attr().empty()) {
      return errors::InvalidArgument(
          "Function ", function_name, " has an ", arg_type, " named \"",
          arg.name(),
          "\" that is a list of tensors."
          " Multi-device functions support only single-tensor inputs "
          " and outputs");
    }
  }
  return OkStatus();
}

Status ValidateMultiDeviceOptions(
    const FunctionDef& fdef,
    const FunctionLibraryRuntime::InstantiateOptions& options) {
  const OpDef& signature = fdef.signature();
  // Multi-device functions currently do not support list inputs or outputs.
  TF_RETURN_IF_ERROR(ValidateNoListArguments(signature.input_arg(), "input",
                                             signature.name()));
  TF_RETURN_IF_ERROR(ValidateNoListArguments(signature.output_arg(), "output",
                                             signature.name()));
  if (fdef.attr().count(FunctionLibraryDefinition::kIntsOnDeviceAttr) != 0 &&
      fdef.attr().at(FunctionLibraryDefinition::kIntsOnDeviceAttr).b()) {
    return errors::Unimplemented(
        "Function '", signature.name(), "' has `",
        FunctionLibraryDefinition::kIntsOnDeviceAttr,
        "` attribute set. This attribute is not currently supported by "
        "multi-device functions.");
  }
  if (options.input_devices.size() != signature.input_arg_size()) {
    return errors::InvalidArgument(
        "InstantiateOptions.input_devices must have the same length "
        "as the number of arguments: input_devices length = ",
        options.input_devices.size(),
        " number of arguments = ", signature.input_arg_size());
  }
  if (!options.output_devices.empty() &&
      options.output_devices.size() != signature.output_arg_size()) {
    return errors::InvalidArgument(
        "InstantiateOptions.output_devices must either be empty or have the "
        "same length as the number of arguments: output_devices length = ",
        options.output_devices.size(),
        " number of arguments = ", signature.output_arg_size());
  }
  return OkStatus();
}

Status SetArgShape(const std::unordered_map<int, DtypeAndPartialTensorShape>&
                       input_resource_dtypes_and_shapes,
                   const std::vector<Node*>& arg_nodes) {
  for (Node* n : arg_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "T", &dtype));
    if (dtype == DT_RESOURCE) {
      auto dtype_and_shape_iter = input_resource_dtypes_and_shapes.find(index);
      if (dtype_and_shape_iter != input_resource_dtypes_and_shapes.end()) {
        AttrValue dtype_attr_value;
        dtype_attr_value.mutable_list()->add_type(
            dtype_and_shape_iter->second.dtype);
        n->AddAttr("_handle_dtypes", dtype_attr_value);
        TensorShapeProto shape_proto;
        dtype_and_shape_iter->second.shape.AsProto(&shape_proto);
        AttrValue shape_attr_value;
        *shape_attr_value.mutable_list()->add_shape() = shape_proto;
        n->AddAttr("_handle_shapes", shape_attr_value);
      }
    }
  }
  return OkStatus();
}

const string* AssignedOrRequestedDeviceName(const Node& node) {
  if (node.has_assigned_device_name()) {
    return &node.assigned_device_name();
  }
  return &node.requested_device();
}

// Sets `group` to the first colocation group specified in `node`. If no
// group is specified, does not touch `group`.
void GetColocationGroup(const Node* node, string* group) {
  // We hoist the conversion from C-style string literal to string here,
  // so that we can avoid the many repeated calls to strlen().
  static const StringPiece kColocationAttrNameStringPiece(kColocationAttrName);
  const AttrValue* attr_value =
      node->attrs().Find(kColocationAttrNameStringPiece);
  if (attr_value != nullptr && attr_value->has_list() &&
      attr_value->list().s_size() > 0) {
    *group = attr_value->list().s(0);
  }
}
}  // namespace

Status GetGraphAndArgRets(
    const string& function_name, AttrSlice attrs, const FunctionDef* fdef,
    const FunctionLibraryDefinition* lib_def, std::unique_ptr<Graph>* graph,
    std::vector<Node*>* arg_nodes, std::vector<Node*>* ret_nodes,
    std::vector<string>* ret_node_names, DataTypeVector* ret_types,
    std::vector<string>* control_ret_node_names) {
  std::unique_ptr<FunctionBody> fbody;
  // TODO(iga): FunctionDefToBodyHelper copies fdef. Avoid this copy.
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, attrs, lib_def, &fbody));
  if (!fbody) {
    LOG(ERROR) << "Failed to get FunctionBody for \"" << function_name << "\"";
    return errors::Internal("Failed to construct FunctionBody for ",
                            function_name);
  }
  *graph = std::unique_ptr<Graph>(fbody->graph);
  arg_nodes->reserve(fbody->arg_nodes.size());
  std::copy(fbody->arg_nodes.begin(), fbody->arg_nodes.end(),
            std::back_inserter(*arg_nodes));
  ret_nodes->reserve(fbody->ret_nodes.size());
  std::copy(fbody->ret_nodes.begin(), fbody->ret_nodes.end(),
            std::back_inserter(*ret_nodes));
  fbody->graph = nullptr;
  ret_node_names->reserve(fbody->ret_nodes.size());
  for (const Node* node : fbody->ret_nodes) {
    ret_node_names->push_back(node->name());
  }
  for (const auto& ret_type : fbody->ret_types) {
    ret_types->push_back(ret_type);
  }
  control_ret_node_names->reserve(fbody->control_ret_nodes.size());
  for (const Node* node : fbody->control_ret_nodes) {
    control_ret_node_names->push_back(node->name());
  }
  return OkStatus();
}

Status PinArgsAndRets(const std::vector<string>& input_devices,
                      const std::vector<string>& output_devices,
                      const DeviceSet& device_set,
                      const std::vector<Node*>& arg_nodes,
                      const std::vector<Node*>& ret_nodes,
                      const FunctionLibraryDefinition* lib_def,
                      Device* default_device) {
  // If output_devices are not specified, we want to set the output device
  // based on the device of the output producing node. The output producing
  // node can be an arg node because functions can simply return their
  // arguments. To make sure that the output producing nodes have assigned
  // devices, we assign them to arguments first.
  for (Node* node : arg_nodes) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
    int64_t index = attr_value->i();
    node->set_assigned_device_name(input_devices[index]);
  }

  for (Node* node : ret_nodes) {
    if (output_devices.empty()) {
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      VLOG(3) << "Trying to determine device for node " << node->name()
              << "[T=" << DataTypeString(dtype) << "]";

      // If output_devices are empty, the node producing retval
      // must have explicitly assigned device or a colocation constraint
      // to a node with explicitly assigned device.
      for (const auto& it : node->in_edges()) {
        if (it->IsControlEdge()) continue;

        Node* src_node = it->src();
        const string* src_device = AssignedOrRequestedDeviceName(*src_node);
        string colocation_group = "";
        GetColocationGroup(src_node, &colocation_group);
        VLOG(3) << "Considering src: " << src_node->name()
                << " src_device: " << *src_device
                << " colo group: " << colocation_group;
        while (src_device->empty() && colocation_group.empty() &&
               src_node->IsIdentity()) {
          // Only follows the real data input of Identity, not control edges.
          Node* input_node;
          TF_RETURN_IF_ERROR(src_node->input_node(0, &input_node));
          src_node = input_node;

          src_device = AssignedOrRequestedDeviceName(*src_node);
          GetColocationGroup(src_node, &colocation_group);
          VLOG(3) << "Considering src: " << src_node->name()
                  << " src_device: " << *src_device
                  << " colo group: " << colocation_group;
        }

        // If resource is produced by a function call node, we can't trust
        // source node device assignment, because multi-device functions can
        // return resource placed on multiple devices. In such case we leave
        // retval device assignment empty, and rely on placer to infer correct
        // assignment based on actual output device.
        const bool can_use_src_node_device =
            !(dtype == DT_RESOURCE && IsFunctionCall(*lib_def, *src_node));

        if (!colocation_group.empty()) {
          AttrValue::ListValue colo_attr;
          colo_attr.add_s(colocation_group);
          std::vector<string> colo_slice = {colocation_group};
          node->AddAttr(kColocationAttrName, colo_slice);
        } else if (!src_device->empty() && can_use_src_node_device) {
          // Do not copy device from src node for variants, unless it is a no-op
          // forward from input to output. This gets handled in
          // colocation_graph.cc which has special logic for correctly placing
          // _Retvals for various variant types.
          if (dtype == DT_VARIANT && !src_node->IsArg()) {
            continue;
          }
          // src_device can be a partially specified device. Find the
          // matching device in the device_set.
          DeviceNameUtils::ParsedName parsed;
          if (!DeviceNameUtils::ParseFullName(*src_device, &parsed)) {
            return errors::InvalidArgument(
                "Failed to parse explicit device specification ", *src_device);
          }
          std::vector<Device*> matching_devices;
          device_set.FindMatchingDevices(parsed, &matching_devices);
          if (matching_devices.empty()) {
            if (default_device != nullptr) {
              matching_devices.push_back(default_device);
            } else {
              return errors::InvalidArgument(
                  "Unable to find any devices for spec ", *src_device);
            }
          } else if (matching_devices.size() != 1) {
            bool on_same_task = true;
            for (int i = 1; i < matching_devices.size(); ++i) {
              if (!DeviceNameUtils::IsSameAddressSpace(
                      matching_devices.at(0)->parsed_name(),
                      matching_devices.at(i)->parsed_name())) {
                on_same_task = false;
                break;
              }
            }
            // If the src node of an output is assigned to a address space (e.g.
            // py_func), rely on placer to assign a device to the output.
            if (on_same_task) {
              continue;
            }
            // Compare with default_device if it has a narrower scope matching
            // requested device.
            if (default_device != nullptr) {
              int colocated_on_default_device = 0;
              for (int i = 0; i < matching_devices.size(); ++i) {
                if (DeviceNameUtils::IsSameAddressSpace(
                        default_device->parsed_name(),
                        matching_devices.at(i)->parsed_name())) {
                  colocated_on_default_device++;
                }
              }
              // Continue to raise error if multiple colocated devices are
              // found.
              if (colocated_on_default_device == 1) {
                continue;
              }
            }
            // Convert a vector of devices to a string.
            // Using absl::StrJoin did not work in Android builds.
            string devices = "[";
            for (Device* device : matching_devices) {
              devices.append(device->name());
              devices.append(", ");
            }
            if (devices.size() > 2) {
              devices.resize(devices.size() - 2);
            }
            devices.append("]");

            return errors::InvalidArgument(
                *src_device,
                "When FunctionLibraryRuntime::Options.output_devices are "
                "not specified for a multi-device function, the device "
                "specification on the output node must match exactly one "
                "device. Matched devices are ",
                devices);
          }
          VLOG(3) << "Setting output device to " << matching_devices[0]->name()
                  << " for node " << SummarizeNode(*node);
          node->set_assigned_device_name(matching_devices[0]->name());
        } else if (!src_device->empty() && !can_use_src_node_device) {
          VLOG(3) << "Did not set device for a resource output node "
                  << SummarizeNode(*node);
        }
      }
    } else {
      const AttrValue* attr_value;
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int64_t index = attr_value->i();
      // output_devices size is checked in InstantiateMultiDevice
      DCHECK_GT(output_devices.size(), index);
      VLOG(3) << "Setting output device to " << output_devices[index]
              << " for return at index " << index;
      node->set_assigned_device_name(output_devices[index]);
    }
  }
  return OkStatus();
}

StatusOr<OptimizedFunctionGraphInfo> OptimizeFunctionGraph(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    const DeviceSet& dev_set, const FunctionLibraryDefinition* input_lib_def,
    const std::vector<CompositeDevice*>& composite_devices, Device* cpu_device,
    Device* default_device, Env* env) {
  const FunctionLibraryDefinition* lib_def =
      options.lib_def == nullptr ? input_lib_def : options.lib_def;

  const FunctionDef* fdef = lib_def->Find(function_name);
  if (fdef == nullptr) {
    return errors::InvalidArgument("Failed to find function \"", function_name,
                                   "\" in function library: ", lib_def);
  }

  TF_RETURN_IF_ERROR(ValidateMultiDeviceOptions(*fdef, options));

  std::unique_ptr<Graph> graph;
  std::vector<Node*> arg_nodes, ret_nodes;
  std::vector<string> ret_node_names;
  DataTypeVector ret_types;
  std::vector<string> control_ret_node_names;

  TF_RETURN_IF_ERROR(GetGraphAndArgRets(
      function_name, attrs, fdef, lib_def, &graph, &arg_nodes, &ret_nodes,
      &ret_node_names, &ret_types, &control_ret_node_names));

  DUMP_OP_CREATION_STACKTRACES(function_name, "op_stacktraces", graph.get());

  GraphDef graph_def;
  graph->ToGraphDef(&graph_def);
  FunctionLibraryDefinition reachable_lib_def =
      lib_def->ReachableDefinitions(graph_def);
  *graph_def.mutable_library() = reachable_lib_def.ToProto();
  if (options.graph_collector != nullptr) {
    options.graph_collector->CollectRawGraph(graph_def);
  }

  // Dump the initial graph.
  DUMP_GRAPH(function_name, "initial", graph.get(), &reachable_lib_def);

  // Mark and assign device for each node in the graph to be compiled by
  // specified device.
  if (!options.xla_compile_device_type.empty()) {
    for (Node* node : graph->op_nodes()) {
      node->AddAttr("_xla_compile_device_type",
                    options.xla_compile_device_type);
      if (default_device) {
        node->set_assigned_device_name(default_device->name());
      }
    }
  }

  TF_RETURN_IF_ERROR(
      SetArgShape(options.input_resource_dtypes_and_shapes, arg_nodes));
  TF_RETURN_IF_ERROR(PinArgsAndRets(
      options.input_devices, options.output_devices, dev_set, arg_nodes,
      ret_nodes, lib_def,
      options.config_proto.allow_soft_placement() ? default_device : nullptr));

  // The runtime shouldn't depend on duplication between the function library
  // owned by the graph and the one owned by the runtime. To ensure this, for
  // now we ensure that the graph function library is empty and the runtime
  // library receives the query from LookUps on the graph function library.
  graph->mutable_flib_def()->set_default_registry(&reachable_lib_def);
  graph->mutable_flib_def()->Clear();

  // Do not run function/graph optimization passes for component functions,
  // since they have already processed the main function.
  const bool should_run_optimization_passes = !options.is_component_function;
  if (!should_run_optimization_passes) {
    VLOG(1) << "Skipping function/graph optimization passes when instantiating "
               "component function "
            << function_name;
  }

  // Mapping from a function body node name to the control output name.
  std::unordered_map<string, string> node_name_to_control_ret;

  bool control_rets_updated = false;
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(FunctionOptimizationPassRegistry::Global().Run(
        function_name, dev_set, options.config_proto,
        options.xla_compile_device_type, &graph, &reachable_lib_def,
        &control_ret_node_names, &control_rets_updated));
  }

  if (control_rets_updated) {
    // Function graph pass may have resulted in different nodes/node names for
    // control rets.
    for (const auto& control_ret : control_ret_node_names) {
      node_name_to_control_ret.emplace(control_ret, control_ret);
    }
  } else {
    for (const auto& control_ret : fdef->control_ret()) {
      node_name_to_control_ret.emplace(control_ret.second, control_ret.first);
    }
  }

  GraphOptimizationPassOptions optimization_options;
  // TODO(iga): Thread other relevant options from SessionOptions.
  SessionOptions session_options;
  session_options.env = env;
  session_options.config = options.config_proto;
  optimization_options.session_options = &session_options;
  optimization_options.graph = &graph;
  optimization_options.flib_def = &reachable_lib_def;
  optimization_options.device_set = &dev_set;
  optimization_options.is_function_graph = true;
  optimization_options.composite_devices = &composite_devices;
  optimization_options.default_function_device = default_device;
  optimization_options.function_def = fdef;
  optimization_options.shape_inference_on_tfe_dialect_import =
      options.shape_inference_on_tfe_dialect_import;
  optimization_options.debug_filename_prefix = "pflr_optmz_";
  env->CreateUniqueFileName(&optimization_options.debug_filename_prefix, "_");

  DUMP_GRAPH(function_name, "before_pre_placement_passes", graph.get(),
             &reachable_lib_def);
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));
  }

  // TODO(b/124993244): Smartly merge options in nested defuns, and raise
  // exceptions/warnings in case where nested function call options are ignored.
  DUMP_GRAPH(function_name, "before_placer", graph.get(), &reachable_lib_def);
  Placer placer(graph.get(), function_name, optimization_options.flib_def,
                &dev_set, default_device,
                options.config_proto.allow_soft_placement(),
                options.config_proto.log_device_placement());
  TF_RETURN_IF_ERROR(placer.Run(optimization_options));

  DUMP_GRAPH(function_name, "before_post_placement_passes", graph.get(),
             &reachable_lib_def);
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::POST_PLACEMENT, optimization_options));
  }

  if (options.optimize_graph_fn) {
    DUMP_GRAPH(function_name, "before_graph_optimization", graph.get(),
               &reachable_lib_def);
    Status status = options.optimize_graph_fn(
        std::move(ret_node_names), std::move(control_ret_node_names),
        &reachable_lib_def, dev_set, cpu_device, &graph);
    if (!status.ok()) {
      LOG(WARNING) << "Ignoring multi-device function optimization failure: "
                   << status.ToString();
    }
    DUMP_GRAPH(function_name, "after_graph_optimization", graph.get(),
               &reachable_lib_def);
  }

  DUMP_GRAPH(function_name, "before_post_rewrite_for_exec_passes", graph.get(),
             &reachable_lib_def);
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));
  }
  DUMP_GRAPH(function_name, "after_post_rewrite_for_exec_passes", graph.get(),
             &reachable_lib_def);

  graph->mutable_flib_def()->set_default_registry(nullptr);
  graph->mutable_flib_def()->Clear();
  return OptimizedFunctionGraphInfo{function_name,
                                    std::move(graph),
                                    std::move(reachable_lib_def),
                                    node_name_to_control_ret,
                                    std::move(ret_types),
                                    ret_nodes.size()};
}

StatusOr<std::unique_ptr<std::unordered_map<string, std::unique_ptr<Graph>>>>
PreprocessAndPartitionGraph(
    OptimizedFunctionGraphInfo& input_optimized_graph,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    const DeviceSet& dev_set, const FunctionLibraryDefinition* input_lib_def,
    const std::vector<CompositeDevice*>& composite_devices, Env* env) {
  std::unique_ptr<Graph>& graph = input_optimized_graph.function_graph;

  // Expand the nodes assigned to a CompositeDevice before graph partition to
  // avoid generating a subgraph on a virtual device for execution.
  // This transformation should happen as late as possible, in order to run as
  // many graph optimization passes (e.g. PRE_PLACEMENT, PLACER,
  // POST_PLACEMENT, POST_REWRITE_FOR_EXEC) on the smallest graph possible.
  TF_RETURN_IF_ERROR(ReplicatePerReplicaNodesInFunctionGraph(
      options.composite_devices, graph.get()));

  const FunctionLibraryDefinition* lib_def =
      options.lib_def == nullptr ? input_lib_def : options.lib_def;
  if (options.graph_collector != nullptr) {
    GraphDef def;
    graph->ToGraphDef(&def);
    *def.mutable_library() = lib_def->ReachableDefinitions(def).ToProto();
    options.graph_collector->CollectOptimizedGraph(def);
  }

  VLOG(4) << "Main function graph to be partitioned:";
  VLOG(4) << DebugString(graph->ToGraphDefDebug());

  auto device_name_to_subgraphs =
      std::make_unique<std::unordered_map<string, std::unique_ptr<Graph>>>();
  TF_RETURN_IF_ERROR(PartitionFunctionGraph(dev_set, std::move(graph),
                                            device_name_to_subgraphs.get()));

  for (const auto& pair : *device_name_to_subgraphs) {
    DumpGraph(strings::StrCat("Before running POST_PARTITIONING passes (",
                              pair.first, ")"),
              pair.second.get());
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.flib_def = &(input_optimized_graph.lib_def);
  optimization_options.is_function_graph = true;
  optimization_options.graph = nullptr;
  optimization_options.device_set = nullptr;
  optimization_options.partition_graphs = device_name_to_subgraphs.get();
  optimization_options.debug_filename_prefix = "pflr_imd_";
  env->CreateUniqueFileName(&optimization_options.debug_filename_prefix, "_");

  // Normally POST_PARTITIONING passes are run by distributed workers.
  // Distributed workers are currently not supported in this code path, so we
  // run the passes here.
  const bool should_run_optimization_passes = !options.is_component_function;
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::POST_PARTITIONING, optimization_options));
  }
  for (const auto& pair : *device_name_to_subgraphs) {
    const auto* optimized_subgraph = pair.second.get();
    DumpGraph(
        strings::StrCat("After all optimization passes (", pair.first, ")"),
        optimized_subgraph);
    if (VLOG_IS_ON(3)) {
      DumpGraphDefToFile(
          strings::StrCat("pflr_after_all_optimization_passes_",
                          reinterpret_cast<uintptr_t>(optimized_subgraph), "_",
                          pair.first),
          optimized_subgraph->ToGraphDefDebug());
    }
  }
  return std::move(device_name_to_subgraphs);
}

}  // namespace tensorflow
