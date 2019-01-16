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
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"

#include <utility>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {

const char ProcessFunctionLibraryRuntime::kDefaultFLRDevice[] = "null";

Status ProcessFunctionLibraryRuntime::FunctionData::DistributedInit(
    DistributedFunctionLibraryRuntime* parent, const string& function_name,
    const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options) {
  mutex_lock l(mu_);
  if (!init_started_) {
    init_started_ = true;
    init_result_ = parent->Instantiate(function_name, lib_def, attrs, options,
                                       &local_handle_);
  }
  return init_result_;
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    thread::ThreadPool* default_thread_pool,
    DistributedFunctionLibraryRuntime* parent)
    : device_mgr_(device_mgr),
      lib_def_(lib_def),
      default_thread_pool_(default_thread_pool),
      next_handle_(0),
      parent_(parent) {
  if (device_mgr == nullptr) {
    flr_map_[nullptr] = NewFunctionLibraryRuntime(
        nullptr, env, nullptr, graph_def_version, lib_def_, default_thread_pool,
        optimizer_options, this);
    return;
  }
  for (Device* d : device_mgr->ListDevices()) {
    flr_map_[d] = NewFunctionLibraryRuntime(
        device_mgr, env, d, graph_def_version, lib_def_, default_thread_pool,
        optimizer_options, this);
  }
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    CustomKernelCreator custom_kernel_creator,
    thread::ThreadPool* default_thread_pool,
    DistributedFunctionLibraryRuntime* parent)
    : device_mgr_(device_mgr),
      lib_def_(lib_def),
      default_thread_pool_(default_thread_pool),
      next_handle_(0),
      parent_(parent) {
  if (device_mgr == nullptr) {
    flr_map_[nullptr] = NewFunctionLibraryRuntime(
        nullptr, env, nullptr, graph_def_version, lib_def_, default_thread_pool,
        optimizer_options, std::move(custom_kernel_creator), this);
    return;
  }
  for (Device* d : device_mgr->ListDevices()) {
    flr_map_[d] = NewFunctionLibraryRuntime(
        device_mgr, env, d, graph_def_version, lib_def_, default_thread_pool,
        optimizer_options, custom_kernel_creator, this);
  }
}

/* static */
Status ProcessFunctionLibraryRuntime::SendTensors(
    const string& source_device, const string& target_device,
    const string& key_prefix, int64 src_incarnation,
    gtl::ArraySlice<Tensor> tensors_to_send, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    Rendezvous* rendezvous) {
  std::vector<string> keys;
  for (int i = 0; i < tensors_to_send.size(); ++i) {
    string name = strings::StrCat(key_prefix, i);
    string key = Rendezvous::CreateKey(source_device, src_incarnation,
                                       target_device, name, FrameAndIter(0, 0));
    keys.push_back(key);
  }
  TF_RETURN_IF_ERROR(SendTensorsToRendezvous(
      rendezvous, device_context, alloc_attrs, keys, tensors_to_send));
  return Status::OK();
}

/* static */
void ProcessFunctionLibraryRuntime::ReceiveTensorsAsync(
    const string& source_device, const string& target_device,
    const string& key_prefix, int64 src_incarnation, int64 num_tensors,
    DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs, Rendezvous* rendezvous,
    std::vector<Tensor>* received_tensors, StatusCallback done) {
  std::vector<string> keys;
  for (int64 i = 0; i < num_tensors; ++i) {
    string name = strings::StrCat(key_prefix, i);
    string key = Rendezvous::CreateKey(source_device, src_incarnation,
                                       target_device, name, FrameAndIter(0, 0));
    keys.push_back(key);
  }
  RecvOutputsFromRendezvousAsync(rendezvous, device_context, alloc_attrs, keys,
                                 received_tensors, std::move(done));
}

Status ProcessFunctionLibraryRuntime::GetDeviceIncarnation(
    const string& device_name, int64* incarnation) const {
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found");
  }
  *incarnation = flr->device()->attributes().incarnation();
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::GetDeviceContext(
    const string& device_name, DeviceContext** device_context) const {
  *device_context = nullptr;
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found.");
  }
  Device* device = flr->device();
  string device_type = device->parsed_name().type;
  if (device_type == "CPU" || device_type == "TPU_SYSTEM") {
    // "TPU_SYSTEM" indicates that `device` is a CPU.
    return Status::OK();
  }
  if (device_type == "GPU" || device_type == "TPU") {
    auto* dev_info = flr->device()->tensorflow_gpu_device_info();
    if (dev_info) {
      *device_context = dev_info->default_context;
      return Status::OK();
    }
  }
  return errors::Internal("Device type: ", device_type,
                          " is currently unsupported for remote ",
                          "function executions");
}

FunctionLibraryRuntime* ProcessFunctionLibraryRuntime::GetFLR(
    const string& device_name) const {
  Device* device = nullptr;
  if (device_name != kDefaultFLRDevice) {
    if (!device_mgr_->LookupDevice(device_name, &device).ok()) {
      VLOG(1) << "Could not find device: " << device_name;
      return nullptr;
    }
  }
  const auto& iter = flr_map_.find(device);
  if (iter == flr_map_.end()) {
    LOG(ERROR) << "Could not find device: " << device_name;
    return nullptr;
  }
  return iter->second.get();
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::AddHandle(
    const string& function_key, const string& device_name,
    FunctionLibraryRuntime::LocalHandle local_handle) {
  mutex_lock l(mu_);
  return AddHandleLocked(function_key, device_name, local_handle);
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::AddHandleLocked(
    const string& function_key, const string& device_name,
    FunctionLibraryRuntime::LocalHandle local_handle) {
  auto h = next_handle_;
  function_data_[h] =
      MakeUnique<FunctionData>(device_name, local_handle, function_key);
  table_[function_key] = h;
  next_handle_++;
  return h;
}

FunctionLibraryRuntime::Handle
ProcessFunctionLibraryRuntime::AddMultiDeviceHandle(
    std::unique_ptr<MultiDeviceFunctionData> data, const string& function_key) {
  mutex_lock l(mu_);
  auto h = next_handle_;
  mdevice_data_[h] = std::move(data);
  table_[function_key] = h;
  next_handle_++;
  return h;
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::GetHandle(
    const string& function_key) const {
  tf_shared_lock l(mu_);
  return gtl::FindWithDefault(table_, function_key, kInvalidHandle);
}

bool ProcessFunctionLibraryRuntime::IsInstantiatedOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle) const {
  return GetHandleOnDevice(device_name, handle) != kInvalidHandle;
}

FunctionLibraryRuntime::LocalHandle
ProcessFunctionLibraryRuntime::GetHandleOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle) const {
  tf_shared_lock l(mu_);

  auto miter = mdevice_data_.find(handle);
  if (miter != mdevice_data_.end()) {
    return kInvalidLocalHandle;
  }

  auto iter = function_data_.find(handle);
  if (iter == function_data_.end()) {
    return kInvalidLocalHandle;
  }
  FunctionData* function_data = iter->second.get();
  if (function_data->target_device() != device_name) {
    return kInvalidLocalHandle;
  }
  return function_data->local_handle();
}

string ProcessFunctionLibraryRuntime::GetDeviceName(
    FunctionLibraryRuntime::Handle handle) const {
  tf_shared_lock l(mu_);
  auto iter = function_data_.find(handle);
  CHECK(iter != function_data_.end());
  FunctionData* function_data = iter->second.get();
  return function_data->target_device();
}

ProcessFunctionLibraryRuntime::MultiDeviceFunctionData*
ProcessFunctionLibraryRuntime::IsMultiDevice(
    FunctionLibraryRuntime::Handle handle) const {
  tf_shared_lock l(mu_);
  const auto& it = mdevice_data_.find(handle);
  if (it != mdevice_data_.end()) {
    return it->second.get();
  }
  return nullptr;
}

namespace {
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

}  // anonymous namespace

Status ProcessFunctionLibraryRuntime::PinArgsAndRets(
    const std::vector<string>& input_devices,
    const std::vector<string>& output_devices, const DeviceSet& device_set,
    Graph* graph) const {
  // If output_devices are not specified, we want to set the output device
  // based on the device of the output producing node. The output producing
  // node can be an arg node because functions can simply return their
  // arguments. To make sure that the output producing nodes have assigned
  // devices, we assign them to arguments first.
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == FunctionLibraryDefinition::kArgOp) {
      const AttrValue* attr_value;
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int64 index = attr_value->i();
      node->set_assigned_device_name(input_devices[index]);
    }
  }

  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == FunctionLibraryDefinition::kRetOp) {
      if (output_devices.empty()) {
        // If output_devices are empty, the node producing retval
        // must have explicitly assigned device or a colocation constraint
        // to a node with explicitly assigned device.
        for (const auto& it : node->in_edges()) {
          if (!it->IsControlEdge()) {
            Node* src_node = it->src();
            const string* src_device = &src_node->requested_device();
            string colocation_group = "";
            GetColocationGroup(src_node, &colocation_group);
            while (src_device->empty() && colocation_group.empty() &&
                   src_node->IsIdentity()) {
              src_node = *src_node->in_nodes().begin();
              src_device = &src_node->requested_device();
              if (src_device->empty()) {
                // Some node (e.g. _Args) can have no requested_device,
                // but have assigned_device.
                src_device = &src_node->assigned_device_name();
              }

              GetColocationGroup(src_node, &colocation_group);
            }

            if (!colocation_group.empty()) {
              AttrValue::ListValue colo_attr;
              colo_attr.add_s(colocation_group);
              std::vector<string> colo_slice = {colocation_group};
              node->AddAttr(kColocationAttrName, colo_slice);
            } else if (!src_device->empty()) {
              // src_device can be a partially specified device. Find the
              // matching device in the device_set.
              DeviceNameUtils::ParsedName parsed;
              if (!DeviceNameUtils::ParseFullName(*src_device, &parsed)) {
                return errors::InvalidArgument(
                    "Failed to parse explicit device specification ",
                    *src_device);
              }
              std::vector<Device*> matching_devices;
              device_set.FindMatchingDevices(parsed, &matching_devices);
              if (matching_devices.size() != 1) {
                // Convert a vector of devices to a string.
                // Using absl::StrJoin did not work in Android builds.
                string devices = "]";
                for (Device* device : matching_devices) {
                  devices.append(device->name());
                  devices.append(", ");
                }
                if (devices.size() > 2) {
                  devices.resize(devices.size() - 2);
                }
                devices.append("]");

                return errors::InvalidArgument(
                    "When FunctionLibraryRuntime::Options.output_devices are "
                    "not specified for a multi-device function, the device "
                    "specification on the output node must match exactly one "
                    "device. Matched devices are ",
                    devices);
              }
              node->set_assigned_device_name(matching_devices[0]->name());
            }
          }
        }
      } else {
        const AttrValue* attr_value;
        TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
        int64 index = attr_value->i();
        // output_devices size is checked in InstantiateMultiDevice
        DCHECK_GT(output_devices.size(), index);
        node->set_assigned_device_name(output_devices[index]);
      }
    }
  }
  return Status::OK();
}

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
  return Status::OK();
}

Status ValidateMultiDeviceOptions(
    const FunctionDef& fdef,
    const FunctionLibraryRuntime::InstantiateOptions& options) {
  const OpDef& signature = fdef.signature();
  // Multi-device functions don't currently support list inputs or outputs
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
        "InstantiateOptions.output_devices must either be empty or have "
        "the same length as the number of arguments: output_devices length "
        "= ",
        options.output_devices.size(),
        " number of arguments = ", signature.output_arg_size());
  }

  if (!options.state_handle.empty()) {
    return errors::Unimplemented(
        "InstantiateOptions.state_handle is not supported for multi-device "
        "functions. Function: ",
        signature.name());
  }
  if (options.create_kernels_eagerly) {
    return errors::Unimplemented(
        "InstantiateOptions.create_kernels_eagerly is not supported for "
        "multi-device functions. Function: ",
        signature.name());
  }

  return Status::OK();
}

Status GetGraphAndRets(const string& function_name, AttrSlice attrs,
                       const FunctionDef* fdef,
                       const FunctionLibraryDefinition* lib_def,
                       std::unique_ptr<Graph>* graph,
                       std::vector<string>* ret_node_names) {
  auto get_func_sig = [lib_def](const string& op, const OpDef** sig) {
    return lib_def->LookUpOpDef(op, sig);
  };
  FunctionBody* tmp_fbody;
  // TODO(iga): FunctionDefToBodyHelper copies fdef. Avoid this copy.
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*fdef, attrs, lib_def, get_func_sig, &tmp_fbody));
  if (tmp_fbody == nullptr) {
    LOG(ERROR) << "Failed to get FunctionBody for \"" << function_name << "\"";
    return errors::Internal("Failed to construct FunctionBody for ",
                            function_name);
  }
  std::unique_ptr<FunctionBody> fbody(tmp_fbody);
  *graph = std::unique_ptr<Graph>(fbody->graph);
  fbody->graph = nullptr;
  ret_node_names->reserve(fbody->ret_nodes.size());
  for (const Node* node : fbody->ret_nodes) {
    ret_node_names->push_back(node->name());
  }
  return Status::OK();
}

}  // anonymous namespace

Status ProcessFunctionLibraryRuntime::InstantiateMultiDevice(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle) {
  // Check if this function has already been instantiated.
  const string& function_key = Canonicalize(function_name, attrs, options);

  {
    mutex_lock l(mu_);
    const auto& it = table_.find(function_key);
    if (it != table_.end()) {
      *handle = it->second;
      ++mdevice_data_[*handle]->instantiation_counter_;
      return Status::OK();
    }
  }

  VLOG(1) << "Instantiating MultiDevice function \"" << function_name
          << "\" on default device " << options.target;

  const FunctionLibraryDefinition* lib_def =
      options.overlay_lib == nullptr ? lib_def_ : options.overlay_lib;

  const FunctionDef* fdef = lib_def->Find(function_name);
  if (fdef == nullptr) {
    return errors::InvalidArgument("Failed to find function \"", function_name,
                                   "\" in function library: ", lib_def);
  }

  TF_RETURN_IF_ERROR(ValidateMultiDeviceOptions(*fdef, options));

  std::unique_ptr<Graph> graph;
  std::vector<string> ret_node_names;

  TF_RETURN_IF_ERROR(GetGraphAndRets(function_name, attrs, fdef, lib_def,
                                     &graph, &ret_node_names));

  DeviceSet device_set;
  for (auto d : device_mgr_->ListDevices()) {
    device_set.AddDevice(d);
  }

  TF_RETURN_IF_ERROR(PinArgsAndRets(
      options.input_devices, options.output_devices, device_set, graph.get()));

  // Make the FunctionLibraryRuntime's device the default device if
  // nothing else is hard coded. This allows the same function definition
  // to be specialized to different devices depending on the
  // PartitionedCallOp's device.
  FunctionLibraryRuntime* flr = GetFLR(options.target);
  if (flr == nullptr) {
    return errors::InvalidArgument(
        "Cannot instantiate multi-device function with target device ",
        options.target);
  }

  std::unique_ptr<MultiDeviceFunctionData> data =
      MakeUnique<MultiDeviceFunctionData>(function_name, function_key,
                                          ret_node_names.size(),
                                          lib_def->ReachableDefinitions(*fdef));

  GraphOptimizationPassOptions optimization_options;
  // TODO(iga): Thread other relevant options from SessionOptions.
  SessionOptions session_options;
  session_options.env = flr->env();
  optimization_options.session_options = &session_options;
  optimization_options.graph = &graph;
  optimization_options.flib_def = &data->overlay_lib_;
  optimization_options.device_set = &device_set;

  DumpGraph("Before running PRE_PLACEMENT passes", graph.get());
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));

  DumpGraph("Before calling Placer", graph.get());
  Placer placer(graph.get(), &device_set, nullptr, /* No session options */
                flr->device() /* Default device */);
  TF_RETURN_IF_ERROR(placer.Run());

  DumpGraph("Before running POST_PLACEMENT passes", graph.get());
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PLACEMENT, optimization_options));
  DumpGraph("Before running POST_REWRITE_FOR_EXEC passes", graph.get());
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));
  DumpGraph("After all optimization passes", graph.get());

  Device* cpu_device;
  TF_RETURN_IF_ERROR(device_mgr_->LookupDevice("CPU:0", &cpu_device));

  if (options.optimize_graph_fn) {
    Status status = options.optimize_graph_fn(std::move(ret_node_names),
                                              &data->overlay_lib_, device_set,
                                              cpu_device, &graph);
    if (!status.ok()) {
      LOG(WARNING) << "Ignoring multi-device function optimization failure: "
                   << status.ToString();
    }
    DumpGraph("After optimization", graph.get());
  }

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  TF_RETURN_IF_ERROR(
      PartitionFunctionGraph(device_set, std::move(graph), &subgraphs));

  if (options.graph_collector != nullptr) {
    for (const auto& pair : subgraphs) {
      GraphDef def;
      pair.second->ToGraphDef(&def);
      options.graph_collector->CollectGraph(def);
    }
  }

  int i = 0;
  FunctionNameGenerator name_generator(&data->overlay_lib_, function_name);
  for (const auto& pair : subgraphs) {
    i += 1;
    // TODO(iga): Fail gracefully if the set of devices corresponds
    // to more than one address space.
    const string& target = pair.first;
    Graph* subgraph = pair.second.get();

    ComponentFunctionData* comp_data = &data->glue_[target];
    TF_RETURN_IF_ERROR(UpdateArgAndRetvalMetadata(
        subgraph, &comp_data->arg_indices_, &comp_data->ret_indices_,
        &comp_data->arg_alloc_attrs_, &comp_data->ret_alloc_attrs_));
    FunctionDef shard;
    string unique_name = name_generator.GetName();
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*subgraph, unique_name, &shard));
    FunctionLibraryRuntime* target_flr = GetFLR(target);
    TF_RETURN_IF_ERROR(data->overlay_lib_.AddFunctionDef(shard));
    FunctionLibraryRuntime::InstantiateOptions opts;
    opts.executor_type = options.executor_type;
    opts.target = target;
    opts.overlay_lib = &data->overlay_lib_;
    FunctionLibraryRuntime::Handle component_handle;

    TF_RETURN_IF_ERROR(target_flr->Instantiate(
        unique_name, AttrSlice(&shard.attr()), opts, &component_handle));
    VLOG(1) << "Instantiated component function " << unique_name
            << " on device " << target << " with component handle "
            << component_handle;
    VLOG(2) << DebugString(shard);
    comp_data->handle_ = component_handle;
  }

  *handle = AddMultiDeviceHandle(std::move(data), function_key);
  VLOG(2) << "Instantiated MultiDevice function \"" << function_name
          << "\" with handle " << *handle;
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::GetOutputDevices(
    FunctionLibraryRuntime::Handle handle,
    std::vector<Device*>* output_devices) const {
  const MultiDeviceFunctionData* data = IsMultiDevice(handle);
  if (data == nullptr) {
    return errors::InvalidArgument(
        "Failed for find multi-device function handle ", handle);
  }

  for (const auto& pair : data->glue_) {
    const ComponentFunctionData& comp_data = pair.second;
    DCHECK(comp_data.ret_alloc_attrs_.size() == comp_data.ret_indices_.size());

    const string& target = pair.first;
    FunctionLibraryRuntime* target_flr = GetFLR(target);
    Device* target_device = target_flr->device();
    const FunctionBody* fbody = target_flr->GetFunctionBody(comp_data.handle_);
    DCHECK(fbody != nullptr);

    output_devices->resize(data->num_outputs_);
    for (int j = 0; j < comp_data.ret_indices_.size(); ++j) {
      int ret_index = comp_data.ret_indices_[j];
      if (fbody->ret_types[j] == DT_RESOURCE) {
        (*output_devices)[ret_index] = target_device;
      } else {
        (*output_devices)[ret_index] =
            comp_data.ret_alloc_attrs_[j].on_host() ? nullptr : target_device;
      }
    }
  }

  return Status::OK();
}

void ProcessFunctionLibraryRuntime::RunMultiDevice(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  if (opts.create_rendezvous) {
    // FLR->Run() is the default entry point. It checks for cancellation,
    // creates rendezvous, etc.
    // Letting create_rendezvous through will do the wrong thing - each
    // component function will get a separate rendezvous created by its FLR.
    done(
        errors::Internal("Cannot call ProcessFunctionLibraryRuntime::Run with "
                         "create_rendezvous=true. Please run the function "
                         "using FunctionLibraryRuntime::Run"));
    return;
  }

  const MultiDeviceFunctionData* data = IsMultiDevice(handle);
  if (data == nullptr) {
    done(
        errors::InvalidArgument("Failed for find multi-device function handle ",
                                handle, ". Was the function instantiated?"));
    return;
  }

  if (data->glue_.empty()) {
    // Trivial case where the function body is empty.
    done(Status::OK());
    return;
  }

  auto* refcounted_done = new ReffedStatusCallback(std::move(done));
  for (int i = 0; i < data->glue_.size(); ++i) {
    refcounted_done->Ref();
  }

  FunctionLibraryRuntime::Options opts_copy = opts;
  for (const auto& pair : data->glue_) {
    const string& target = pair.first;
    const ComponentFunctionData& comp_data = pair.second;
    FunctionLibraryRuntime::Handle handle = pair.second.handle_;
    VLOG(1) << "Running function shard on device " << target << " with handle "
            << handle;

    opts_copy.args_alloc_attrs = comp_data.arg_alloc_attrs_;
    opts_copy.rets_alloc_attrs = comp_data.ret_alloc_attrs_;
    opts_copy.remote_execution = false;
    std::vector<Tensor> comp_args =
        GetArgsForIndices(comp_data.arg_indices_, args);
    std::vector<Tensor>* comp_rets = new std::vector<Tensor>;
    rets->resize(data->num_outputs_);
    GetFLR(target)->Run(
        opts_copy, handle, comp_args, comp_rets,
        [comp_rets, rets, comp_data, refcounted_done](const Status& status) {
          if (!status.ok()) {
            LOG(ERROR) << "Component function execution failed: " << status;
            refcounted_done->UpdateStatus(status);
          } else {
            for (int i = 0; i < comp_rets->size(); ++i) {
              (*rets)[comp_data.ret_indices_[i]] = (*comp_rets)[i];
            }
          }
          delete comp_rets;
          // refcounted_done is thread-safe
          refcounted_done->Unref();
        });
  }
  refcounted_done->Unref();
}

Status ProcessFunctionLibraryRuntime::Instantiate(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle) {
  if (options.is_multi_device_function) {
    return InstantiateMultiDevice(function_name, attrs, options, handle);
  }

  *handle = kInvalidHandle;
  FunctionLibraryRuntime* flr = GetFLR(options.target);
  if (flr != nullptr) {
    return flr->Instantiate(function_name, attrs, options, handle);
  }
  if (parent_ == nullptr) {
    return errors::Internal(
        "Currently don't support instantiating functions on device: ",
        options.target);
  }
  VLOG(1) << "ProcessFLR Instantiate: " << function_name
          << " on: " << options.target;
  string function_key = Canonicalize(function_name, attrs, options);
  FunctionData* f;
  {
    mutex_lock l(mu_);
    FunctionLibraryRuntime::Handle h =
        gtl::FindWithDefault(table_, function_key, kInvalidHandle);
    if (h == kInvalidHandle || function_data_.count(h) == 0) {
      h = AddHandleLocked(function_key, options.target, kInvalidHandle);
    }
    f = function_data_[h].get();
    *handle = h;
  }
  TF_RETURN_IF_ERROR(
      f->DistributedInit(parent_, function_name, *lib_def_, attrs, options));
  VLOG(1) << "ProcessFLR Instantiate [success]: " << function_name
          << " on: " << options.target << " with handle: " << *handle
          << " (this: " << this << ")";
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::RemoveHandle(
    FunctionLibraryRuntime::Handle handle) {
  mutex_lock l(mu_);
  table_.erase(function_data_[handle]->function_key());
  function_data_.erase(handle);
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::ReleaseMultiDeviceHandle(
    FunctionLibraryRuntime::Handle handle) {
  std::unique_ptr<MultiDeviceFunctionData> mdata;
  {
    mutex_lock l(mu_);
    auto it = mdevice_data_.find(handle);
    --it->second->instantiation_counter_;
    if (it->second->instantiation_counter_ != 0) {
      return Status::OK();
    }
    mdata = std::move(it->second);
    table_.erase(mdata->function_key_);
    mdevice_data_.erase(it);
  }

  // If we are here we are releasing the last instantiation of `handle`.
  // Release all component function handles.
  Status overall_status;
  for (const auto& it : mdata->glue_) {
    const string& device = it.first;
    FunctionLibraryRuntime::Handle flr_handle = it.second.handle_;
    FunctionLibraryRuntime* flr = GetFLR(device);
    if (flr == nullptr) {
      return errors::InvalidArgument(
          "Failed to find FunctionLibraryRuntime for device ", device,
          " when releasing multi-device function handle ", handle);
    }
    Status status = flr->ReleaseHandle(flr_handle);
    if (!status.ok()) {
      overall_status = status;
    }
  }

  return overall_status;
}

Status ProcessFunctionLibraryRuntime::ReleaseHandle(
    FunctionLibraryRuntime::Handle handle) {
  if (IsMultiDevice(handle)) {
    return ReleaseMultiDeviceHandle(handle);
  }

  FunctionLibraryRuntime* flr = nullptr;
  string target_device;
  {
    mutex_lock l(mu_);
    CHECK_EQ(1, function_data_.count(handle)) << " handle: " << handle;
    target_device = function_data_[handle]->target_device();
  }
  flr = GetFLR(target_device);
  if (flr != nullptr) {
    return flr->ReleaseHandle(handle);
  }
  return errors::InvalidArgument("Handle not found: ", handle);
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  bool multi_device;
  {
    tf_shared_lock l(mu_);
    multi_device = mdevice_data_.find(handle) != mdevice_data_.end();
  }
  if (multi_device) {
    return RunMultiDevice(opts, handle, args, rets, done);
  }

  FunctionLibraryRuntime* flr = nullptr;
  string target_device;
  FunctionLibraryRuntime::LocalHandle local_handle;
  {
    tf_shared_lock l(mu_);
    auto iter = function_data_.find(handle);
    if (iter == function_data_.end()) {
      done(errors::NotFound("Handle: ", handle, " not found."));
      return;
    }
    FunctionData* function_data = iter->second.get();
    target_device = function_data->target_device();
    local_handle = function_data->local_handle();
  }

  if (!opts.remote_execution) {
    done(
        errors::InvalidArgument("ProcessFunctionLibraryRuntime::Run should "
                                "only be called for multi-device functions or "
                                "for remote execution."));
    return;
  }

  flr = GetFLR(target_device);
  if (flr != nullptr) {
    auto rendezvous = opts.rendezvous;
    string source_device = opts.source_device;
    DeviceContext* device_context;
    Status s = GetDeviceContext(source_device, &device_context);
    if (!s.ok()) {
      done(s);
      return;
    }
    int64 src_incarnation, target_incarnation;
    s = GetDeviceIncarnation(source_device, &src_incarnation);
    s.Update(GetDeviceIncarnation(target_device, &target_incarnation));
    if (!s.ok()) {
      done(s);
      return;
    }

    // Send the args over to the target device.
    s = SendTensors(source_device, target_device, "arg_", src_incarnation, args,
                    device_context, opts.args_alloc_attrs, rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
    const std::vector<AllocatorAttributes>& rets_alloc_attrs =
        opts.rets_alloc_attrs;
    std::vector<Tensor>* remote_rets = new std::vector<Tensor>;
    flr->Run(opts, handle, args, remote_rets,
             std::bind(
                 [source_device, target_device, target_incarnation, rendezvous,
                  device_context, rets_alloc_attrs, remote_rets,
                  rets](const Status& status,
                        FunctionLibraryRuntime::DoneCallback& done) {
                   if (!status.ok()) {
                     delete remote_rets;
                     done(status);
                     return;
                   }
                   int64 num_returns = remote_rets->size();
                   delete remote_rets;
                   // Now receive the return values from the target.
                   ReceiveTensorsAsync(target_device, source_device, "ret_",
                                       target_incarnation, num_returns,
                                       device_context, rets_alloc_attrs,
                                       rendezvous, rets, std::move(done));
                 },
                 std::placeholders::_1, std::move(done)));
    return;
  }
  if (parent_ != nullptr) {
    parent_->Run(opts, local_handle, args, rets, std::move(done));
    return;
  }
  done(errors::Internal("Could not find device"));
}

Status ProcessFunctionLibraryRuntime::Clone(
    Env* env, int graph_def_version, const OptimizerOptions& optimizer_options,
    CustomKernelCreator custom_kernel_creator,
    std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr) const {
  out_lib_def->reset(new FunctionLibraryDefinition(*lib_def_));
  out_pflr->reset(new ProcessFunctionLibraryRuntime(
      device_mgr_, env, graph_def_version, out_lib_def->get(),
      optimizer_options, std::move(custom_kernel_creator), default_thread_pool_,
      parent_));
  return Status::OK();
}

}  // namespace tensorflow
