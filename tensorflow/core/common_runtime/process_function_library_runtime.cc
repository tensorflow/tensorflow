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

#include <iterator>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/reffed_status_callback.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

const char ProcessFunctionLibraryRuntime::kDefaultFLRDevice[] = "null";

void ProcessFunctionLibraryRuntime::FunctionData::DistributedInit(
    DistributedFunctionLibraryRuntime* parent, const string& function_name,
    const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::DoneCallback done) {
  {
    mutex_lock l(mu_);
    is_cross_process_ = true;
    if (init_started_) {
      init_done_.WaitForNotification();
      done(init_result_);
      return;
    }
    init_started_ = true;
  }
  parent->Instantiate(function_name, lib_def, attrs, options, &local_handle_,
                      [this, done](const Status& s) {
                        init_done_.Notify();
                        done(s);
                      });
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
    int graph_def_version, const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    thread::ThreadPool* default_thread_pool,
    DistributedFunctionLibraryRuntime* parent,
    const SessionMetadata* session_metadata,
    Rendezvous::Factory rendezvous_factory)
    : parent_(parent),
      env_(env),
      config_(config ? absl::make_optional(*config) : absl::nullopt),
      device_mgr_(device_mgr),
      lib_def_(lib_def),
      default_thread_pool_(default_thread_pool),
      flr_map_(new std::unordered_map<Device*,
                                      std::unique_ptr<FunctionLibraryRuntime>>),
      next_handle_(0),
      session_metadata_(session_metadata),
      rendezvous_factory_(std::move(rendezvous_factory)),
      optimizer_options_(optimizer_options),
      graph_def_version_(graph_def_version) {
  if (device_mgr == nullptr) {
    (*flr_map_)[nullptr] = NewFunctionLibraryRuntime(
        nullptr, env, config_ ? &(*config_) : nullptr, nullptr,
        graph_def_version, lib_def_, default_thread_pool, optimizer_options,
        session_metadata_, this);
    return;
  }
  InitializeDeviceAndFlr();
}

/* static */
Status ProcessFunctionLibraryRuntime::SendTensors(
    const string& source_device, const string& target_device,
    const string& key_prefix, int64 src_incarnation,
    gtl::ArraySlice<Tensor> tensors_to_send, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    RendezvousInterface* rendezvous) {
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
    const std::vector<AllocatorAttributes>& alloc_attrs,
    RendezvousInterface* rendezvous, std::vector<Tensor>* received_tensors,
    StatusCallback done) {
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

Status ProcessFunctionLibraryRuntime::GetRetTypes(
    FunctionLibraryRuntime::Handle h, DataTypeVector* ret_types) {
  FunctionLibraryRuntime* flr = nullptr;
  {
    tf_shared_lock l(mu_);
    auto miter = mdevice_data_.find(h);
    if (miter != mdevice_data_.end()) {
      *ret_types = miter->second->ret_types_;
      return Status::OK();
    }
    auto fiter = function_data_.find(h);
    if (fiter != function_data_.end()) {
      flr = GetFLR(fiter->second->target_device());
    }
  }
  if (flr != nullptr) {
    return flr->GetRetTypes(h, ret_types);
  }
  return errors::InvalidArgument("Handle ", h, " not found.");
}

Status ProcessFunctionLibraryRuntime::GetDeviceIncarnation(
    const string& device_name, int64* incarnation) const {
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found.");
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

  if (device->IsRemoteCallAllowed()) {
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

void ProcessFunctionLibraryRuntime::InitializeDeviceAndFlr() {
  DeviceMgr const* all_devices = device_mgr_;
  if (parent_ != nullptr && parent_->remote_device_mgr() != nullptr) {
    all_devices = parent_->remote_device_mgr();
  }

  mutex_lock l(mu_);
  device_set_ = std::make_shared<DeviceSet>();
  for (auto d : all_devices->ListDevices()) {
    device_set_->AddDevice(d);
  }
  for (Device* d : device_mgr_->ListDevices()) {
    if ((*flr_map_)[d] == nullptr) {
      (*flr_map_)[d] = NewFunctionLibraryRuntime(
          device_mgr_, env_, config_ ? &(*config_) : nullptr, d,
          graph_def_version_, lib_def_, default_thread_pool_,
          optimizer_options_, session_metadata_, this);
    }
  }
}

FunctionLibraryRuntime* ProcessFunctionLibraryRuntime::GetFLR(
    const string& device_name) const {
  Device* device = nullptr;
  if (device_name != kDefaultFLRDevice) {
    if (!device_mgr_->LookupDevice(device_name, &device).ok()) {
      VLOG(4) << "Could not find device: " << device_name;
      return nullptr;
    }
  }
  const auto& iter = flr_map_->find(device);
  if (iter == flr_map_->end()) {
    VLOG(1) << "Could not find device: " << device_name
            << "in the local process.";
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
      absl::make_unique<FunctionData>(device_name, local_handle, function_key);
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
    const string& device_name, FunctionLibraryRuntime::Handle handle,
    bool include_multi_device) const {
  tf_shared_lock l(mu_);

  auto miter = mdevice_data_.find(handle);
  if (miter != mdevice_data_.end()) {
    if (!include_multi_device) return kInvalidLocalHandle;

    const MultiDeviceFunctionData& data = *miter->second;
    if (data.glue_.size() != 1) return kInvalidLocalHandle;

    const auto& pair = *data.glue_.begin();
    const string& func_device_name = pair.first;
    const ComponentFunctionData& component_data = pair.second;
    if (func_device_name != device_name) return kInvalidLocalHandle;

    // Replace the given handle with the handle for the single component
    // function.
    handle = component_data.handle;
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

const string* AssignedOrRequestedDeviceName(const Node& node) {
  if (node.has_assigned_device_name()) {
    return &node.assigned_device_name();
  }
  return &node.requested_device();
}

Status SetArgShape(
    const std::unordered_map<int, DtypeAndPartialTensorShape>&
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
  return Status::OK();
}

// Returns the local tensors referred by `args`.
std::vector<Tensor> GetLocalArgs(gtl::ArraySlice<FunctionArg> args) {
  std::vector<Tensor> tensors;
  for (const auto& arg : args) {
    if (arg.index() == 0) {
      tensors.push_back(absl::get<Tensor>(arg));
    }
  }
  return tensors;
}

// Update the done callback to push Tensors in `tensors` into `rets`.
FunctionLibraryRuntime::DoneCallback TensorsToFunctionRetsDoneCallback(
    std::vector<FunctionRet>* rets, std::vector<Tensor>* tensors,
    FunctionLibraryRuntime::DoneCallback done) {
  return [rets, tensors, done = std::move(done)](const Status& s) {
    if (s.ok()) {
      for (const auto& t : *tensors) {
        rets->push_back(t);
      }
    }
    delete tensors;
    done(s);
  };
}

}  // anonymous namespace

Status ProcessFunctionLibraryRuntime::PinArgsAndRets(
    const std::vector<string>& input_devices,
    const std::vector<string>& output_devices, const DeviceSet& device_set,
    const std::vector<Node*>& arg_nodes, const std::vector<Node*>& ret_nodes,
    Device* default_device) const {
  // If output_devices are not specified, we want to set the output device
  // based on the device of the output producing node. The output producing
  // node can be an arg node because functions can simply return their
  // arguments. To make sure that the output producing nodes have assigned
  // devices, we assign them to arguments first.
  for (Node* node : arg_nodes) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
    int64 index = attr_value->i();
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
            !(dtype == DT_RESOURCE && IsFunctionCall(*lib_def_, *src_node));

        if (!colocation_group.empty()) {
          AttrValue::ListValue colo_attr;
          colo_attr.add_s(colocation_group);
          std::vector<string> colo_slice = {colocation_group};
          node->AddAttr(kColocationAttrName, colo_slice);
        } else if (!src_device->empty() && can_use_src_node_device) {
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
      int64 index = attr_value->i();
      // output_devices size is checked in InstantiateMultiDevice
      DCHECK_GT(output_devices.size(), index);
      VLOG(3) << "Setting output device to " << output_devices[index]
              << " for return at index " << index;
      node->set_assigned_device_name(output_devices[index]);
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
  return Status::OK();
}

}  // anonymous namespace

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
  return Status::OK();
}

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
          << "\" on default device \"" << options.target << "\"";
  if (VLOG_IS_ON(3)) {
    int index = 0;
    VLOG(3) << "Requested input devices:";
    for (const string& device : options.input_devices) {
      VLOG(3) << "    [input " << index++ << "] " << device;
    }
    index = 0;
    VLOG(3) << "Requested output devices:";
    for (const string& device : options.output_devices) {
      VLOG(3) << "    [output " << index++ << "] " << device;
    }
  }

  const FunctionLibraryDefinition* lib_def =
      options.lib_def == nullptr ? lib_def_ : options.lib_def;

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

  if (options.graph_collector != nullptr) {
    GraphDef def;
    graph->ToGraphDef(&def);
    *def.mutable_library() = lib_def->ReachableDefinitions(def).ToProto();
    options.graph_collector->CollectRawGraph(def);
  }

  Device* default_device = nullptr;
  if (options.default_device_to_target && !options.target.empty()) {
    // Make the `target` device the default device if nothing else is hard
    // coded. This allows the same function definition to be specialized to
    // different devices depending on the `PartitionedCallOp` device.
    FunctionLibraryRuntime* flr = GetFLR(options.target);
    if (flr == nullptr) {
      return errors::InvalidArgument(
          "Cannot instantiate multi-device function with target device ",
          options.target);
    }
    default_device = flr->device();
  }
  const std::shared_ptr<DeviceSet> dev_set = device_set();

  TF_RETURN_IF_ERROR(
      SetArgShape(options.input_resource_dtypes_and_shapes, arg_nodes));
  TF_RETURN_IF_ERROR(PinArgsAndRets(
      options.input_devices, options.output_devices, *dev_set, arg_nodes,
      ret_nodes,
      options.config_proto.allow_soft_placement() ? default_device : nullptr));

  auto data = absl::make_unique<MultiDeviceFunctionData>(
      function_name, function_key, ret_node_names.size(),
      lib_def->ReachableDefinitions(*fdef), std::move(ret_types));

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
        *dev_set, options.config_proto, &graph, &data->lib_def_,
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
  session_options.env = env_;
  session_options.config = options.config_proto;
  optimization_options.session_options = &session_options;
  optimization_options.graph = &graph;
  optimization_options.flib_def = &data->lib_def_;
  optimization_options.device_set = dev_set.get();
  optimization_options.is_function_graph = true;

  DumpGraph("Before running PRE_PLACEMENT passes", graph.get());
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));
  }

  // TODO(b/124993244): Smartly merge options in nested defuns, and raise
  // exceptions/warnings in case where nested function call options are ignored.
  DumpGraph("Before calling Placer", graph.get());
  Placer placer(graph.get(), function_name, optimization_options.flib_def,
                dev_set.get(), default_device,
                options.config_proto.allow_soft_placement(),
                options.config_proto.log_device_placement());
  TF_RETURN_IF_ERROR(placer.Run());

  DumpGraph("Before running POST_PLACEMENT passes", graph.get());
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::POST_PLACEMENT, optimization_options));
  }

  Device* cpu_device;
  TF_RETURN_IF_ERROR(device_mgr_->LookupDevice("CPU:0", &cpu_device));

  if (options.optimize_graph_fn) {
    DumpGraph("Before running graph optimization fn", graph.get());
    Status status = options.optimize_graph_fn(
        std::move(ret_node_names), std::move(control_ret_node_names),
        &data->lib_def_, *dev_set, cpu_device, &graph);
    if (!status.ok()) {
      LOG(WARNING) << "Ignoring multi-device function optimization failure: "
                   << status.ToString();
    }
    DumpGraph("After optimization", graph.get());
  }

  DumpGraph("Before running POST_REWRITE_FOR_EXEC passes", graph.get());
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));
  }

  // Expand the nodes assigned to a CompositeDevice before graph partition to
  // avoid generating a subgraph on a virtual device for execution.
  // This transformation should happen as late as possible, in order to run as
  // more graph optimization passes (e.g. PRE_PLACEMENT, PLACER,
  // POST_PLACEMENT, POST_REWRITE_FOR_EXEC) on a smaller graph as possible.
  TF_RETURN_IF_ERROR(ReplicatePerReplicaNodesInFunctionGraph(
      options.composite_devices, graph.get()));

  if (options.graph_collector != nullptr) {
    GraphDef def;
    graph->ToGraphDef(&def);
    *def.mutable_library() = lib_def->ReachableDefinitions(def).ToProto();
    options.graph_collector->CollectOptimizedGraph(def);
  }

  VLOG(4) << "Main function graph to be partitioned:";
  VLOG(4) << DebugString(graph->ToGraphDefDebug());

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  TF_RETURN_IF_ERROR(
      PartitionFunctionGraph(*dev_set, std::move(graph), &subgraphs));

  for (const auto& pair : subgraphs) {
    DumpGraph(strings::StrCat("Before running POST_PARTITIONING passes (",
                              pair.first, ")"),
              pair.second.get());
  }
  optimization_options.graph = nullptr;
  optimization_options.device_set = nullptr;
  optimization_options.partition_graphs = &subgraphs;
  // Normally POST_PARTITIONING passes are run by distributed workers.
  // Distributed workers are currently not supported in this code path, so we
  // run the passes here.
  if (should_run_optimization_passes) {
    TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
        OptimizationPassRegistry::POST_PARTITIONING, optimization_options));
  }
  for (const auto& pair : subgraphs) {
    const auto* optimized_subgraph = pair.second.get();
    DumpGraph(
        strings::StrCat("After all optimization passes (", pair.first, ")"),
        optimized_subgraph);
    if (VLOG_IS_ON(1)) {
      DumpGraphDefToFile(
          strings::StrCat("pflr_after_all_optimization_passes_",
                          reinterpret_cast<uintptr_t>(optimized_subgraph), "_",
                          pair.first),
          optimized_subgraph->ToGraphDefDebug());
    }
  }

  if (options.graph_collector != nullptr) {
    for (const auto& pair : subgraphs) {
      GraphDef def;
      pair.second->ToGraphDef(&def);
      *def.mutable_library() = lib_def->ReachableDefinitions(def).ToProto();
      options.graph_collector->CollectPartitionedGraph(def);
    }
  }

  // We must preserve control returns in each of the function components,
  // otherwise after function inlining we might prune side-effectful nodes.
  const auto control_ret =
      [&node_name_to_control_ret](const Node* n) -> absl::optional<string> {
    const auto it = node_name_to_control_ret.find(n->name());
    return it != node_name_to_control_ret.end()
               ? absl::make_optional<string>(it->second)
               : absl::nullopt;
  };

  int i = 0;
  // Generate a random function_name to avoid one function reuse the partition
  // function instantiated by another function.
  FunctionLibraryDefinition* data_lib_def = &data->lib_def_;
  FunctionNameGenerator name_generator(
      data_lib_def, absl::StrCat(function_name, "_", random::New64()));
  auto subgraph_size = subgraphs.size();
  gtl::InlinedVector<Status, 4> instantiate_status(subgraph_size);
  BlockingCounter counter(static_cast<int>(subgraph_size));
  auto runner = [this, subgraph_size](std::function<void()> fn) {
    // NOTE: Only use thread pool to instantiate sub-function when there are
    // more than 8 sub-functions. We want to avoid cost of switching thread when
    // there are only a few sub-functions.
    if (default_thread_pool_ != nullptr && subgraph_size > 8) {
      default_thread_pool_->Schedule(fn);
    } else {
      fn();
    }
  };
  for (const auto& pair : subgraphs) {
    Status* status = &instantiate_status[i];
    string unique_name = name_generator.GetName();
    ComponentFunctionData* comp_data = &data->glue_[pair.first];
    runner([this, &pair, dev_set, comp_data, unique_name, data_lib_def,
            &control_ret, &options, status, &counter, &data] {
      const string& target = pair.first;

      const string& device_type =
          dev_set->FindDeviceByName(target)->device_type();
      Graph* subgraph = pair.second.get();

      status->Update(UpdateArgAndRetvalMetadata(
          subgraph, device_type, &comp_data->arg_indices,
          &comp_data->ret_indices, &comp_data->arg_alloc_attrs,
          &comp_data->ret_alloc_attrs));
      if (!status->ok()) {
        counter.DecrementCount();
        return;
      }
      FunctionDef shard;
      status->Update(
          GraphToFunctionDef(*subgraph, unique_name, control_ret, &shard));
      if (!status->ok()) {
        counter.DecrementCount();
        return;
      }
      status->Update(data_lib_def->AddFunctionDef(shard));
      if (!status->ok()) {
        counter.DecrementCount();
        return;
      }
      FunctionLibraryRuntime::InstantiateOptions opts;
      opts.executor_type = options.executor_type;
      opts.target = target;
      opts.lib_def = data_lib_def;
      opts.create_kernels_eagerly = options.create_kernels_eagerly;
      opts.state_handle = options.state_handle;
      auto attrs = AttrSlice(&shard.attr());
      VLOG(1) << "Start instantiating component function " << unique_name
              << " on device " << target;
      VLOG(4) << DebugString(shard);

      auto* component_handle = new FunctionLibraryRuntime::Handle;
      auto done = [this, status, unique_name, comp_data, component_handle,
                   &data, &counter](const Status& s) {
        status->Update(s);

        VLOG(1) << "Finished instantiating component function " << unique_name
                << " with handle " << *component_handle << " status: " << s;
        if (status->ok()) {
          {
            mutex_lock l(mu_);
            if (function_data_[*component_handle]->is_cross_process()) {
              data->is_cross_process_ = true;
            }
          }
          comp_data->handle = *component_handle;
        }
        delete component_handle;
        counter.DecrementCount();
      };

      FunctionLibraryRuntime* flr = GetFLR(opts.target);
      if (flr != nullptr) {
        // Initialize local function synchronously.
        Status s = flr->Instantiate(unique_name, attrs, opts, component_handle);
        done(s);
      } else {
        opts.ret_indices = comp_data->ret_indices;
        // Initialize remote function asynchronously.
        InstantiateRemote(unique_name, attrs, opts, component_handle, done);
      }
    });
    i += 1;
  }
  counter.Wait();
  StatusGroup group;
  for (auto& status : instantiate_status) {
    group.Update(status);
  }
  TF_RETURN_IF_ERROR(group.as_summary_status());

  *handle = AddMultiDeviceHandle(std::move(data), function_key);
  VLOG(2) << "Instantiated MultiDevice function \"" << function_name
          << "\" with handle " << *handle;
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::GetOutputDevices(
    FunctionLibraryRuntime::Handle handle,
    std::vector<Device*>* output_devices) const {
  MultiDeviceFunctionData* data = IsMultiDevice(handle);
  if (data == nullptr) {
    return errors::InvalidArgument(
        "Failed for find multi-device function handle ", handle);
  }

  for (const auto& pair : data->glue_) {
    const ComponentFunctionData& comp_data = pair.second;
    DCHECK(comp_data.ret_alloc_attrs.size() == comp_data.ret_indices.size());
    if (comp_data.ret_indices.empty()) {
      continue;
    }

    const string& target = pair.first;
    FunctionLibraryRuntime* target_flr = GetFLR(target);
    Device* target_device = nullptr;
    Device* host = nullptr;
    if (target_flr == nullptr) {
      if (!data->has_remote_outputs) {
        data->has_remote_outputs = true;
      }
      target_device = device_set()->FindDeviceByName(target);
      string remote_host;
      TF_RETURN_IF_ERROR(
          DeviceNameUtils::DeviceNameToCpuDeviceName(target, &remote_host));
      host = device_set()->FindDeviceByName(remote_host);
    } else {
      target_device = target_flr->device();
    }
    output_devices->resize(data->num_outputs_);
    for (int j = 0; j < comp_data.ret_indices.size(); ++j) {
      int ret_index = comp_data.ret_indices[j];
      if (data->ret_types_[ret_index] == DT_RESOURCE) {
        (*output_devices)[ret_index] = target_device;
      } else {
        (*output_devices)[ret_index] =
            comp_data.ret_alloc_attrs[j].on_host() ? host : target_device;
      }
    }
  }

  return Status::OK();
}

void ProcessFunctionLibraryRuntime::RunMultiDevice(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, std::vector<FunctionRet>* rets,
    std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
    FunctionLibraryRuntime::DoneCallback done,
    std::function<Status(const ComponentFunctionData& comp_data,
                         InternalArgs* args)>
        get_component_args) const {
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
    done(errors::NotFound("Multi-device function handle ", handle,
                          "not found. Was the function instantiated?"));
    return;
  }

  VLOG(1) << "Running multi-device function " << data->function_name_;
  VLOG(4) << "    with " << opts.DebugString();

  if (data->glue_.empty()) {
    // Trivial case where the function body is empty.
    done(Status::OK());
    return;
  }

  // Check whether we have the right rendezvous.
  if (opts.rendezvous && data->is_cross_process_ &&
      !opts.rendezvous->is_cross_process()) {
    done(errors::InvalidArgument(
        "Running a cross process function ", data->function_name_,
        " without an appropriate cross process Rendezvous."));
    return;
  }

  // A locally created cancellation manager, used only when the caller does not
  // provide one in argument.
  std::shared_ptr<CancellationManager> local_cm;
  CancellationManager* cm = opts.cancellation_manager;
  if (cm == nullptr) {
    local_cm = std::make_shared<CancellationManager>();
    cm = local_cm.get();
  }

  auto* refcounted_done = new ReffedStatusCallback(std::move(done));
  for (int i = 0; i < data->glue_.size(); ++i) {
    refcounted_done->Ref();
  }

  FunctionLibraryRuntime::Options opts_copy = opts;
  for (const auto& pair : data->glue_) {
    const string& target = pair.first;
    const ComponentFunctionData& comp_data = pair.second;
    FunctionLibraryRuntime::Handle handle = pair.second.handle;

    opts_copy.args_alloc_attrs = comp_data.arg_alloc_attrs;
    opts_copy.rets_alloc_attrs = comp_data.ret_alloc_attrs;
    opts_copy.cancellation_manager = cm;

    InternalArgs comp_args;
    Status s = get_component_args(comp_data, &comp_args);
    if (!s.ok()) {
      VLOG(2) << "Failed to get component function arguments: " << s;
      refcounted_done->UpdateStatus(s);
      refcounted_done->Unref();
      cm->StartCancel();
      continue;
    }
    std::vector<FunctionRet>* comp_rets = new std::vector<FunctionRet>;
    rets->resize(data->num_outputs_);

    auto component_fn_callback = [comp_rets, rets, comp_data, refcounted_done,
                                  cm, local_cm, data, handle,
                                  target](const Status& status) {
      if (!status.ok()) {
        VLOG(2) << "Component function execution on target " << target
                << " from " << data->function_name_ << " with handle " << handle
                << " failed: " << status;
        const string function_and_msg = strings::StrCat(
            errors::FormatFunctionForError(data->function_name_), " ",
            status.error_message());
        refcounted_done->UpdateStatus(Status(status.code(), function_and_msg));
        // Cancel the execution of other component functions.
        cm->StartCancel();
      } else {
        VLOG(2) << "Component function execution on target " << target
                << " from " << data->function_name_ << " with handle " << handle
                << " succeeded.";
        for (int i = 0; i < comp_rets->size(); ++i) {
          (*rets)[comp_data.ret_indices[i]] = (*comp_rets)[i];
        }
      }
      delete comp_rets;
      // refcounted_done is thread-safe
      refcounted_done->Unref();
    };

    FunctionLibraryRuntime* flr = GetFLR(target);
    if (flr != nullptr) {
      opts_copy.remote_execution = false;
      // When target device has private thread pool, use the target device
      // runner
      thread::ThreadPool* pool = flr->device()->tensorflow_device_thread_pool();
      opts_copy.runner = (pool == nullptr) ? opts_copy.runner : flr->runner();

      VLOG(1) << "Running component function on device " << target << " from "
              << data->function_name_ << " with handle " << handle;
      VLOG(4) << "    with " << opts_copy.DebugString();

      std::vector<Tensor>* comp_tensor_rets = new std::vector<Tensor>;
      flr->Run(
          opts_copy, handle, GetLocalArgs(comp_args.args), comp_tensor_rets,
          TensorsToFunctionRetsDoneCallback(comp_rets, comp_tensor_rets,
                                            std::move(component_fn_callback)));
    } else {
      opts_copy.remote_execution = true;

      VLOG(1) << "Running component function on device " << target << " from "
              << data->function_name_ << " with handle " << handle;
      VLOG(4) << "    with " << opts_copy.DebugString();

      RunInternal(opts_copy, handle, comp_args.args, comp_rets, cleanup_items,
                  std::move(component_fn_callback));
    }
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

  Status status;
  Notification notification;
  InstantiateRemote(function_name, attrs, options, handle,
                    [&status, &notification](const Status& s) {
                      status = s;
                      notification.Notify();
                    });
  notification.WaitForNotification();
  return status;
}

Status ProcessFunctionLibraryRuntime::IsCrossProcess(
    FunctionLibraryRuntime::Handle handle, bool* is_cross_process) const {
  tf_shared_lock l(mu_);
  const auto& mdevice_it = mdevice_data_.find(handle);
  if (mdevice_it != mdevice_data_.end()) {
    *is_cross_process = mdevice_it->second->is_cross_process_;
    return Status::OK();
  }
  const auto& it = function_data_.find(handle);
  if (it != function_data_.end()) {
    *is_cross_process = it->second->is_cross_process();
    return Status::OK();
  }
  return errors::InvalidArgument("Handle ", handle, " not found.");
}

void ProcessFunctionLibraryRuntime::InstantiateRemote(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle,
    FunctionLibraryRuntime::DoneCallback done) {
  if (parent_ == nullptr) {
    done(errors::Internal(
        "Currently don't support instantiating functions on device: ",
        options.target));
    return;
  }
  auto target = options.target;
  VLOG(1) << "ProcessFLR Instantiate: " << function_name << " on: " << target;
  string function_key = Canonicalize(function_name, attrs, options);
  FunctionData* f;
  {
    mutex_lock l(mu_);
    FunctionLibraryRuntime::Handle h =
        gtl::FindWithDefault(table_, function_key, kInvalidHandle);
    if (h == kInvalidHandle || function_data_.count(h) == 0) {
      h = AddHandleLocked(function_key, target, kInvalidHandle);
    }
    f = function_data_[h].get();
    *handle = h;
  }
  f->DistributedInit(
      parent_, function_name,
      options.lib_def == nullptr ? *lib_def_ : *options.lib_def, attrs, options,
      [this, function_name, target, handle, done](const Status& s) {
        VLOG(1) << "ProcessFLR Instantiate [success]: " << function_name
                << " on: " << target << " with handle: " << *handle
                << " (this: " << this << ")";
        done(s);
      });
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
    FunctionLibraryRuntime::Handle flr_handle = it.second.handle;
    FunctionLibraryRuntime* flr = GetFLR(device);
    if (flr == nullptr) {
      // TODO(nareshmodi): Implement DeregisterGraph call to remote device if
      // parent is not null.
      if (parent_ != nullptr) {
        return errors::Unimplemented(
            "Releasing a multi-device component handle on a remote device is "
            "not yet implemented.");
      }
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
  // Return directly if all function handles has already been released.
  if (flr_map_ == nullptr) return Status::OK();

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

FunctionLibraryRuntime::DoneCallback
ProcessFunctionLibraryRuntime::ApplyCleanUpToDoneCallback(
    std::vector<std::unique_ptr<CleanUpItem>>* items,
    FunctionLibraryRuntime::DoneCallback done, const int64 step_id,
    const Rendezvous* created_rendezvous) const {
  return
      [this, items, done = std::move(done), step_id,
       created_rendezvous](const Status& status) {
        if (created_rendezvous) {
          DCHECK(rendezvous_factory_);
          created_rendezvous->Unref();
          Status s = rendezvous_factory_.CleanUp(step_id);
          if (!s.ok()) {
            LOG(ERROR) << s;
          }
        }
        auto* local_status = new Status(status);
        CleanUp(items, [local_status, done](const Status& cleanup_status) {
          local_status->Update(cleanup_status);
          done(*local_status);
          delete local_status;
        });
        delete items;
      };
}

Status ProcessFunctionLibraryRuntime::CreateRendezvous(
    const FunctionLibraryRuntime::Options& opts,
    Rendezvous** created_rendezvous) const {
  if (rendezvous_factory_) {
    return rendezvous_factory_(opts.step_id, device_mgr_, created_rendezvous);
  } else {
    return errors::FailedPrecondition(
        "The caller does not provide a rendezvous and "
        "ProcessFunctionLibraryRuntime was created without a rendezvous "
        "factory.");
  }
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  FunctionLibraryRuntime::Options new_opts = opts;
  Rendezvous* created_rendezvous = nullptr;
  if (!opts.rendezvous) {
    Status s = CreateRendezvous(opts, &created_rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
    new_opts.rendezvous = created_rendezvous;
    new_opts.create_rendezvous = false;
  }

  auto* cleanup_items = new std::vector<std::unique_ptr<CleanUpItem>>;
  done = ApplyCleanUpToDoneCallback(cleanup_items, std::move(done),
                                    new_opts.step_id, created_rendezvous);
  std::vector<FunctionRet>* function_rets = new std::vector<FunctionRet>;
  done = [rets, function_rets, done = std::move(done)](const Status& s) {
    Status status = s;
    if (status.ok()) {
      for (const auto& ret : *function_rets) {
        if (ret.index() == 0) {
          rets->push_back(absl::get<Tensor>(ret));
        } else {
          status.Update(errors::Internal(
              "Expect a Tensor as a function output but got a TensorShape."));
          break;
        }
      }
    }
    delete function_rets;
    done(status);
  };
  bool multi_device;
  {
    tf_shared_lock l(mu_);
    multi_device = mdevice_data_.find(handle) != mdevice_data_.end();
  }
  if (multi_device) {
    auto get_component_args = [&args](const ComponentFunctionData& comp_data,
                                      InternalArgs* comp_args) -> Status {
      // "Index"s of _Arg nodes are unique when all arguments are local Tensors.
      for (const auto& it : comp_data.arg_indices) {
        if (it.index >= args.size()) {
          return errors::InvalidArgument(
              "index ", it.index, " is out of range [0, ", args.size(), ")");
        }
        if (it.sub_index >= 0) {
          const Tensor& t = args[it.index];
          if (t.dtype() != DT_RESOURCE) {
            return errors::InvalidArgument("Got unexpected sub_index ",
                                           it.sub_index, " for argument ",
                                           it.index);
          }
          const auto& handles = t.flat<ResourceHandle>();
          if (it.sub_index >= handles.size()) {
            return errors::InvalidArgument(
                "Sub_index ", it.sub_index, "is out of range [0,",
                handles.size(), ") for argument ", it.index);
          }
          comp_args->args.push_back(Tensor(handles(it.sub_index)));
        } else {
          comp_args->args.push_back(args[it.index]);
        }
      }
      return Status::OK();
    };
    return RunMultiDevice(new_opts, handle, function_rets, cleanup_items,
                          std::move(done), std::move(get_component_args));
  }
  std::vector<FunctionArg> local_args;
  for (const auto& tensor : args) {
    local_args.push_back(tensor);
  }
  RunInternal(new_opts, handle, local_args, function_rets, cleanup_items,
              std::move(done));
}

void ProcessFunctionLibraryRuntime::RunInternal(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<FunctionArg> args,
    std::vector<FunctionRet>* rets,
    std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
    FunctionLibraryRuntime::DoneCallback done) const {
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

    std::vector<Tensor> local_args = GetLocalArgs(args);

    // Send the args over to the target device.
    s = SendTensors(source_device, target_device, "arg_", src_incarnation,
                    local_args, device_context, opts.args_alloc_attrs,
                    rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
    const std::vector<AllocatorAttributes>& rets_alloc_attrs =
        opts.rets_alloc_attrs;
    std::vector<Tensor>* remote_rets = new std::vector<Tensor>;
    flr->Run(opts, handle, local_args, remote_rets,
             [source_device, target_device, target_incarnation, rendezvous,
              device_context, rets_alloc_attrs, remote_rets, rets,
              done = std::move(done)](const Status& status) mutable {
               if (!status.ok()) {
                 delete remote_rets;
                 done(status);
                 return;
               }
               int64 num_returns = remote_rets->size();
               delete remote_rets;
               // Now receive the return values from the target.
               std::vector<Tensor>* recv_tensors = new std::vector<Tensor>;
               ReceiveTensorsAsync(target_device, source_device, "ret_",
                                   target_incarnation, num_returns,
                                   device_context, rets_alloc_attrs, rendezvous,
                                   recv_tensors,
                                   TensorsToFunctionRetsDoneCallback(
                                       rets, recv_tensors, std::move(done)));
             });
    return;
  }
  if (parent_ != nullptr) {
    auto cleanup_item = absl::make_unique<CleanUpItem>();
    cleanup_item->device = target_device;
    cleanup_item->step_id = opts.step_id;
    cleanup_item->local_handle = local_handle;
    cleanup_items->emplace_back(std::move(cleanup_item));
    parent_->Run(opts, local_handle, args, rets, std::move(done));
    return;
  }
  done(errors::Internal("Could not find device"));
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, CallFrameInterface* frame,
    FunctionLibraryRuntime::DoneCallback done) const {
  std::vector<Tensor> args;
  args.reserve(frame->num_args());
  for (size_t i = 0; i < frame->num_args(); ++i) {
    const Tensor* arg;
    Status s = frame->GetArg(i, &arg);
    args.emplace_back(*arg);
    if (!s.ok()) {
      done(s);
    }
  }
  std::vector<Tensor>* rets = new std::vector<Tensor>;
  rets->reserve(frame->num_retvals());

  Run(opts, handle, args, rets,

      [frame, rets, done = std::move(done)](const Status& status) {
        std::unique_ptr<std::vector<Tensor>> rets_releaser(rets);

        if (!status.ok()) {
          done(status);
          return;
        }

        if (rets->size() != frame->num_retvals()) {
          done(errors::Internal(
              "Number of return values from function (", rets->size(),
              ") did not match expected number of return values (",
              frame->num_retvals(), ")."));
          return;
        }

        for (size_t i = 0; i < frame->num_retvals(); ++i) {
          Status s = frame->SetRetval(i, (*rets)[i]);
          if (!s.ok()) {
            done(s);
            return;
          }
        }
        done(Status::OK());
      });
}

Status ProcessFunctionLibraryRuntime::RunSync(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets) const {
  Notification n;
  Status s;
  Run(opts, handle, args, rets, [&n, &s](const Status& status) {
    s.Update(status);
    n.Notify();
  });
  n.WaitForNotification();
  return s;
}

Status ProcessFunctionLibraryRuntime::RunSync(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, CallFrameInterface* frame) const {
  Notification n;
  Status s;
  Run(opts, handle, frame, [&n, &s](const Status& status) {
    s.Update(status);
    n.Notify();
  });
  n.WaitForNotification();
  return s;
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, const FunctionArgsInterface& args,
    std::vector<FunctionRet>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  bool has_remote_outputs = false;
  const MultiDeviceFunctionData* data = IsMultiDevice(handle);
  if (data != nullptr) {
    has_remote_outputs = data->has_remote_outputs;
  }
  if (!args.HasRemoteOrPackedInputs() && !has_remote_outputs) {
    const std::vector<Tensor> local_inputs = args.GetLocalTensors();
    std::vector<Tensor>* tensor_rets = new std::vector<Tensor>;
    return Run(
        opts, handle, local_inputs, tensor_rets,
        TensorsToFunctionRetsDoneCallback(rets, tensor_rets, std::move(done)));
  }

  FunctionLibraryRuntime::Options new_opts = opts;
  Rendezvous* created_rendezvous = nullptr;
  if (!opts.rendezvous) {
    Status s = CreateRendezvous(opts, &created_rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
    new_opts.rendezvous = created_rendezvous;
    new_opts.create_rendezvous = false;
  }

#if defined(IS_MOBILE_PLATFORM)
  done(errors::Unimplemented(
      "Remote inputs are not available on mobile devices."));
  return;
#else   // !IS_MOBILE_PLATFORM
  auto* cleanup_items = new std::vector<std::unique_ptr<CleanUpItem>>;
  done = ApplyCleanUpToDoneCallback(cleanup_items, done, opts.step_id,
                                    created_rendezvous);

  auto get_component_args = [&args](const ComponentFunctionData& comp_data,
                                    InternalArgs* comp_args) -> Status {
    for (int i = 0; i < comp_data.arg_indices.size(); ++i) {
      const FunctionArgIndex index = comp_data.arg_indices.at(i);
      Tensor tensor;
      if (args.GetLocalArg(index, &tensor).ok()) {
        comp_args->args.push_back(std::move(tensor));
      } else {
        eager::RemoteTensorHandle remote_handle;
        TF_RETURN_IF_ERROR(args.GetRemoteArg(index, &remote_handle));
        comp_args->remote_args.emplace_back(
            absl::make_unique<eager::RemoteTensorHandle>(
                std::move(remote_handle)));
        comp_args->args.push_back(comp_args->remote_args.back().get());
      }
    }
    return Status::OK();
  };
  return RunMultiDevice(new_opts, handle, rets, cleanup_items, std::move(done),
                        std::move(get_component_args));
#endif  // !IS_MOBILE_PLATFORM
}

void ProcessFunctionLibraryRuntime::CleanUp(
    std::vector<std::unique_ptr<CleanUpItem>>* items,
    FunctionLibraryRuntime::DoneCallback done) const {
  auto* refcounted_done = new ReffedStatusCallback(std::move(done));
  for (auto& item : *items) {
    refcounted_done->Ref();
    auto* flr = GetFLR(item->device);
    if (flr != nullptr) {
      // TODO(fishx): cleanup state for local execution.
      refcounted_done->UpdateStatus(
          errors::Internal("Cleanup items shouldn't contain local item."));
      refcounted_done->Unref();
    } else if (parent_ != nullptr) {
      parent_->CleanUp(item->step_id, item->local_handle,
                       [refcounted_done](const Status& status) {
                         if (!status.ok()) {
                           refcounted_done->UpdateStatus(status);
                         }
                         // refcounted_done is thread-safe
                         refcounted_done->Unref();
                       });
    } else {
      refcounted_done->UpdateStatus(
          errors::Internal("Could not find device in cleanup."));
      refcounted_done->Unref();
    }
  }
  refcounted_done->Unref();
}

Status ProcessFunctionLibraryRuntime::Clone(
    Env* env, int graph_def_version, const OptimizerOptions& optimizer_options,
    std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
    bool skip_flib_def) const {
  if (skip_flib_def) {
    *out_lib_def = absl::make_unique<FunctionLibraryDefinition>(
        lib_def_->default_registry(), FunctionDefLibrary{});
  } else {
    *out_lib_def = absl::make_unique<FunctionLibraryDefinition>(*lib_def_);
  }
  *out_pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_, env, config_ ? &(*config_) : nullptr, graph_def_version,
      out_lib_def->get(), optimizer_options, default_thread_pool_, parent_,
      session_metadata_, rendezvous_factory_);
  {
    tf_shared_lock l(mu_);
    for (auto* d : composite_devices_) (*out_pflr)->AddCompositeDevice(d);
  }
  return Status::OK();
}

}  // namespace tensorflow
