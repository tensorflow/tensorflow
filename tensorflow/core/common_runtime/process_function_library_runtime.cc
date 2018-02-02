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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

const char ProcessFunctionLibraryRuntime::kDefaultFLRDevice[] = "null";

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    DistributedFunctionLibraryRuntime* parent)
    : device_mgr_(device_mgr),
      lib_def_(lib_def),
      next_handle_(0),
      parent_(parent) {
  if (device_mgr == nullptr) {
    flr_map_[nullptr] =
        NewFunctionLibraryRuntime(nullptr, env, nullptr, graph_def_version,
                                  lib_def, optimizer_options, this);
    return;
  }
  for (Device* d : device_mgr->ListDevices()) {
    flr_map_[d] =
        NewFunctionLibraryRuntime(device_mgr, env, d, graph_def_version,
                                  lib_def, optimizer_options, this);
  }
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    CustomKernelCreator custom_kernel_creator,
    DistributedFunctionLibraryRuntime* parent)
    : device_mgr_(device_mgr),
      lib_def_(lib_def),
      next_handle_(0),
      parent_(parent) {
  if (device_mgr == nullptr) {
    flr_map_[nullptr] = NewFunctionLibraryRuntime(
        nullptr, env, nullptr, graph_def_version, lib_def, optimizer_options,
        std::move(custom_kernel_creator), this);
    return;
  }
  for (Device* d : device_mgr->ListDevices()) {
    flr_map_[d] = NewFunctionLibraryRuntime(
        device_mgr, env, d, graph_def_version, lib_def, optimizer_options,
        custom_kernel_creator, this);
  }
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options)
    : ProcessFunctionLibraryRuntime(device_mgr, env, graph_def_version, lib_def,
                                    optimizer_options,
                                    nullptr /* cluster_flr */) {}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    CustomKernelCreator custom_kernel_creator)
    : ProcessFunctionLibraryRuntime(
          device_mgr, env, graph_def_version, lib_def, optimizer_options,
          std::move(custom_kernel_creator), nullptr /* cluster_flr */) {}

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
    std::vector<Tensor>* received_tensors, const StatusCallback& done) {
  std::vector<string> keys;
  for (int64 i = 0; i < num_tensors; ++i) {
    string name = strings::StrCat(key_prefix, i);
    string key = Rendezvous::CreateKey(source_device, src_incarnation,
                                       target_device, name, FrameAndIter(0, 0));
    keys.push_back(key);
  }
  RecvOutputsFromRendezvousAsync(
      rendezvous, device_context, alloc_attrs, keys, received_tensors,
      [done](const Status& status) { done(status); });
}

Status ProcessFunctionLibraryRuntime::GetDeviceIncarnation(
    const string& device_name, int64* incarnation) {
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found");
  }
  *incarnation = flr->device()->attributes().incarnation();
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::GetDeviceContext(
    const string& device_name, DeviceContext** device_context) {
  *device_context = nullptr;
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found.");
  }
  Device* device = flr->device();
  string device_type = device->parsed_name().type;
  if (device_type == "CPU") return Status::OK();
  if (device_type == "GPU") {
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
      LOG(ERROR) << "Could not find device: " << device_name;
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
  FunctionLibraryRuntime::Handle h =
      gtl::FindWithDefault(table_, function_key, kInvalidHandle);
  if (h != kInvalidHandle) {
    if (function_data_.count(h) != 0) return h;
  }
  h = next_handle_;
  function_data_.insert({h, FunctionData(device_name, local_handle)});
  table_[function_key] = h;
  next_handle_++;
  return h;
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::GetHandle(
    const string& function_key) const {
  mutex_lock l(mu_);
  FunctionLibraryRuntime::Handle h =
      gtl::FindWithDefault(table_, function_key, kInvalidHandle);
  if (h != kInvalidHandle) {
    if (function_data_.count(h) == 0) return kInvalidHandle;
  }
  return h;
}

bool ProcessFunctionLibraryRuntime::IsInstantiatedOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle) {
  return GetHandleOnDevice(device_name, handle) != kInvalidHandle;
}

FunctionLibraryRuntime::LocalHandle
ProcessFunctionLibraryRuntime::GetHandleOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle) {
  mutex_lock l(mu_);
  if (function_data_.count(handle) == 0) {
    return kInvalidLocalHandle;
  }
  const FunctionData& function_data = function_data_[handle];
  if (function_data.target_device != device_name) {
    return kInvalidLocalHandle;
  }
  return function_data.local_handle;
}

string ProcessFunctionLibraryRuntime::GetDeviceName(
    FunctionLibraryRuntime::Handle handle) {
  mutex_lock l(mu_);
  CHECK_EQ(1, function_data_.count(handle));
  const FunctionData& function_data = function_data_[handle];
  return function_data.target_device;
}

Status ProcessFunctionLibraryRuntime::Instantiate(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle) {
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
  FunctionLibraryRuntime::Handle cluster_handle;
  TF_RETURN_IF_ERROR(parent_->Instantiate(function_name, *lib_def_, attrs,
                                          options, &cluster_handle));
  string function_key = Canonicalize(function_name, attrs);
  *handle = AddHandle(function_key, options.target, cluster_handle);
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::RemoveHandle(
    FunctionLibraryRuntime::Handle handle) {
  mutex_lock l(mu_);
  function_data_.erase(handle);
  return Status::OK();
}

Status ProcessFunctionLibraryRuntime::ReleaseHandle(
    FunctionLibraryRuntime::Handle handle) {
  FunctionLibraryRuntime* flr = nullptr;
  string target_device;
  {
    mutex_lock l(mu_);
    CHECK_EQ(1, function_data_.count(handle));
    target_device = function_data_[handle].target_device;
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
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
  if (!opts.remote_execution) {
    done(errors::InvalidArgument(
        "ProcessFunctionLibraryRuntime::Run should only be called when there ",
        "is a remote execution."));
    return;
  }

  FunctionLibraryRuntime* flr = nullptr;
  string target_device;
  FunctionLibraryRuntime::LocalHandle local_handle;
  {
    mutex_lock l(mu_);
    if (function_data_.count(handle) == 0) {
      done(errors::NotFound("Handle: ", handle, " not found."));
      return;
    }
    target_device = function_data_[handle].target_device;
    local_handle = function_data_[handle].local_handle;
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
             [source_device, target_device, target_incarnation, rendezvous,
              device_context, rets_alloc_attrs, remote_rets, rets,
              done](const Status& status) {
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
                                   device_context, rets_alloc_attrs, rendezvous,
                                   rets, done);
             });
    return;
  }
  if (parent_ != nullptr) {
    parent_->Run(opts, local_handle, args, rets, done);
    return;
  }
  done(errors::Internal("Could not find device"));
}

}  // namespace tensorflow
