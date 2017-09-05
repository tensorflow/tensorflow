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
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

const char ProcessFunctionLibraryRuntime::kDefaultFLRDevice[] = "null";

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options) {
  if (device_mgr == nullptr) {
    flr_map_[kDefaultFLRDevice] =
        NewFunctionLibraryRuntime(nullptr, env, nullptr, graph_def_version,
                                  lib_def, optimizer_options, this);
    return;
  }
  for (Device* d : device_mgr->ListDevices()) {
    flr_map_[d->name()] =
        NewFunctionLibraryRuntime(device_mgr, env, d, graph_def_version,
                                  lib_def, optimizer_options, this);
  }
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    CustomKernelCreator custom_kernel_creator) {
  if (device_mgr == nullptr) {
    flr_map_[kDefaultFLRDevice] = NewFunctionLibraryRuntime(
        nullptr, env, nullptr, graph_def_version, lib_def, optimizer_options,
        custom_kernel_creator, this);
  }
  for (Device* d : device_mgr->ListDevices()) {
    flr_map_[d->name()] = NewFunctionLibraryRuntime(
        device_mgr, env, d, graph_def_version, lib_def, optimizer_options,
        custom_kernel_creator, this);
  }
}

string ProcessFunctionLibraryRuntime::ObtainFunctionTarget(
    const AttrSlice& attrs) {
  const AttrValue* value;
  if (!attrs.Find("_target", &value).ok()) {
    return "";
  }
  return value->s();
}

FunctionLibraryRuntime* ProcessFunctionLibraryRuntime::GetFLR(
    const string& device_name) {
  if (flr_map_.find(device_name) == flr_map_.end()) {
    LOG(ERROR) << "Could not find device: " << device_name;
    return nullptr;
  }
  return flr_map_[device_name].get();
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::AddHandle(
    const string& function_key, const string& device_name,
    FunctionLibraryRuntime::LocalHandle local_handle) {
  mutex_lock l(mu_);
  FunctionLibraryRuntime::Handle h =
      gtl::FindWithDefault(table_, function_key, kInvalidHandle);
  if (h != kInvalidHandle) {
    return h;
  }
  h = function_data_.size();
  function_data_.emplace_back(device_name, local_handle);
  table_[function_key] = h;
  return h;
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::GetHandle(
    const string& function_key) const {
  mutex_lock l(mu_);
  return gtl::FindWithDefault(table_, function_key, kInvalidHandle);
}

bool ProcessFunctionLibraryRuntime::IsInstantiatedOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle) {
  return GetHandleOnDevice(device_name, handle) != -1;
}

FunctionLibraryRuntime::LocalHandle
ProcessFunctionLibraryRuntime::GetHandleOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle) {
  mutex_lock l(mu_);
  std::pair<string, FunctionLibraryRuntime::LocalHandle> p =
      function_data_[handle];
  if (p.first != device_name) {
    return kInvalidLocalHandle;
  }
  return p.second;
}

Status ProcessFunctionLibraryRuntime::Instantiate(
    const string& function_name, AttrSlice attrs,
    FunctionLibraryRuntime::Handle* handle) {
  string target = ObtainFunctionTarget(attrs);

  FunctionLibraryRuntime* flr = GetFLR(target);
  if (flr != nullptr) {
    return flr->Instantiate(function_name, attrs, handle);
  }
  return errors::InvalidArgument("Target: ", target, " is not supported");
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
  FunctionLibraryRuntime* flr = nullptr;
  {
    mutex_lock l(mu_);
    std::pair<string, FunctionLibraryRuntime::LocalHandle> p =
        function_data_[handle];
    flr = GetFLR(p.first);
  }
  if (flr != nullptr) {
    return flr->Run(opts, handle, args, rets, std::move(done));
  }
}

}  // namespace tensorflow
