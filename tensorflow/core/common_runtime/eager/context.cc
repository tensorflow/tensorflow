/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/context.h"

#include "tensorflow/core/common_runtime/process_util.h"

namespace tensorflow {

EagerContext::EagerContext(const SessionOptions& opts,
                           ContextDevicePlacementPolicy default_policy,
                           bool async, std::unique_ptr<DeviceMgr> device_mgr,
                           Rendezvous* rendezvous)
    : policy_(default_policy),
      device_manager_(std::move(device_mgr)),
      devices_(device_manager_->ListDevices()),
      rendezvous_(rendezvous),
      thread_pool_(NewThreadPoolFromSessionOptions(opts)),
      pflr_(new ProcessFunctionLibraryRuntime(
          device_manager_.get(), opts.env, TF_GRAPH_DEF_VERSION, &func_lib_def_,
          {}, thread_pool_.get())),
      log_device_placement_(opts.config.log_device_placement()),
      async_default_(async) {
  if (async_default_) {
    executor_.EnableAsync();
  }

  for (auto* device : devices_) {
    devices_map_[device->name()] = device;
  }
}

bool EagerContext::Async() const {
  mutex_lock l(async_map_mu_);
  return gtl::FindWithDefault(thread_local_async_, std::this_thread::get_id(),
                              async_default_);
}

Status EagerContext::SetAsyncForThread(bool async) {
  {
    tensorflow::mutex_lock l(async_map_mu_);
    thread_local_async_[std::this_thread::get_id()] = async;
  }
  if (async) {
    executor_.EnableAsync();
  } else {
    // TODO(agarwal): Currently we add a wait here to handle cases where a
    // sync op has a control dependency on an async op, and the latter has not
    // executed yet. This wait can be removed by storing all the control
    // inputs and waiting for them when executing ops.
    return executor_.WaitForAllPendingNodes();
  }
  return Status::OK();
}

void EagerContext::ClearCaches() {
  mutex_lock ml(cache_mu_);
  gtl::STLDeleteValues(&kernel_cache_);
}

void EagerContext::SetThreadLocalDevicePlacementPolicy(
    ContextDevicePlacementPolicy policy) {
  mutex_lock ml(policy_map_mu_);
  thread_local_policies_[std::this_thread::get_id()] = policy;
}

ContextDevicePlacementPolicy EagerContext::GetDevicePlacementPolicy() {
  mutex_lock ml(policy_map_mu_);
  auto policy_map_it = thread_local_policies_.find(std::this_thread::get_id());
  if (policy_map_it != thread_local_policies_.end()) {
    return policy_map_it->second;
  }
  return policy_;
}

EagerContext::~EagerContext() {
  executor_.WaitForAllPendingNodes().IgnoreError();
  ClearCaches();
  rendezvous_->Unref();
}

bool EagerContext::FindFunctionByName(const string& name) {
  mutex_lock l(functions_mu_);
  return func_lib_def_.Find(name) != nullptr;
}

Status EagerContext::FindFunctionOpData(
    const string& name, const tensorflow::OpRegistrationData** op_data) {
  mutex_lock l(functions_mu_);
  return func_lib_def_.LookUp(name, op_data);
}

const FunctionDef* EagerContext::FindFunctionDef(const string& name) {
  mutex_lock l(functions_mu_);
  return func_lib_def_.Find(name);
}

Status EagerContext::FindDeviceByName(const string& name, Device** result) {
  auto it = devices_map_.find(name);
  if (it == devices_map_.end()) {
    return errors::InvalidArgument(name, " unknown device.");
  }
  *result = it->second;
  return Status::OK();
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef) {
  mutex_lock l(functions_mu_);
  return func_lib_def_.AddFunctionDef(fdef);
}

KernelAndDevice* EagerContext::GetCachedKernel(Fprint128 cache_key) {
  tf_shared_lock l(cache_mu_);
  return gtl::FindPtrOrNull(kernel_cache_, cache_key);
}

void EagerContext::AddKernelToCache(Fprint128 cache_key,
                                    KernelAndDevice* kernel) {
  mutex_lock ml(cache_mu_);
  gtl::InsertOrUpdate(&kernel_cache_, cache_key, kernel);
}

void EagerContext::SetShouldStoreMetadata(bool value) {
  should_store_metadata_.store(value);
  if (!value) {
    mutex_lock ml(metadata_mu_);
    run_metadata_.Clear();
  }
}

}  // namespace tensorflow
