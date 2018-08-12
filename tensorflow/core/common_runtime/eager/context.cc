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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

bool ReadBoolFromEnvVar(StringPiece env_var_name, bool default_val) {
  bool val;
  if (ReadBoolFromEnvVar(env_var_name, default_val, &val).ok()) {
    return val;
  }
  return default_val;
}

}  // namespace

EagerContext::EagerContext(const SessionOptions& opts,
                           ContextDevicePlacementPolicy default_policy,
                           bool async, std::unique_ptr<DeviceMgr> device_mgr,
                           Rendezvous* rendezvous)
    : policy_(default_policy),
      local_device_manager_(std::move(device_mgr)),
      local_unowned_device_manager_(nullptr),
      devices_(local_device_manager_->ListDevices()),
      rendezvous_(rendezvous),
      thread_pool_(NewThreadPoolFromSessionOptions(opts)),
      pflr_(new ProcessFunctionLibraryRuntime(
          local_device_manager_.get(), opts.env, TF_GRAPH_DEF_VERSION,
          &func_lib_def_, {}, thread_pool_.get())),
      log_device_placement_(opts.config.log_device_placement()),
      num_active_steps_(0),
      async_default_(async),
      env_(opts.env),
      use_send_tensor_rpc_(false) {
  InitDeviceMapAndAsync();
  if (opts.config.inter_op_parallelism_threads() > 0) {
    runner_ = [this](std::function<void()> closure) {
      this->thread_pool_->Schedule(closure);
    };
  } else {
    runner_ = [](std::function<void()> closure) { closure(); };
  }
}

void EagerContext::InitDeviceMapAndAsync() {
  if (async_default_) {
    executor_.EnableAsync();
  }

  for (auto* device : devices_) {
    devices_map_[device->name()] = device;
  }

  if (remote_device_manager_ != nullptr) {
    for (auto* device : remote_device_manager_->ListDevices()) {
      if (devices_map_.find(device->name()) == devices_map_.end()) {
        devices_map_[device->name()] = device;
        devices_.push_back(device);
      }
    }
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

#ifndef __ANDROID__
void EagerContext::CloseRemoteContexts() {
  // Close all remote contexts.
  std::vector<eager::CloseContextRequest> requests(remote_contexts_.size());
  std::vector<eager::CloseContextResponse> responses(remote_contexts_.size());
  BlockingCounter counter(static_cast<int>(remote_contexts_.size()));

  int i = 0;
  for (const auto& worker_and_context_id : remote_contexts_) {
    auto* client =
        remote_eager_workers_->GetClient(worker_and_context_id.first);

    requests[i].set_context_id(worker_and_context_id.second);
    client->CloseContextAsync(
        &requests[i], &responses[i],
        [&worker_and_context_id, &counter](const Status& s) {
          if (!s.ok()) {
            LOG(ERROR) << "Unable to close remote context with ID "
                       << worker_and_context_id.second
                       << " for worker: " << worker_and_context_id.first
                       << " due to " << s.error_message();
          }
          counter.DecrementCount();
        });
    i++;
  }

  counter.Wait();
}
#endif

EagerContext::~EagerContext() {
#ifndef __ANDROID__
  if (server_) {
    // TODO(nareshmodi): Fix this.
    LOG(WARNING) << "Unable to destroy server_ object, so releasing instead. "
                    "Servers don't support clean shutdown.";
    server_.release();
  }

  {
    mutex_lock l(keep_alive_thread_shutdown_mu_);
    shutting_down_ = true;
    keep_alive_thread_cv_.notify_all();
  }
  keep_alive_thread_.reset();

  CloseRemoteContexts();
#endif

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

void EagerContext::StartStep() {
  mutex_lock ml(metadata_mu_);
  num_active_steps_++;
  if (step_container_ == nullptr) {
    step_container_.reset(
        new ScopedStepContainer(0, [this](const string& name) {
          for (Device* device : devices_) {
            device->resource_manager()->Cleanup(name).IgnoreError();
          }
        }));
  }
}

void EagerContext::EndStep() {
  mutex_lock ml(metadata_mu_);
  num_active_steps_--;
  if (num_active_steps_ == 0) {
    step_container_.reset();
  }
}

ScopedStepContainer* EagerContext::StepContainer() {
  if (num_active_steps_.load() == 0) {
    return nullptr;
  }
  mutex_lock ml(metadata_mu_);
  return step_container_.get();
}

Status EagerContext::MaybeRegisterFunctionRemotely(const FunctionDef& fdef) {
  if (remote_device_manager_ == nullptr) return Status::OK();
#ifndef __ANDROID__
  BlockingCounter blocking_counter(static_cast<int>(remote_contexts_.size()));

  std::vector<eager::RegisterFunctionRequest> requests(remote_contexts_.size());
  std::vector<eager::RegisterFunctionResponse> responses(
      remote_contexts_.size());
  std::vector<Status> statuses(remote_contexts_.size());

  int i = 0;
  for (const auto& target_and_context_id : remote_contexts_) {
    requests[i].set_context_id(target_and_context_id.second);
    *requests[i].mutable_function_def() = fdef;

    auto* eager_client =
        remote_eager_workers_->GetClient(target_and_context_id.first);

    eager_client->RegisterFunctionAsync(
        &requests[i], &responses[i],
        [i, &statuses, &blocking_counter](const Status& status) {
          statuses[i] = status;
          blocking_counter.DecrementCount();
        });

    i++;
  }
  blocking_counter.Wait();

  for (int i = 0; i < remote_contexts_.size(); i++) {
    TF_RETURN_IF_ERROR(statuses[i]);
  }
#endif
  return Status::OK();
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef) {
  mutex_lock l(functions_mu_);
  TF_RETURN_IF_ERROR(func_lib_def_.AddFunctionDef(fdef));

  return MaybeRegisterFunctionRemotely(fdef);
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

namespace {
Status GetTaskName(Device* d, string* task_name) {
  string ignored;
  if (!DeviceNameUtils::SplitDeviceName(d->name(), task_name, &ignored)) {
    return errors::InvalidArgument("Unable to parse device name: ", d->name());
  }

  return Status::OK();
}
}  // namespace

#ifndef __ANDROID__
Status EagerContext::GetClientAndContextID(Device* device,
                                           eager::EagerClient** client,
                                           uint64* context_id) {
  auto it = device_to_client_cache_.find(device);
  if (it != device_to_client_cache_.end()) {
    *client = it->second.first;
    *context_id = it->second.second;
  }
  string device_task_name;
  TF_RETURN_IF_ERROR(GetTaskName(device, &device_task_name));

  *client = remote_eager_workers_->GetClient(device_task_name);

  if (*client == nullptr) {
    return errors::InvalidArgument(
        "Unable to find eager client corresponding to device ", device->name());
  }

  auto context_iterator = remote_contexts_.find(device_task_name);
  if (context_iterator == remote_contexts_.end()) {
    return errors::Internal("Unable to find a context for handle on task: ",
                            device_task_name, ". This should not be possible");
  }
  *context_id = context_iterator->second;

  device_to_client_cache_.insert({device, {*client, *context_id}});

  return Status::OK();
}

void EagerContext::InitializeRemote(
    std::unique_ptr<ServerInterface> server,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    std::unique_ptr<DeviceMgr> remote_device_manager,
    const gtl::FlatMap<string, uint64>& remote_contexts, Rendezvous* r,
    DeviceMgr* local_device_mgr, int keep_alive_secs) {
  mutex_lock l(remote_state_mu_);

  if (!remote_contexts_.empty()) {
    CloseRemoteContexts();
  }
  remote_contexts_ = remote_contexts;

  use_send_tensor_rpc_ =
      ReadBoolFromEnvVar("TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC", false);

  local_unowned_device_manager_ = local_device_mgr;
  local_device_manager_ = nullptr;
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      local_unowned_device_manager_, env_, TF_GRAPH_DEF_VERSION, &func_lib_def_,
      {}, thread_pool_.get()));

  devices_ = local_unowned_device_manager_->ListDevices();
  devices_map_.clear();

  if (rendezvous_ != nullptr) rendezvous_->Unref();
  rendezvous_ = r;

  // Memory leak!
  if (server_ != nullptr) {
    LOG(WARNING) << "Unable to destroy server_ object, so releasing instead. "
                    "Servers don't support clean shutdown.";
    server_.release();
  }

  server_ = std::move(server);
  remote_eager_workers_ = std::move(remote_eager_workers);

  active_remote_contexts_.clear();
  for (const auto& remote_context : remote_contexts_) {
    active_remote_contexts_.insert(remote_context.second);
  }

  device_to_client_cache_.clear();
  remote_device_manager_ = std::move(remote_device_manager);

  InitDeviceMapAndAsync();

  ClearCaches();

  keep_alive_secs_ = keep_alive_secs;

  sleep_for_secs_ = std::max(1, keep_alive_secs_ / 2);

  // Only schedule a single closure.
  if (keep_alive_thread_ == nullptr) {
    keep_alive_thread_.reset(
        env_->StartThread({}, "EagerKeepAliveThread", [this]() {
          while (true) {
            {
              {
                mutex_lock l(keep_alive_thread_shutdown_mu_);
                keep_alive_thread_cv_.wait_for(
                    l, std::chrono::seconds(sleep_for_secs_));

                if (shutting_down_) {
                  return;
                }
              }
              {
                mutex_lock l(remote_state_mu_);
                if (keep_alive_secs_ > 0) {
                  {
                    for (const auto& worker_and_context_id : remote_contexts_) {
                      auto* client = remote_eager_workers_->GetClient(
                          worker_and_context_id.first);

                      eager::KeepAliveRequest* request =
                          new eager::KeepAliveRequest;
                      eager::KeepAliveResponse* response =
                          new eager::KeepAliveResponse;

                      request->set_context_id(worker_and_context_id.second);
                      client->KeepAliveAsync(
                          request, response,
                          [request, response](const Status& s) {
                            delete request;
                            delete response;
                          });
                    }
                  }
                }
              }
            }
          }
        }));
  }
}
#endif

}  // namespace tensorflow
