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

#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#endif  // !IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/monitoring.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

bool ReadBoolFromEnvVar(StringPiece env_var_name, bool default_val) {
  bool val;
  if (tensorflow::ReadBoolFromEnvVar(env_var_name, default_val, &val).ok()) {
    return val;
  }
  return default_val;
}

auto* eager_context_created =
    monitoring::Gauge<bool, 0>::New("/tensorflow/core/eager_context_created",
                                    "True if an eager context was created.");

}  // namespace

EagerContext::EagerContext(
    const SessionOptions& opts,
    ContextDevicePlacementPolicy default_device_placement_policy,
    ContextMirroringPolicy default_mirroring_policy, bool async,
    const DeviceMgr* device_mgr, bool device_mgr_owned, Rendezvous* rendezvous,
    const CustomKernelCreator* custom_kernel_creator,
    DistributedFunctionLibraryRuntime* cluster_flr,
    std::function<Rendezvous*(const int64)> rendezvous_creator,
    const DeviceMgr* remote_device_mgr)
    : default_device_placement_policy_(default_device_placement_policy),
      default_mirroring_policy_(default_mirroring_policy),
      remote_unowned_device_manager_(remote_device_mgr),
      devices_(device_mgr->ListDevices()),
      rendezvous_(rendezvous),
      rendezvous_creator_(std::move(rendezvous_creator)),
      thread_pool_(NewThreadPoolFromSessionOptions(opts)),
      custom_kernel_creator_(custom_kernel_creator),
      pflr_(new ProcessFunctionLibraryRuntime(
          device_mgr, opts.env, TF_GRAPH_DEF_VERSION, &func_lib_def_,
          opts.config.graph_options().optimizer_options(), thread_pool_.get(),
          cluster_flr, custom_kernel_creator_)),
      log_device_placement_(opts.config.log_device_placement()),
      num_active_steps_(0),
      async_default_(async),
      log_memory_(LogMemory::IsEnabled()),
      env_(opts.env),
      use_send_tensor_rpc_(false),
      pin_small_ops_to_cpu_(ReadBoolFromEnvVar(
          "TF_EAGER_ENABLE_SMALL_TENSOR_CPU_PINNING", false)) {
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  eager_context_created->GetCell()->Set(true);
  monitoring::StartExporter();
  if (device_mgr_owned) {
    local_device_manager_.reset(device_mgr);
    local_unowned_device_manager_ = nullptr;
  } else {
    local_unowned_device_manager_ = device_mgr;
  }
  InitDeviceMapAndAsync();
  runner_ = [this](std::function<void()> closure) {
    this->thread_pool_->Schedule(std::move(closure));
  };

  std::unique_ptr<DeviceResolverInterface> drl(
      new DeviceResolverLocal(local_device_mgr()));
  std::unique_ptr<ParamResolverInterface> cprl(new CollectiveParamResolverLocal(
      opts.config, local_device_mgr(), drl.get(),
      "/job:localhost/replica:0/task:0"));
  collective_executor_mgr_.reset(new CollectiveExecutorMgr(
      opts.config, local_device_mgr(), std::move(drl), std::move(cprl)));
}

void EagerContext::InitDeviceMapAndAsync() {
  if (async_default_) {
    executor_.EnableAsync();
  }

  for (auto* device : devices_) {
    devices_map_[device->name()] = device;
  }

  if (remote_device_mgr() != nullptr) {
    for (auto* device : remote_device_mgr()->ListDevices()) {
      if (devices_map_.find(device->name()) == devices_map_.end()) {
        devices_map_[device->name()] = device;
        devices_.push_back(device);
      }
    }
  }

  DeviceSet ds;
  for (Device* d : devices_) {
    ds.AddDevice(d);
  }
  prioritized_device_type_list_ = ds.PrioritizedDeviceTypeList();
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
  // The executor stores pointers to kernels, so we need to make sure that no
  // async eager ops are still executing. We lock the cache during this time as
  // well.
  mutex_lock ml(cache_mu_);
  executor_.WaitForAllPendingNodes().IgnoreError();
  kernel_cache_.clear();
  for (auto& entry : registered_functions_) {
    entry.second->cached_kernel_keys->clear();
  }
}

void EagerContext::SetThreadLocalDevicePlacementPolicy(
    ContextDevicePlacementPolicy policy) {
  mutex_lock ml(policy_map_mu_);
  device_placement_policy_[std::this_thread::get_id()] = policy;
}

ContextDevicePlacementPolicy EagerContext::GetDevicePlacementPolicy() const {
  tf_shared_lock l(policy_map_mu_);
  auto policy_map_it =
      device_placement_policy_.find(std::this_thread::get_id());
  if (policy_map_it != device_placement_policy_.end()) {
    return policy_map_it->second;
  }
  return default_device_placement_policy_;
}

void EagerContext::SetThreadLocalMirroringPolicy(
    ContextMirroringPolicy policy) {
  mutex_lock ml(policy_map_mu_);
  mirroring_policy_[std::this_thread::get_id()] = policy;
}

ContextMirroringPolicy EagerContext::GetMirroringPolicy() const {
  tf_shared_lock l(policy_map_mu_);
  auto policy_map_it = mirroring_policy_.find(std::this_thread::get_id());
  if (policy_map_it != mirroring_policy_.end()) {
    return policy_map_it->second;
  }
  return default_mirroring_policy_;
}

bool EagerContext::MirrorTensors() const {
  return GetMirroringPolicy() == MIRRORING_ALL;
}

#if !defined(IS_MOBILE_PLATFORM)
void EagerContext::CloseRemoteContexts() {
  // Close all remote contexts.
  std::vector<eager::CloseContextRequest> requests(remote_contexts_.size());
  std::vector<eager::CloseContextResponse> responses(remote_contexts_.size());
  BlockingCounter counter(static_cast<int>(remote_contexts_.size()));

  int i = 0;
  for (const auto& worker_and_context_id : remote_contexts_) {
    eager::EagerClient* client;
    Status s =
        remote_eager_workers_->GetClient(worker_and_context_id.first, &client);

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
#endif  // !IS_MOBILE_PLATFORM

EagerContext::~EagerContext() {
#if !defined(IS_MOBILE_PLATFORM)
  ClearCaches();
  for (auto& entry : registered_functions_) {
    while (!entry.second->Unref()) {
      // remove all references.
    }
  }
  registered_functions_.clear();

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
#endif  // !IS_MOBILE_PLATFORM

  executor_.WaitForAllPendingNodes().IgnoreError();
  rendezvous_->Unref();

  for (auto& thread : child_threads_) {
    thread.reset();
  }

  // Release resources ahead of destroying the device manager as the resource
  // destructors (e.g. ~IteratorResource) assume devices still exist.
  for (auto device : local_device_mgr()->ListDevices()) {
    device->ClearResourceMgr();
  }
}

void EagerContext::AddChildThread(std::unique_ptr<Thread> thread) {
  child_threads_.push_back(std::move(thread));
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

// TODO(gjn): Delete in favour of FindDeviceFromName
Status EagerContext::FindDeviceByName(const string& name,
                                      Device** result) const {
  auto it = devices_map_.find(name);
  if (it == devices_map_.end()) {
    return errors::InvalidArgument(name, " unknown device.");
  }
  *result = it->second;
  return Status::OK();
}

void EagerContext::ClearRunMetadata() {
  if (metadata_listener_ != nullptr) {
    metadata_listener_->BeforeClearRunMetadata();
  }
  run_metadata_.Clear();
}

Status EagerContext::RegisterRunMetadataListener(
    RunMetadataListener* listener) {
  mutex_lock l(metadata_mu_);
  if (metadata_listener_ != nullptr) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Cannot run two eager profiler at the same time");
  }
  metadata_listener_ = listener;
  return Status::OK();
}

void EagerContext::ClearRunMetadataListener() {
  mutex_lock l(metadata_mu_);
  metadata_listener_ = nullptr;
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
  // Only client context can register function on remote worker context.
  if (remote_device_manager_ == nullptr) return Status::OK();
#if !defined(IS_MOBILE_PLATFORM)
  BlockingCounter blocking_counter(static_cast<int>(remote_contexts_.size()));

  std::vector<eager::RegisterFunctionRequest> requests(remote_contexts_.size());
  std::vector<eager::RegisterFunctionResponse> responses(
      remote_contexts_.size());
  std::vector<Status> statuses(remote_contexts_.size());

  int i = 0;
  for (const auto& target_and_context_id : remote_contexts_) {
    requests[i].set_context_id(target_and_context_id.second);
    *requests[i].mutable_function_def() = fdef;

    eager::EagerClient* eager_client;
    TF_RETURN_IF_ERROR(remote_eager_workers_->GetClient(
        target_and_context_id.first, &eager_client));

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
#endif  // !IS_MOBILE_PLATFORM
  return Status::OK();
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef) {
  bool is_first_ref = false;
  {
    mutex_lock l(cache_mu_);
    auto* registered_function =
        gtl::FindPtrOrNull(registered_functions_, fdef.signature().name());
    if (registered_function == nullptr) {
      registered_function = new RegisteredFunction;
      registered_function->cached_kernel_keys =
          absl::make_unique<std::vector<Fprint128>>();
      gtl::InsertOrUpdate(&registered_functions_, fdef.signature().name(),
                          registered_function);
    } else {
      registered_function->Ref();
    }
    is_first_ref = registered_function->RefCountIsOne();
  }
  if (is_first_ref) {
    mutex_lock l(functions_mu_);
    TF_RETURN_IF_ERROR(func_lib_def_.AddFunctionDef(fdef));
    // TODO(fishx): Avoid holding lock when sending RPCs.
    return MaybeRegisterFunctionRemotely(fdef);
  }
  return Status::OK();
}

Status EagerContext::RemoveFunction(const string& func) {
  bool is_last_ref = false;
  {
    mutex_lock l(cache_mu_);
    auto* registered_function = gtl::FindPtrOrNull(registered_functions_, func);
    if (registered_function == nullptr) {
      return errors::InvalidArgument("Tried to remove non-existent function '",
                                     func, "'.");
    }
    is_last_ref = registered_function->RefCountIsOne();
    if (is_last_ref) {
      for (auto& key : *registered_function->cached_kernel_keys) {
        kernel_cache_.erase(key);
      }
      registered_functions_.erase(func);
    }
    registered_function->Unref();
  }
  if (is_last_ref) {
    mutex_lock l(functions_mu_);
    // TODO(fishx): Remove remote function as well.
    return func_lib_def_.RemoveFunction(func);
  }
  return Status::OK();
}

core::RefCountPtr<KernelAndDevice> EagerContext::GetCachedKernel(
    Fprint128 cache_key) {
  tf_shared_lock l(cache_mu_);
  auto iter = kernel_cache_.find(cache_key);
  if (iter == kernel_cache_.end()) {
    return nullptr;
  }
  core::RefCountPtr<KernelAndDevice> new_ref(iter->second.get());
  new_ref->Ref();
  return new_ref;
}

void EagerContext::AddKernelToCache(Fprint128 cache_key,
                                    KernelAndDevice* kernel) {
  mutex_lock ml(cache_mu_);
  core::RefCountPtr<KernelAndDevice> new_ref(kernel);
  new_ref->Ref();
  kernel_cache_[cache_key] = std::move(new_ref);
  auto* registered_function =
      gtl::FindPtrOrNull(registered_functions_, kernel->name());
  // The kernel name can be either a primitive op or a function.
  if (registered_function != nullptr) {
    registered_function->cached_kernel_keys->emplace_back(cache_key);
  }
}

bool EagerContext::ShouldStoreGraphs() {
  mutex_lock ml(metadata_mu_);
  return should_store_graphs_.load() || metadata_listener_ != nullptr;
}

bool EagerContext::ShouldStoreStepStats() {
  mutex_lock ml(metadata_mu_);
  return should_store_step_stats_.load() || metadata_listener_ != nullptr;
}

void EagerContext::SetShouldStoreGraphs(bool value) {
  mutex_lock ml(metadata_mu_);
  should_store_graphs_.store(value);
  if (!value || metadata_listener_ != nullptr) {
    run_metadata_.Clear();
  }
}

void EagerContext::SetShouldStoreStepStats(bool value) {
  mutex_lock ml(metadata_mu_);
  should_store_step_stats_.store(value);
  if (!value || metadata_listener_ != nullptr) {
    run_metadata_.Clear();
  }
}

Status EagerContext::FindDeviceFromName(const char* device_name,
                                        Device** device) const {
  *device = HostCPU();
  if (device_name == nullptr || strlen(device_name) == 0) {
    return Status::OK();
  }

  auto status = local_device_mgr()->LookupDevice(device_name, device);
  if (status.ok()) {
    return status;
  }

  if (remote_device_mgr() != nullptr) {
    return remote_device_mgr()->LookupDevice(device_name, device);
  }

  return status;
}

bool EagerContext::IsLocal(const Device* d) const {
  if (d == nullptr || remote_device_mgr() == nullptr) return true;
  tensorflow::Device* tmp;
  return local_device_mgr()->LookupDevice(d->name(), &tmp).ok();
}

bool EagerContext::OnSameTask(const Device* first, const Device* second) const {
  if (first == nullptr) first = HostCPU();
  if (second == nullptr) second = HostCPU();
  return first->parsed_name().job == second->parsed_name().job &&
         first->parsed_name().replica == second->parsed_name().replica &&
         first->parsed_name().task == second->parsed_name().task;
}

// Gets the CPU device on the task of device.
Status EagerContext::CPUDeviceOnTask(const Device* device,
                                     Device** cpu_device) const {
  string cpu_device_name;
  TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
      device->name(), &cpu_device_name));

  return FindDeviceByName(cpu_device_name, cpu_device);
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

#if !defined(IS_MOBILE_PLATFORM)
Status EagerContext::GetClientAndContextID(Device* device,
                                           eager::EagerClient** client,
                                           uint64* context_id) {
  if (remote_eager_workers_ == nullptr) {
    return errors::Internal(
        "Haven't set up remote eager worker in this eager context yet.");
  }
  auto it = device_to_client_cache_.find(device);
  if (it != device_to_client_cache_.end()) {
    *client = it->second.first;
    *context_id = it->second.second;
  }
  string device_task_name;
  TF_RETURN_IF_ERROR(GetTaskName(device, &device_task_name));

  TF_RETURN_IF_ERROR(
      remote_eager_workers_->GetClient(device_task_name, client));

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

Status EagerContext::StoreCollectiveOpsServer(
    std::unique_ptr<ServerInterface> server, DeviceMgr* device_mgr,
    CollectiveExecutorMgrInterface* rpc_collective_executor_mgr) {
  collective_executor_mgr_.reset(nullptr);
  unowned_collective_executor_mgr_ = rpc_collective_executor_mgr;

  local_device_manager_.reset(nullptr);
  local_unowned_device_manager_ = device_mgr;

  devices_ = local_unowned_device_manager_->ListDevices();
  devices_map_.clear();

  InitDeviceMapAndAsync();
  ClearCaches();

  pflr_.reset(new ProcessFunctionLibraryRuntime(
      local_unowned_device_manager_, env_, TF_GRAPH_DEF_VERSION, &func_lib_def_,
      {}, thread_pool_.get()));

  // Memory leak!
  if (server_ != nullptr) {
    LOG(WARNING) << "Unable to destroy server_ object, so releasing instead. "
                    "Servers don't support clean shutdown.";
    server_.release();
  }
  server_ = std::move(server);

  return Status::OK();
}

Status EagerContext::InitializeRemote(
    std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
    std::shared_ptr<WorkerSession> worker_session,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    std::unique_ptr<DeviceMgr> remote_device_manager,
    const gtl::FlatMap<string, uint64>& remote_contexts, Rendezvous* r,
    DeviceMgr* local_device_mgr, int keep_alive_secs,
    DistributedFunctionLibraryRuntime* cluster_flr) {
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
      {}, thread_pool_.get(), cluster_flr, custom_kernel_creator_));

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
  worker_env_ = worker_env;
  worker_session_ = worker_session;
  remote_eager_workers_ = std::move(remote_eager_workers);

  active_remote_contexts_.clear();
  for (const auto& remote_context : remote_contexts_) {
    active_remote_contexts_.insert(remote_context.second);
  }

  device_to_client_cache_.clear();
  remote_device_manager_ = std::move(remote_device_manager);

  InitDeviceMapAndAsync();

  ClearCaches();
  executor_.ClearError();

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

                if (shutting_down_) {
                  return;
                }

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
                      eager::EagerClient* client;
                      Status s = remote_eager_workers_->GetClient(
                          worker_and_context_id.first, &client);

                      if (!s.ok()) {
                        LOG(WARNING) << "Keep-alive thread was unable to find "
                                        "a client for target "
                                     << worker_and_context_id.first
                                     << ". Got error: " << s;
                        continue;
                      }

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
  return Status::OK();
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace tensorflow
