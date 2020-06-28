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

#include <memory>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#endif  // !IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
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
    const bool lazy_copy_function_remote_inputs, const DeviceMgr* device_mgr,
    bool device_mgr_owned, Rendezvous* rendezvous,
    const CustomKernelCreator* custom_kernel_creator,
    DistributedFunctionLibraryRuntime* cluster_flr)
    : ImmediateExecutionContext(kEager),
      opts_(opts),
      default_device_placement_policy_(default_device_placement_policy),
      default_mirroring_policy_(default_mirroring_policy),
      local_device_manager_(device_mgr, device_mgr_owned),
      host_cpu_device_(device_mgr->HostCPU()),
      rendezvous_(rendezvous),
      thread_pool_(NewThreadPoolFromSessionOptions(opts)),
      custom_kernel_creator_(custom_kernel_creator),
      cluster_flr_(cluster_flr),
      log_device_placement_(opts.config.log_device_placement()),
      allow_soft_placement_(opts.config.allow_soft_placement()),
      num_active_steps_(0),
      default_executor_(async),
      log_memory_(LogMemory::IsEnabled()),
      env_(opts.env),
      lazy_copy_function_remote_inputs_(lazy_copy_function_remote_inputs),
      use_send_tensor_rpc_(false),
      pin_small_ops_to_cpu_(ReadBoolFromEnvVar(
          "TF_EAGER_ENABLE_SMALL_TENSOR_CPU_PINNING", false)) {
  ResetPFLR(device_mgr, opts.env, &opts.config, TF_GRAPH_DEF_VERSION,
            &func_lib_def_, opts.config.graph_options().optimizer_options(),
            thread_pool_.get(), cluster_flr, custom_kernel_creator_);
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  eager_context_created->GetCell()->Set(true);
  InitPrioritizedDeviceTypeList();
  runner_ = [this](std::function<void()> closure) {
    this->thread_pool_->Schedule(std::move(closure));
  };

#if !defined(IS_MOBILE_PLATFORM)
  context_id_ = kInvalidContextId;
  context_view_id_ = 0;
#endif  // IS_MOBILE_PLATFORM

  std::unique_ptr<DeviceResolverInterface> drl(
      new DeviceResolverLocal(local_device_mgr()));
  std::unique_ptr<ParamResolverInterface> cprl(new CollectiveParamResolverLocal(
      opts.config, local_device_mgr(), drl.get(),
      "/job:localhost/replica:0/task:0"));
  collective_executor_mgr_.Reset(
      new CollectiveExecutorMgr(opts.config, local_device_mgr(), std::move(drl),
                                std::move(cprl)),
      /*owned=*/true);
}

AbstractTensorInterface* EagerContext::CreateInt64Scalar(int64 value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateUint64Scalar(uint64 value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateInt32Scalar(int32 value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateFloatScalar(float value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateDoubleScalar(double value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateHalfScalar(Eigen::half value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateStringScalar(tstring value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateComplex128Scalar(
    complex128 value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateBoolScalar(bool value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateTensor(
    DataType dtype, absl::Span<const int64> dim_sizes) {
  return new TensorInterface(Tensor(dtype, TensorShape(dim_sizes)));
}

AbstractTensorInterface* EagerContext::CreateTensor(
    DataType dtype, const int64_t* dims, int num_dims, void* data, size_t len,
    bool convert_string, MemoryReleaser memory_releaser,
    void* memory_releaser_arg) {
  TF_Tensor* tensor_wrapper =
      TF_NewTensor(static_cast<TF_DataType>(dtype), dims, num_dims, data, len,
                   memory_releaser, memory_releaser_arg);

  if (convert_string) {
    tensorflow::Tensor tensor;
    Status status = TF_TensorToTensor(tensor_wrapper, &tensor);
    TF_DeleteTensor(tensor_wrapper);
    if (!status.ok()) return nullptr;
    return new TensorInterface(std::move(tensor));
  } else {
    AbstractTensorInterface* result = nullptr;
    std::swap(result, tensor_wrapper->tensor);
    TF_DeleteTensor(tensor_wrapper);
    return result;
  }
}

void EagerContext::ResetPFLR(const DeviceMgr* device_mgr, Env* env,
                             const ConfigProto* config, int graph_def_version,
                             const FunctionLibraryDefinition* lib_def,
                             const OptimizerOptions& optimizer_options,
                             thread::ThreadPool* thread_pool,
                             DistributedFunctionLibraryRuntime* cluster_flr,
                             const CustomKernelCreator* custom_kernel_creator) {
  Rendezvous::Factory rendezvous_factory{
      [this](const int64 step_id, const DeviceMgr*, Rendezvous** r) {
        *r = CreateRendezvous(step_id);
        return Status::OK();
      }};
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      device_mgr, env, config, graph_def_version, lib_def, optimizer_options,
      thread_pool, cluster_flr, custom_kernel_creator,
      /*session_metadata=*/nullptr, std::move(rendezvous_factory)));
}

void EagerContext::InitPrioritizedDeviceTypeList() {
  DeviceSet ds;
  for (Device* d : local_device_mgr()->ListDevices()) {
    ds.AddDevice(d);
  }
  auto remote_device_manager = remote_device_mgr();
  if (remote_device_manager != nullptr) {
    for (Device* d : remote_device_manager->ListDevices()) {
      ds.AddDevice(d);
    }
  }
  mutex_lock l(device_type_list_mu_);
  prioritized_device_type_list_ =
      std::make_shared<std::vector<DeviceType>>(ds.PrioritizedDeviceTypeList());
}

namespace {
// Using absl::StrJoin with lambda does not work in tf-lite builds.
// TODO(b/148160441): Replace with absl::StrJoin once DeviceBase has operator<<.
std::vector<string> DevicesToString(const PrioritizedDeviceVector& devices) {
  std::vector<string> v;
  v.reserve(devices.size());
  for (const auto& p : devices) {
    v.push_back(p.first->name());
  }
  return v;
}

std::vector<string> DeviceTypesToString(
    const PrioritizedDeviceTypeVector& types) {
  std::vector<string> v;
  v.reserve(types.size());
  for (const auto& p : types) {
    v.push_back(p.first.type_string());
  }
  return v;
}

// Selects the "best" device that both exists and is supported.
//
// The `existing` argument specifies the available devices in the system, in
// priority order. The `supported` argument specifies the supported device types
// and their priorities, lower index types having higher priority.
// Currently the type priority defined by the `supported` parameter takes
// precedence over system device priorities from `existing`.
//
// TODO(b/148213212): Allow setting default device in eager context.
Device* SelectBestMatchingDevice(const DeviceNameUtils::ParsedName& pattern,
                                 const PrioritizedDeviceVector& existing,
                                 const PrioritizedDeviceTypeVector& supported) {
  for (const std::pair<DeviceType, int32>& prioritized_type : supported) {
    for (const std::pair<Device*, int32>& prioritized_device : existing) {
      Device* dev = prioritized_device.first;
      if (DeviceType(dev->attributes().device_type()) ==
              prioritized_type.first &&
          DeviceNameUtils::IsCompleteSpecification(pattern,
                                                   dev->parsed_name())) {
        return dev;
      }
    }
  }
  return nullptr;
}

}  // namespace

Status EagerContext::SelectDevice(DeviceNameUtils::ParsedName preferred,
                                  const PrioritizedDeviceTypeVector& supported,
                                  const DataType dtype, Device** out) const {
  DCHECK(out != nullptr);

  // We always place string tensors on the CPU device if we're allowed to.
  if (dtype == DT_STRING && AllowSoftPlacement()) {
    preferred = HostCPU()->parsed_name();
  }

  // Select the first matching registered device from the supported device
  // list. If nothing matches and soft placement is enabled, pick a suitable
  // device from the available ones.
  const auto pflr_device_set = pflr()->device_set();
  const PrioritizedDeviceVector& existing =
      pflr_device_set->prioritized_devices();
  *out = SelectBestMatchingDevice(preferred, existing, supported);
  if (*out != nullptr) {
    return Status::OK();
  }

  if (AllowSoftPlacement()) {
    DeviceNameUtils::ParsedName soft_device_name = preferred;
    soft_device_name.type.clear();
    soft_device_name.has_type = false;
    soft_device_name.has_id = false;
    // TODO(b/148213746): Soft placement logic picks up another task if the
    // requested does not exist.
    *out = SelectBestMatchingDevice(soft_device_name, existing, supported);
    if (*out != nullptr) {
      return Status::OK();
    }
  }

  if (DeviceNameUtils::HasSomeDetails(preferred)) {
    return errors::InvalidArgument(
        "Could not satisfy device specification '", preferred,
        "'. enable_soft_placement=", AllowSoftPlacement(),
        ". Supported device types [",
        absl::StrJoin(DeviceTypesToString(supported), ", "),
        "]. All available devices [",
        absl::StrJoin(DevicesToString(existing), ", "), "].");
  }
  return errors::InvalidArgument(
      "No supported device found in available devices [",
      absl::StrJoin(DevicesToString(existing), ", "),
      "]. enable_soft_placement=", AllowSoftPlacement(),
      ". Supported devices types [",
      absl::StrJoin(DeviceTypesToString(supported), ", "), "].");
}

void EagerContext::ResetClusterFLR(
    DistributedFunctionLibraryRuntime* cluster_flr) {
  cluster_flr_.Reset(cluster_flr, lazy_copy_function_remote_inputs_);
}

EagerExecutor& EagerContext::Executor() {
  tf_shared_lock l(executor_map_mu_);
  return *gtl::FindWithDefault(thread_local_executor_,
                               std::this_thread::get_id(), &default_executor_);
}

void EagerContext::SetExecutorForThread(EagerExecutor* executor) {
  tensorflow::mutex_lock l(executor_map_mu_);
  if (executor == &default_executor_) {
    thread_local_executor_.erase(std::this_thread::get_id());
  } else {
    auto thread_id = std::this_thread::get_id();
    thread_local_executor_[thread_id] = executor;
    auto& executors_with_cleanups = has_cleanup_[thread_id];
    if (executors_with_cleanups.find(executor) ==
        executors_with_cleanups.end()) {
      executors_with_cleanups.insert(executor);
      // If the executor is deleted before this context, we need to remove it
      // from the map to avoid attempting to sync it in our destructor.
      std::function<void()> cleanup([this, thread_id, executor]() {
        {
          tensorflow::mutex_lock l(executor_map_mu_);
          auto existing = thread_local_executor_.find(thread_id);
          if (existing != thread_local_executor_.end() &&
              existing->second == executor) {
            thread_local_executor_.erase(thread_id);
          }
          has_cleanup_[thread_id].erase(executor);
        }
      });
      executor->AddCleanup(reinterpret_cast<intptr_t>(this),
                           std::move(cleanup));
    }
  }
}

void EagerContext::ClearCachesAndThreadExecutors() {
  std::unordered_map<std::thread::id, EagerExecutor*> executors_copy;
  {
    mutex_lock l(executor_map_mu_);
    executors_copy = thread_local_executor_;
  }
  for (const auto& entry : executors_copy) {
    entry.second->WaitForAllPendingNodes().IgnoreError();
  }
  ClearCachesAndDefaultExecutor();
}

void EagerContext::ClearCachesAndDefaultExecutor() {
  // The executor stores pointers to kernels, so we need to make sure that no
  // async eager ops are still executing. We lock the cache during this time
  // as well.
  mutex_lock ml(cache_mu_);
  default_executor_.WaitForAllPendingNodes().IgnoreError();
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

bool EagerContext::LazyCopyFunctionRemoteInputs() const {
  return lazy_copy_function_remote_inputs_;
}

#if !defined(IS_MOBILE_PLATFORM)
std::vector<string> EagerContext::GetRemoteContexts() {
  tf_shared_lock l(remote_state_mu_);
  return remote_contexts_;
}

bool EagerContext::IsRemoteContextsEmpty() {
  tf_shared_lock l(remote_state_mu_);
  return remote_contexts_.empty();
}

void EagerContext::CloseAndClearAllRemoteContexts() {
  uint64 context_id;
  uint64 context_view_id;
  std::vector<string> remote_contexts_copy;
  {
    mutex_lock l(remote_state_mu_);
    if (!is_master_) return;
    context_id = context_id_;
    context_view_id = context_view_id_;
    context_id_ = kInvalidContextId;
    // Forget the current view id and reset to the starting value 0.
    context_view_id_ = 0;

    // Make a copy of remote targets to avoid holding the lock when sending
    // close context requests.
    remote_contexts_copy = remote_contexts_;
    remote_contexts_.clear();
  }
  CloseRemoteContexts(remote_contexts_copy, context_id, context_view_id);
}

void EagerContext::CloseRemoteContexts(
    const std::vector<string>& remote_contexts, uint64 context_id,
    uint64 context_view_id) {
  // Close all remote contexts.
  eager::CloseContextRequest request;
  request.set_context_id(context_id);
  request.set_context_view_id(context_view_id);
  // Setting context_id to a new value can avoid us issuing DestroyTensorHandle
  // request to closed remote workers.
  std::vector<eager::CloseContextResponse> responses(remote_contexts.size());
  BlockingCounter counter(static_cast<int>(remote_contexts.size()));

  int i = 0;
  for (const auto& worker : remote_contexts) {
    core::RefCountPtr<eager::EagerClient> client;
    Status s = GetClient(worker, &client);

    client->CloseContextAsync(
        &request, &responses[i],
        [&worker, &counter, context_id](const Status& s) {
          if (!s.ok()) {
            LOG(ERROR) << "Unable to close remote context with ID "
                       << context_id << " for worker: " << worker << " due to "
                       << s.error_message();
          }
          counter.DecrementCount();
        });
    i++;
  }

  counter.Wait();
}

#endif  // !IS_MOBILE_PLATFORM

void EagerContext::WaitForAndCloseRemoteContexts() {
  ClearCachesAndThreadExecutors();

#if !defined(IS_MOBILE_PLATFORM)
  {
    mutex_lock l(keep_alive_thread_shutdown_mu_);
    shutting_down_ = true;
    keep_alive_thread_cv_.notify_all();
  }
  keep_alive_thread_.reset();

  if (!IsRemoteContextsEmpty()) {
    CloseAndClearAllRemoteContexts();
  }

  {
    mutex_lock l(remote_state_mu_);

    default_executor_.ShutDown().IgnoreError();
    std::unordered_map<std::thread::id, EagerExecutor*> executors_copy;
    {
      mutex_lock l(executor_map_mu_);
      executors_copy = thread_local_executor_;
    }
    for (const auto& it : executors_copy) {
      it.second->ShutDown().IgnoreError();
    }

    // This shuts down the completion queue and joins the thread polling it.
    // The thread exits only after the completion queue has been drained of all
    // the events. These events' completion should invoke all remaining RPC
    // callbacks.
    // This also deletes all EagerClient instances. There should not be any
    // references to EagerClients left after all RPCs and async ops have been
    // finished.
    remote_eager_workers_ = nullptr;
  }
#endif  // !IS_MOBILE_PLATFORM
}

EagerContext::~EagerContext() {
  // TODO(iga): Add a separate API method to shutdown EagerContext so that we
  // don't send RPCs and block in destructor.
  WaitForAndCloseRemoteContexts();

  // Custom devices may have obtained references to various context components
  // (executors, thread pool). It's safer to run their destructors early.
  custom_devices_.clear();

  ClearCachesAndThreadExecutors();
  std::unordered_map<std::thread::id, EagerExecutor*> executors_copy;
  {
    mutex_lock l(executor_map_mu_);
    executors_copy = thread_local_executor_;
  }
  for (const auto& entry : executors_copy) {
    // Let the executor know that its cleanup closure is no longer valid.
    entry.second->RemoveCleanups(reinterpret_cast<intptr_t>(this));
  }
  for (auto& entry : registered_functions_) {
    while (!entry.second->Unref()) {
      // remove all references.
    }
  }
  registered_functions_.clear();

#if !defined(IS_MOBILE_PLATFORM)
  if (server_) {
    // TODO(b/136478427): Fix this.
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
  if (!remote_contexts_.empty()) {
    CloseAndClearAllRemoteContexts();
  }
#endif  // !IS_MOBILE_PLATFORM

  if (rendezvous_) {
    rendezvous_->Unref();
  }
  if (resource_deallocator_ != nullptr) {
    resource_deallocator_();
  }
}

bool EagerContext::FindFunctionByName(const string& name) const {
  return func_lib_def_.Find(name) != nullptr;
}

Status EagerContext::FindFunctionOpData(
    const string& name, const tensorflow::OpRegistrationData** op_data) {
  return func_lib_def_.LookUp(name, op_data);
}

const FunctionDef* EagerContext::FindFunctionDef(const string& name) {
  return func_lib_def_.Find(name);
}

void EagerContext::ClearRunMetadata() { run_metadata_.Clear(); }

bool EagerContext::UsesTFRT() { return false; }

void EagerContext::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* devices) {
  local_device_mgr()->ListDeviceAttributes(devices);
  if (remote_device_mgr()) {
    remote_device_mgr()->ListDeviceAttributes(devices);
  }
}

void EagerContext::StartStep() {
  mutex_lock ml(metadata_mu_);
  num_active_steps_++;
  if (step_container_ == nullptr) {
    step_container_.reset(
        new ScopedStepContainer(0, [this](const string& name) {
          auto local_devices = local_device_mgr()->ListDevices();
          for (Device* device : local_devices) {
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
  if (!remote_device_manager_.Owned()) return Status::OK();
#if !defined(IS_MOBILE_PLATFORM)
  std::shared_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(GetContextId());

  eager::RegisterFunctionOp* register_function =
      request->add_queue()->mutable_register_function();
  *register_function->mutable_function_def() = fdef;
  StripDefaultAttributes(
      *OpRegistry::Global(),
      register_function->mutable_function_def()->mutable_node_def());

  auto remote_contexts = GetRemoteContexts();
  for (const auto& target : remote_contexts) {
    core::RefCountPtr<eager::EagerClient> eager_client;
    TF_RETURN_IF_ERROR(GetClient(target, &eager_client));

    eager::EnqueueResponse* response = new eager::EnqueueResponse();
    eager_client->StreamingEnqueueAsync(
        request.get(), response, [request, response](const Status& status) {
          if (!status.ok()) {
            LOG(ERROR) << "Failed to register function remotely due to "
                       << status.error_message()
                       << "\nThis shouldn't happen, please file a bug to "
                          "tensorflow team.";
          }
          delete response;
        });
  }
#endif  // !IS_MOBILE_PLATFORM
  return Status::OK();
}

Status EagerContext::RegisterExistingFunctionsOnRemoteWorkers(
    const std::vector<string>& remote_workers) {
#if !defined(IS_MOBILE_PLATFORM)
  // Register multiple functions on selected remote workers.
  uint64 context_id = GetContextId();
  FunctionDefLibrary function_defs = func_lib_def_.ToProto();
  std::vector<std::shared_ptr<eager::EnqueueRequest>> requests(
      function_defs.function_size());
  for (int i = 0; i < function_defs.function_size(); i++) {
    requests[i] = std::make_shared<eager::EnqueueRequest>();
    requests[i]->set_context_id(context_id);
    eager::RegisterFunctionOp* register_function =
        requests[i]->add_queue()->mutable_register_function();
    *register_function->mutable_function_def() =
        std::move(*function_defs.mutable_function(i));
    StripDefaultAttributes(
        *OpRegistry::Global(),
        register_function->mutable_function_def()->mutable_node_def());
  }

  for (auto& remote_worker : remote_workers) {
    core::RefCountPtr<eager::EagerClient> eager_client;
    Status s = GetClient(remote_worker, &eager_client);
    if (!s.ok()) {
      continue;
    }
    for (int i = 0; i < requests.size(); i++) {
      auto response = std::make_shared<eager::EnqueueResponse>();
      eager_client->StreamingEnqueueAsync(
          requests[i].get(), response.get(),
          [request = requests[i], response](const Status& s) {
            if (!s.ok()) {
              LOG(ERROR) << "Failed to register function remotely due to "
                         << s.error_message()
                         << "\nThis shouldn't happen, please file a bug to "
                            "tensorflow team.";
            }
          });
    }
  }
#endif  // !IS_MOBILE_PLATFORM
  return Status::OK();
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef) {
  return AddFunctionDef(fdef, FunctionDefLibrary(),
                        /* add_to_local_only=*/false);
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef,
                                    const FunctionDefLibrary& library,
                                    const bool add_to_local_only) {
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
    TF_RETURN_IF_ERROR(func_lib_def_.AddFunctionDef(fdef));
    TF_RETURN_IF_ERROR(func_lib_def_.AddLibrary(library));
    if (!add_to_local_only) {
      return MaybeRegisterFunctionRemotely(fdef);
    }
  }
  return Status::OK();
}

const FunctionDef* EagerContext::GetFunctionDef(const string& function_name) {
  return func_lib_def_.Find(function_name);
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
    // TODO(fishx): Remove remote function as well.
    return func_lib_def_.RemoveFunction(func);
  }
  return Status::OK();
}

Status EagerContext::SyncExecutors() {
  StatusGroup sg;
  // Synchronize on context default executor
  sg.Update(default_executor_.WaitForAllPendingNodes());
  default_executor_.ClearError();

  // Synchronize thread local executors on client
  std::unordered_map<std::thread::id, EagerExecutor*> executors_copy;
  {
    mutex_lock l(executor_map_mu_);
    executors_copy = thread_local_executor_;
  }
  for (const auto& entry : executors_copy) {
    sg.Update(entry.second->WaitForAllPendingNodes());
    entry.second->ClearError();
  }

#if !defined(IS_MOBILE_PLATFORM)
  auto remote_contexts = GetRemoteContexts();
  // Synchronize executors on remote workers
  eager::EnqueueRequest request;
  request.set_context_id(GetContextId());
  request.add_queue()->mutable_sync_remote_executor_for_stream();
  BlockingCounter counter(static_cast<int>(remote_contexts.size()));
  std::vector<Status> statuses(remote_contexts.size());

  for (int i = 0; i < remote_contexts.size(); i++) {
    const auto& target = remote_contexts[i];
    core::RefCountPtr<eager::EagerClient> eager_client;
    TF_RETURN_IF_ERROR(GetClient(target, &eager_client));

    eager::EnqueueResponse* response = new eager::EnqueueResponse();
    eager_client->StreamingEnqueueAsync(
        &request, response,
        [response, target, &counter, &s = statuses[i]](const Status& status) {
          s = status;
          delete response;
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (const Status& s : statuses) {
    sg.Update(s);
  }
#endif  // !IS_MOBILE_PLATFORM
  return sg.as_summary_status();
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

bool EagerContext::ShouldStoreGraphs() { return should_store_graphs_.load(); }

void EagerContext::SetShouldStoreGraphs(bool value) {
  mutex_lock ml(metadata_mu_);
  should_store_graphs_.store(value);
  if (!value) {
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

Status EagerContext::FindCompositeDeviceFromName(
    StringPiece device_name, CompositeDevice** device) const {
  tf_shared_lock l(composite_devices_mu_);
  for (const auto& d : composite_devices_) {
    if (d.second->name() == device_name) {
      *device = d.second.get();
      return Status::OK();
    }
  }
  return errors::NotFound("Unknown composite device: ", device_name);
}

Status EagerContext::FindCustomDeviceFromName(const string& device_name,
                                              CustomDevice** dev) const {
  auto dev_it = custom_devices_.find(device_name);
  if (dev_it == custom_devices_.end()) {
    return errors::InvalidArgument(device_name, " unknown device.");
  }
  *dev = dev_it->second.get();
  return Status::OK();
}

Status EagerContext::RegisterCustomDevice(
    const string& device_name, std::unique_ptr<CustomDevice> device) {
  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed) ||
      !parsed.has_job || !parsed.has_replica || !parsed.has_task ||
      !parsed.has_type || !parsed.has_id) {
    return errors::InvalidArgument(
        device_name,
        " could not be parsed as a device name. Use the full "
        "/job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num> "
        "format.");
  }
  Device* existing_physical_device = nullptr;
  if (FindDeviceFromName(device_name.c_str(), &existing_physical_device).ok()) {
    return errors::AlreadyExists(device_name,
                                 " already registered as a physical device.");
  }
  if (!custom_devices_.emplace(device_name, std::move(device)).second) {
    return errors::AlreadyExists(device_name,
                                 " already registered as a custom device.");
  }
  return Status::OK();
}

Status EagerContext::FindOrCreateCompositeDevice(
    const std::vector<string>& underlying_devices, const string& device_name,
    CompositeDevice** composite_device) {
  if (!device_name.empty() &&
      FindCompositeDeviceFromName(device_name, composite_device).ok()) {
    return Status::OK();
  }

  const uint64 hash_key = Fingerprint64(absl::StrJoin(underlying_devices, ","));

  mutex_lock l(composite_devices_mu_);
  auto iter = composite_devices_.find(hash_key);
  if (iter != composite_devices_.end()) {
    *composite_device = iter->second.get();
    return Status::OK();
  }

  Status s;
  std::unique_ptr<CompositeDevice> device;
  if (device_name.empty()) {
    // Create a CompositeDevice on the same task as the host CPU, in order to
    // trigger packed TensorHandle copy from a client to a remote worker.
    device = CompositeDevice::MakeDevice(underlying_devices,
                                         composite_devices_.size(),
                                         HostCPU()->parsed_name(), &s);
  } else {
    device = CompositeDevice::MakeDevice(underlying_devices, device_name, &s);
  }
  TF_RETURN_IF_ERROR(s);
  *composite_device = device.get();
  pflr_->AddCompositeDevice(*composite_device);
  composite_devices_.emplace(hash_key, std::move(device));
  return Status::OK();
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

  return FindDeviceFromName(cpu_device_name.c_str(), cpu_device);
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
Status EagerContext::GetClient(Device* device,
                               core::RefCountPtr<eager::EagerClient>* client) {
  return GetClient(device->parsed_name(), client);
}

Status EagerContext::GetClient(const DeviceNameUtils::ParsedName& device_name,
                               core::RefCountPtr<eager::EagerClient>* client) {
  string device_task_name;
  if (!DeviceNameUtils::GetTaskName(device_name, &device_task_name)) {
    return errors::InvalidArgument(
        "Task is not fully specified in device name: ",
        DeviceNameUtils::ParsedNameToString(device_name));
  }

  {
    tf_shared_lock l(remote_state_mu_);
    if (remote_eager_workers_ == nullptr) {
      return errors::Internal(
          "Haven't set up remote eager worker in this eager context yet.");
    }
    TF_RETURN_IF_ERROR(
        remote_eager_workers_->GetClient(device_task_name, client));

    if (*client == nullptr) {
      return errors::InvalidArgument(
          "Unable to find eager client corresponding to device ",
          DeviceNameUtils::ParsedNameToString(device_name));
    }
    if (std::find(remote_contexts_.begin(), remote_contexts_.end(),
                  device_task_name) == remote_contexts_.end()) {
      return errors::Internal("Unable to find a context for handle on task: ",
                              device_task_name, ". This should not happen.");
    }
  }

  return Status::OK();
}

Status EagerContext::GetClient(const string& remote_task,
                               core::RefCountPtr<eager::EagerClient>* client) {
  {
    tf_shared_lock l(remote_state_mu_);
    if (remote_eager_workers_ == nullptr) {
      return errors::Internal(
          "Haven't set up remote eager worker in this eager context yet.");
    }
    TF_RETURN_IF_ERROR(remote_eager_workers_->GetClient(remote_task, client));
  }

  if (*client == nullptr) {
    return errors::InvalidArgument(
        "Unable to find eager client corresponding to target ", remote_task);
  }
  return Status::OK();
}

uint64 EagerContext::GetContextId() const {
  tf_shared_lock l(remote_state_mu_);
  return context_id_;
}

uint64 EagerContext::GetContextViewId() const {
  tf_shared_lock l(remote_state_mu_);
  return context_view_id_;
}

void EagerContext::IncrementContextViewId() {
  mutex_lock l(remote_state_mu_);
  context_view_id_ += 1;
}

// Set collective ops related state in the context. Passing nullptr to
// `new_server` will reuse the existing GRPC server in context.
Status EagerContext::StoreCollectiveOpsServer(
    std::unique_ptr<ServerInterface> new_server, const DeviceMgr* device_mgr,
    CollectiveExecutorMgrInterface* rpc_collective_executor_mgr) {
  collective_executor_mgr_.Reset(rpc_collective_executor_mgr);

  local_device_manager_.Reset(device_mgr);
  host_cpu_device_ = local_device_manager_.Get()->HostCPU();

  InitPrioritizedDeviceTypeList();
  ClearCachesAndThreadExecutors();
  default_executor_.ClearError();
  {
    tensorflow::mutex_lock l(executor_map_mu_);
    for (auto& entry : thread_local_executor_) {
      entry.second->ClearError();
    }
  }

  const ConfigProto* config = pflr_ ? pflr_->config() : nullptr;
  ResetPFLR(
      local_device_manager_.Get(), env_, /*config=*/config,
      TF_GRAPH_DEF_VERSION, &func_lib_def_,
      /*optimizer_options=*/
      config ? config->graph_options().optimizer_options() : OptimizerOptions(),
      thread_pool_.get());

  if (new_server != nullptr) {
    // Memory leak!
    if (server_ != nullptr) {
      LOG(WARNING) << "Unable to destroy server_ object, so releasing instead. "
                      "Servers don't support clean shutdown.";
      server_.release();
    }
    server_ = std::move(new_server);
  }
  DCHECK(server_ != nullptr);

  return Status::OK();
}

Status EagerContext::SetRemoteDeviceFilters(
    const string& remote_worker, const std::vector<string>& device_filters) {
  // Get fully specified task name for remote worker
  string remote_worker_task_name;
  DeviceNameUtils::ParsedName pw;
  if (!DeviceNameUtils::ParseFullName(remote_worker, &pw)) {
    return tensorflow::errors::InvalidArgument(
        "Remote worker task name is invalid ", remote_worker);
  }
  // Force set a replica as the key in cluster device filters map. I.e., if the
  // remote worker is `/job:worker/task:0` it then becomes
  // `/job:worker/replica:0/task:0`.
  pw.has_replica = true;
  if (!DeviceNameUtils::GetTaskName(pw, &remote_worker_task_name)) {
    return tensorflow::errors::InvalidArgument(
        "Job name and task index must be specified for worker ", remote_worker);
  }

  std::vector<DeviceNameUtils::ParsedName> parsed_filters;
  for (auto& filter : device_filters) {
    DeviceNameUtils::ParsedName parsed_filter;
    if (DeviceNameUtils::ParseFullName(filter, &parsed_filter)) {
      parsed_filters.emplace_back(parsed_filter);
    } else {
      return tensorflow::errors::InvalidArgument("Invalid filter: ", filter);
    }
  }

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Setting device filters for " << remote_worker << ":";
    for (auto& filter : device_filters) {
      VLOG(1) << "  " << filter;
    }
  }
  mutex_lock l(remote_state_mu_);
  cluster_device_filters_.emplace(remote_worker_task_name, parsed_filters);
  return Status::OK();
}

void EagerContext::FilterDevicesForRemoteWorkers(
    const string& remote_worker,
    const protobuf::RepeatedPtrField<DeviceAttributes>& device_attrs,
    std::vector<bool>* filtered_device_mask) {
  filtered_device_mask->resize(device_attrs.size());
  std::fill(filtered_device_mask->begin(), filtered_device_mask->end(), false);

  tf_shared_lock l(remote_state_mu_);
  auto it = cluster_device_filters_.find(remote_worker);
  // If no filters were specified, all devices should be visible to the worker
  if (it == cluster_device_filters_.end() || it->second.empty()) {
    std::fill(filtered_device_mask->begin(), filtered_device_mask->end(), true);
    return;
  }

  const std::vector<DeviceNameUtils::ParsedName>& parsed_filters = it->second;
  DeviceNameUtils::ParsedName parsed_remote_worker;
  DeviceNameUtils::ParseFullName(remote_worker, &parsed_remote_worker);
  for (int i = 0; i < device_attrs.size(); i++) {
    DeviceNameUtils::ParsedName pn;
    DeviceNameUtils::ParseFullName(device_attrs[i].name(), &pn);
    if (DeviceNameUtils::IsSameAddressSpace(parsed_remote_worker, pn)) {
      // If this device is on the remote worker itself, it should be visible
      // regardless of device filters
      filtered_device_mask->at(i) = true;
      continue;
    }
    for (const auto& pf : parsed_filters) {
      if ((!pn.has_job || !pf.has_job || pn.job == pf.job) &&
          (!pn.has_replica || !pf.has_replica || pn.replica == pf.replica) &&
          (!pn.has_task || !pf.has_task || pn.task == pf.task) &&
          (!pn.has_type || !pf.has_type || pn.type == pf.type) &&
          (!pn.has_id || !pf.has_id || pn.id == pf.id)) {
        // Found a match, make it visible, stop processing more device filters
        filtered_device_mask->at(i) = true;
        break;
      }
    }
  }
}

Status EagerContext::InitializeRemoteMaster(
    std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
    std::shared_ptr<WorkerSession> worker_session,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    std::unique_ptr<DynamicDeviceMgr> remote_device_manager,
    const std::vector<string>& remote_contexts, uint64 context_id,
    Rendezvous* r, const DeviceMgr* local_device_mgr, int keep_alive_secs,
    DistributedFunctionLibraryRuntime* cluster_flr,
    std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
        remote_mgr) {
  if (context_id == kInvalidContextId) {
    return errors::InvalidArgument(
        "Failed to initialize remote for master context due to invalid ",
        "context id");
  }

  if (!IsRemoteContextsEmpty()) {
    CloseAndClearAllRemoteContexts();
  }
  {
    mutex_lock l(remote_state_mu_);
    remote_contexts_ = remote_contexts;
  }

  return SetMasterContextState(
      std::move(server), worker_env, std::move(worker_session),
      std::move(remote_eager_workers), std::move(remote_device_manager),
      context_id, 0, r, local_device_mgr, keep_alive_secs, cluster_flr,
      std::move(remote_mgr));
}

Status EagerContext::UpdateRemoteMaster(
    uint64 context_id,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    const std::vector<string>& add_remote_contexts,
    const std::vector<string>& remove_remote_contexts) {
  {
    tf_shared_lock l(remote_state_mu_);
    if (context_id != context_id_) {
      return errors::InvalidArgument(
          "Failed to update remote remote master context due to invalid ",
          "context id. Request id = ", context_id,
          " but current id = ", context_id_);
    }
  }

  if (!remove_remote_contexts.empty()) {
    // N.B. remove_remote_contexts include both removed and replaced workers.
    // In the case where a worker is replaced by one that resolves to the same
    // `hostname:port`, it is safe to close context with the current view id,
    // since the newly created context on the remote worker will be holding
    // a larger view id and ignores this request.
    CloseRemoteContexts(remove_remote_contexts, context_id, GetContextViewId());
    mutex_lock l(remote_state_mu_);
    for (const string& remote_context : remove_remote_contexts) {
      remote_contexts_.erase(
          std::remove(remote_contexts_.begin(), remote_contexts_.end(),
                      remote_context),
          remote_contexts_.end());
    }
  }
  if (!add_remote_contexts.empty()) {
    mutex_lock l(remote_state_mu_);
    remote_contexts_.insert(std::end(remote_contexts_),
                            std::begin(add_remote_contexts),
                            std::end(add_remote_contexts));
  }

  {
    mutex_lock l(remote_state_mu_);
    context_view_id_++;

    remote_eager_workers_ = std::move(remote_eager_workers);
    pflr_->InitializeDeviceSet();
    InitPrioritizedDeviceTypeList();

    default_executor_.ClearError();
    {
      tensorflow::mutex_lock l(executor_map_mu_);
      for (auto& entry : thread_local_executor_) {
        entry.second->ClearError();
      }
    }
  }

  // Register existing functions to the newly added remote workers. Note that
  // this should happen only after updating `remote_contexts_` because new
  // functions might be registered while we update the context. When that
  // happens, this ordering ensures that `MaybeRegisterFunctionRemotely` will
  // register the new functions on all remote workers (including the newly added
  // ones), and `RegisterExistingFunctionsOnRemoteWorkers` will take care of
  // registering existing functions, where duplicate registrations will be
  // ignored by the remote workers.
  TF_RETURN_IF_ERROR(
      RegisterExistingFunctionsOnRemoteWorkers(add_remote_contexts));
  return Status::OK();
}

// Set distributed execution related state in the master context.
Status EagerContext::SetMasterContextState(
    std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
    std::shared_ptr<WorkerSession> worker_session,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    std::unique_ptr<DynamicDeviceMgr> remote_device_manager, uint64 context_id,
    uint64 context_view_id, Rendezvous* r, const DeviceMgr* local_device_mgr,
    int keep_alive_secs, DistributedFunctionLibraryRuntime* cluster_flr,
    std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
        remote_mgr) {
  mutex_lock l(remote_state_mu_);
  is_master_ = true;
  context_id_ = context_id;
  context_view_id_ = context_view_id;

  use_send_tensor_rpc_ =
      ReadBoolFromEnvVar("TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC", true);

  if (local_device_mgr != local_device_manager_.Get()) {
    if (local_device_manager_.Owned()) {
      old_local_device_managers_.push_back(
          std::move(local_device_manager_.owned_object));
    }
    local_device_manager_.Reset(local_device_mgr);
  }
  host_cpu_device_ = local_device_manager_.Get()->HostCPU();

  if (rendezvous_ != nullptr) rendezvous_->Unref();
  rendezvous_ = r;

  // Memory leak!
  if (server_ != nullptr) {
    LOG(WARNING) << "Unable to destroy server_ object, so releasing instead. "
                    "Servers don't support clean shutdown.";
    server_.release();
  }
  server_ = std::move(server);

  remote_mgr_ = std::move(remote_mgr);
  worker_env_ = worker_env;
  worker_session_ = std::move(worker_session);
  remote_eager_workers_ = std::move(remote_eager_workers);

  remote_device_manager_.Reset(std::move(remote_device_manager));
  ResetClusterFLR(cluster_flr);

  InitPrioritizedDeviceTypeList();

  ClearCachesAndThreadExecutors();
  default_executor_.ClearError();
  {
    tensorflow::mutex_lock l(executor_map_mu_);
    for (auto& entry : thread_local_executor_) {
      entry.second->ClearError();
    }
  }
  const auto* config = pflr_->config();
  ResetPFLR(local_device_manager_.Get(), env_, config, TF_GRAPH_DEF_VERSION,
            &func_lib_def_, config->graph_options().optimizer_options(),
            thread_pool_.get(), cluster_flr_.Get(), custom_kernel_creator_);

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
                    for (const auto& worker : remote_contexts_) {
                      core::RefCountPtr<eager::EagerClient> client;
                      Status s =
                          remote_eager_workers_->GetClient(worker, &client);

                      if (!s.ok()) {
                        LOG(WARNING) << "Keep-alive thread was unable to find "
                                        "a client for target "
                                     << worker << ". Got error: " << s;
                        continue;
                      }

                      eager::KeepAliveRequest* request =
                          new eager::KeepAliveRequest;
                      eager::KeepAliveResponse* response =
                          new eager::KeepAliveResponse;

                      request->set_context_id(context_id_);
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

Status EagerContext::InitializeRemoteWorker(
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    DynamicDeviceMgr* remote_device_mgr,
    const std::vector<string>& remote_contexts, uint64 context_id,
    uint64 context_view_id,
    std::function<Rendezvous*(const int64)> rendezvous_creator,
    DistributedFunctionLibraryRuntime* cluster_flr,
    std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
        remote_mgr,
    std::function<void()> resource_deallocator) {
  if (context_id == kInvalidContextId) {
    return errors::InvalidArgument(
        "Failed to initialize remote for worker context due to invalid ",
        "context id");
  }
  mutex_lock l(remote_state_mu_);

  if (remote_device_manager_.Owned() || server_ != nullptr ||
      keep_alive_thread_ != nullptr) {
    return errors::FailedPrecondition(
        "EagerContext::InitializeRemoteWorker Failed. ",
        "Already initialized remote as a master context.");
  }
  is_master_ = false;

  remote_contexts_ = remote_contexts;
  context_id_ = context_id;
  context_view_id_ = context_view_id;

  rendezvous_creator_ = std::move(rendezvous_creator);
  remote_eager_workers_ = std::move(remote_eager_workers);
  remote_mgr_ = std::move(remote_mgr);
  ResetClusterFLR(cluster_flr);

  remote_device_manager_.Reset(remote_device_mgr);

  const auto* config = pflr_->config();
  ResetPFLR(local_device_manager_.Get(), env_, config, TF_GRAPH_DEF_VERSION,
            &func_lib_def_, config->graph_options().optimizer_options(),
            thread_pool_.get(), cluster_flr_.Get(), custom_kernel_creator_);
  InitPrioritizedDeviceTypeList();

  ClearCachesAndThreadExecutors();
  default_executor_.ClearError();
  {
    tensorflow::mutex_lock l(executor_map_mu_);
    for (auto& entry : thread_local_executor_) {
      entry.second->ClearError();
    }
  }

  resource_deallocator_ = std::move(resource_deallocator);

  return Status::OK();
}

Status EagerContext::UpdateRemoteWorker(
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    const std::vector<string>& remote_contexts, uint64 context_id) {
  {
    mutex_lock l(remote_state_mu_);
    if (context_id != context_id_) {
      return errors::InvalidArgument(
          "Failed to update remote for worker context due to invalid ",
          "context id. Request id = ", context_id,
          " but current id = ", context_id_);
    }
    context_view_id_++;

    remote_contexts_ = remote_contexts;
    remote_eager_workers_ = std::move(remote_eager_workers);
    InitPrioritizedDeviceTypeList();
    pflr_->InitializeDeviceSet();
  }

  // No need to update remote_device_manager_ since it's not owned for remote
  // worker context (owned by the corresponding worker session).
  if (remote_device_manager_.Owned()) {
    return errors::FailedPrecondition(
        "EagerContext::UpdateRemoteWorker failed because the context was "
        "initialized as a master context.");
  }

  ClearCachesAndThreadExecutors();
  default_executor_.ClearError();
  {
    tensorflow::mutex_lock l(executor_map_mu_);
    for (auto& entry : thread_local_executor_) {
      entry.second->ClearError();
    }
  }
  return Status::OK();
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace tensorflow
