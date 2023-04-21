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

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/common_runtime/eager/rendezvous_cache.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
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
#include "tensorflow/core/common_runtime/eager/small_constants_optimizer.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/tsl/platform/refcount.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#endif  // !IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {
// This object tracks the EagerContext owned by global_py_eager_context in
// pywrap_tfe_src.cc. Since the vast majority of the Python API is dependent on
// that global_py_eager_context (including memory management), the Py object
// owns the C object, so this pointer is non-owning.
EagerContext* global_c_eager_context = nullptr;

}  // namespace

void SetCEagerContext(EagerContext* ctx) { global_c_eager_context = ctx; }

EagerContext* GetCEagerContext() { return global_c_eager_context; }

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

const int64_t EagerContext::kGlobalRendezvousId = -1;

// Find the rendezvous instance corresponding to the step id, or create a
// new instance if not existing.
tsl::core::RefCountPtr<IntraProcessRendezvous>
EagerContext::LocalRendezvousCache::FindOrCreate(int64_t step_id,
                                                 DeviceMgr* device_mgr) {
  return cache_->FindOrCreate(step_id, [&]() {
    return tsl::core::RefCountPtr<IntraProcessRendezvous>(
        new IntraProcessRendezvous(device_mgr));
  });
}

EagerContext::EagerContext(
    const SessionOptions& opts,
    ContextDevicePlacementPolicy default_device_placement_policy, bool async,
    DeviceMgr* device_mgr, bool device_mgr_owned,
    tsl::core::RefCountPtr<Rendezvous> rendezvous,
    DistributedFunctionLibraryRuntime* cluster_flr,
    CollectiveExecutorMgrInterface* collective_executor_mgr,
    bool run_eager_op_as_function, bool jit_compile_rewrite)
    : ImmediateExecutionContext(kEager),
      opts_(opts),
      default_device_placement_policy_(default_device_placement_policy),
      local_device_manager_(device_mgr, device_mgr_owned),
      host_cpu_device_(device_mgr->HostCPU()),
      rendezvous_(std::move(rendezvous)),
      thread_pool_(NewThreadPoolFromSessionOptions(opts)),
      cluster_flr_(cluster_flr),
      log_device_placement_(opts.config.log_device_placement()),
      allow_soft_placement_(opts.config.allow_soft_placement()),
      num_active_steps_(0),
      step_container_(std::make_unique<ScopedStepContainer>(
          0, [this](const string& name) { ClearResourceContainer(name); })),
      default_executor_(async, /*enable_streaming_enqueue=*/true),
      log_memory_(LogMemory::IsEnabled()),
      env_(opts.env),
      collective_executor_mgr_(collective_executor_mgr, /*owned=*/false),
      use_send_tensor_rpc_(false),
      pin_small_ops_to_cpu_(ReadBoolFromEnvVar(
          "TF_EAGER_ENABLE_SMALL_TENSOR_CPU_PINNING", false)),
      run_eager_op_as_function_(run_eager_op_as_function),
      jit_compile_rewrite_(jit_compile_rewrite) {
  ResetPFLR(device_mgr, opts.env, &opts.config, TF_GRAPH_DEF_VERSION,
            &func_lib_def_, opts.config.graph_options().optimizer_options(),
            thread_pool_.get(), cluster_flr);
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/tsl/platform/default", this is
  // currently a no-op.
  eager_context_created->GetCell()->Set(true);
  InitPrioritizedDeviceTypeList();
  runner_ = [this](std::function<void()> closure) {
    this->thread_pool_->Schedule(std::move(closure));
  };

  run_metadata_ = std::make_unique<RunMetadata>();

#if !defined(IS_MOBILE_PLATFORM)
  context_id_ = kInvalidContextId;
  context_view_id_ = 0;
#endif  // IS_MOBILE_PLATFORM

  // TODO(b/278898454): Consider consolidating Local and RPC to a unified API,
  // like CreateCollectiveExecutorMgr.
  if (collective_executor_mgr_.Get() == nullptr) {
    collective_executor_mgr_.Reset(
        CreateProdLocalCollectiveExecutorMgr(opts.config, local_device_mgr()));
  }

  ResetGlobalRendezvousForFunction();
}

AbstractTensorInterface* EagerContext::CreateInt64Scalar(int64_t value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateUint64Scalar(uint64 value) {
  return new TensorInterface(Tensor(value));
}

AbstractTensorInterface* EagerContext::CreateInt32Scalar(int32_t value) {
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
    DataType dtype, absl::Span<const int64_t> dim_sizes) {
  return new TensorInterface(Tensor(dtype, TensorShape(dim_sizes)));
}

AbstractTensorInterface* EagerContext::CreateTensor(
    DataType dtype, const int64_t* dims, int num_dims, void* data, size_t len,
    MemoryReleaser memory_releaser, void* memory_releaser_arg) {
  TF_Tensor* tensor_wrapper =
      TF_NewTensor(static_cast<TF_DataType>(dtype), dims, num_dims, data, len,
                   memory_releaser, memory_releaser_arg);

  AbstractTensorInterface* result = nullptr;
  std::swap(result, tensor_wrapper->tensor);
  TF_DeleteTensor(tensor_wrapper);
  return result;
}

void EagerContext::ResetPFLR(const DeviceMgr* device_mgr, Env* env,
                             const ConfigProto* config, int graph_def_version,
                             const FunctionLibraryDefinition* lib_def,
                             const OptimizerOptions& optimizer_options,
                             thread::ThreadPool* thread_pool,
                             DistributedFunctionLibraryRuntime* cluster_flr) {
  Rendezvous::Factory rendezvous_factory = CreateRendezvousFactory();
  const tensorflow::SessionMetadata* session_metadata = nullptr;
  if (opts_.config.experimental().has_session_metadata()) {
    session_metadata = &opts_.config.experimental().session_metadata();
  }
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      device_mgr, env, config, graph_def_version, lib_def, optimizer_options,
      thread_pool, cluster_flr, session_metadata, std::move(rendezvous_factory),
      StatsPublisherInterface::GetStatsPublisherFactory()));
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
                                  const NodeDef& ndef, Device** out) const {
  DCHECK(out != nullptr);

  PrioritizedDeviceTypeVector supported_devs;
  auto device_type_list = prioritized_device_type_list();
  TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
      *device_type_list, ndef, &supported_devs, &HostCPU()->parsed_name()));
  if (supported_devs.empty()) {
    return errors::NotFound("Could not find device for node: ",
                            errors::FormatNodeNameForError(ndef.name()), " = ",
                            ndef.op(), "[", SummarizeAttrs(ndef), "]",
                            "\nAll kernels registered for op ", ndef.op(),
                            ":\n", KernelsRegisteredForOp(ndef.op()));
  }

  // Select the first matching registered device from the supported device
  // list. If nothing matches and soft placement is enabled, pick a suitable
  // device from the available ones.
  const auto pflr_device_set = pflr()->device_set();
  const PrioritizedDeviceVector& existing =
      pflr_device_set->prioritized_devices();
  *out = SelectBestMatchingDevice(preferred, existing, supported_devs);
  if (*out != nullptr) {
    return OkStatus();
  }

  if (AllowSoftPlacement()) {
    DeviceNameUtils::ParsedName soft_device_name = preferred;
    soft_device_name.type.clear();
    soft_device_name.has_type = false;
    soft_device_name.has_id = false;
    // TODO(b/148213746): Soft placement logic picks up another task if the
    // requested does not exist.
    *out = SelectBestMatchingDevice(soft_device_name, existing, supported_devs);
    if (*out != nullptr) {
      return OkStatus();
    }
  }

  if (DeviceNameUtils::HasSomeDetails(preferred)) {
    return errors::InvalidArgument(
        "Could not satisfy device specification '", preferred,
        "'. enable_soft_placement=", AllowSoftPlacement(),
        ". Supported device types [",
        absl::StrJoin(DeviceTypesToString(supported_devs), ", "),
        "]. All available devices [",
        absl::StrJoin(DevicesToString(existing), ", "), "].");
  }
  return errors::InvalidArgument(
      "No supported device found in available devices [",
      absl::StrJoin(DevicesToString(existing), ", "),
      "]. enable_soft_placement=", AllowSoftPlacement(),
      ". Supported devices types [",
      absl::StrJoin(DeviceTypesToString(supported_devs), ", "), "].");
}

void EagerContext::ResetClusterFLR(
    DistributedFunctionLibraryRuntime* cluster_flr) {
  cluster_flr_.Reset(cluster_flr, /*owned=*/true);
}

void EagerContext::UpdateClusterFLRAndInitDevices(
    DistributedFunctionLibraryRuntime* cluster_flr) {
  ResetClusterFLR(cluster_flr);

  const ConfigProto* config = pflr_ ? pflr_->config() : nullptr;
  ResetPFLR(
      local_device_manager_.Get(), env_, /*config=*/config,
      TF_GRAPH_DEF_VERSION, &func_lib_def_,
      /*optimizer_options=*/
      config ? config->graph_options().optimizer_options() : OptimizerOptions(),
      thread_pool_.get(), cluster_flr_.Get());
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
          // Clears the global rendezvous after cleaning up the executor. This
          // is needed when running in eager op as function mode because it
          // re-uses the EagerContext's global_rendezvous_for_functions. The
          // global rendezvous can end up in a bad state if any op ends in a
          // bad state after execution.
          if (!GetGlobalRendezvousForFunctionLocalRendezvousStatus().ok()) {
            VLOG(6) << "global_rendezvous_for_functions_ is in bad state. "
                       "Resetting.";
            ResetGlobalRendezvousForFunction();
          }
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
  {
    // The executor stores pointers to kernels, so we need to make sure that no
    // async eager ops are still pending to be executed. We lock the cache
    // during this time as well.
    mutex_lock ml(cache_mu_);
    default_executor_.WaitForAllPendingNodes().IgnoreError();
    kernel_cache_.clear();
    for (auto& entry : registered_functions_) {
      entry.second->cached_kernel_keys->clear();
    }
  }
  {
    mutex_lock dl(device_cache_mu_);
    device_cache_.clear();
  }
  {
    mutex_lock ml(metadata_mu_);
    step_container_ = std::make_unique<ScopedStepContainer>(
        0, [this](const string& name) { ClearResourceContainer(name); });
  }
}

void EagerContext::SetThreadLocalDevicePlacementPolicy(
    ContextDevicePlacementPolicy policy) {
  mutex_lock ml(policy_map_mu_);
  VLOG(6) << "Setting device placement policy to: " << policy;
  device_placement_policy_[std::this_thread::get_id()] = policy;
}

ContextDevicePlacementPolicy EagerContext::GetDevicePlacementPolicy() const {
  tf_shared_lock l(policy_map_mu_);
  auto policy_map_it =
      device_placement_policy_.find(std::this_thread::get_id());
  if (policy_map_it != device_placement_policy_.end()) {
    VLOG(6) << "ContextDevicePlacementPolicy: " << policy_map_it->second;
    return policy_map_it->second;
  }
  VLOG(6) << "ContextDevicePlacementPolicy not found; returning default.";
  return default_device_placement_policy_;
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
                       << s.message();
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
  custom_device_op_handler_.Clear();

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
    // TODO(hanyangtay): Remove this teardown logic once gRPC server clean
    // shutdown is supported.
    if (server_->worker_env()->session_mgr != nullptr) {
      // Tear down coordination service.
      Status s = server_->StopCoordinationService();
      if (!s.ok()) {
        LOG(ERROR) << "Failed to stop coordination service: " << s;
      }
    }
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

  // Clean up all the rendezvous instances created via EagerContext.
  // Currently there are 3 cases in which a rendezvous instances is created:
  // (1). Created through rendezvous_mgr.
  // (2). Created within EagerContext using LocalRendezvousCache.
  //
  // Currently case-(2) is taken care of automatically when an EagerContext
  // instance is deleted. The following code takes care of case-(1).
  if (worker_env_ != nullptr && worker_env_->rendezvous_mgr != nullptr) {
    worker_env_->rendezvous_mgr->CleanupAll();
  }
#endif  // !IS_MOBILE_PLATFORM

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

const FunctionDef* EagerContext::FindFunctionDef(const string& name) const {
  return func_lib_def_.Find(name);
}

core::RefCountPtr<FunctionRecord> EagerContext::FindRecord(
    const string& name) const {
  return func_lib_def_.FindRecord(name);
}

std::unique_ptr<RunMetadata> EagerContext::ExportRunMetadata() {
  mutex_lock ml(metadata_mu_);
  auto result = std::make_unique<RunMetadata>();
  run_metadata_.swap(result);
  return result;
}

ImmediateExecutionTensorHandle* EagerContext::TFTensorHandleFromInterface(
    ImmediateExecutionTensorHandle* handle) {
  return handle;
}

Status EagerContext::RegisterFunction(AbstractFunction* f) {
  FunctionDef* fdef;
  TF_RETURN_IF_ERROR(f->GetFunctionDef(&fdef));
  if (!fdef) {
    return errors::InvalidArgument("GetFunctionDef returned nullptr.");
  }
  return AddFunctionDef(*fdef);
}

bool EagerContext::UsesTFRT() { return false; }

bool EagerContext::RunEagerOpAsFunction() const {
  VLOG(3) << "RunEagerOpAsFunction: " << run_eager_op_as_function_;
  return run_eager_op_as_function_;
}

void EagerContext::SetRunEagerOpAsFunction(bool enable) {
  run_eager_op_as_function_ = enable;
}

bool EagerContext::JitCompileRewrite() const {
  VLOG(3) << "JitCompileRewrite: " << jit_compile_rewrite_;
  return jit_compile_rewrite_;
}

void EagerContext::SetJitCompileRewrite(bool enable) {
  jit_compile_rewrite_ = enable;
}

void EagerContext::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* device_attributes) {
  std::vector<Device*> devices = ListAllTfDevices();
  device_attributes->reserve(devices.size());
  for (const auto& dev : devices) {
    device_attributes->emplace_back(dev->attributes());
  }
}

std::vector<Device*> EagerContext::ListAllTfDevices() {
  // Since remote_device_mgr may also contain local devices, make sure no
  // duplicated device is returned.
  std::vector<Device*> devices;
  std::unordered_set<string> dev_names;

  if (local_device_mgr()) {
    for (const auto& dev : local_device_mgr()->ListDevices()) {
      devices.emplace_back(dev);
      dev_names.emplace(dev->attributes().name());
    }
  }

  // TODO (b/197281777): Include local devices in remote_device_mgr on the
  // client-side in single-client deployment.
  if (remote_device_mgr()) {
    for (const auto& dev : remote_device_mgr()->ListDevices()) {
      Device* device = nullptr;
      if (local_device_mgr()->LookupDevice(dev->name(), &device) !=
          OkStatus()) {
        // Include this device from remote_device_mgr only if it does not exist
        // in local_device_mgr.
        devices.emplace_back(dev);
      }
    }
  }

  return devices;
}

Status EagerContext::AddDevices(std::vector<std::unique_ptr<Device>> devices) {
  std::vector<std::unique_ptr<Device>> local_devices, remote_devices;
  while (!devices.empty()) {
    if (devices.front()->IsLocal()) {
      local_devices.push_back(std::move(devices.front()));
    } else {
      remote_devices.push_back(std::move(devices.front()));
    }
    devices.erase(devices.begin());
  }
  TF_RETURN_IF_ERROR(
      reinterpret_cast<DynamicDeviceMgr*>(local_device_manager_.Get())
          ->AddDevices(std::move(local_devices)));

  if (!remote_devices.empty()) {
    if (!remote_device_mgr()) {
      remote_device_manager_.Reset(
          std::make_unique<tensorflow::DynamicDeviceMgr>());
    }
    TF_RETURN_IF_ERROR(
        reinterpret_cast<DynamicDeviceMgr*>(remote_device_manager_.Get())
            ->AddDevices(std::move(remote_devices)));
  }

  // Add the devices to pflr's device set.
  pflr_->InitializeDeviceAndFlr();
  InitPrioritizedDeviceTypeList();
  return OkStatus();
}

void EagerContext::StartStep() {
  mutex_lock ml(metadata_mu_);
  num_active_steps_++;
}

void EagerContext::EndStep() {
  mutex_lock ml(metadata_mu_);
  num_active_steps_--;
  if (num_active_steps_ == 0) {
    // TODO(b/139809335): This does not properly clean up remote resources
    // Clean up the previous step container and create a new one.
    step_container_.reset(new ScopedStepContainer(
        0, [this](const string& name) { ClearResourceContainer(name); }));
  }
}

ScopedStepContainer* EagerContext::StepContainer() {
  mutex_lock ml(metadata_mu_);
  return step_container_.get();
}

Status EagerContext::MaybeRegisterFunctionRemotely(const FunctionDef& fdef) {
  // Only client context can register function on remote worker context.
  if (!remote_device_manager_.Owned()) return OkStatus();
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
        this->Executor().StreamingEnqueue(),
        /*call_opts=*/nullptr, request.get(), response,
        [request, response](const Status& status) {
          if (!status.ok()) {
            LOG(ERROR) << "Failed to register function remotely due to "
                       << status.message()
                       << "\nThis could happen if the remote target has been "
                          "disconnected from the client.";
          }
          delete response;
        });
  }
#endif  // !IS_MOBILE_PLATFORM
  return OkStatus();
}

Status EagerContext::MaybeRemoveFunctionRemotely(const string& function_name) {
  // Only client context can remove function on remote worker context.
  if (!remote_device_manager_.Owned()) {
    return OkStatus();
  }

#if !defined(IS_MOBILE_PLATFORM)
  auto request = std::make_shared<eager::EnqueueRequest>();
  request->set_context_id(GetContextId());

  eager::RemoveFunctionOp* remove_function =
      request->add_queue()->mutable_remove_function();
  *remove_function->mutable_function_name() = function_name;

  auto remote_contexts = GetRemoteContexts();
  for (const auto& target : remote_contexts) {
    core::RefCountPtr<eager::EagerClient> eager_client;
    TF_RETURN_IF_ERROR(GetClient(target, &eager_client));

    auto response = std::make_shared<eager::EnqueueResponse>();
    eager_client->StreamingEnqueueAsync(
        this->Executor().StreamingEnqueue(),
        /*call_opts=*/nullptr, request.get(), response.get(),
        [request, response](const Status& status) {
          if (!status.ok()) {
            LOG(ERROR) << "Failed to remove function remotely due to "
                       << status.message()
                       << "\nThis could happen if the remote target has been "
                          "disconnected from the client.";
          }
        });
  }
#endif  // !IS_MOBILE_PLATFORM
  return OkStatus();
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
          this->Executor().StreamingEnqueue(),
          /*call_opts=*/nullptr, requests[i].get(), response.get(),
          [request = requests[i], response](const Status& s) {
            if (!s.ok()) {
              LOG(ERROR) << "Failed to register function remotely due to "
                         << s.message()
                         << "\nThis could happen if the remote target has been "
                            "disconnected from the client.";
            }
          });
    }
  }
#endif  // !IS_MOBILE_PLATFORM
  return OkStatus();
}

Status EagerContext::AddFunctionDefWithStackTraces(
    const FunctionDef& fdef, const StackTracesMap& stack_traces) {
  return AddFunctionDef(fdef, FunctionDefLibrary(),
                        /* add_to_local_only=*/false, stack_traces);
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef) {
  return AddFunctionDef(fdef, FunctionDefLibrary(),
                        /* add_to_local_only=*/false);
}

Status EagerContext::AddFunctionDef(const FunctionDef& fdef,
                                    const FunctionDefLibrary& library,
                                    const bool add_to_local_only,
                                    const StackTracesMap& stack_traces) {
  auto fdefs_to_add =
      small_constants_optimizer::FoldInputTensors(fdef, func_lib_def_);
  for (const auto& fdef_to_add : fdefs_to_add) {
    TF_RETURN_IF_ERROR(
        AddFunctionDef(fdef_to_add, library, add_to_local_only, stack_traces));
  }

  bool is_first_ref = false;
  {
    mutex_lock l(cache_mu_);
    auto* registered_function =
        gtl::FindPtrOrNull(registered_functions_, fdef.signature().name());
    if (registered_function == nullptr) {
      registered_function = new RegisteredFunction;
      registered_function->cached_kernel_keys =
          std::make_unique<std::vector<Fprint128>>();
      gtl::InsertOrUpdate(&registered_functions_, fdef.signature().name(),
                          registered_function);
    } else {
      // The function has been registered before. If the function is the same,
      // then we take a Ref() otherwise we error out.
      const FunctionDef* prev_fdef =
          func_lib_def_.Find(fdef.signature().name());
      if (prev_fdef == nullptr) {
        return errors::Internal("Function: ", fdef.signature().name(),
                                " is in the cache but not in the library");
      }
      if (!FunctionDefsEqual(fdef, *prev_fdef)) {
        return errors::InvalidArgument(
            "Attempting to add a duplicate function with name: ",
            fdef.signature().name(), " where the previous and current ",
            "definitions differ. Previous definition: ",
            prev_fdef->DebugString(),
            " and current definition: ", fdef.DebugString());
      }
      registered_function->Ref();
    }
    is_first_ref = registered_function->RefCountIsOne();
    if (is_first_ref) {
      TF_RETURN_IF_ERROR(func_lib_def_.AddFunctionDef(fdef, stack_traces));
      TF_RETURN_IF_ERROR(func_lib_def_.AddLibrary(library));
    }
  }
  if (is_first_ref && !add_to_local_only) {
    return MaybeRegisterFunctionRemotely(fdef);
  }
  return OkStatus();
}

const FunctionDef* EagerContext::GetFunctionDef(const string& function_name) {
  return func_lib_def_.Find(function_name);
}

std::vector<string> EagerContext::ListFunctionNames() {
  return func_lib_def_.ListFunctionNames();
}

Status EagerContext::AddRemoveFunctionNotifier(const string& func,
                                               std::function<void()> notifier) {
  mutex_lock l(remove_function_notifiers_mu_);
  auto iter = remove_function_notifiers_.find(func);
  if (iter != remove_function_notifiers_.end()) {
    iter->second.push_back(notifier);
  } else {
    std::vector<std::function<void()>> notifiers = {notifier};
    remove_function_notifiers_.insert({func, notifiers});
  }
  return OkStatus();
}

tensorflow::ImmediateExecutionContext::CacheStats
EagerContext::GetCacheStats() {
  CacheStats stats;
  {
    mutex_lock l(cache_mu_);
    stats.kernel_cache_size = kernel_cache_.size();
    for (const auto& iter : registered_functions_) {
      stats.func_kernel_cache_entries[iter.first] =
          iter.second->cached_kernel_keys->size();
    }
  }
  {
    mutex_lock dl(device_cache_mu_);
    stats.device_cache_size = device_cache_.size();
  }
  {
    stats.local_rendezvous_cache_active_size =
        local_rendezvous_cache_.GetActiveStepIds().size();
  }
  return stats;
}

Status EagerContext::RemoveFunction(const string& func) {
  // TODO(mdan): The context owns these functions. Why check refcount then?
  std::vector<std::function<void()>> notifiers;
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
    if (is_last_ref) {
      TF_RETURN_IF_ERROR(func_lib_def_.RemoveFunction(func));

      mutex_lock l(remove_function_notifiers_mu_);
      auto iter = remove_function_notifiers_.find(func);
      if (iter != remove_function_notifiers_.end()) {
        notifiers = std::move(iter->second);
      }
      remove_function_notifiers_.erase(func);
    }
  }
  // MaybeRemoveFunctionRemotely contains rpc calls. Including it to mutex lock
  // will cause error.
  if (is_last_ref) {
    for (const auto& notifier : notifiers) {
      notifier();
    }
    return MaybeRemoveFunctionRemotely(func);
  }
  return OkStatus();
}

Status EagerContext::SyncExecutors() {
  VLOG(6) << "Calling SyncExecutors";
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
        this->Executor().StreamingEnqueue(),
        /*call_opts=*/nullptr, &request, response,
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

  // Reset the global rendezvous, which otherwise stores a failure state.
  ResetGlobalRendezvousForFunction();

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

Device* EagerContext::GetCachedDevice(Fprint128 device_cache_key) {
  tf_shared_lock l(device_cache_mu_);
  auto iter = device_cache_.find(device_cache_key);
  if (iter == device_cache_.end()) return nullptr;
  return iter->second;
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
    VLOG(5) << "Cached key size of kernel " << kernel->name()
            << " is: " << registered_function->cached_kernel_keys->size();
  }
}

void EagerContext::AddDeviceToCache(Fprint128 device_cache_key,
                                    Device* device) {
  mutex_lock l(device_cache_mu_);
  device_cache_[device_cache_key] = device;
}

bool EagerContext::ShouldStoreGraphs() { return should_store_graphs_.load(); }

void EagerContext::SetShouldStoreGraphs(bool value) {
  mutex_lock ml(metadata_mu_);
  should_store_graphs_.store(value);
  if (!value) {
    run_metadata_.reset(new RunMetadata);
  }
}

Status EagerContext::FindDeviceFromName(const char* device_name,
                                        Device** device) const {
  *device = HostCPU();
  if (device_name == nullptr || strlen(device_name) == 0) {
    return OkStatus();
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
      return OkStatus();
    }
  }
  return errors::NotFound("Unknown composite device: ", device_name);
}

bool EagerContext::IsCustomDevice(const string& device_name) {
  CustomDevice* device = nullptr;
  return custom_device_op_handler_.FindCustomDeviceFromName(device_name,
                                                            &device);
}

Status EagerContext::RegisterCustomDevice(
    const string& device_name, std::unique_ptr<CustomDevice> device) {
  Device* existing_physical_device = nullptr;
  if (FindDeviceFromName(device_name.c_str(), &existing_physical_device).ok()) {
    return errors::AlreadyExists(device_name,
                                 " already registered as a physical device.");
  }
  return custom_device_op_handler_.RegisterCustomDevice(device_name,
                                                        std::move(device));
}

Status EagerContext::FindOrCreateCompositeDevice(
    const std::vector<string>& underlying_devices, const string& device_name,
    CompositeDevice** composite_device) {
  if (!device_name.empty() &&
      FindCompositeDeviceFromName(device_name, composite_device).ok()) {
    return OkStatus();
  }

  const uint64 hash_key = Fingerprint64(absl::StrJoin(underlying_devices, ","));

  mutex_lock l(composite_devices_mu_);
  auto iter = composite_devices_.find(hash_key);
  if (iter != composite_devices_.end()) {
    *composite_device = iter->second.get();
    return OkStatus();
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
  return OkStatus();
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

void EagerContext::ClearResourceContainer(const string& name) {
  // TODO(b/139809335): This does not properly clean up remote resources
  auto local_devices = local_device_mgr()->ListDevices();
  for (Device* device : local_devices) {
    // Only ignore container not found errors.
    device->resource_manager()->Cleanup(name).IgnoreError();
  }
}

Status EagerContext::GetGlobalRendezvousForFunctionLocalRendezvousStatus() {
  mutex_lock l(global_rendezvous_mu_);
  tsl::core::RefCountPtr<IntraProcessRendezvous> rendezvous =
      local_rendezvous_cache_.Find(kGlobalRendezvousId);
  if (rendezvous == nullptr) return OkStatus();
  return rendezvous->GetLocalRendezvousStatus();
}

void EagerContext::UpdateGlobalRendezvousDeviceManager(
    tensorflow::DeviceMgr* device_mgr) {
  mutex_lock l(global_rendezvous_mu_);
  tsl::core::RefCountPtr<IntraProcessRendezvous> rendezvous =
      local_rendezvous_cache_.Find(kGlobalRendezvousId);
  if (rendezvous == nullptr) return;
  rendezvous->UpdateDeviceManager(device_mgr);
}

namespace {
Status GetTaskName(Device* d, string* task_name) {
  string ignored;
  if (!DeviceNameUtils::SplitDeviceName(d->name(), task_name, &ignored)) {
    return errors::InvalidArgument("Unable to parse device name: ", d->name());
  }

  return OkStatus();
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

  return OkStatus();
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
  return OkStatus();
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

Status EagerContext::EnableCollectiveOps(const ServerDef& server_def) {
  return distributed_manager_->EnableCollectiveOps(server_def);
}

// Set collective ops related state in the context. Passing nullptr to
// `new_server` will reuse the existing GRPC server in context.
Status EagerContext::StoreCollectiveOpsServer(
    std::unique_ptr<ServerInterface> new_server, DeviceMgr* device_mgr,
    CollectiveExecutorMgrInterface* rpc_collective_executor_mgr) {
  collective_executor_mgr_.Reset(rpc_collective_executor_mgr);

  if (device_mgr != local_device_manager_.Get()) {
    if (local_device_manager_.Owned()) {
      old_local_device_managers_.push_back(
          std::move(local_device_manager_.owned_object));
    }
    local_device_manager_.Reset(device_mgr);
    UpdateGlobalRendezvousDeviceManager(local_device_manager_.Get());
    TF_RETURN_IF_ERROR(RendezvousFactory()(-1, nullptr, &rendezvous_));
  }
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

  return OkStatus();
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
  return OkStatus();
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

void EagerContext::SetWorkerEnv(const WorkerEnv* worker_env,
                                std::shared_ptr<WorkerSession> worker_session) {
  worker_env_ = worker_env;
  worker_session_ = worker_session;
}

Status EagerContext::InitializeRemoteMaster(
    std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
    std::shared_ptr<WorkerSession> worker_session,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    std::unique_ptr<DynamicDeviceMgr> remote_device_manager,
    const std::vector<string>& remote_contexts, uint64 context_id,
    tsl::core::RefCountPtr<Rendezvous> r, DeviceMgr* local_device_mgr,
    int keep_alive_secs, DistributedFunctionLibraryRuntime* cluster_flr,
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
      context_id, 0, std::move(r), local_device_mgr, keep_alive_secs,
      cluster_flr, std::move(remote_mgr));
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
          "Failed to update remote master context due to invalid context id. ",
          "Request id = ", context_id, " but current id = ", context_id_);
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
    pflr_->InitializeDeviceAndFlr();
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
  return OkStatus();
}

// Set distributed execution related state in the master context.
Status EagerContext::SetMasterContextState(
    std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
    std::shared_ptr<WorkerSession> worker_session,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    std::unique_ptr<DynamicDeviceMgr> remote_device_manager, uint64 context_id,
    uint64 context_view_id, tsl::core::RefCountPtr<Rendezvous> r,
    DeviceMgr* local_device_mgr, int keep_alive_secs,
    DistributedFunctionLibraryRuntime* cluster_flr,
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
    UpdateGlobalRendezvousDeviceManager(local_device_manager_.Get());
  }
  host_cpu_device_ = local_device_manager_.Get()->HostCPU();

  rendezvous_ = std::move(r);

  // Memory leak!
  if (server_ != nullptr) {
    LOG(WARNING) << "Unable to destroy server_ object, so releasing instead. "
                    "Servers don't support clean shutdown.";
    server_.release();
  }
  server_ = std::move(server);

  remote_mgr_ = std::move(remote_mgr);

  SetWorkerEnv(worker_env, std::move(worker_session));

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
            thread_pool_.get(), cluster_flr_.Get());

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
  return OkStatus();
}

Status EagerContext::InitializeRemoteWorker(
    const WorkerEnv* worker_env, std::shared_ptr<WorkerSession> worker_session,
    std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
    DynamicDeviceMgr* remote_device_mgr,
    const std::vector<string>& remote_contexts, uint64 context_id,
    uint64 context_view_id, DistributedFunctionLibraryRuntime* cluster_flr,
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

  remote_eager_workers_ = std::move(remote_eager_workers);
  remote_mgr_ = std::move(remote_mgr);

  SetWorkerEnv(worker_env, std::move(worker_session));

  ResetClusterFLR(cluster_flr);

  remote_device_manager_.Reset(remote_device_mgr);

  const auto* config = pflr_->config();
  ResetPFLR(local_device_manager_.Get(), env_, config, TF_GRAPH_DEF_VERSION,
            &func_lib_def_, config->graph_options().optimizer_options(),
            thread_pool_.get(), cluster_flr_.Get());
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

  return OkStatus();
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
    pflr_->InitializeDeviceAndFlr();
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
  return OkStatus();
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace tensorflow
