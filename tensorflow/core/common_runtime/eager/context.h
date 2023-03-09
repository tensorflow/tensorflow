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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_H_

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/common_runtime/eager/custom_device_op_handler.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

// "tensorflow/core/platform/platform.h" must be included first before using
// IS_MOBILE_PLATFORM.
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {

namespace eager {
// We need this forward declaration because we have circular dependency:
// Context -> RemoteMgr -> TensorHandle -> Context.
// TODO(fishx): Remove this once we remove Context dependency in TensorHandle.
class RemoteMgr;
}  // namespace eager

class EagerContext : public ImmediateExecutionContext, public core::RefCounted {
 public:
  static constexpr uint64 kInvalidContextId = 0;

  static uint64 NewContextId() {
    uint64 context_id = random::New64();
    while (context_id == kInvalidContextId) {
      context_id = random::New64();
    }
    return context_id;
  }

  EagerContext(
      const SessionOptions& opts,
      ContextDevicePlacementPolicy default_device_placement_policy, bool async,
      /*const*/ DeviceMgr* device_mgr, bool device_mgr_owned,
      /*const*/ Rendezvous* rendezvous,
      DistributedFunctionLibraryRuntime* cluster_flr = nullptr,
      CollectiveExecutorMgrInterface* collective_executor_mgr = nullptr,
      bool run_eager_op_as_function = false, bool jit_compile_rewrite = false);

  void Release() override { Unref(); }

  AbstractTensorInterface* CreateInt64Scalar(int64_t value) override;
  AbstractTensorInterface* CreateUint64Scalar(uint64 value) override;
  AbstractTensorInterface* CreateInt32Scalar(int32_t value) override;
  AbstractTensorInterface* CreateFloatScalar(float value) override;
  AbstractTensorInterface* CreateDoubleScalar(double value) override;
  AbstractTensorInterface* CreateHalfScalar(Eigen::half value) override;
  AbstractTensorInterface* CreateStringScalar(
      tensorflow::tstring value) override;
  AbstractTensorInterface* CreateComplex128Scalar(
      tensorflow::complex128 value) override;
  AbstractTensorInterface* CreateBoolScalar(bool value) override;

  AbstractTensorInterface* CreateTensor(
      DataType dtype, absl::Span<const int64_t> dim_sizes) override;
  AbstractTensorInterface* CreateTensor(DataType dtype, const int64_t* dims,
                                        int num_dims, void* data, size_t len,
                                        MemoryReleaser memory_releaser,
                                        void* memory_releaser_arg) override;

  ImmediateExecutionTensorHandle* CreateLocalHandle(
      AbstractTensorInterface* t) override;
  // Create an abstract tensor handle from tensorflow::Tensor.
  ImmediateExecutionTensorHandle* CreateLocalHandleFromTFTensor(
      tensorflow::Tensor& t, const char* d_name) override;
  ImmediateExecutionTensorHandle* CopyTensorHandleToDevice(
      ImmediateExecutionTensorHandle* handle, const char* device_name,
      Status* status) override;
  ImmediateExecutionOperation* CreateOperation() override;

  // This is a virtual helper function to convert TFRT TensorHandle to
  // tensorflow::TensorHandle. In current runtime EagerContext, just forward
  // the input since the input tensor handle is already a
  // tensorflow::TensorHandle.
  ImmediateExecutionTensorHandle* TFTensorHandleFromInterface(
      ImmediateExecutionTensorHandle* handle) override;

  Status RegisterFunction(AbstractFunction* f) override;

  bool UsesTFRT() override;

  bool RunEagerOpAsFunction() const;

  void SetRunEagerOpAsFunction(bool enable) override;

  bool JitCompileRewrite() const;

  void SetJitCompileRewrite(bool enable) override;

  void ListDevices(std::vector<DeviceAttributes>* devices) override;

  Status AddDevices(std::vector<std::unique_ptr<Device>> devices) override;

  thread::ThreadPool* GetThreadPool() { return thread_pool_.get(); }

  // Returns the function library runtime for the given device.
  FunctionLibraryRuntime* func_lib(const Device* d) const {
    return pflr_->GetFLR(d->name());
  }

  ProcessFunctionLibraryRuntime* pflr() const { return pflr_.get(); }

  std::function<void(std::function<void()>)>* runner() { return &runner_; }

  // Specify a executor for this thread.
  void SetExecutorForThread(EagerExecutor* executor) override;

  const std::shared_ptr<std::vector<DeviceType>> prioritized_device_type_list()
      const {
    mutex_lock l(device_type_list_mu_);
    return prioritized_device_type_list_;
  }

  // Clear pending nodes in thread executors and kernel caches.
  void ClearCachesAndThreadExecutors() override;
  // Clear pending nodes in default executor and kernel caches.
  void ClearCachesAndDefaultExecutor();

  // Sets the device placement policy for the current thread.
  void SetThreadLocalDevicePlacementPolicy(
      ContextDevicePlacementPolicy policy) override;

  // Returns the device placement policy for the current thread.
  ContextDevicePlacementPolicy GetDevicePlacementPolicy() const override;

  // Select an appropriate device for an operation.
  //
  // Given the preferred device for the operation, and the node_def, finds the
  // best suitable device for the operation in this context.
  //
  // The preferred device is specified as a `ParsedName` containing the elements
  // (details) that the resulting device should match. If there are no such
  // devices, and the context currently allows soft device placement, a suitable
  // device not matching `preferred` will be chosen.
  //
  // The chosen device is stored in the `device` argument. The argument is not
  // modified unless this method returns `OkStatus()`.
  Status SelectDevice(DeviceNameUtils::ParsedName preferred,
                      const NodeDef& ndef, Device** out) const;

  // TODO(mdan): Rename to ContainsFunction.
  bool FindFunctionByName(const string& name) const;

  Status FindFunctionOpData(const string& name,
                            const tensorflow::OpRegistrationData** op_data);

  const FunctionDef* FindFunctionDef(const string& name) const override;

  Device* HostCPU() const { return host_cpu_device_; }
  Device* CanonicalDevice(Device* d) const {
    return HostCPU() == d ? nullptr : d;
  }
  const DeviceNameUtils::ParsedName& HostCPUParsedName() const override {
    return HostCPU()->parsed_name();
  }

  const string& HostCPUName() const override { return HostCPU()->name(); }

  GraphCollector* GetGraphCollector() { return &graph_collector_; }

  EagerExecutor& Executor() override;

  // Add the given `fdef` to the local FunctionLibraryDefinition. And add an
  // entry to the KernelAndDevice cache for it if it's not exist.
  Status AddFunctionDef(const FunctionDef& fdef) override;

  Status AddFunctionDefWithStackTraces(
      const FunctionDef& fdef, const StackTracesMap& stack_traces) override;

  // `library` contains all FunctionDefs and GradientDefs to expand `fdef`. Add
  // it to the local FunctionLibraryDefinition as well, but no need to add it
  // to the KernelAndDevice cache since they won't be executed as
  // KernelAndDevices.
  Status AddFunctionDef(const FunctionDef& fdef,
                        const FunctionDefLibrary& library,
                        bool add_to_local_only = false,
                        const StackTracesMap& stack_traces = {});

  const FunctionDef* GetFunctionDef(const string& function_name);

  std::vector<string> ListFunctionNames() override;

  Status RemoveFunction(const string& func) override;

  // Wait for pending nodes to be finished in local executors (including context
  // default executor and thread executors) and executors on remote workers.
  // Return combined status of remote executors. If there are multiple errors,
  // the Status code will be the same as the first remote executor that has
  // errors, and the error message will be combined from all executors.
  Status SyncExecutors();

  Status AsyncWait() override { return SyncExecutors(); }

  core::RefCountPtr<KernelAndDevice> GetCachedKernel(Fprint128 cache_key);
  Device* GetCachedDevice(Fprint128 device_cache_key);

  void AddKernelToCache(Fprint128 cache_key, KernelAndDevice* kernel);
  void AddDeviceToCache(Fprint128 device_cache_key, Device* device);

  bool LogDevicePlacement() const { return log_device_placement_; }
  void SetLogDevicePlacement(bool enable) override {
    log_device_placement_ = enable;
  }

  // When tensor transfer across functions/eager executions using send/recv ops
  // are required, `reuse_rendezvous_for_functions_` can be set to true so that
  // function executions and eager executions use the same rendezvous instance,
  // instead of creating new instance per function calls.
  void SetReuseRendezvousForFunctions(
      bool reuse_rendezvous_for_functions) override {
    reuse_rendezvous_for_functions_ = reuse_rendezvous_for_functions;
  }
  bool GetReuseRendezvousForFunctions() const {
    return reuse_rendezvous_for_functions_;
  }
  mutex* reuse_rendezvous_for_functions_mu() {
    return &reuse_rendezvous_for_functions_mu_;
  }

  bool AllowSoftPlacement() const { return allow_soft_placement_; }
  void SetAllowSoftPlacement(bool enable) override {
    allow_soft_placement_ = enable;
  }
  bool LogMemory() const { return log_memory_; }

  Rendezvous* GetRendezvous() const { return rendezvous_; }

  void ResetGlobalRendezvousForFunction() override {
    mutex_lock l(global_rendezvous_mu_);
    // Remove the global rendezvous instance from the local rendezvous table
    // if it uses local rendezvous type, which forces EagerContext to create a
    // new local rendezvous instance in the table.
    local_rendezvous_table_->Remove(-1);
    Rendezvous* rendezvous;
    TF_CHECK_OK(CreateRendezvousFactory()(-1, nullptr, &rendezvous));
    global_rendezvous_for_functions_ =
        core::RefCountPtr<Rendezvous>(rendezvous);
  }

  // Returns the global_rendezvous_for_functions' underlying LocalRendezvous'
  // status. If the underlying Rendezvous is not in the local_rendezvous_table_
  // returns OK.
  Status GetGlobalRendezvousForFunctionLocalRendezvousStatus();

  // Returns a factory which maps from step_id to rendezvous. This closure
  // respects the value of `SetReuseRendezvousForFunctions` at the time the
  // closure was created, which allows the setting to be toggled around async op
  // launches.
  //
  // The caller of the returned function owns a reference to the resulting
  // Rendezvous.
  Rendezvous::Factory RendezvousFactory() {
    Rendezvous::Factory factory = CreateRendezvousFactory();

    if (!factory && reuse_rendezvous_for_functions_) {
      // There is an implicit assumption that the
      // global_rendezvous_for_functions_ is always an IntraProcessRendezvous to
      // match the behaviour of the EagerContext's rendezvous. Ref:
      // tensorflow/c/eager/c_api.cc;l=143;rcl=396387348 If a cross process
      // kernel needs a rendezvous a new InterProcessRendezvous should be
      // created.
      factory = Rendezvous::Factory{[this](const int64_t step_id,
                                           const DeviceMgr* device_mgr,
                                           Rendezvous** r) {
        mutex_lock l(global_rendezvous_mu_);
        // Increase the ref that owned by the caller.
        global_rendezvous_for_functions_->Ref();
        *r = global_rendezvous_for_functions_.get();
        return OkStatus();
      }};
    }

    return factory;
  }

  CollectiveExecutorMgrInterface* collective_executor_mgr() {
    return collective_executor_mgr_.Get();
  }
  std::unique_ptr<CollectiveExecutor::Handle> GetCollectiveExecutorHandle() {
    return std::unique_ptr<CollectiveExecutor::Handle>(
        new CollectiveExecutor::Handle(
            collective_executor_mgr()->FindOrCreate(0), true /*inherit_ref*/));
  }

  tensorflow::DeviceMgr* local_device_mgr() const {
    return local_device_manager_.Get();
  }
  const tensorflow::DynamicDeviceMgr* remote_device_mgr() const {
    return remote_device_manager_.Get();
  }

  tensorflow::DynamicDeviceMgr* GetOwnedRemoteDeviceMgr() {
    return remote_device_manager_.GetOwned();
  }

  std::vector<Device*> ListLocalTfDevices() override {
    return local_device_mgr()->ListDevices();
  }

  std::vector<Device*> ListAllTfDevices() override;

  // TODO(apassos) clean up RunMetadata storage.
  mutex* MetadataMu() TF_LOCK_RETURNED(metadata_mu_) { return &metadata_mu_; }
  bool ShouldStoreGraphs() TF_LOCKS_EXCLUDED(metadata_mu_);
  void SetShouldStoreGraphs(bool value) override;
  RunMetadata* RunMetadataProto() TF_EXCLUSIVE_LOCKS_REQUIRED(metadata_mu_) {
    return run_metadata_.get();
  }
  std::unique_ptr<RunMetadata> ExportRunMetadata() override
      TF_LOCKS_EXCLUDED(metadata_mu_);

  void StartStep() override;
  void EndStep() override;
  ScopedStepContainer* StepContainer();

  FunctionLibraryDefinition* FuncLibDef() { return &func_lib_def_; }

#if !defined(IS_MOBILE_PLATFORM)
  // Assign the EagerClient pointer to `client` based on the given device / task
  // name, and increment the refcount of the client. The reference ownership is
  // transferred to the caller, and the unref should automatically happen when
  // destructing the RefCountPtr object at the caller's side.
  // `client` must not be initialized or holding a reference of another object
  // before calling this method.
  Status GetClient(Device* device,
                   core::RefCountPtr<eager::EagerClient>* client);
  Status GetClient(const DeviceNameUtils::ParsedName& device_name,
                   core::RefCountPtr<eager::EagerClient>* client);
  Status GetClient(const string& remote_task,
                   core::RefCountPtr<eager::EagerClient>* client);

  uint64 GetContextId() const;
  uint64 GetContextViewId() const;
  void IncrementContextViewId();

  Status EnableCollectiveOps(const ServerDef& server_def) override;

  // TODO(nareshmodi): Encapsulate remote state into a separate
  // class/struct.
  //
  // Enables the eager context to communicate with remote devices. When
  // initializing with this method, this context will be the primary context,
  // which will kill all its remote contexts in shutdown.
  //
  // - server: A ServerInterface that exports the tensorflow.WorkerService.
  // Note that this class expects the server to already have been started.
  // - remote_eager_workers: A cache from which we can get "EagerClient"s to
  // communicate with remote eager services.
  // - remote_device_mgr: A DeviceMgr* which contains all remote devices
  // (should contain no local devices).
  // - remote_contexts: A vector containing task names.
  // TODO(b/184375824): clean up parameter order for better readability.
  Status InitializeRemoteMaster(
      std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
      std::shared_ptr<WorkerSession> worker_session,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      std::unique_ptr<DynamicDeviceMgr> remote_device_manager,
      const std::vector<string>& remote_contexts, uint64 context_id,
      /*const*/ Rendezvous* r, /*const*/ DeviceMgr* local_device_mgr,
      int keep_alive_secs, DistributedFunctionLibraryRuntime* cluster_flr,
      std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
          remote_mgr);

  // Update an existing master context with a new set of remote workers (i.e., a
  // new "view" of cluster membership. Similar to InitializeRemoteMaster but
  // this will keep the current context_id and increment a context_view_id, will
  // keep the current resource manager so that resources from the previous view
  // can still be accessed, and will automatically register existing functions
  // if there are newly added hosts.
  Status UpdateRemoteMaster(
      uint64 context_id,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      const std::vector<string>& add_remote_contexts,
      const std::vector<string>& remove_remote_contexts);

  // Similar with InitializeRemoteMaster but this context will not kill remote
  // contexts in shutdown.
  Status InitializeRemoteWorker(
      const WorkerEnv* worker_env,
      std::shared_ptr<WorkerSession> worker_session,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      DynamicDeviceMgr* remote_device_mgr,
      const std::vector<string>& remote_contexts, uint64 context_id,
      uint64 context_view_id, DistributedFunctionLibraryRuntime* cluster_flr,
      std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
          remote_mgr,
      std::function<void()> resource_deallocator);

  // Similar with InitializeRemoteWorker but will reuse existing context and
  // increment context_view_id.
  Status UpdateRemoteWorker(
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      const std::vector<string>& remote_contexts, uint64 context_id);

  Status StoreCollectiveOpsServer(
      std::unique_ptr<ServerInterface> new_server, DeviceMgr* device_mgr,
      CollectiveExecutorMgrInterface* rpc_collective_executor_mgr);

  // For the specified remote worker, preprocess and set its device filters.
  Status SetRemoteDeviceFilters(const string& remote_worker,
                                const std::vector<string>& device_filters);

  // For the specified remote worker, apply the stored device filters to the
  // list of device attributes following these rules:
  // (1) if the remote worker does not have device filters, all devices are
  //     visible to the worker;
  // (2) if the device is on the remote worker, then it is visible;
  // (3) if the device matches at least one device filter, then it is visible.
  // The result is saved as a boolean vector of the same length (i.e.,
  // filtered_device_mask) indicating whether each of the devices is visible to
  // the remote worker.
  void FilterDevicesForRemoteWorkers(
      const string& remote_worker,
      const protobuf::RepeatedPtrField<DeviceAttributes>& device_attrs,
      std::vector<bool>* filtered_device_mask);

  // TODO(fishx): Remove the custom deleter once we remove forward declaration.
  const std::unique_ptr<eager::RemoteMgr,
                        std::function<void(eager::RemoteMgr*)>>&
  RemoteMgr() {
    return remote_mgr_;
  }

  // If true, then tensors should be shipped across processes via the
  // EagerService.Enqueue(SendTensorOp). If false, _Send/_Recv ops should be
  // used instead (which in-turn use WorkerService.RecvTensor RPCs).
  bool UseSendTensorRPC() { return use_send_tensor_rpc_; }

  tensorflow::ServerInterface* GetServer() { return server_.get(); }

  // For LLVM style RTTI.
  static bool classof(const AbstractContext* ptr) {
    return ptr->getKind() == kEager;
  }

  // Function to support distributed C API.
  void SetDistributedManager(
      std::unique_ptr<ImmediateExecutionDistributedManager> distributed)
      override {
    distributed_manager_ = std::move(distributed);
  }
  ImmediateExecutionDistributedManager* GetDistributedManager() override {
    return distributed_manager_.get();
  }

  // May only be used during multi-client setup so that a RemoteRendezvous
  // can be initialized instead of defaulting to the IntraProcessRendezvous.
  void SetWorkerEnv(WorkerEnv* worker_env,
                    std::shared_ptr<WorkerSession> worker_session);
#endif  // IS_MOBILE_PLATFORM

  // Closes remote eager contexts, waits for all RPCs to finish, and
  // destroys the EagerClientCache. No RPCs can be made through this context
  // after this method has been called.
  // This method exists to aid a clean shutdown. It causes all RPCs to finish
  // and remote TensorHandles to release their references to this context.
  // To avoid deadlocks, this method must not be called on the thread
  // processing RPCs because it makes RPCs and waits for their completion.
  //
  // On mobile, it just cleans the caches.
  void WaitForAndCloseRemoteContexts();

  bool PinSmallOpsToCPU() const { return pin_small_ops_to_cpu_; }

  tensorflow::Env* TFEnv() const { return env_; }

  Status FindDeviceFromName(const char* device_name, Device** device) const;

  Status FindCompositeDeviceFromName(StringPiece device_name,
                                     CompositeDevice** device) const;

  bool IsCustomDevice(const string& device_name) override;

  Status RegisterCustomDevice(const string& name,
                              std::unique_ptr<CustomDevice> device) override;

  CustomDeviceOpHandler& GetCustomDeviceOpHandler() override {
    return custom_device_op_handler_;
  };

  // Find or create a composite device with the given `underlying_devices` and
  // `device_name` (if not empty).
  Status FindOrCreateCompositeDevice(
      const std::vector<string>& underlying_devices, const string& device_name,
      CompositeDevice** composite_device);

  bool OnSameTask(const Device* first, const Device* second) const;
  // Gets the CPU device on the task of device.
  Status CPUDeviceOnTask(const Device* device, Device** cpu_device) const;

  const SessionOptions& session_options() const { return opts_; }
  void InitPrioritizedDeviceTypeList();

  // Re-assign cluster-FLR and re-initialize devices and FLR in process-FLR
  void UpdateClusterFLRAndInitDevices(
      DistributedFunctionLibraryRuntime* cluster_flr);

  // A constant representing the step id used for the global rendezvous.
  // This is used to distibguish whether a user-specified step id should be set.
  // Step id value of kGlobalRendezvous is reserved and should not be specified
  // by the user.
  static const int64_t kGlobalRendezvousId;

 private:
  // The class for wrapping a map of step_id to local rendezvous instances.
  class LocalRendezvousTable {
   public:
    LocalRendezvousTable() = default;
    ~LocalRendezvousTable();

    IntraProcessRendezvous* FindOrCreate(int64_t step_id,
                                         DeviceMgr* device_mgr);
    IntraProcessRendezvous* Find(int64_t step_id);
    void Remove(int64_t step_id);
    void CleanUpAll();

   private:
    mutable mutex table_lock_;
    absl::flat_hash_map<int64_t, IntraProcessRendezvous*> table_
        TF_GUARDED_BY(table_lock_);
  };

  Rendezvous::Factory CreateRendezvousFactory() const {
#if !defined(IS_MOBILE_PLATFORM)
    if (worker_env_ != nullptr && worker_env_->rendezvous_mgr != nullptr) {
      return Rendezvous::Factory{
          [this](const int64_t step_id, const DeviceMgr* device_mgr,
                 Rendezvous** r) {
            VLOG(6)
                << "Creating rendezvous using the worker_env's rendezvous_mgr.";
            // TODO(hhb): Add a Create method and use it here.
            auto* remote_r = worker_env_->rendezvous_mgr->Find(step_id);
            remote_r->Initialize(worker_session_.get()).IgnoreError();
            *r = remote_r;
            return OkStatus();
          },
          [this](const int64_t step_id) {
            VLOG(6) << "Cleaning up rendezvous from the rendezvous_mgr. "
                    << "Step id: " << step_id;
            worker_env_->rendezvous_mgr->Cleanup(step_id);
            return OkStatus();
          }};
    }
#endif

    if (remote_device_mgr() == nullptr) {
      return Rendezvous::Factory{
          [this](const int64_t step_id, const DeviceMgr* device_mgr,
                 Rendezvous** r) {
            VLOG(6) << "Creating rendezvous using local_device_mgr.";
            *r = local_rendezvous_table_->FindOrCreate(step_id,
                                                       local_device_mgr());
            return OkStatus();
          },
          [this](const int64_t step_id) {
            VLOG(6) << "Cleaning up rendezvous from local_device_mgr.";
            local_rendezvous_table_->Remove(step_id);
            return OkStatus();
          }};
    }

    return Rendezvous::Factory();
  }

  ~EagerContext() override;

  Status MaybeRegisterFunctionRemotely(const FunctionDef& fdef);
  Status MaybeRemoveFunctionRemotely(const string& function_name);
  Status RegisterExistingFunctionsOnRemoteWorkers(
      const std::vector<string>& remote_workers);

  void ResetPFLR(const DeviceMgr* device_mgr, Env* env,
                 const ConfigProto* config, int graph_def_version,
                 const FunctionLibraryDefinition* lib_def,
                 const OptimizerOptions& optimizer_options,
                 thread::ThreadPool* thread_pool = nullptr,
                 DistributedFunctionLibraryRuntime* cluster_flr = nullptr);

  void ResetClusterFLR(DistributedFunctionLibraryRuntime* cluster_flr);
  void UpdateGlobalRendezvousDeviceManager(tensorflow::DeviceMgr* device_mgr);

  void ClearResourceContainer(const string& name);

  template <typename T>
  struct OwnedOrUnownedHelper {
   public:
    OwnedOrUnownedHelper() {}
    explicit OwnedOrUnownedHelper(T* object, const bool owned = false) {
      Reset(object, owned);
    }

    void Reset(std::unique_ptr<T> object) {
      owned_object = std::move(object);
      unowned_object_ptr = nullptr;
    }

    void Reset(T* object, const bool owned = false) {
      if (owned) {
        owned_object.reset(object);
        unowned_object_ptr = nullptr;
      } else {
        owned_object.reset(nullptr);
        unowned_object_ptr = object;
      }
    }

    bool Owned() const { return owned_object != nullptr; }

    T* GetOwned() const { return owned_object.get(); }
    T* Get() const {
      return owned_object ? owned_object.get() : unowned_object_ptr;
    }

    std::unique_ptr<T> owned_object = nullptr;
    T* unowned_object_ptr = nullptr;
  };

  SessionOptions opts_;
  const ContextDevicePlacementPolicy default_device_placement_policy_;

  // Note: we cannot use C++11 thread_local here as there is no concept of a
  // thread-local-object-local variable in C++11.
  mutable mutex policy_map_mu_;
  std::unordered_map<std::thread::id, ContextDevicePlacementPolicy>
      device_placement_policy_ TF_GUARDED_BY(policy_map_mu_);

  // This device manager maintains only the local devices on this worker.
  OwnedOrUnownedHelper<DeviceMgr> local_device_manager_;
  // Maintain copy of all previously created local device managers.
  std::vector<std::unique_ptr<DeviceMgr>> old_local_device_managers_;

  // Unowned DynamicDeviceMgr is set on remote worker to allow running
  // multi-device function on remote worker.
  // This device manager maintains all the devices (including both local and
  // remote to this worker) in the cluster.
  OwnedOrUnownedHelper<DynamicDeviceMgr> remote_device_manager_;

  Device* host_cpu_device_;  // Owned by device_manager
  mutable mutex device_type_list_mu_;
  std::shared_ptr<std::vector<DeviceType>> prioritized_device_type_list_
      TF_GUARDED_BY(device_type_list_mu_);
  Rendezvous* rendezvous_;
  CustomDeviceOpHandler custom_device_op_handler_;

  mutable mutex composite_devices_mu_;
  // Maps from the fingerprint of a set of device names to a virtual
  // CompositeDevice.
  // TODO(b/145922293): Consider taking device names as keys.
  absl::flat_hash_map<uint64, std::unique_ptr<CompositeDevice>>
      composite_devices_ ABSL_GUARDED_BY(composite_devices_mu_);

  FunctionLibraryDefinition func_lib_def_{OpRegistry::Global(), {}};

  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // EagerContext owns the DistributedFunctionLibraryRuntime(
  // EagerClusterFunctionLibraryRuntime) if using EagerService for remote
  // function execution (lazy_copy_function_remote_inputs_=true).
  OwnedOrUnownedHelper<DistributedFunctionLibraryRuntime> cluster_flr_;
  // One FunctionLibraryRuntime per device.
  // func_libs[i] is the FunctionLibraryRuntime corresponding to
  // session->devices[i].
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;

  std::function<void(std::function<void()>)> runner_;

  mutex cache_mu_;
  mutex device_cache_mu_;
  struct RegisteredFunction : public core::RefCounted {
    ~RegisteredFunction() override {}

    std::unique_ptr<std::vector<Fprint128>> cached_kernel_keys;
  };
  std::unordered_map<Fprint128, core::RefCountPtr<KernelAndDevice>,
                     Fprint128Hasher>
      kernel_cache_ TF_GUARDED_BY(cache_mu_);
  std::unordered_map<string, RegisteredFunction*> registered_functions_
      TF_GUARDED_BY(cache_mu_);
  absl::flat_hash_map<Fprint128, Device*, Fprint128Hasher> device_cache_
      TF_GUARDED_BY(device_cache_mu_);

  // Whether we should compute RunMetadata.
  std::atomic<bool> should_store_graphs_{false};
  mutex metadata_mu_;
  std::unique_ptr<RunMetadata> run_metadata_ TF_GUARDED_BY(metadata_mu_);
  GraphCollector graph_collector_;
  std::atomic<bool> log_device_placement_;
  std::atomic<bool> allow_soft_placement_;

  // Information related to step containers.
  std::atomic<int> num_active_steps_;
  std::unique_ptr<ScopedStepContainer> step_container_
      TF_GUARDED_BY(metadata_mu_);

  EagerExecutor default_executor_;
  mutable mutex executor_map_mu_;
  // Not owned.
  std::unordered_map<std::thread::id, EagerExecutor*> thread_local_executor_
      TF_GUARDED_BY(executor_map_mu_);
  std::unordered_map<std::thread::id, std::unordered_set<EagerExecutor*>>
      has_cleanup_ TF_GUARDED_BY(executor_map_mu_);

  const bool log_memory_;

  // The table of local rendezvous instances for intra-process communication.
  // This make sures only one local rendezvous instance exists per step id.
  std::unique_ptr<LocalRendezvousTable> local_rendezvous_table_;

  // Whether to use same rendezvous instance across function/eager executions.
  std::atomic<bool> reuse_rendezvous_for_functions_{false};
  mutable mutex global_rendezvous_mu_;
  core::RefCountPtr<Rendezvous> global_rendezvous_for_functions_
      TF_GUARDED_BY(global_rendezvous_mu_);
  mutex reuse_rendezvous_for_functions_mu_;

  Env* const env_;

  OwnedOrUnownedHelper<CollectiveExecutorMgrInterface> collective_executor_mgr_;

#if !defined(IS_MOBILE_PLATFORM)
  std::vector<string> GetRemoteContexts() TF_LOCKS_EXCLUDED(remote_state_mu_);
  bool IsRemoteContextsEmpty() TF_LOCKS_EXCLUDED(remote_state_mu_);
  void CloseAndClearAllRemoteContexts();
  void CloseRemoteContexts(const std::vector<string>& remote_contexts,
                           uint64 context_id, uint64 context_view_id);

  // TODO(b/184375824): clean up parameter order for better readability.
  Status SetMasterContextState(
      std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
      std::shared_ptr<WorkerSession> worker_session,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      std::unique_ptr<DynamicDeviceMgr> remote_device_manager,
      uint64 context_id, uint64 context_view_id, /*const*/ Rendezvous* r,
      /*const*/ DeviceMgr* local_device_mgr, int keep_alive_secs,
      DistributedFunctionLibraryRuntime* cluster_flr,
      std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
          remote_mgr);

  // The server_ is not const since we release it when the context is destroyed.
  // Therefore the server_ object is not marked as const (even though it should
  // be).
  std::unique_ptr<ServerInterface> server_;
  const WorkerEnv* worker_env_ = nullptr;
  std::shared_ptr<WorkerSession> worker_session_;

  mutable mutex remote_state_mu_;

  uint64 context_id_ TF_GUARDED_BY(remote_state_mu_);
  // The view id of an eager context should be set to 0 when context is created,
  // and continuously incremented when context with the same context_id gets
  // updated. The view id should be consistent between master and workers.
  uint64 context_view_id_ TF_GUARDED_BY(remote_state_mu_);
  std::vector<string> remote_contexts_ TF_GUARDED_BY(remote_state_mu_);
  std::unique_ptr<eager::EagerClientCache> remote_eager_workers_
      TF_GUARDED_BY(remote_state_mu_);

  int keep_alive_secs_ TF_GUARDED_BY(remote_state_mu_);
  std::atomic<int> sleep_for_secs_;

  std::unique_ptr<Thread> keep_alive_thread_;
  mutex keep_alive_thread_shutdown_mu_;
  condition_variable keep_alive_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(keep_alive_thread_shutdown_mu_) = false;

  std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
      remote_mgr_;
  bool is_master_ TF_GUARDED_BY(remote_state_mu_);

  // Maps from a remote worker to a list of parsed device filters.
  std::unordered_map<string, std::vector<DeviceNameUtils::ParsedName>>
      cluster_device_filters_ TF_GUARDED_BY(remote_state_mu_);

  // A distributed manager that helps setup, update, and check liveness of
  // member tasks in the cluster.
  std::unique_ptr<ImmediateExecutionDistributedManager> distributed_manager_;

#endif  // IS_MOBILE_PLATFORM

  // For a multi device function, the target device of each input is unknown
  // until the function is instantiated on the default function device.
  // If false, eagerly copy all remote inputs to the default function device;
  // if true, lazily copy remote inputs to their target devices to avoid
  // redundant copies.
  bool lazy_copy_function_remote_inputs_ = false;
  bool use_send_tensor_rpc_;
  const bool pin_small_ops_to_cpu_;

  // Function that will be invoked in destructor to deallocate resources related
  // to this context.
  std::function<void()> resource_deallocator_ = nullptr;
  bool run_eager_op_as_function_;
  bool jit_compile_rewrite_;
};

inline EagerContext* ContextFromInterface(ImmediateExecutionContext* context) {
  return down_cast<EagerContext*>(context);
}

namespace internal {
struct EagerContextDeleter {
  void operator()(EagerContext* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using EagerContextPtr =
    std::unique_ptr<EagerContext, internal::EagerContextDeleter>;

// Sets the EagerContext owned by the current Python eager Context (see
// TFE_Py_SetEagerContext in python/eager/pywrap_tfe.h). This is always called
// in tandem with TFE_Py_SetEagerContext (but not called by it, because its
// py_context argument is opaque).
//
// Do not use this function in production. It is only intended for testing.
// (see _reset_context in context.py).
//
// Not thread-safe.
void SetCEagerContext(EagerContext* ctx);

// Returns the EagerContext owned by the current Python eager Context (see
// TFE_Py_SetEagerContext in pywrap_tfe.h).
//
// Not thread-safe.
EagerContext* GetCEagerContext();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_H_
