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
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/device_name_utils.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#endif  // !IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"

#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace eager {
// We need this forward declaration because we have circular dependency:
// Context -> RemoteMgr -> TensorHandle -> Context.
// TODO(fishx): Remove this once we remove Context dependency in TensorHandle.
class RemoteMgr;
}  // namespace eager

// LINT.IfChange
// Note: Keep in sync with exported copy of enum in eager/c_api.h.
enum ContextDevicePlacementPolicy {
  // Running operations with input tensors on the wrong device will fail.
  DEVICE_PLACEMENT_EXPLICIT = 0,
  // Copy the tensor to the right device but log a warning.
  DEVICE_PLACEMENT_WARN = 1,
  // Silently copy the tensor, which has a performance cost since the operation
  // will be blocked till the copy completes. This is the default policy.
  DEVICE_PLACEMENT_SILENT = 2,
  // Placement policy which silently copies int32 tensors but not other dtypes.
  DEVICE_PLACEMENT_SILENT_FOR_INT32 = 3,
};
// LINT.ThenChange(//tensorflow/c/eager/c_api.h)

// LINT.IfChange
// Note: Keep in sync with exported copy of enum in eager/c_api_experimental.h.
enum ContextMirroringPolicy {
  // Do not maintain mirrors in a TensorHandle, instead make new TensorHandle
  // copies with their own lifetime.
  MIRRORING_NONE = 0,
  // Mirroring any remote tensor handles, associating them with the lifetime of
  // the local TensorHandle.
  MIRRORING_ALL = 1,
};
// LINT.ThenChange(//tensorflow/c/eager/c_api_experimental.h)

class RunMetadataListener {
 public:
  virtual ~RunMetadataListener() {}
  virtual void BeforeClearRunMetadata() = 0;
};

class EagerContext : public core::RefCounted {
 public:
  static const uint64 kInvalidContextId = 0;

  static uint64 NewContextId() {
    uint64 context_id = random::New64();
    while (context_id == kInvalidContextId) {
      context_id = random::New64();
    }
    return context_id;
  }

  EagerContext(const SessionOptions& opts,
               ContextDevicePlacementPolicy default_device_placement_policy,
               ContextMirroringPolicy default_mirroring_policy, bool async,
               const bool lazy_copy_function_remote_inputs,
               const DeviceMgr* device_mgr, bool device_mgr_owned,
               Rendezvous* rendezvous,
               const CustomKernelCreator* custom_kernel_creator,
               DistributedFunctionLibraryRuntime* cluster_flr = nullptr);

  ~EagerContext() override;

  // Returns the function library runtime for the given device.
  FunctionLibraryRuntime* func_lib(const Device* d) const {
    return pflr_->GetFLR(d->name());
  }

  ProcessFunctionLibraryRuntime* pflr() const { return pflr_.get(); }

  std::function<void(std::function<void()>)>* runner() { return &runner_; }

  // Specify a executor for this thread.
  void SetExecutorForThread(EagerExecutor* executor);

  // TODO(apassos) make this return a constant reference
  gtl::FlatMap<string, Device*, StringPieceHasher>* device_map() {
    return &devices_map_;
  }

  // TODO(apassos) make this return a constant reference
  std::vector<Device*>* devices() { return &devices_; }
  const std::vector<DeviceType>& prioritized_device_type_list() {
    return prioritized_device_type_list_;
  }

  // Clears the kernel caches.
  void ClearCaches();

  // Sets the device placement policy for the current thread.
  void SetThreadLocalDevicePlacementPolicy(ContextDevicePlacementPolicy policy);

  // Returns the device placement policy for the current thread.
  ContextDevicePlacementPolicy GetDevicePlacementPolicy() const;

  // Sets the implicit copy policy for the current thread.
  void SetThreadLocalMirroringPolicy(ContextMirroringPolicy);

  // Returns the implicit copy policy for the current thread.
  ContextMirroringPolicy GetMirroringPolicy() const;

  bool MirrorTensors() const;

  bool LazyCopyFunctionRemoteInputs() const;

  bool FindFunctionByName(const string& name);

  Status FindFunctionOpData(const string& name,
                            const tensorflow::OpRegistrationData** op_data);

  const FunctionDef* FindFunctionDef(const string& name);

  Status FindDeviceByName(const string& name, Device** result) const;

  Device* HostCPU() const { return devices_[0]; }
  Device* CanonicalDevice(Device* d) const {
    return HostCPU() == d ? nullptr : d;
  }

  GraphCollector* GetGraphCollector() { return &graph_collector_; }

  EagerExecutor& Executor();

  // Add the given `fdef` to the local FunctionLibraryDefinition. And add an
  // entry to the KernelAndDevice cache for it if it's not exist.
  Status AddFunctionDef(const FunctionDef& fdef);
  // `library` contains all FunctionDefs and GradientDefs to expand `fdef`. Add
  // it to the local FunctionLibraryDefinition as well, but no need to add it
  // to the KernelAndDevice cache since they won't be executed as
  // KernelAndDevices.
  Status AddFunctionDef(const FunctionDef& fdef,
                        const FunctionDefLibrary& library,
                        const bool add_to_local_only = false);

  Status RemoveFunction(const string& func);

  core::RefCountPtr<KernelAndDevice> GetCachedKernel(Fprint128 cache_key);

  void AddKernelToCache(Fprint128 cache_key, KernelAndDevice* kernel);

  bool LogDevicePlacement() const { return log_device_placement_; }
  bool AllowSoftPlacement() const { return allow_soft_placement_; }
  bool LogMemory() const { return log_memory_; }

  Rendezvous* GetRendezvous() const { return rendezvous_; }
  Rendezvous* CreateRendezvous(const int64 step_id) const {
    if (rendezvous_creator_ != nullptr) {
      return rendezvous_creator_(step_id);
    }

#if !defined(IS_MOBILE_PLATFORM)
    if (worker_env_ != nullptr && worker_env_->rendezvous_mgr != nullptr) {
      auto* remote_r = worker_env_->rendezvous_mgr->Find(step_id);
      remote_r->Initialize(worker_session_.get()).IgnoreError();
      return remote_r;
    }
#endif

    if (remote_device_mgr() == nullptr) {
      return new IntraProcessRendezvous(local_device_mgr());
    }

    return nullptr;
  }

  CollectiveExecutorMgrInterface* collective_executor_mgr() {
    return collective_executor_mgr_.Get();
  }
  std::unique_ptr<CollectiveExecutor::Handle> GetCollectiveExecutorHandle() {
    return std::unique_ptr<CollectiveExecutor::Handle>(
        new CollectiveExecutor::Handle(
            collective_executor_mgr()->FindOrCreate(0), true /*inherit_ref*/));
  }

  const tensorflow::DeviceMgr* local_device_mgr() const {
    return local_device_manager_.Get();
  }
  const tensorflow::DynamicDeviceMgr* remote_device_mgr() const {
    return remote_device_manager_.Get();
  }

  tensorflow::DynamicDeviceMgr* GetOwnedRemoteDeviceMgr() {
    return remote_device_manager_.GetOwned();
  }

  // TODO(apassos) clean up RunMetadata storage.
  mutex* MetadataMu() LOCK_RETURNED(metadata_mu_) { return &metadata_mu_; }
  bool ShouldStoreGraphs() LOCKS_EXCLUDED(metadata_mu_);
  void SetShouldStoreGraphs(bool value);
  RunMetadata* RunMetadataProto() { return &run_metadata_; }
  void ClearRunMetadata() EXCLUSIVE_LOCKS_REQUIRED(metadata_mu_);

  void StartStep();
  void EndStep();
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

  uint64 GetContextId();
  uint64 GetContextViewId();

  // TODO(nareshmodi): Encapsulate remote state into a separate
  // class/struct.
  //
  // Enables the eager context to communicate with remote devices. When
  // initializing with this method, this context will be the master context,
  // which will kill all its slaves in shutdown.
  //
  // - server: A ServerInterface that exports the tensorflow.WorkerService.
  // Note that this class expects the server to already have been started.
  // - remote_eager_workers: A cache from which we can get "EagerClient"s to
  // communicate with remote eager services.
  // - remote_device_mgr: A DeviceMgr* which contains all remote devices
  // (should contain no local devices).
  // - remote_contexts: A vector containing task names.
  Status InitializeRemoteMaster(
      std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
      std::shared_ptr<WorkerSession> worker_session,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      std::unique_ptr<DynamicDeviceMgr> remote_device_manager,
      const std::vector<string>& remote_contexts, uint64 context_id,
      Rendezvous* r, DeviceMgr* local_device_mgr, int keep_alive_secs,
      DistributedFunctionLibraryRuntime* cluster_flr,
      std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
          remote_mgr);

  // Update an existing master context with a new set of remote workers (i.e., a
  // new "view" of cluster membership. Similar to InitializeRemoteMaster but
  // this will keep the current context_id and increment a context_view_id, will
  // keep the current resource manager so that resources from the previous view
  // can still be accessed, and will automatically register existing functions
  // if there are newly added hosts.
  Status UpdateRemoteMaster(
      WorkerEnv* worker_env,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      const std::vector<string>& add_remote_contexts,
      const std::vector<string>& remove_remote_contexts, uint64 context_id,
      Rendezvous* r, DeviceMgr* local_device_mgr, int keep_alive_secs,
      DistributedFunctionLibraryRuntime* cluster_flr);

  // Similar with InitializeRemoteMaster but this context will not kill remote
  // contexts in shutdown.
  Status InitializeRemoteWorker(
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      DynamicDeviceMgr* remote_device_mgr,
      const std::vector<string>& remote_contexts, uint64 context_id,
      uint64 context_view_id,
      std::function<Rendezvous*(const int64)> rendezvous_creator,
      DistributedFunctionLibraryRuntime* cluster_flr,
      std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
          remote_mgr,
      std::function<void()> resource_deallocator);

  // Similar with InitializeRemoteWorker but will reuse existing context and
  // increment context_view_id.
  Status UpdateRemoteWorker(
      const DeviceMgr* worker_session_device_mgr,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      DynamicDeviceMgr* remote_device_mgr,
      const std::vector<string>& remote_contexts, uint64 context_id,
      DistributedFunctionLibraryRuntime* cluster_flr);

  Status StoreCollectiveOpsServer(
      std::unique_ptr<ServerInterface> new_server, DeviceMgr* device_mgr,
      CollectiveExecutorMgrInterface* rpc_collective_executor_mgr);

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

  bool PinSmallOpsToCPU() { return pin_small_ops_to_cpu_; }

  tensorflow::Env* TFEnv() const { return env_; }

  std::vector<const FunctionDef*> ListRegisteredFunctions();

  Status FindDeviceFromName(const char* device_name, Device** device) const;

  bool OnSameTask(const Device* first, const Device* second) const;
  // Gets the CPU device on the task of device.
  Status CPUDeviceOnTask(const Device* device, Device** cpu_device) const;

 private:
  void InitDeviceMapAndAsync();
  Status MaybeRegisterFunctionRemotely(const FunctionDef& fdef);
  Status RegisterExistingFunctionsOnRemoteWorkers(
      const std::vector<const FunctionDef*>& function_defs,
      const std::vector<string>& remote_workers);

  void ResetPFLR(const DeviceMgr* device_mgr, Env* env,
                 const ConfigProto* config, int graph_def_version,
                 const FunctionLibraryDefinition* lib_def,
                 const OptimizerOptions& optimizer_options,
                 thread::ThreadPool* thread_pool = nullptr,
                 DistributedFunctionLibraryRuntime* cluster_flr = nullptr,
                 const CustomKernelCreator* custom_kernel_creator = nullptr);

  void ResetClusterFLR(DistributedFunctionLibraryRuntime* cluster_flr);

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

  const ContextDevicePlacementPolicy default_device_placement_policy_;
  const ContextMirroringPolicy default_mirroring_policy_;

  // Note: we cannot use C++11 thread_local here as there is no concept of a
  // thread-local-object-local variable in C++11.
  mutable mutex policy_map_mu_;
  std::unordered_map<std::thread::id, ContextDevicePlacementPolicy>
      device_placement_policy_ GUARDED_BY(policy_map_mu_);
  std::unordered_map<std::thread::id, ContextMirroringPolicy> mirroring_policy_
      GUARDED_BY(policy_map_mu_);

  OwnedOrUnownedHelper<const DeviceMgr> local_device_manager_;

  // Unowned DynamicDeviceMgr is set on remote worker to allow running
  // multi-device function on remote worker.
  OwnedOrUnownedHelper<DynamicDeviceMgr> remote_device_manager_;

  // Devices owned by device_manager
  std::vector<Device*> devices_;
  std::vector<DeviceType> prioritized_device_type_list_;
  // All devices are not owned.
  gtl::FlatMap<string, Device*, StringPieceHasher> devices_map_;
  Rendezvous* rendezvous_;
  std::function<Rendezvous*(const int64)> rendezvous_creator_;

  FunctionLibraryDefinition func_lib_def_{OpRegistry::Global(), {}};

  std::unique_ptr<thread::ThreadPool> thread_pool_;

  const CustomKernelCreator* const custom_kernel_creator_;

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
  struct RegisteredFunction : public core::RefCounted {
    ~RegisteredFunction() override {}

    std::unique_ptr<std::vector<Fprint128>> cached_kernel_keys;
  };
  std::unordered_map<Fprint128, core::RefCountPtr<KernelAndDevice>,
                     Fprint128Hasher>
      kernel_cache_ GUARDED_BY(cache_mu_);
  std::unordered_map<string, RegisteredFunction*> registered_functions_
      GUARDED_BY(cache_mu_);

  // Whether we should compute RunMetadata.
  std::atomic<bool> should_store_graphs_{false};
  mutex metadata_mu_;
  RunMetadata run_metadata_ GUARDED_BY(metadata_mu_);
  GraphCollector graph_collector_;
  // TODO(fishx): Allow update following two bool after context creation.
  const bool log_device_placement_;
  const bool allow_soft_placement_;

  // Information related to step containers.
  std::atomic<int> num_active_steps_;
  std::unique_ptr<ScopedStepContainer> step_container_ GUARDED_BY(metadata_mu_);

  EagerExecutor default_executor_;
  mutable mutex executor_map_mu_;
  // Not owned.
  std::unordered_map<std::thread::id, EagerExecutor*> thread_local_executor_
      GUARDED_BY(executor_map_mu_);

  const bool log_memory_;

  Env* const env_;

  OwnedOrUnownedHelper<CollectiveExecutorMgrInterface> collective_executor_mgr_;

#if !defined(IS_MOBILE_PLATFORM)
  void CloseAndClearAllRemoteContexts();
  void CloseRemoteContexts(const std::vector<string>& remote_contexts,
                           uint64 context_id, uint64 context_view_id);

  Status SetMasterContextState(
      std::unique_ptr<ServerInterface> server, WorkerEnv* worker_env,
      std::shared_ptr<WorkerSession> worker_session,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      std::unique_ptr<DynamicDeviceMgr> remote_device_manager,
      uint64 context_id, uint64 context_view_id, Rendezvous* r,
      DeviceMgr* local_device_mgr, int keep_alive_secs,
      DistributedFunctionLibraryRuntime* cluster_flr,
      std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
          remote_mgr);

  // The server_ is not const since we release it when the context is destroyed.
  // Therefore the server_ object is not marked as const (even though it should
  // be).
  std::unique_ptr<ServerInterface> server_;
  WorkerEnv* worker_env_ = nullptr;
  std::shared_ptr<WorkerSession> worker_session_;
  std::unique_ptr<eager::EagerClientCache> remote_eager_workers_;

  mutex remote_state_mu_;

  uint64 context_id_ GUARDED_BY(remote_state_mu_);
  // The view id of an eager context should be set to 0 when context is created,
  // and continously incremented when context with the same context_id gets
  // updated. The view id should be consistent between master and workers.
  uint64 context_view_id_ GUARDED_BY(remote_state_mu_);
  std::vector<string> remote_contexts_;

  int keep_alive_secs_ GUARDED_BY(remote_state_mu_);
  std::atomic<int> sleep_for_secs_;

  std::unique_ptr<Thread> keep_alive_thread_;
  mutex keep_alive_thread_shutdown_mu_;
  condition_variable keep_alive_thread_cv_;
  bool shutting_down_ GUARDED_BY(keep_alive_thread_shutdown_mu_) = false;

  std::unique_ptr<eager::RemoteMgr, std::function<void(eager::RemoteMgr*)>>
      remote_mgr_;
  bool is_master_ GUARDED_BY(remote_state_mu_);
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
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_H_
