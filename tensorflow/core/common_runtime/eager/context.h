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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// Note: there's a copy enum in eager/c_api.h. It should be kept in sync.
enum ContextDevicePlacementPolicy {
  // Running operations with input tensors on the wrong device will fail.
  DEVICE_PLACEMENT_EXPLICIT = 0,
  // Copy the tensor to the right device but log a warning.
  DEVICE_PLACEMENT_WARN = 1,
  // Silently copy the tensor, which has a performance cost since the operation
  // will be blocked till the copy completes. This is the default policy.
  DEVICE_PLACEMENT_SILENT = 2,
  // Default placement policy which silently copies int32 tensors but not other
  // dtypes.
  DEVICE_PLACEMENT_SILENT_FOR_INT32 = 3,
};

class EagerContext {
 public:
  explicit EagerContext(const SessionOptions& opts,
                        ContextDevicePlacementPolicy default_policy, bool async,
                        std::unique_ptr<DeviceMgr> device_mgr,
                        Rendezvous* rendezvous);

  // TODO(nareshmodi): Split this into 2 classes and hide functionality behind
  // an interface. Alternatively, encapsulate remote state into a separate
  // class/struct.
  //
  // Constructs an eager context that is able to communicate with remote
  // workers.
  //
  // Additional remote-specific args are:
  //  - server: A GrpcServer that exports the tensorflow.WorkerService. Note
  //  that this class expects the server to already have been started.
  //  - remote_eager_workers: A cache from which we can get "EagerClient"s to
  //  communicate with remote eager services.
  //  - remote_device_mgr: A DeviceMgr* which contains all remote devices
  //  (should contain no local devices).
  //  - remote_contexts: A map containing task name to remote context ID.
  explicit EagerContext(
      const SessionOptions& opts, ContextDevicePlacementPolicy default_policy,
      bool async, DeviceMgr* local_device_mgr, Rendezvous* rendezvous,
      std::unique_ptr<GrpcServer> server,
      std::unique_ptr<eager::EagerClientCache> remote_eager_workers,
      std::unique_ptr<DeviceMgr> remote_device_manager,
      const gtl::FlatMap<string, uint64>& remote_contexts);

  ~EagerContext();

  // Returns the function library runtime for the given device.
  FunctionLibraryRuntime* func_lib(Device* d) const {
    return pflr_->GetFLR(d->name());
  }

  // True if running in asynchronous mode.
  bool Async() const;

  EagerExecutor* Executor() { return &executor_; }

  // Sets whether this thread should run in synchronous or asynchronous mode.
  Status SetAsyncForThread(bool async);

  // TODO(apassos) make this return a constant reference
  gtl::FlatMap<string, Device*, StringPieceHasher>* device_map() {
    return &devices_map_;
  }

  // TODO(apassos) make this return a constant reference
  std::vector<Device*>* devices() { return &devices_; }

  // Clears the kernel caches.
  void ClearCaches();

  // Sets the device placement policy for the current thread.
  void SetThreadLocalDevicePlacementPolicy(ContextDevicePlacementPolicy policy);

  // Returns the device placement policy for the current thread.
  ContextDevicePlacementPolicy GetDevicePlacementPolicy();

  Status AsyncWait() { return executor_.WaitForAllPendingNodes(); }

  Status GetStatus() { return executor_.status(); }

  void ClearAsyncError() { executor_.ClearError(); }

  bool FindFunctionByName(const string& name);

  Status FindFunctionOpData(const string& name,
                            const tensorflow::OpRegistrationData** op_data);

  const FunctionDef* FindFunctionDef(const string& name);

  Status FindDeviceByName(const string& name, Device** result);

  Device* HostCPU() { return devices_[0]; }

  uint64 NextId() { return executor_.NextId(); }

  void ExecutorAdd(EagerNode* node) { executor_.Add(node); }

  Status AddFunctionDef(const FunctionDef& fdef);

  KernelAndDevice* GetCachedKernel(Fprint128 cache_key);

  void AddKernelToCache(Fprint128 cache_key, KernelAndDevice* kernel);

  bool LogDevicePlacement() { return log_device_placement_; }

  Rendezvous* GetRendezvous() { return rendezvous_; }

  mutex* FunctionsMu() { return &functions_mu_; }

  const tensorflow::DeviceMgr* local_device_mgr() const {
    return (local_device_manager_ != nullptr) ? local_device_manager_.get()
                                              : local_unowned_device_manager_;
  }
  const tensorflow::DeviceMgr* remote_device_mgr() {
    return remote_device_manager_.get();
  }

  // TODO(apassos) remove the need for this
  void ReleaseDeviceMgr() { local_device_manager_.release(); }

  // TODO(apassos) clean up RunMetadata storage.
  mutex* MetadataMu() { return &metadata_mu_; }
  bool ShouldStoreMetadata() { return should_store_metadata_.load(); }
  void SetShouldStoreMetadata(bool value);
  RunMetadata* RunMetadataProto() { return &run_metadata_; }

  FunctionLibraryDefinition* FuncLibDef() { return &func_lib_def_; }

  Status GetClientAndContextID(Device* device, eager::EagerClient** client,
                               uint64* context_id);

 private:
  void InitDeviceMapAndAsync();

  const ContextDevicePlacementPolicy policy_;

  // Note: we cannot use C++11 thread_local here as there is no concept of a
  // thread-local-object-local variable in C++11.
  mutex policy_map_mu_;
  std::unordered_map<std::thread::id, ContextDevicePlacementPolicy>
      thread_local_policies_ GUARDED_BY(policy_map_mu_);

  // Only one of the below is set.
  std::unique_ptr<DeviceMgr> local_device_manager_;
  const DeviceMgr* local_unowned_device_manager_;

  // Devices owned by device_manager
  std::vector<Device*> devices_;
  // All devices are not owned.
  gtl::FlatMap<string, Device*, StringPieceHasher> devices_map_;
  Rendezvous* const rendezvous_;

  mutex functions_mu_;
  FunctionLibraryDefinition func_lib_def_ GUARDED_BY(functions_mu_){
      OpRegistry::Global(), {}};

  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // One FunctionLibraryRuntime per device.
  // func_libs[i] is the FunctionLibraryRuntime corresponding to
  // session->devices[i].
  const std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;

  mutex cache_mu_;
  std::unordered_map<Fprint128, KernelAndDevice*, Fprint128Hasher> kernel_cache_
      GUARDED_BY(cache_mu_);

  // Whether we should compute RunMetadata.
  std::atomic<bool> should_store_metadata_{false};
  mutex metadata_mu_;
  RunMetadata run_metadata_ GUARDED_BY(metadata_mu_);
  const bool log_device_placement_;
  // EagerExecutor for async execution.
  EagerExecutor executor_;

  // True if the default value for execution mode is async. Note that this value
  // can be overridden per thread based on `thread_local_async` overrides.
  const bool async_default_;
  mutable mutex async_map_mu_;
  std::unordered_map<std::thread::id, bool> thread_local_async_
      GUARDED_BY(async_map_mu_);

  // The server_ is not const since we release it when the context is destroyed.
  // Therefore the server_ object is not marked as const (even though it should
  // be).
  std::unique_ptr<GrpcServer> server_;
  const std::unique_ptr<eager::EagerClientCache> remote_eager_workers_;
  const std::unique_ptr<DeviceMgr> remote_device_manager_;

  const gtl::FlatMap<string, uint64> remote_contexts_;
  gtl::FlatMap<Device*, std::pair<eager::EagerClient*, uint64>>
      device_to_client_cache_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_H_
