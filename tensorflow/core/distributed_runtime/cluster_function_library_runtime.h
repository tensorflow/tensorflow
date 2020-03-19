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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_

#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {

class WorkerSession;

// ClusterFunctionLibraryRuntime contains methods to Instantiate and Run
// functions across processes by making RPCs.
class ClusterFunctionLibraryRuntime : public DistributedFunctionLibraryRuntime {
 public:
  ClusterFunctionLibraryRuntime(WorkerSession* worker_session,
                                bool create_worker_session_called,
                                DeviceMgr* remote_device_mgr)
      : worker_session_(worker_session),
        create_worker_session_called_(create_worker_session_called),
        remote_device_mgr_(remote_device_mgr) {}

  ~ClusterFunctionLibraryRuntime() override;

  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* handle,
                   FunctionLibraryRuntime::DoneCallback done) override;

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) override;

  void CleanUp(uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
               FunctionLibraryRuntime::DoneCallback done) override;

  DeviceMgr* remote_device_mgr() const override { return remote_device_mgr_; }

 private:
  static Status ConstructFunctionGraph(
      const OpDef& sig, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const FunctionLibraryDefinition& flib_def, GraphDef* g,
      std::vector<string>* send_keys, std::vector<string>* recv_keys);
  friend class ClusterFunctionLibraryRuntimeTest;

  mutable mutex mu_;
  WorkerSession* const worker_session_ = nullptr;  // not owned.
  const bool create_worker_session_called_;

  DeviceMgr* remote_device_mgr_;  // not owned.

  struct FunctionData {
    const string graph_handle;
    const string target;
    // Hold a shared pointer to the underlying worker cache to avoid it being
    // deleted in potential cluster update.
    const std::shared_ptr<WorkerCacheInterface> worker_cache;
    WorkerInterface* wi = nullptr;
    const std::vector<string> send_keys;
    const std::vector<string> recv_keys;

    FunctionData(const string& graph_handle, const string& target,
                 std::shared_ptr<WorkerCacheInterface> worker_cache,
                 WorkerInterface* wi, const std::vector<string>& send_keys,
                 const std::vector<string>& recv_keys)
        : graph_handle(graph_handle),
          target(target),
          worker_cache(std::move(worker_cache)),
          wi(wi),
          send_keys(send_keys),
          recv_keys(recv_keys) {}
  };

  std::vector<FunctionData> function_data_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
