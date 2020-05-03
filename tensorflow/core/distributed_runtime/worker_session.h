/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_SESSION_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_SESSION_H_

#include <string>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"

namespace tensorflow {

class ClusterFunctionLibraryRuntime;
class GraphMgr;
class WorkerCacheInterface;

// WorkerSession encapsulates all of the state relating to a given session.
class WorkerSession {
 public:
  // Collection of local devices. These devices are typically
  // RenamedDevices in all except the SessionMgr.legacy_session_ and
  // sessions created with `isolate_session_state == false`. In the
  // those cases, this method returns a pointer to a borrowed
  // DeviceMgr (typically the `worker_env.device_mgr`).
  DeviceMgr* device_mgr() {
    return device_mgr_ ? device_mgr_.get() : borrowed_device_mgr_;
  }

  DynamicDeviceMgr* remote_device_mgr() { return remote_device_mgr_.get(); }

  const string& session_name() const { return session_name_; }
  const string& worker_name() const { return worker_name_; }

  WorkerCacheInterface* worker_cache() const {
    tf_shared_lock l(worker_session_state_mu_);
    return worker_cache_.get();
  }
  GraphMgr* graph_mgr() const { return graph_mgr_.get(); }

  ClusterFunctionLibraryRuntime* cluster_flr() const {
    return cluster_flr_.get();
  }

  WorkerSession(const string& session_name, const string& worker_name,
                std::unique_ptr<WorkerCacheInterface> worker_cache,
                std::unique_ptr<DeviceMgr> device_mgr,
                std::unique_ptr<GraphMgr> graph_mgr,
                std::unique_ptr<DynamicDeviceMgr> remote_device_mgr);

  static std::shared_ptr<WorkerSession> CreateWithBorrowedDeviceMgr(
      const string& session_name, const string& worker_name,
      std::unique_ptr<WorkerCacheInterface> worker_cache,
      DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
      std::unique_ptr<DynamicDeviceMgr> remote_device_mgr);

  // In the eager runtime we allow WorkerSession to be updated, where the
  // worker cache will be recreated. If WorkerSession upate is expected and a
  // worker in the cache is used in RPCs, the caller should hold a shared
  // pointer to avoid the workers getting deleted.
  std::shared_ptr<WorkerCacheInterface> GetSharedWorkerCache() {
    tf_shared_lock l(worker_session_state_mu_);
    return worker_cache_;
  }

  // Update an existing worker session with new set of remote workers and
  // devices. Added devices will be owned by the worker session, and removed
  // devices will be freed by their names.
  Status UpdateWorkerCacheAndDevices(
      std::unique_ptr<WorkerCacheInterface> new_worker_cache,
      std::vector<std::unique_ptr<Device>> added_remote_devices,
      const std::vector<Device*>& removed_remote_devices);

  ~WorkerSession();

 private:
  WorkerSession(const string& session_name, const string& worker_name,
                std::unique_ptr<WorkerCacheInterface> worker_cache,
                DeviceMgr* borrowed_device_mgr,
                std::unique_ptr<GraphMgr> graph_mgr,
                std::unique_ptr<DynamicDeviceMgr> remote_device_mgr);

  // The name of the session.
  const string session_name_;

  // The name of the worker. E.g., /job:mnist/replica:0/task:1.
  const string worker_name_;

  mutable mutex worker_session_state_mu_;
  // Object from which WorkerInterface instances can be obtained.
  std::shared_ptr<WorkerCacheInterface> worker_cache_
      TF_GUARDED_BY(worker_session_state_mu_);

  // graph_mgr keeps track of the registered graphs of this session.
  //
  // Note: graph_mgr must be deleted before rendezvous_mgr!
  // Note: graph_mgr must be deleted before device_mgr!
  const std::unique_ptr<GraphMgr> graph_mgr_;

  std::unique_ptr<ClusterFunctionLibraryRuntime> cluster_flr_;

  const std::unique_ptr<DeviceMgr> device_mgr_;
  DeviceMgr* const borrowed_device_mgr_;  // Not owned.
  std::unique_ptr<DynamicDeviceMgr> remote_device_mgr_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_SESSION_H_
