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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_

#include <functional>
#include <string>

#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_rpc_handler.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class WorkerCacheInterface;
struct WorkerEnv;

// SessionMgr keeps track of information related to a given session.
//
// SessionMgr runs on the workers.
//
// SessionMgr is threadsafe.
class SessionMgr {
 public:
  typedef std::function<absl::Status(const ServerDef&, WorkerCacheInterface**)>
      WorkerCacheFactory;

  explicit SessionMgr(
      WorkerEnv* worker_env, const std::string& default_worker_name,
      std::unique_ptr<WorkerCacheInterface> default_worker_cache,
      WorkerCacheFactory worker_cache_factory,
      tsl::CoordinationServiceRpcHandler* coordination_handler);
  ~SessionMgr() {}

  // Allocates state for a new session.
  absl::Status CreateSession(
      const std::string& session, const ServerDef& server_def,
      bool isolate_session_state,
      StatusCallback coordination_error_callback = [](absl::Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      });
  absl::Status CreateSession(
      const std::string& session, const ServerDef& server_def,
      const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
      bool isolate_session_state);

  // Create WorkerSession from the master with the given `master_task` and
  // `master_incarnation`. We first look for existing WorkerSessions associated
  // with the specified master task. If there are sessions created by the same
  // master but with a different incarnation, it indicates that the remote
  // master has restarted before deleting the sessions on worker. When it
  // happens, old sessions associated with the master will be automatically
  // removed before the new session is created.
  absl::Status CreateSession(
      const std::string& session, const ServerDef& server_def,
      const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
      bool isolate_session_state, std::string master_task,
      int64_t master_incarnation,
      StatusCallback coordination_error_callback = [](absl::Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      });

  void ResetDefaultWorkerCache(WorkerCacheInterface* worker_cache);

  // Updates state (worker cache, devices) of worker session identified by
  // session name (`session`) based on a new server_def and set of devices.
  absl::Status UpdateSession(const std::string& session,
                             const ServerDef& server_def,
                             const protobuf::RepeatedPtrField<DeviceAttributes>&
                                 cluster_device_attributes);

  // Locates the worker session for a given session handle
  absl::Status WorkerSessionForSession(
      const std::string& session_handle,
      std::shared_ptr<WorkerSession>* out_session);
  std::shared_ptr<WorkerSession> LegacySession();

  absl::Status DeleteSession(const std::string& session);

  // Deletes all existing sessions.
  absl::Status DeleteAllSessions();

  // Provides access to the coordination service agent. This method should only
  // be called after the agent has been initialized during session creation, or
  // an invalid nullptr is returned. Note: the agent is thread-safe and mutable.
  tsl::CoordinationServiceAgent* GetCoordinationServiceAgent();

  static std::string WorkerNameFromServerDef(const ServerDef& server_def);

  void SetLogging(bool active);

  void RetrieveLogs(int64_t step_id, LoggingResponse* response);

  void ClearLogs();

  // Agent should be torn down before service as it needs to disconnect first.
  void TeardownCoordinationServiceAgent();
  void TeardownCoordinationService();

 private:
  WorkerEnv* const worker_env_;  // Not owned.

  // A note about destruction:
  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  //
  // legacy_session_ owns the worker_env_.device_mgr, and so we must ensure
  // that sessions_'s WorkerSessions are deleted (which do not own the
  // underlying devices, but instead own RenamedDevices) before
  // legacy_session_ is deleted. Further, we must ensure that WorkerSession's
  // device_mgr is deleted after WorkerSession's graph_mgr.

  std::unique_ptr<WorkerCacheInterface> default_worker_cache_;
  std::shared_ptr<WorkerSession> legacy_session_;
  std::unique_ptr<tsl::CoordinationServiceInterface> coordination_service_;
  std::unique_ptr<tsl::CoordinationServiceAgent> coordination_service_agent_;

  bool is_logging_active_ = false;

  const WorkerCacheFactory worker_cache_factory_;

  // Not owned. And should only be used for setting the coordination service.
  tsl::CoordinationServiceRpcHandler* coordination_handler_ = nullptr;

  absl::Status WorkerSessionForSessionLocked(
      const std::string& session_handle,
      std::shared_ptr<WorkerSession>* out_session)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  // A map from session identifier to internal session structure.
  std::map<std::string, std::shared_ptr<WorkerSession>> sessions_
      TF_GUARDED_BY(mu_);

  // Incarnation and WorkerSession handle associated with a master task.
  struct MasterAssociatedSession {
    const int64_t master_incarnation;
    const std::string session_handle;
  };
  // A map from master task name to its associated worker sessions.
  std::unordered_multimap<std::string, MasterAssociatedSession>
      master_to_associated_sessions_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
