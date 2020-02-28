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

#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
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
  typedef std::function<Status(const ServerDef&, WorkerCacheInterface**)>
      WorkerCacheFactory;

  explicit SessionMgr(
      WorkerEnv* worker_env, const string& default_worker_name,
      std::unique_ptr<WorkerCacheInterface> default_worker_cache,
      WorkerCacheFactory worker_cache_factory);
  ~SessionMgr() {}

  // Allocates state for a new session.
  Status CreateSession(const string& session, const ServerDef& server_def,
                       bool isolate_session_state);
  Status CreateSession(
      const string& session, const ServerDef& server_def,
      const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
      bool isolate_session_state);

  // Updates state (worker cache, devices) of worker session identified by
  // session name (`session`) based on a new server_def and set of devices.
  Status UpdateSession(const string& session, const ServerDef& server_def,
                       const protobuf::RepeatedPtrField<DeviceAttributes>&
                           cluster_device_attributes,
                       bool isolate_session_state);

  // Locates the worker session for a given session handle
  Status WorkerSessionForSession(const string& session_handle,
                                 std::shared_ptr<WorkerSession>* out_session);
  std::shared_ptr<WorkerSession> LegacySession();

  Status DeleteSession(const string& session);

  static string WorkerNameFromServerDef(const ServerDef& server_def);

  void SetLogging(bool active);

  void RetrieveLogs(tensorflow::int64 step_id, LoggingResponse* response);

  void ClearLogs();

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

  bool is_logging_active_ = false;

  const WorkerCacheFactory worker_cache_factory_;

  Status WorkerSessionForSessionLocked(
      const string& session_handle, std::shared_ptr<WorkerSession>* out_session)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  // A map from session identifier to internal session structure.
  std::map<string, std::shared_ptr<WorkerSession>> sessions_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
