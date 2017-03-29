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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_

#include <functional>

#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

class WorkerCacheInterface;
struct WorkerEnv;

// SessionMgr keeps track of information related to a given session.
//
// SessionMgr is threadsafe.
class SessionMgr {
 public:
  typedef std::function<Status(const ServerDef&, WorkerCacheInterface**)>
      WorkerCacheFactory;

  explicit SessionMgr(
      WorkerEnv* worker_env, const string& default_worker_name,
      std::unique_ptr<WorkerCacheInterface> default_worker_cache,
      std::unique_ptr<RendezvousMgrInterface> default_rendezvous_mgr,
      WorkerCacheFactory worker_cache_factory);
  ~SessionMgr() {}

  // Allocates state for a new session.
  Status CreateSession(const string& session, const ServerDef& server_def);

  // Locates the worker session for a given session handle
  WorkerSession* WorkerSessionForSession(const string& session);
  WorkerSession* LegacySession();

  // Locates the worker session for a given graph handle
  WorkerSession* WorkerSessionForGraphHandle(const string& graph_handle);
  void AssociateGraphWithSession(const string& session,
                                 const string& graph_handle);
  void DisassociateGraphFromSession(const string& graph_handle);

  // Locates a worker session for a given step id
  WorkerSession* WorkerSessionForStepId(const int64 step_id);
  void AssociateStepIdWithGraph(const string& graph_handle,
                                const int64 step_id);
  void DisassociateStepIdFromGraph(const int64 step_id);

  Status DeleteSession(const string& session);

  static string WorkerNameFromServerDef(const ServerDef& server_def);

 private:
  // Private constructor to work around std::unique_ptr ownership issues.
  explicit SessionMgr(
      WorkerEnv* worker_env, const string& default_worker_name,
      std::unique_ptr<WorkerCacheInterface> default_worker_cache,
      RendezvousMgrInterface* default_rendezvous_mgr,
      WorkerCacheFactory worker_cache_factory);

  const WorkerEnv* const worker_env_;  // Not owned.
  WorkerSession legacy_session_;

  const WorkerCacheFactory worker_cache_factory_;

  WorkerSession* WorkerSessionForSessionUnlocked(const string& session)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);
  WorkerSession* WorkerSessionForGraphHandleUnlocked(const string& graph_handle)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  // A map from session identifier to internal session structure.
  std::map<string, std::unique_ptr<WorkerSession>> sessions_ GUARDED_BY(mu_);

  // A map from graph handles to the session that they belong to.
  std::map<string, string> sessions_by_graph_handle_ GUARDED_BY(mu_);

  // A map from globally-unique step id's to the corresponding graph handles.
  std::map<int64, string> graphs_by_step_id_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
