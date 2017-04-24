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

#include "tensorflow/core/distributed_runtime/session_mgr.h"

#include <utility>

#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

SessionMgr::SessionMgr(
    WorkerEnv* worker_env, const string& default_worker_name,
    std::unique_ptr<WorkerCacheInterface> default_worker_cache,
    std::unique_ptr<RendezvousMgrInterface> default_rendezvous_mgr,
    WorkerCacheFactory worker_cache_factory)
    : SessionMgr(
          worker_env, default_worker_name, std::move(default_worker_cache),
          default_rendezvous_mgr.release(), std::move(worker_cache_factory)) {}

SessionMgr::SessionMgr(
    WorkerEnv* worker_env, const string& default_worker_name,
    std::unique_ptr<WorkerCacheInterface> default_worker_cache,
    RendezvousMgrInterface* default_rendezvous_mgr,
    WorkerCacheFactory worker_cache_factory)
    : worker_env_(worker_env),
      legacy_session_(
          default_worker_name, std::move(default_worker_cache),
          std::unique_ptr<RendezvousMgrInterface>(default_rendezvous_mgr),
          std::unique_ptr<GraphMgr>(
              new GraphMgr(worker_env, default_rendezvous_mgr))),
      worker_cache_factory_(std::move(worker_cache_factory)) {}

string SessionMgr::WorkerNameFromServerDef(const ServerDef& server_def) {
  return strings::StrCat("/job:", server_def.job_name(),
                         "/replica:0/task:", server_def.task_index());
}

Status SessionMgr::CreateSession(const string& session,
                                 const ServerDef& server_def) {
  mutex_lock l(mu_);
  const string worker_name = WorkerNameFromServerDef(server_def);

  WorkerCacheInterface* worker_cache = nullptr;
  TF_RETURN_IF_ERROR(worker_cache_factory_(server_def, &worker_cache));

  std::unique_ptr<RendezvousMgrInterface> rendezvous_mgr(
      new RpcRendezvousMgr(worker_env_, worker_name, worker_cache));

  std::unique_ptr<GraphMgr> graph_mgr(
      new GraphMgr(worker_env_, rendezvous_mgr.get()));

  std::unique_ptr<WorkerSession> worker_session(new WorkerSession(
      worker_name, std::unique_ptr<WorkerCacheInterface>(worker_cache),
      std::move(rendezvous_mgr), std::move(graph_mgr)));

  sessions_.insert(std::make_pair(session, std::move(worker_session)));
  return Status::OK();
}

Status SessionMgr::DeleteSession(const string& session) {
  mutex_lock l(mu_);
  auto it = sessions_.find(session);
  if (it != sessions_.end()) {
    sessions_.erase(it);
  }
  std::set<string> graph_handles;
  for (auto graph_handle_it = sessions_by_graph_handle_.begin();
       graph_handle_it != sessions_by_graph_handle_.end(); ++graph_handle_it) {
    if (graph_handle_it->second == session) {
      graph_handles.insert(graph_handle_it->first);
      graph_handle_it = sessions_by_graph_handle_.erase(graph_handle_it);
      if (graph_handle_it == sessions_by_graph_handle_.end()) break;
    }
  }
  for (auto step_id_it = graphs_by_step_id_.begin();
       step_id_it != graphs_by_step_id_.end(); ++step_id_it) {
    if (graph_handles.find(step_id_it->second) != graph_handles.end()) {
      step_id_it = graphs_by_step_id_.erase(step_id_it);
      if (step_id_it == graphs_by_step_id_.end()) break;
    }
  }
  return Status::OK();
}

WorkerSession* SessionMgr::WorkerSessionForSessionUnlocked(
    const string& session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return &legacy_session_;
  } else {
    return it->second.get();
  }
}

WorkerSession* SessionMgr::WorkerSessionForSession(const string& session) {
  mutex_lock l(mu_);
  return WorkerSessionForSessionUnlocked(session);
}

WorkerSession* SessionMgr::LegacySession() { return &legacy_session_; }

WorkerSession* SessionMgr::WorkerSessionForGraphHandleUnlocked(
    const string& graph_handle) {
  auto it = sessions_by_graph_handle_.find(graph_handle);
  if (it == sessions_by_graph_handle_.end()) {
    return &legacy_session_;
  } else {
    return WorkerSessionForSessionUnlocked(it->second);
  }
}

WorkerSession* SessionMgr::WorkerSessionForGraphHandle(
    const string& graph_handle) {
  mutex_lock l(mu_);
  return WorkerSessionForGraphHandleUnlocked(graph_handle);
}

WorkerSession* SessionMgr::WorkerSessionForStepId(const int64 step_id) {
  mutex_lock l(mu_);
  auto it = graphs_by_step_id_.find(step_id);
  if (it == graphs_by_step_id_.end()) {
    return &legacy_session_;
  } else {
    return WorkerSessionForGraphHandleUnlocked(it->second);
  }
}

void SessionMgr::AssociateGraphWithSession(const string& session,
                                           const string& graph_handle) {
  mutex_lock l(mu_);
  sessions_by_graph_handle_[graph_handle] = session;
}

void SessionMgr::DisassociateGraphFromSession(const string& graph_handle) {
  mutex_lock l(mu_);
  auto it = sessions_by_graph_handle_.find(graph_handle);
  if (it != sessions_by_graph_handle_.end()) {
    sessions_by_graph_handle_.erase(it);
  }
}

void SessionMgr::AssociateStepIdWithGraph(const string& graph_handle,
                                          const int64 step_id) {
  mutex_lock l(mu_);
  graphs_by_step_id_[step_id] = graph_handle;
}

void SessionMgr::DisassociateStepIdFromGraph(const int64 step_id) {
  mutex_lock l(mu_);
  auto it = graphs_by_step_id_.find(step_id);
  if (it != graphs_by_step_id_.end()) {
    graphs_by_step_id_.erase(it);
  }
}

}  // namespace tensorflow
