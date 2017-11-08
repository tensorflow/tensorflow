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

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

SessionMgr::SessionMgr(
    WorkerEnv* worker_env, const string& default_worker_name,
    std::unique_ptr<WorkerCacheInterface> default_worker_cache,
    WorkerCacheFactory worker_cache_factory)
    : worker_env_(worker_env),
      legacy_session_("", default_worker_name, std::move(default_worker_cache),
                      std::unique_ptr<DeviceMgr>(worker_env->device_mgr),
                      std::unique_ptr<GraphMgr>(
                          new GraphMgr(worker_env, worker_env->device_mgr))),
      worker_cache_factory_(std::move(worker_cache_factory)) {}

string SessionMgr::WorkerNameFromServerDef(const ServerDef& server_def) {
  return strings::StrCat("/job:", server_def.job_name(),
                         "/replica:0/task:", server_def.task_index());
}

Status SessionMgr::CreateSession(const string& session,
                                 const ServerDef& server_def) {
  mutex_lock l(mu_);
  if (session.empty()) {
    return errors::InvalidArgument("Session must be non-empty.");
  }

  const string worker_name = WorkerNameFromServerDef(server_def);

  WorkerCacheInterface* worker_cache = nullptr;
  TF_RETURN_IF_ERROR(worker_cache_factory_(server_def, &worker_cache));

  std::vector<Device*> renamed_devices;
  for (Device* d : worker_env_->local_devices) {
    renamed_devices.push_back(
        RenamedDevice::NewRenamedDevice(worker_name, d, false));
  }
  std::unique_ptr<DeviceMgr> device_mgr(new DeviceMgr(renamed_devices));

  std::unique_ptr<GraphMgr> graph_mgr(
      new GraphMgr(worker_env_, device_mgr.get()));

  std::unique_ptr<WorkerSession> worker_session(new WorkerSession(
      session, worker_name, std::unique_ptr<WorkerCacheInterface>(worker_cache),
      std::move(device_mgr), std::move(graph_mgr)));

  sessions_.insert(std::make_pair(session, std::move(worker_session)));
  return Status::OK();
}

Status SessionMgr::DeleteSession(const string& session) {
  mutex_lock l(mu_);
  auto it = sessions_.find(session);
  if (it != sessions_.end()) {
    sessions_.erase(it);
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

}  // namespace tensorflow
