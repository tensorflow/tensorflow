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
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

SessionMgr::SessionMgr(
    WorkerEnv* worker_env, const string& default_worker_name,
    std::unique_ptr<WorkerCacheInterface> default_worker_cache,
    WorkerCacheFactory worker_cache_factory)
    : worker_env_(worker_env),
      default_worker_cache_(std::move(default_worker_cache)),
      legacy_session_(new WorkerSession(
          "", default_worker_name,
          std::unique_ptr<WorkerCacheInterface>(
              new WorkerCacheWrapper(default_worker_cache_.get())),
          std::unique_ptr<DeviceMgr>(worker_env->device_mgr),
          std::unique_ptr<GraphMgr>(
              new GraphMgr(worker_env, worker_env->device_mgr)))),
      worker_cache_factory_(std::move(worker_cache_factory)) {}

string SessionMgr::WorkerNameFromServerDef(const ServerDef& server_def) {
  return strings::StrCat("/job:", server_def.job_name(), "/replica:0/task:",
                         server_def.task_index());
}

Status SessionMgr::CreateSession(const string& session,
                                 const ServerDef& server_def,
                                 bool isolate_session_state) {
  mutex_lock l(mu_);
  if (session.empty()) {
    return errors::InvalidArgument("Session must be non-empty.");
  }

  const string worker_name = WorkerNameFromServerDef(server_def);

  WorkerCacheInterface* worker_cache = nullptr;
  if (server_def.cluster().job().empty()) {
    worker_cache = new WorkerCacheWrapper(default_worker_cache_.get());
  } else {
    TF_RETURN_IF_ERROR(worker_cache_factory_(server_def, &worker_cache));
  }

  if (worker_cache != nullptr & default_worker_cache_.get() != nullptr) {
    worker_cache->SetLogging(this->is_logging_active_);
  }

  CHECK(!worker_env_->local_devices.empty())
      << "The WorkerEnv must have at least one device in `local_devices`.";

  std::vector<Device*> renamed_devices;
  for (Device* d : worker_env_->local_devices) {
    renamed_devices.push_back(RenamedDevice::NewRenamedDevice(
        worker_name, d, false, isolate_session_state));
  }
  std::unique_ptr<DeviceMgr> device_mgr(new DeviceMgr(renamed_devices));

  std::unique_ptr<GraphMgr> graph_mgr(
      new GraphMgr(worker_env_, device_mgr.get()));

  std::shared_ptr<WorkerSession> worker_session(new WorkerSession(
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

std::shared_ptr<WorkerSession> SessionMgr::WorkerSessionForSessionUnlocked(
    const string& session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return legacy_session_;
  } else {
    return it->second;
  }
}

std::shared_ptr<WorkerSession> SessionMgr::WorkerSessionForSession(
    const string& session) {
  mutex_lock l(mu_);
  return WorkerSessionForSessionUnlocked(session);
}

std::shared_ptr<WorkerSession> SessionMgr::LegacySession() {
  return legacy_session_;
}

void SessionMgr::SetLogging(bool active) {
  mutex_lock l(mu_);
  this->is_logging_active_ = active;
  // Legacy Session
  if (legacy_session_) {
    auto* worker_cache = legacy_session_->worker_cache.get();
    if (worker_cache) {
      worker_cache->SetLogging(active);
    }
  }

  for (const auto& session_kv : sessions_) {
    auto session = session_kv.second.get();
    if (session) {
      auto* worker_cache = session->worker_cache.get();
      if (worker_cache) {
        worker_cache->SetLogging(active);
      }
    }
  }
}

void SessionMgr::RetrieveLogs(tensorflow::int64 step_id,
                              LoggingResponse* response) {
  mutex_lock l(mu_);
  // Legacy Session
  if (legacy_session_) {
    auto* worker_cache = legacy_session_->worker_cache.get();
    if (worker_cache) {
      auto step_stats = StepStats();
      if (worker_cache->RetrieveLogs(step_id, &step_stats)) {
        auto* labeled_step_stats = response->add_step();
        labeled_step_stats->set_step_id(step_id);
        labeled_step_stats->mutable_step_stats()->Swap(&step_stats);
      }
    }
  }
  for (const auto& session_kv : sessions_) {
    auto session = session_kv.second.get();
    if (session) {
      auto* worker_cache = session->worker_cache.get();
      if (worker_cache) {
        auto step_stats = StepStats();
        if (worker_cache->RetrieveLogs(step_id, &step_stats)) {
          auto* labeled_step_stats = response->add_step();
          labeled_step_stats->set_step_id(step_id);
          labeled_step_stats->mutable_step_stats()->Swap(&step_stats);
        }
      }
    }
  }
}

void SessionMgr::ClearLogs() {
  mutex_lock l(mu_);
  // Legacy Session
  if (legacy_session_) {
    auto* worker_cache = legacy_session_->worker_cache.get();
    if (worker_cache) {
      worker_cache->ClearLogs();
    }
  }

  for (const auto& session_kv : sessions_) {
    auto session = session_kv.second.get();
    if (session) {
      auto* worker_cache = session->worker_cache.get();
      if (worker_cache) {
        worker_cache->ClearLogs();
      }
    }
  }
}
}  // namespace tensorflow
