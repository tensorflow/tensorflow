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
#include "tensorflow/core/distributed_runtime/worker_session.h"

namespace tensorflow {

namespace {

// A private cache that wraps worker_cache and allows reuse of
// WorkerInterface objects.
class WorkerFreeListCache : public WorkerCacheInterface {
 public:
  explicit WorkerFreeListCache(std::unique_ptr<WorkerCacheInterface> w)
      : wrapped_(std::move(w)) {}

  ~WorkerFreeListCache() final {
    for (auto& p : workers_) {
      wrapped_->ReleaseWorker(p.first, p.second.worker);
    }
  }

  void ListWorkers(std::vector<string>* workers) const override {
    wrapped_->ListWorkers(workers);
  }

  WorkerInterface* CreateWorker(const string& target) override {
    mutex_lock l(mu_);
    auto p = workers_.find(target);
    if (p != workers_.end()) {
      return p->second.worker;
    }
    WorkerState state;
    state.worker = wrapped_->CreateWorker(target);
    if (state.worker != nullptr) {
      workers_.insert(std::make_pair(target, state));
    }
    return state.worker;
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
    // TODO(jeff,sanjay): Should decrement ref-count when we implement eviction.
  }

  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
    return wrapped_->GetDeviceLocalityNonBlocking(device, locality);
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
    wrapped_->GetDeviceLocalityAsync(device, locality, done);
  }

  void SetLogging(bool active) override { wrapped_->SetLogging(active); }

  void ClearLogs() override { wrapped_->ClearLogs(); }

  bool RetrieveLogs(int64 step_id, StepStats* ss) override {
    return wrapped_->RetrieveLogs(step_id, ss);
  }

 private:
  std::unique_ptr<WorkerCacheInterface> wrapped_;

  // Information kept per created WorkerInterface.
  struct WorkerState {
    WorkerInterface* worker;
    // TODO(jeff,sanjay): Add reference count if we support eviction.
  };

  // TODO(jeff,sanjay): Eviction when the map becomes too big.
  mutex mu_;
  std::unordered_map<string, WorkerState> workers_ GUARDED_BY(mu_);
};

}  // namespace

WorkerSession::WorkerSession(const string& session_name,
                             const string& worker_name,
                             std::unique_ptr<WorkerCacheInterface> worker_cache,
                             std::unique_ptr<DeviceMgr> device_mgr,
                             std::unique_ptr<GraphMgr> graph_mgr)
    : session_name(session_name),
      worker_name(worker_name),
      worker_cache(new WorkerFreeListCache(std::move(worker_cache))),
      graph_mgr(std::move(graph_mgr)),
      cluster_flr(
          new ClusterFunctionLibraryRuntime(this, !session_name.empty())),
      device_mgr_(std::move(device_mgr)),
      borrowed_device_mgr_(nullptr) {}

/* static */
std::shared_ptr<WorkerSession> WorkerSession::CreateWithBorrowedDeviceMgr(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr) {
  return std::shared_ptr<WorkerSession>(
      new WorkerSession(session_name, worker_name, std::move(worker_cache),
                        borrowed_device_mgr, std::move(graph_mgr)));
}

WorkerSession::WorkerSession(const string& session_name,
                             const string& worker_name,
                             std::unique_ptr<WorkerCacheInterface> worker_cache,
                             DeviceMgr* borrowed_device_mgr,
                             std::unique_ptr<GraphMgr> graph_mgr)
    : session_name(session_name),
      worker_name(worker_name),
      worker_cache(new WorkerFreeListCache(std::move(worker_cache))),
      graph_mgr(std::move(graph_mgr)),
      cluster_flr(
          new ClusterFunctionLibraryRuntime(this, !session_name.empty())),
      device_mgr_(nullptr),
      borrowed_device_mgr_(borrowed_device_mgr) {}

WorkerSession::~WorkerSession() {
  if (graph_mgr) {
    Status s = graph_mgr->DeregisterAll();
    if (!s.ok()) {
      LOG(WARNING) << "Error during worker session deletion: " << s;
    }
  }
}

}  // namespace tensorflow
