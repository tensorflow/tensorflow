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

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tsl/platform/stacktrace.h"

namespace tensorflow {

namespace {

auto* worker_session_created =
    monitoring::Gauge<bool, 0>::New("/tensorflow/core/worker_session_created",
                                    "True if a worker session was created.");

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

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    wrapped_->ListWorkersInJob(job_name, workers);
  }

  WorkerInterface* GetOrCreateWorker(const string& target) override {
    {
      // Fast path if worker has been created.
      tf_shared_lock l(mu_);
      auto p = workers_.find(target);
      if (p != workers_.end()) {
        return p->second.worker;
      }
    }
    {
      // Slow path if worker hasn't been created.
      mutex_lock l(mu_);
      auto p = workers_.find(target);
      if (p != workers_.end()) {
        return p->second.worker;
      }
      WorkerState state;
      state.worker = wrapped_->GetOrCreateWorker(target);
      if (state.worker != nullptr) {
        workers_.insert(std::make_pair(target, state));
      }
      return state.worker;
    }
  }

  absl::Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
    return wrapped_->GetEagerClientCache(eager_client_cache);
  }

  absl::Status GetCoordinationClientCache(
      std::unique_ptr<CoordinationClientCache>* coordination_client_cache)
      override {
    return wrapped_->GetCoordinationClientCache(coordination_client_cache);
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

  bool RetrieveLogs(int64_t step_id, StepStats* ss) override {
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
  std::unordered_map<string, WorkerState> workers_ TF_GUARDED_BY(mu_);
};

}  // namespace

WorkerSession::WorkerSession(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    std::unique_ptr<DeviceMgr> device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
    std::unique_ptr<DynamicDeviceMgr> remote_device_mgr,
    DistributedFunctionLibraryRuntimeCreator cluster_flr_creator)
    : session_name_(session_name),
      worker_name_(worker_name),
      worker_cache_(new WorkerFreeListCache(std::move(worker_cache))),
      graph_mgr_(std::move(graph_mgr)),
      cluster_flr_(cluster_flr_creator(
          this, !session_name.empty(),
          remote_device_mgr ? remote_device_mgr.get() : nullptr)),
      device_mgr_(std::move(device_mgr)),
      borrowed_device_mgr_(nullptr),
      remote_device_mgr_(std::move(remote_device_mgr)) {
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/compiler/xla/tsl/platform/default",
  // this is currently a no-op.
  worker_session_created->GetCell()->Set(true);
}

absl::Status WorkerSession::UpdateWorkerCacheAndDevices(
    std::unique_ptr<WorkerCacheInterface> new_worker_cache,
    std::vector<std::unique_ptr<Device>> added_remote_devices,
    const std::vector<Device*>& removed_remote_devices) {
  {
    mutex_lock l(worker_session_state_mu_);
    worker_cache_ = std::shared_ptr<WorkerCacheInterface>(
        new WorkerFreeListCache(std::move(new_worker_cache)));
  }
  TF_RETURN_IF_ERROR(remote_device_mgr_->RemoveDevices(removed_remote_devices));
  TF_RETURN_IF_ERROR(
      remote_device_mgr_->AddDevices(std::move(added_remote_devices)));
  return absl::OkStatus();
}

/* static */
std::shared_ptr<WorkerSession> WorkerSession::CreateWithBorrowedDeviceMgr(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
    std::unique_ptr<DynamicDeviceMgr> remote_device_mgr,
    DistributedFunctionLibraryRuntimeCreator cluster_flr_creator) {
  return std::shared_ptr<WorkerSession>(new WorkerSession(
      session_name, worker_name, std::move(worker_cache), borrowed_device_mgr,
      std::move(graph_mgr), std::move(remote_device_mgr),
      std::move(cluster_flr_creator)));
}

WorkerSession::WorkerSession(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
    std::unique_ptr<DynamicDeviceMgr> remote_device_mgr,
    DistributedFunctionLibraryRuntimeCreator cluster_flr_creator)
    : session_name_(session_name),
      worker_name_(worker_name),
      worker_cache_(new WorkerFreeListCache(std::move(worker_cache))),
      graph_mgr_(std::move(graph_mgr)),
      cluster_flr_(cluster_flr_creator(this, !session_name.empty(),
                                       remote_device_mgr.get())),
      device_mgr_(nullptr),
      borrowed_device_mgr_(borrowed_device_mgr),
      remote_device_mgr_(std::move(remote_device_mgr)) {
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/compiler/xla/tsl/platform/default",
  // this is currently a no-op.
  worker_session_created->GetCell()->Set(true);
}

WorkerSession::~WorkerSession() {
  VLOG(1) << "WorkerSession::~WorkerSession @@stacktrace\n "
          << tsl::CurrentStackTrace();
  if (graph_mgr_) {
    absl::Status s = graph_mgr_->DeregisterAll();
    if (!s.ok()) {
      LOG(WARNING) << "Error during worker session deletion: " << s;
    }
  }
}

}  // namespace tensorflow
