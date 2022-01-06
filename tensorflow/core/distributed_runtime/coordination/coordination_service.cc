/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

constexpr int kDefaultHeartbeatTimeoutMs = 10 * 1000;  // 10 seconds
constexpr char kHealthCheckThread[] = "CoordinationServiceHealthCheck";

std::string GetTaskName(const std::string& job_name, int task_id) {
  return strings::StrCat("/job:", job_name, "/replica:", 0, "/task:", task_id);
}

bool is_multi_client_leader(const ServerDef& server_def) {
  const auto& config = server_def.default_session_config();
  const std::string& leader =
      config.experimental().coordination_config().service_leader();
  const std::string& collective_leader =
      config.experimental().collective_group_leader();
  DeviceNameUtils::ParsedName leader_pn;
  if (!leader.empty()) {
    DeviceNameUtils::ParseFullName(leader, &leader_pn);
  } else if (!collective_leader.empty()) {
    LOG(INFO) << "No coordination leader is set, using the collective leader "
              << collective_leader;
    DeviceNameUtils::ParseFullName(collective_leader, &leader_pn);
  } else {
    LOG(INFO) << "No coordination leader is set, using the default /job:"
              << server_def.job_name() << "/replica:0/task:0";
    return server_def.task_index() == 0;
  }
  return server_def.job_name() == leader_pn.job &&
         server_def.task_index() == leader_pn.task;
}

// Standalone implementation of the coordination service.
class CoordinationServiceStandaloneImpl : public CoordinationServiceInterface {
 public:
  CoordinationServiceStandaloneImpl(
      std::unique_ptr<CoordinationClientCache> client_cache, Env* env,
      const ServerDef& server_def);
  ~CoordinationServiceStandaloneImpl() override { Stop(); }

  void RegisterWorker(const std::string& job_name, int task_id,
                      uint64 incarnation, StatusCallback done) override;
  void WaitForAllTasks(const std::string& job_name, int task_id,
                       std::vector<DeviceAttributes> devices,
                       StatusCallback done) override;
  Status RecordHeartbeat(const std::string& job_name, int task_id,
                         uint64 incarnation) override;
  Status ReportTaskError(const std::string& job_name, int task_id,
                         Status error) override;
  Status InsertKeyValue(const std::string& key,
                        const std::string& value) override;
  StatusOr<std::string> GetKeyValue(const std::string& key) override;
  void GetKeyValueAsync(const std::string& key,
                        StatusOrValueCallback done) override;
  Status DeleteKeyValue(const std::string& key) override;

 private:
  const std::vector<DeviceAttributes>& ListClusterDevices() override
      TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void StartCheckStaleness();
  void Stop();
  void PropagateError(const std::string& job, int task_id, Status error)
      TF_LOCKS_EXCLUDED(state_mu_);
  void DoneClusterRegistration(Status s) TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  class TaskState {
   public:
    // Task state maintained on the coordination service side.
    // State transition:
    //                Register           Heartbeat
    //   DISCONNECTED -------> CONNECTED --------> ERROR (timeout)
    //                              |   ReportError
    //                              +--------------> ERROR
    //                              |    Register
    //                              ---------------> RESTARTED
    //
    // When task state becomes ERROR or RESTARTED, propagate this status to
    // other CONNECTED tasks in the cluster.
    enum class State {
      DISCONNECTED,
      CONNECTED,
      ERROR,
      RESTARTED,
    };

    State GetState() { return state_; }
    Status GetStatus() { return status_; }
    void SetConnected(uint64 task_incarnation);
    void SetRegisteredCallback(StatusCallback cb);
    Status RecordHeartbeat(uint64 task_incarnation);
    int64 TimeSinceLastHeartbeatMs();
    void InvokeRegisteredCallback(Status s);
    void SetError(Status status);

   private:
    // Incarnation ID for CPU:0 on remote task.
    uint64 task_incarnation_ = 0;
    // WaitForAllTasks callback invoked when all tasks are registered. Must be
    // invoked exactly once.
    StatusCallback registered_callback_;
    std::atomic_bool is_callback_invoked_{true};

    State state_ = State::DISCONNECTED;
    Status status_;
    mutex last_heartbeat_mu_;
    int64 last_heartbeat_us_ TF_GUARDED_BY(last_heartbeat_mu_);
  };

  std::unique_ptr<CoordinationClientCache> client_cache_;
  Env& env_;
  const uint64 heartbeat_timeout_ms_;

  mutex state_mu_;
  condition_variable cluster_registered_cv_;
  absl::flat_hash_map<std::string, std::unique_ptr<TaskState>> cluster_state_
      TF_GUARDED_BY(state_mu_);
  std::vector<DeviceAttributes> cluster_devices_ TF_GUARDED_BY(state_mu_);
  int cluster_pending_workers_ TF_GUARDED_BY(state_mu_);

  mutex kv_mu_;
  // Ordered map to store config key-values
  std::map<std::string, std::string> kv_store_ TF_GUARDED_BY(kv_mu_);
  absl::flat_hash_map<std::string, std::vector<StatusOrValueCallback>> get_cb_
      TF_GUARDED_BY(kv_mu_);

  mutex check_staleness_thread_shutdown_mu_;
  condition_variable check_staleness_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(check_staleness_thread_shutdown_mu_) =
      false;
  std::unique_ptr<Thread> check_staleness_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(CoordinationServiceStandaloneImpl);
};

void CoordinationServiceStandaloneImpl::TaskState::SetConnected(
    uint64 task_incarnation) {
  state_ = State::CONNECTED;
  status_ = Status::OK();
  task_incarnation_ = task_incarnation;
  mutex_lock l(last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
}

void CoordinationServiceStandaloneImpl::TaskState::SetRegisteredCallback(
    StatusCallback cb) {
  is_callback_invoked_.store(false);
  registered_callback_ = cb;
}

void CoordinationServiceStandaloneImpl::TaskState::SetError(
    const Status status) {
  if (state_ == State::ERROR) return;
  state_ = State::ERROR;
  status_ = status;
}

Status CoordinationServiceStandaloneImpl::TaskState::RecordHeartbeat(
    uint64 task_incarnation) {
  if (!status_.ok()) return status_;
  if (task_incarnation != task_incarnation_) {
    return errors::Aborted("Incarnation ID mismatch: expecting ",
                           task_incarnation_, " but got ", task_incarnation,
                           ". This means the remote task has restarted.");
  }
  mutex_lock l(last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
  return Status::OK();
}

int64 CoordinationServiceStandaloneImpl::TaskState::TimeSinceLastHeartbeatMs() {
  mutex_lock l(last_heartbeat_mu_);
  return (Env::Default()->NowMicros() - last_heartbeat_us_) / 1000;
}

void CoordinationServiceStandaloneImpl::TaskState::InvokeRegisteredCallback(
    Status s) {
  if (!is_callback_invoked_.exchange(true, std::memory_order_acq_rel)) {
    registered_callback_(s);
    mutex_lock l(last_heartbeat_mu_);
    last_heartbeat_us_ = Env::Default()->NowMicros();
  }
}

CoordinationServiceStandaloneImpl::CoordinationServiceStandaloneImpl(
    std::unique_ptr<CoordinationClientCache> client_cache, Env* env,
    const ServerDef& server_def)
    : client_cache_(std::move(client_cache)),
      env_(*env),
      heartbeat_timeout_ms_([&server_def]() -> uint64 {
        const auto& configs = server_def.default_session_config()
                                  .experimental()
                                  .coordination_config();
        return configs.heartbeat_timeout_in_ms() > 0
                   ? configs.heartbeat_timeout_in_ms()
                   : kDefaultHeartbeatTimeoutMs;
      }()) {
  const auto& configs =
      server_def.default_session_config().experimental().coordination_config();
  const std::unordered_set<std::string> coordinated_jobs(
      configs.coordinated_jobs().cbegin(), configs.coordinated_jobs().cend());
  const auto& cluster_def = server_def.cluster();
  for (const auto& job : cluster_def.job()) {
    // If `coordinated_jobs` is specified, skip jobs that are not included there
    if (!coordinated_jobs.empty() &&
        coordinated_jobs.find(job.name()) == coordinated_jobs.end()) {
      continue;
    }
    for (const auto& task : job.tasks()) {
      const std::string& task_name = GetTaskName(job.name(), task.first);
      cluster_state_.emplace(task_name, std::make_unique<TaskState>());
    }
  }
  cluster_pending_workers_ = cluster_state_.size();
  StartCheckStaleness();
}

void CoordinationServiceStandaloneImpl::StartCheckStaleness() {
  check_staleness_thread_.reset(
      env_.StartThread({}, kHealthCheckThread, [this]() {
        // Used to store the job and task info if a task becomes stale
        DeviceNameUtils::ParsedName parsed;
        while (true) {
          {
            mutex_lock l(check_staleness_thread_shutdown_mu_);
            check_staleness_thread_cv_.wait_for(l, std::chrono::seconds(1));
            if (shutting_down_) {
              return;
            }
          }
          Status status = Status::OK();
          {
            mutex_lock l(state_mu_);
            for (const auto& worker_state : cluster_state_) {
              // Skip workers that are not registered or in error state
              if (worker_state.second->GetState() !=
                  TaskState::State::CONNECTED) {
                continue;
              }
              const bool is_stale =
                  worker_state.second->TimeSinceLastHeartbeatMs() >
                  heartbeat_timeout_ms_;
              VLOG(1) << "Checking staleness for " << worker_state.first
                      << " stale?=" << is_stale;
              if (is_stale) {
                status = errors::Unavailable(
                    "Task ", worker_state.first,
                    " heartbeat timeout. This indicates that the remote task "
                    "has failed, got preempted, or crashed unexpectedly.");
                worker_state.second->SetError(status);
                DeviceNameUtils::ParseFullName(worker_state.first, &parsed);
                break;
              }
            }
          }
          if (!status.ok()) {
            PropagateError(parsed.job, parsed.task, status);
          }
        }
      }));
}

void CoordinationServiceStandaloneImpl::Stop() {
  {
    mutex_lock l(kv_mu_);
    get_cb_.clear();
  }
  {
    mutex_lock l(state_mu_);
    cluster_state_.clear();
  }
  {
    mutex_lock l(check_staleness_thread_shutdown_mu_);
    shutting_down_ = true;
    check_staleness_thread_cv_.notify_all();
    cluster_registered_cv_.notify_all();
  }
  check_staleness_thread_.reset();
}

void CoordinationServiceStandaloneImpl::RegisterWorker(
    const std::string& job_name, int task_id, uint64 incarnation,
    StatusCallback done) {
  const std::string& task_name = GetTaskName(job_name, task_id);

  Status status;
  {
    mutex_lock l(state_mu_);
    if (!cluster_state_.contains(task_name)) {
      done(errors::InvalidArgument(
          "Unexpected worker registered with job_name=", job_name,
          ", task_id=", task_id));
      // Note: unexpected task register should not be propagated to other tasks
      return;
    } else if (cluster_state_[task_name]->GetState() ==
               TaskState::State::CONNECTED) {
      Status s = errors::Aborted("Duplicate worker registration with job_name=",
                                 job_name, ", task_id=", task_id);
      cluster_state_[task_name]->SetError(s);
      status = s;
      DoneClusterRegistration(s);
    } else {
      // Hit this path when the task is registering itself for the first time,
      // or it's already in ERROR state and now register again. In both cases,
      // the service allows it to be registered.
      cluster_state_[task_name]->SetConnected(incarnation);
    }
  }
  if (!status.ok()) PropagateError(job_name, task_id, status);
  done(status);
}

void CoordinationServiceStandaloneImpl::WaitForAllTasks(
    const std::string& job_name, int task_id,
    std::vector<DeviceAttributes> devices, StatusCallback done) {
  const std::string& task_name = GetTaskName(job_name, task_id);
  mutex_lock l(state_mu_);
  if (!cluster_state_.contains(task_name)) {
    done(errors::InvalidArgument("Unexpected worker request with job_name=",
                                 job_name, ", task_id=", task_id));
    return;
  }
  DCHECK_GT(cluster_pending_workers_, 0);
  cluster_state_[task_name]->SetRegisteredCallback(std::move(done));
  cluster_devices_.insert(cluster_devices_.end(),
                          std::make_move_iterator(devices.begin()),
                          std::make_move_iterator(devices.end()));
  cluster_pending_workers_--;
  if (cluster_pending_workers_ == 0) {
    DoneClusterRegistration(Status::OK());
  }
}

const std::vector<DeviceAttributes>&
CoordinationServiceStandaloneImpl::ListClusterDevices() {
  return cluster_devices_;
}

void CoordinationServiceStandaloneImpl::DoneClusterRegistration(Status s) {
  for (const auto& task_state : cluster_state_) {
    if (task_state.second != nullptr) {
      task_state.second->InvokeRegisteredCallback(s);
    }
  }
  cluster_registered_cv_.notify_all();
}

Status CoordinationServiceStandaloneImpl::ReportTaskError(
    const std::string& job_name, int task_id, Status error) {
  const std::string& task_name = GetTaskName(job_name, task_id);
  {
    mutex_lock l(state_mu_);
    if (!cluster_state_.contains(task_name)) {
      return errors::InvalidArgument("Unexpected worker request with job_name=",
                                     job_name, ", task_id=", task_id);
    } else if (cluster_state_[task_name]->GetState() !=
               TaskState::State::CONNECTED) {
      return errors::FailedPrecondition(
          "The task is not connected or already has an error.");
    } else {
      cluster_state_[task_name]->SetError(error);
    }
  }
  PropagateError(job_name, task_id, error);
  return Status::OK();
}

Status CoordinationServiceStandaloneImpl::RecordHeartbeat(
    const std::string& job_name, int task_id, uint64 incarnation) {
  const std::string& task_name = GetTaskName(job_name, task_id);
  Status s = Status::OK();
  {
    mutex_lock l(state_mu_);
    if (!cluster_state_.contains(task_name)) {
      return errors::InvalidArgument(
          "Unexpected worker heartbeat with job_name=", job_name,
          ", task_id=", task_id);
    } else if (!cluster_state_[task_name]->GetStatus().ok()) {
      return cluster_state_[task_name]->GetStatus();
    } else if (cluster_state_[task_name]->GetState() ==
               TaskState::State::DISCONNECTED) {
      return errors::InvalidArgument(
          "Task with job_name=", job_name, ", task_id=", task_id,
          " must be registered before sending heartbeat messages");
    }
    s = cluster_state_[task_name]->RecordHeartbeat(incarnation);
  }
  if (!s.ok()) {
    PropagateError(job_name, task_id, s);
  }
  return s;
}

void CoordinationServiceStandaloneImpl::PropagateError(
    const std::string& job_name, int task_id, Status error) {
  assert(!error.ok());
  ReportErrorToAgentRequest request;
  request.set_source_job(job_name);
  request.set_source_task(task_id);
  request.set_error_code(error.code());
  request.set_error_message(error.error_message());
  std::vector<std::shared_ptr<Notification>> notifications;

  std::vector<absl::string_view> task_names;
  {
    tf_shared_lock l(state_mu_);
    task_names.reserve(cluster_state_.size());
    for (const auto& pair : cluster_state_) {
      task_names.emplace_back(pair.first);
    }
  }
  for (absl::string_view task : task_names) {
    {
      mutex_lock l(state_mu_);
      // Propagate error only to workers that are connected
      if (cluster_state_[task]->GetState() != TaskState::State::CONNECTED)
        continue;
    }

    CoordinationClient* client = client_cache_->GetClient(std::string(task));
    auto response = std::make_shared<ReportErrorToAgentResponse>();
    auto n = std::make_shared<Notification>();
    client->ReportErrorToAgentAsync(
        &request, response.get(), [response, n, task](Status s) {
          if (!s.ok()) {
            LOG(ERROR) << "Encountered another error while reporting to "
                       << task << ": " << s;
          }
          n->Notify();
        });
    notifications.push_back(n);
  }
  for (auto& n : notifications) {
    n->WaitForNotification();
  }
}

// Utility for normalizing structured config key string.
// The normalized key will not have leading or trailing slashes, and all parts
// in the key path are separated by exactly one slack ('/').
// E.g., ///a//b/c// --> a/b/c
std::string NormalizeKey(const StringPiece orig_key) {
  std::string norm_key = std::string(orig_key);
  const char* src = norm_key.c_str();
  std::string::iterator dst = norm_key.begin();

  // Parse all characters
  while (*src) {
    // Skip leading slashes
    while (*src == '/') src++;
    // Copy over all non-slash characters
    while (*src && *src != '/') {
      *dst++ = *src++;
    }
    // Allow one slash at the end of current directory
    if (*src) {
      *dst++ = *src++;
    }
  }
  // If ending with slash, remove the trailing slash
  if (dst > norm_key.begin() && *(dst - 1) == '/') dst--;
  norm_key.resize(dst - norm_key.begin());
  return norm_key;
}

Status CoordinationServiceStandaloneImpl::InsertKeyValue(
    const std::string& key, const std::string& value) {
  const std::string& norm_key = NormalizeKey(key);
  mutex_lock l(kv_mu_);
  if (kv_store_.find(norm_key) != kv_store_.end()) {
    return errors::InvalidArgument("Config key ", key, " already exists.");
  }
  kv_store_.emplace(norm_key, value);
  auto iter = get_cb_.find(norm_key);
  if (iter != get_cb_.end()) {
    for (const auto& cb : iter->second) {
      cb(value);
    }
    get_cb_.erase(iter);
  }
  return Status::OK();
}

StatusOr<std::string> CoordinationServiceStandaloneImpl::GetKeyValue(
    const std::string& key) {
  absl::Notification n;
  StatusOr<std::string> result;
  GetKeyValueAsync(key, [&](const StatusOr<std::string>& status_or_value) {
    result = status_or_value;
    n.Notify();
  });
  n.WaitForNotification();
  return result;
}

void CoordinationServiceStandaloneImpl::GetKeyValueAsync(
    const std::string& key, StatusOrValueCallback done) {
  const std::string& norm_key = NormalizeKey(key);
  mutex_lock l(kv_mu_);
  const auto& iter = kv_store_.find(norm_key);
  if (iter != kv_store_.end()) {
    done(iter->second);
    return;
  }
  auto cb_iter = get_cb_.find(norm_key);
  if (cb_iter == get_cb_.end()) {
    cb_iter =
        get_cb_.emplace(norm_key, std::vector<StatusOrValueCallback>()).first;
  }
  cb_iter->second.emplace_back(std::move(done));
}

Status CoordinationServiceStandaloneImpl::DeleteKeyValue(
    const std::string& key) {
  const std::string& norm_key = NormalizeKey(key);
  mutex_lock l(kv_mu_);
  // Delete directory: find key range that match directory prefix
  const std::string& dir = strings::StrCat(norm_key, "/");
  auto begin = kv_store_.lower_bound(dir);
  std::map<std::string, std::string>::iterator end;
  for (end = begin; end != kv_store_.end(); end++) {
    if (std::mismatch(dir.begin(), dir.end(), end->first.begin()).first !=
        dir.end())
      break;
  }
  kv_store_.erase(begin, end);
  auto iter = kv_store_.find(norm_key);
  if (iter != kv_store_.end()) {
    kv_store_.erase(iter);
  }
  return Status::OK();
}

}  // namespace

std::unique_ptr<CoordinationServiceInterface> EnableCoordinationService(
    Env* env, const ServerDef& server_def,
    std::unique_ptr<CoordinationClientCache> cache) {
  std::unique_ptr<CoordinationServiceInterface> coord_service;
  if (is_multi_client_leader(server_def)) {
    coord_service = std::make_unique<CoordinationServiceStandaloneImpl>(
        std::move(cache), env, server_def);
  }
  return coord_service;
}

// Register standalone coordination service implementation.
REGISTER_COORDINATION_SERVICE("standalone", EnableCoordinationService);

}  // namespace tensorflow
