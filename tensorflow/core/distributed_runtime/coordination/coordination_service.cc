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
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.pb.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

constexpr int kDefaultHeartbeatTimeoutMs = 10 * 1000;  // 10 seconds
constexpr int kServiceToClientTimeoutMs = 10 * 1000;   // 10 seconds
constexpr size_t kOngoingBarriersSoftLimit = 20;
constexpr char kHealthCheckThread[] = "CoordinationServiceHealthCheck";

std::string GetTaskName(absl::string_view job_name, int task_id) {
  return strings::StrCat("/job:", job_name, "/replica:", 0, "/task:", task_id);
}

std::string GetTaskName(const CoordinatedTask& task) {
  return GetTaskName(task.job_name(), task.task_id());
}

CoordinatedTask GetTaskFromName(absl::string_view task_name) {
  DeviceNameUtils::ParsedName parsed;
  DeviceNameUtils::ParseFullName(task_name, &parsed);
  CoordinatedTask task;
  task.set_job_name(parsed.job);
  task.set_task_id(parsed.task);
  return task;
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

// Convenience structs to allow using CoordinatedTask as container keys.
struct CoordinatedTaskHash {
  uint64_t operator()(const CoordinatedTask& task) const {
    return absl::HashOf(task.job_name(), task.task_id());
  }
};
struct CoordinatedTaskEqual {
  bool operator()(const CoordinatedTask& lhs,
                  const CoordinatedTask& rhs) const {
    return lhs.job_name() == rhs.job_name() && lhs.task_id() == rhs.task_id();
  }
};

// Standalone implementation of the coordination service.
class CoordinationServiceStandaloneImpl : public CoordinationServiceInterface {
 public:
  CoordinationServiceStandaloneImpl(
      std::unique_ptr<CoordinationClientCache> client_cache, Env* env,
      const ServerDef& server_def);
  ~CoordinationServiceStandaloneImpl() override { Stop(); }

  Status RegisterTask(const CoordinatedTask& task,
                      uint64_t incarnation) override;
  void WaitForAllTasks(const CoordinatedTask& task,
                       const CoordinationServiceDeviceInfo& devices,
                       StatusCallback done) override;
  void ShutdownTaskAsync(const CoordinatedTask& task,
                         StatusCallback done) override;
  Status ResetTask(const CoordinatedTask& task) override;
  Status RecordHeartbeat(const CoordinatedTask& task,
                         uint64_t incarnation) override;
  Status ReportTaskError(const CoordinatedTask& task, Status error) override;
  Status InsertKeyValue(const std::string& key,
                        const std::string& value) override;
  StatusOr<std::string> GetKeyValue(const std::string& key) override;
  void GetKeyValueAsync(const std::string& key,
                        StatusOrValueCallback done) override;
  std::vector<KeyValueEntry> GetKeyValueDir(
      absl::string_view directory_key) override;
  Status DeleteKeyValue(const std::string& key) override;
  void BarrierAsync(const std::string& barrier_id, absl::Duration timeout,
                    const CoordinatedTask& task,
                    const std::vector<CoordinatedTask>& participating_tasks,
                    StatusCallback done) override;
  Status CancelBarrier(const std::string& barrier_id,
                       const CoordinatedTask& task) override;

 private:
  const CoordinationServiceDeviceInfo& ListClusterDevices() override
      TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  uint64_t GetServiceIncarnation() override;
  void StartCheckStaleness();  // Checks both heartbeat and barrier timeouts.
  void Stop(bool shut_staleness_thread = true);
  // Report service error to a specified task.
  void ReportServiceErrorToTaskAsync(const CoordinatedTask& destination_task,
                                     Status error);
  // Report error from a task to all other connected tasks.
  // Note: SetTaskError() must be called before propagating its error.
  void PropagateError(const CoordinatedTask& source_task,
                      bool is_reported_by_task = false)
      TF_LOCKS_EXCLUDED(state_mu_);
  void SetTaskError(absl::string_view task_name, Status error)
      TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void SetXlaGlobalDeviceIds() TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  Status DisconnectTask(const CoordinatedTask& task)
      TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  struct BarrierState {
    bool passed = false;
    Status result = errors::Unknown(
        "Invalid barrier result.");  // Only valid if `passed` is true.
    uint64_t deadline_in_micros = 0;
    int num_pending_tasks = 0;
    // Specifies which tasks have called the barrier so far.
    absl::flat_hash_map<CoordinatedTask, bool, CoordinatedTaskHash,
                        CoordinatedTaskEqual>
        tasks_at_barrier;
    std::vector<StatusCallback> done_callbacks;
  };
  void PassBarrier(absl::string_view barrier_id, Status result,
                   BarrierState* barrier)
      TF_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Check if participating tasks are specified correctly across barrier calls.
  bool ValidateTaskArgs(
      const std::vector<CoordinatedTask>& tasks_args,
      const absl::flat_hash_map<CoordinatedTask, bool, CoordinatedTaskHash,
                                CoordinatedTaskEqual>& tasks_at_barrier,
      int64_t cluster_size);

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
    void SetConnected(uint64_t task_incarnation);
    void Disconnect(uint64_t grace_period_duration_us);
    Status RecordHeartbeat(uint64_t task_incarnation);
    int64_t TimeSinceLastHeartbeatMs();
    // This denotes the deadline after which we stop accepting heartbeats from a
    // disconnected task. This grace period accounts for the lag time between
    // the service recording the state change and the agent stopping heartbeats.
    uint64_t GetDisconnectedGracePeriodMicros();
    void SetError(Status status);
    absl::flat_hash_set<std::string> GetOngoingBarriers();
    void JoinBarrier(absl::string_view barrier_id);
    void ExitBarrier(absl::string_view barrier_id);

   private:
    // Incarnation ID for CPU:0 on remote task.
    uint64_t task_incarnation_ = 0;

    State state_ = State::DISCONNECTED;
    Status status_;
    mutex last_heartbeat_mu_;
    uint64_t last_heartbeat_us_ TF_GUARDED_BY(last_heartbeat_mu_);
    // This denotes the deadline after which we stop accepting heartbeats from a
    // disconnected task. This grace period accounts for the lag time between
    // the service recording the state change and the agent stopping heartbeats.
    uint64_t disconnect_grace_period_us_ = 0;
    // For now, we assume there won't be many simultaneous barriers so we simply
    // use a set.
    absl::flat_hash_set<std::string> ongoing_barriers_for_task_;
  };

  std::unique_ptr<CoordinationClientCache> client_cache_;
  Env& env_;
  const uint64_t service_incarnation_ = random::New64();
  const uint64_t heartbeat_timeout_ms_;
  const absl::Duration shutdown_barrier_timeout_;

  const std::string device_propagation_barrier_id_ =
      absl::StrCat("WaitForAllTasks::", std::to_string(service_incarnation_));
  const std::string shutdown_barrier_id_ =
      absl::StrCat("Shutdown::", std::to_string(service_incarnation_));

  mutex state_mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<TaskState>> cluster_state_
      TF_GUARDED_BY(state_mu_);
  CoordinationServiceDeviceInfo cluster_devices_ TF_GUARDED_BY(state_mu_);

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

  absl::flat_hash_map<std::string, BarrierState> barriers_
      TF_GUARDED_BY(state_mu_);
  // For now, we assume there won't be many simultaneous barriers so we simply
  // use a set.
  absl::flat_hash_set<std::string> ongoing_barriers_ TF_GUARDED_BY(state_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CoordinationServiceStandaloneImpl);
};

void CoordinationServiceStandaloneImpl::TaskState::SetConnected(
    uint64_t task_incarnation) {
  state_ = State::CONNECTED;
  status_ = Status::OK();
  task_incarnation_ = task_incarnation;
  mutex_lock l(last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
}

void CoordinationServiceStandaloneImpl::TaskState::Disconnect(
    uint64_t grace_period_duration_us) {
  disconnect_grace_period_us_ =
      Env::Default()->NowMicros() + grace_period_duration_us;
  state_ = State::DISCONNECTED;
  status_ = Status::OK();
}

void CoordinationServiceStandaloneImpl::TaskState::SetError(
    const Status status) {
  if (state_ == State::ERROR) return;
  state_ = State::ERROR;
  status_ = status;
}

Status CoordinationServiceStandaloneImpl::TaskState::RecordHeartbeat(
    uint64_t task_incarnation) {
  if (!status_.ok()) return status_;
  if (task_incarnation != task_incarnation_) {
    return MakeCoordinationError(errors::Aborted(
        "Incarnation ID mismatch: expecting ", task_incarnation_, " but got ",
        task_incarnation, ". This means the remote task has restarted."));
  }
  mutex_lock l(last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
  return Status::OK();
}

int64_t
CoordinationServiceStandaloneImpl::TaskState::TimeSinceLastHeartbeatMs() {
  mutex_lock l(last_heartbeat_mu_);
  return (Env::Default()->NowMicros() - last_heartbeat_us_) / 1000;
}

uint64_t CoordinationServiceStandaloneImpl::TaskState::
    GetDisconnectedGracePeriodMicros() {
  return disconnect_grace_period_us_;
}

absl::flat_hash_set<std::string>
CoordinationServiceStandaloneImpl::TaskState::GetOngoingBarriers() {
  return ongoing_barriers_for_task_;
}

void CoordinationServiceStandaloneImpl::TaskState::JoinBarrier(
    absl::string_view barrier_id) {
  ongoing_barriers_for_task_.emplace(barrier_id);
}

void CoordinationServiceStandaloneImpl::TaskState::ExitBarrier(
    absl::string_view barrier_id) {
  ongoing_barriers_for_task_.erase(barrier_id);
}
CoordinationServiceStandaloneImpl::CoordinationServiceStandaloneImpl(
    std::unique_ptr<CoordinationClientCache> client_cache, Env* env,
    const ServerDef& server_def)
    : client_cache_(std::move(client_cache)),
      env_(*env),
      heartbeat_timeout_ms_([&server_def]() -> uint64_t {
        const auto& configs = server_def.default_session_config()
                                  .experimental()
                                  .coordination_config();
        return configs.heartbeat_timeout_in_ms() > 0
                   ? configs.heartbeat_timeout_in_ms()
                   : kDefaultHeartbeatTimeoutMs;
      }()),
      shutdown_barrier_timeout_(
          absl::Milliseconds(server_def.default_session_config()
                                 .experimental()
                                 .coordination_config()
                                 .shutdown_barrier_timeout_in_ms())) {
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
  StartCheckStaleness();
}

// Checks both heartbeat and barrier timeouts in the same thread, since threads
// are a constrained resource.
void CoordinationServiceStandaloneImpl::StartCheckStaleness() {
  check_staleness_thread_.reset(
      env_.StartThread({}, kHealthCheckThread, [this]() {
        const bool has_service_to_client_connection = client_cache_ != nullptr;
        // Used to store stale tasks and barriers.
        std::vector<absl::string_view> stale_task_names;
        absl::flat_hash_map<std::string, BarrierState*> expired_barriers;
        while (true) {
          {
            mutex_lock l(check_staleness_thread_shutdown_mu_);
            check_staleness_thread_cv_.wait_for(l, std::chrono::seconds(1));
            if (shutting_down_) {
              return;
            }
          }
          // Heartbeat check.
          Status status = Status::OK();
          {
            mutex_lock l(state_mu_);
            for (const auto& task_state : cluster_state_) {
              // Skip tasks that are not registered or in error state
              if (task_state.second->GetState() !=
                  TaskState::State::CONNECTED) {
                continue;
              }
              const bool is_stale =
                  task_state.second->TimeSinceLastHeartbeatMs() >
                  heartbeat_timeout_ms_;
              VLOG(1) << "Checking staleness for " << task_state.first
                      << " stale?=" << is_stale;
              if (is_stale) {
                absl::string_view stale_task_name = task_state.first;
                stale_task_names.push_back(stale_task_name);
                status = MakeCoordinationError(errors::Unavailable(
                    "Task ", stale_task_name,
                    " heartbeat timeout. This indicates that the remote task "
                    "has failed, got preempted, or crashed unexpectedly."));
                SetTaskError(stale_task_name, status);
              }
            }
          }
          // Propagate heartbeat timeout errors to other connected tasks.
          if (!stale_task_names.empty()) {
            if (!has_service_to_client_connection) {
              // Error cannot be propagated since there is no service-to-client
              // connection, so shut down service instead. Note: we cannot
              // destroy the thread within its own function. However, this
              // thread will be destroyed once the function returns.
              LOG(ERROR) << "Stopping coordination service as heartbeat has "
                            "timed out for "
                         << stale_task_names[0]
                         << " and there is no service-to-client connection";
              Stop(/*shut_staleness_thread=*/false);
              return;
            }
            for (const auto& stale_task_name : stale_task_names) {
              PropagateError(GetTaskFromName(stale_task_name));
            }
            stale_task_names.clear();
          }

          // Barrier timeout check.
          uint64_t current_time_micros = Env::Default()->NowMicros();
          {
            mutex_lock l(state_mu_);
            // Gather barriers which have timed out.
            for (const std::string& barrier_id : ongoing_barriers_) {
              auto* barrier = &barriers_[barrier_id];
              if (current_time_micros > barrier->deadline_in_micros) {
                expired_barriers[barrier_id] = barrier;
              }
            }
            // Pass these barriers with the time out error.
            for (const std::pair<const std::string, BarrierState*>& barrier_kv :
                 expired_barriers) {
              absl::string_view barrier_id = barrier_kv.first;
              BarrierState* barrier = barrier_kv.second;
              const Status error =
                  MakeCoordinationError(errors::DeadlineExceeded(absl::StrCat(
                      "Barrier timed out. Barrier_id: ", barrier_id)));
              PassBarrier(barrier_id, error, barrier);
            }
          }
          if (!has_service_to_client_connection &&
              expired_barriers.contains(shutdown_barrier_id_)) {
            // Error cannot be propagated since there is no service-to-client
            // connection, so shut down service instead. Note: we cannot
            // destroy the thread within its own function. However, this
            // thread will be destroyed once the function returns.
            LOG(ERROR)
                << "Stopping coordination service as shutdown barrier "
                   "timed out and there is no service-to-client connection.";
            Stop(/*shut_staleness_thread=*/false);
          }
          // Reset this for the next barrier check.
          expired_barriers.clear();
        }
      }));
}

void CoordinationServiceStandaloneImpl::Stop(bool shut_staleness_thread) {
  {
    mutex_lock l(kv_mu_);
    get_cb_.clear();
  }
  {
    mutex_lock l(state_mu_);
    cluster_state_.clear();
    for (auto& barrier_state : barriers_) {
      absl::string_view barrier_id = barrier_state.first;
      auto* barrier = &barrier_state.second;
      if (!barrier->passed) {
        Status error = MakeCoordinationError(errors::Aborted(absl::StrCat(
            "Barrier failed because service is shutting down. Barrier_id: ",
            barrier_id)));
        PassBarrier(barrier_id, error, barrier);
      }
    }
    barriers_.clear();
  }
  {
    mutex_lock l(check_staleness_thread_shutdown_mu_);
    shutting_down_ = true;
    check_staleness_thread_cv_.notify_all();
  }
  if (shut_staleness_thread) {
    check_staleness_thread_.reset();
  }
}

Status CoordinationServiceStandaloneImpl::RegisterTask(
    const CoordinatedTask& task, uint64_t incarnation) {
  const std::string& task_name = GetTaskName(task);

  Status status;
  {
    mutex_lock l(state_mu_);
    if (!cluster_state_.contains(task_name)) {
      // Note: return early here as unexpected task register errors should not
      // be propagated to other tasks.
      return MakeCoordinationError(errors::InvalidArgument(
          "Unexpected task registered with task_name=", task_name));
    } else if (cluster_state_[task_name]->GetState() ==
               TaskState::State::CONNECTED) {
      Status s = MakeCoordinationError(
          errors::Aborted("Duplicate task registration with task_name=",
                          task_name),
          task);
      status = s;
      SetTaskError(task_name, status);
    } else {
      // Hit this path when the task is registering itself for the first time,
      // or it's already in ERROR state and now register again. In both cases,
      // the service allows it to be registered.
      cluster_state_[task_name]->SetConnected(incarnation);
      LOG(INFO) << task_name
                << " has connected to coordination service. Incarnation: "
                << incarnation;
    }
  }
  if (!status.ok()) {
    PropagateError(task);
  }
  return status;
}

void CoordinationServiceStandaloneImpl::WaitForAllTasks(
    const CoordinatedTask& task, const CoordinationServiceDeviceInfo& devices,
    StatusCallback done) {
  {
    mutex_lock l(state_mu_);
    cluster_devices_.MergeFrom(devices);
  }
  BarrierAsync(device_propagation_barrier_id_, absl::InfiniteDuration(), task,
               {}, std::move(done));
}

void CoordinationServiceStandaloneImpl::ShutdownTaskAsync(
    const CoordinatedTask& task, StatusCallback done) {
  if (shutdown_barrier_timeout_ > absl::ZeroDuration()) {
    // Impose shutdown barrier so that all tasks can disconnect together.
    BarrierAsync(shutdown_barrier_id_, shutdown_barrier_timeout_, task, {},
                 done);
  } else {
    Status status;
    {
      mutex_lock l(state_mu_);
      // Disconnect task from service individually.
      status = DisconnectTask(task);
    }
    done(status);
  }
}

Status CoordinationServiceStandaloneImpl::ResetTask(
    const CoordinatedTask& task) {
  mutex_lock l(state_mu_);
  return DisconnectTask(task);
}

Status CoordinationServiceStandaloneImpl::DisconnectTask(
    const CoordinatedTask& task) {
  const std::string task_name = GetTaskName(task);
  // Check if task is valid and not already disconnected.
  if (!cluster_state_.contains(task_name)) {
    return MakeCoordinationError(errors::InvalidArgument(
        "Unexpected disconnect request with task_name=", task_name));
  } else if (cluster_state_[task_name]->GetState() ==
             TaskState::State::DISCONNECTED) {
    return MakeCoordinationError(errors::FailedPrecondition(
        "The task is already disconnected: ", task_name));
  }

  // Disconnect task and fail any ongoing barriers.
  cluster_state_[task_name]->Disconnect(
      /*grace_period_duration_us=*/heartbeat_timeout_ms_ * 1000);
  for (const auto& barrier_id :
       cluster_state_[task_name]->GetOngoingBarriers()) {
    Status error = MakeCoordinationError(errors::Internal(absl::StrCat(
        "Barrier failed from a disconnected task. Barrier Id: ", barrier_id,
        ", Task: ", task_name)));
    PassBarrier(barrier_id, error, &barriers_[barrier_id]);
  }

  LOG(INFO) << task_name << " has disconnected from coordination service.";
  return Status::OK();
}

const CoordinationServiceDeviceInfo&
CoordinationServiceStandaloneImpl::ListClusterDevices() {
  return cluster_devices_;
}

uint64_t CoordinationServiceStandaloneImpl::GetServiceIncarnation() {
  return service_incarnation_;
}

Status CoordinationServiceStandaloneImpl::ReportTaskError(
    const CoordinatedTask& task, Status error) {
  const std::string& task_name = GetTaskName(task);
  {
    mutex_lock l(state_mu_);
    if (!cluster_state_.contains(task_name)) {
      return MakeCoordinationError(
          errors::InvalidArgument("Unexpected request from task ", task_name));
    } else if (cluster_state_[task_name]->GetState() !=
               TaskState::State::CONNECTED) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "The task is not connected or already has an error."));
    } else {
      SetTaskError(task_name, error);
    }
  }
  PropagateError(task, /*is_reported_by_task=*/true);
  return Status::OK();
}

Status CoordinationServiceStandaloneImpl::RecordHeartbeat(
    const CoordinatedTask& task, uint64_t incarnation) {
  const std::string& task_name = GetTaskName(task);
  Status s = Status::OK();
  {
    mutex_lock l(state_mu_);
    if (!cluster_state_.contains(task_name)) {
      return MakeCoordinationError(errors::InvalidArgument(
          "Unexpected task request with task_name=", task_name));
    }
    if (!cluster_state_[task_name]->GetStatus().ok()) {
      return cluster_state_[task_name]->GetStatus();
    } else if (cluster_state_[task_name]->GetState() ==
                   TaskState::State::DISCONNECTED &&
               // We accept heartbeats for a short grace period to account for
               // the lag time between the service recording the state change
               // and the agent stopping heartbeats.
               Env::Default()->NowMicros() >
                   cluster_state_[task_name]
                       ->GetDisconnectedGracePeriodMicros()) {
      return MakeCoordinationError(errors::InvalidArgument(
          "Task with task_name=", task_name,
          " must be registered before sending heartbeat messages"));
    }
    s = cluster_state_[task_name]->RecordHeartbeat(incarnation);
  }

  // Set and propagate any heartbeat errors.
  if (!s.ok()) {
    {
      mutex_lock l(state_mu_);
      SetTaskError(task_name, s);
    }
    PropagateError(task);
  }

  return s;
}

void CoordinationServiceStandaloneImpl::ReportServiceErrorToTaskAsync(
    const CoordinatedTask& destination_task, Status error) {
  assert(!error.ok());

  // Don't report error if there is no service-to-client connection.
  if (client_cache_ == nullptr) {
    LOG(ERROR) << error;
    return;
  }

  auto request = std::make_shared<ReportErrorToTaskRequest>();
  auto response = std::make_shared<ReportErrorToTaskResponse>();
  request->set_error_code(error.code());
  request->set_error_message(error.error_message());
  CoordinatedTask* error_source =
      request->mutable_error_payload()->mutable_source_task();
  error_source->set_job_name("coordination_service");
  auto call_opts = std::make_shared<CallOptions>();
  call_opts->SetTimeout(kServiceToClientTimeoutMs);

  const std::string task_name = GetTaskName(destination_task);
  CoordinationClient* client = client_cache_->GetClient(task_name);
  client->ReportErrorToTaskAsync(
      call_opts.get(), request.get(), response.get(),
      [request, response, task_name, call_opts](Status s) {
        if (!s.ok()) {
          LOG(ERROR) << "Encountered another error while reporting to "
                     << task_name << ": " << s;
        }
      });
}

void CoordinationServiceStandaloneImpl::PropagateError(
    const CoordinatedTask& source_task, bool is_reported_by_task) {
  Status error;
  {
    mutex_lock l(state_mu_);
    error = cluster_state_[GetTaskName(source_task)]->GetStatus();
  }
  assert(!error.ok());
  ReportErrorToTaskRequest request;
  request.set_error_code(error.code());
  request.set_error_message(error.error_message());
  CoordinationServiceError* payload = request.mutable_error_payload();
  *payload->mutable_source_task() = source_task;
  payload->set_is_reported_error(is_reported_by_task);
  CallOptions call_opts;
  call_opts.SetTimeout(kServiceToClientTimeoutMs);
  std::vector<std::shared_ptr<absl::Notification>> notifications;

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
      // Propagate error only to tasks that are connected
      if (cluster_state_[task]->GetState() != TaskState::State::CONNECTED)
        continue;
    }

    // Don't propagate error if there is no service-to-client connection.
    if (client_cache_ == nullptr) {
      LOG(ERROR)
          << "Stopping coordination service as there is no "
             "service-to-client connection, but we encountered an error: "
          << error;
      Stop(/*shut_staleness_thread=*/false);
      return;
    }
    CoordinationClient* client = client_cache_->GetClient(std::string(task));
    auto response = std::make_shared<ReportErrorToTaskResponse>();
    auto n = std::make_shared<absl::Notification>();
    client->ReportErrorToTaskAsync(
        &call_opts, &request, response.get(), [response, n, task](Status s) {
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
    return MakeCoordinationError(
        errors::AlreadyExists("Config key ", key, " already exists."));
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

std::vector<KeyValueEntry> CoordinationServiceStandaloneImpl::GetKeyValueDir(
    absl::string_view directory_key) {
  std::vector<KeyValueEntry> kvs_in_directory;
  const std::string norm_key = NormalizeKey(directory_key);
  const std::string dir = absl::StrCat(norm_key, "/");

  mutex_lock l(kv_mu_);
  // Find first key in ordered map that has the directory prefix.
  auto begin = kv_store_.lower_bound(dir);
  std::map<std::string, std::string>::iterator it;
  // Iterate through key range that match directory prefix.
  for (it = begin; it != kv_store_.end(); ++it) {
    // Stop once the next key does not have the directory prefix. Since keys are
    // ordered, none of the other keys would have a matching prefix.
    if (std::mismatch(dir.begin(), dir.end(), it->first.begin()).first !=
        dir.end()) {
      break;
    }
    KeyValueEntry kv;
    kv.set_key(it->first);
    kv.set_value(it->second);
    kvs_in_directory.push_back(kv);
  }

  return kvs_in_directory;
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

void CoordinationServiceStandaloneImpl::SetTaskError(
    absl::string_view task_name, Status error) {
  cluster_state_[task_name]->SetError(error);
  for (const auto& barrier_id :
       cluster_state_[task_name]->GetOngoingBarriers()) {
    Status error = MakeCoordinationError(errors::Internal(absl::StrCat(
        "Barrier failed from a task error. Barrier Id: ", barrier_id,
        ", Task: ", task_name)));
    PassBarrier(barrier_id, error, &barriers_[barrier_id]);
  }

  LOG(ERROR) << task_name << " has been set to ERROR: " << error;
}

void CoordinationServiceStandaloneImpl::BarrierAsync(
    const std::string& barrier_id, absl::Duration timeout,
    const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    StatusCallback done) {
  mutex_lock l(state_mu_);
  auto pair = barriers_.try_emplace(barrier_id);
  auto it = pair.first;
  bool inserted = pair.second;
  auto* barrier = &it->second;
  // Create barrier for the first time.
  if (inserted) {
    // Initialize barrier state.
    barrier->passed = false;
    // Assume barrier is for entire cluster if no tasks are specified.
    if (participating_tasks.empty()) {
      for (const auto& task_state : cluster_state_) {
        absl::string_view task_name = task_state.first;
        barrier->tasks_at_barrier[GetTaskFromName(task_name)] = false;
      }
    } else {
      for (const auto& task : participating_tasks) {
        // Fail the barrier immediately if unexpected task is included in the
        // barrier.
        const std::string task_name = GetTaskName(task);
        if (!cluster_state_.contains(task_name)) {
          Status error = MakeCoordinationError(errors::InvalidArgument(
              absl::StrCat("Unexpected task (", task_name,
                           ") that is not in the cluster called the barrier. "
                           "Barrier Id: ",
                           barrier_id)));
          PassBarrier(barrier_id, error, barrier);
          done(error);
          return;
        }
        barrier->tasks_at_barrier[task] = false;
      }
    }
    barrier->num_pending_tasks = barrier->tasks_at_barrier.size();

    // Fail the barrier immediately if any tasks are already in error.
    for (const auto& pending_task : barrier->tasks_at_barrier) {
      const std::string task_name = GetTaskName(pending_task.first);
      if (cluster_state_[task_name]->GetState() == TaskState::State::ERROR) {
        Status error = MakeCoordinationError(errors::Internal(
            absl::StrCat("Task (", task_name,
                         ") is already in error before the barrier "
                         "was called. Barrier Id: ",
                         barrier_id)));
        PassBarrier(barrier_id, error, barrier);
        done(error);
        return;
      }
    }
    barrier->deadline_in_micros =
        Env::Default()->NowMicros() + (timeout / absl::Microseconds(1));

    // Add ongoing barrier to cluster state.
    ongoing_barriers_.emplace(barrier_id);
    const size_t num_ongoing_barriers = ongoing_barriers_.size();
    if (num_ongoing_barriers > kOngoingBarriersSoftLimit) {
      LOG(WARNING) << "There is a high number of ongoing barriers in "
                      "coordination service: "
                   << num_ongoing_barriers;
    }
    for (const auto& pending_task : barrier->tasks_at_barrier) {
      const CoordinatedTask& task = pending_task.first;
      cluster_state_[GetTaskName(task)]->JoinBarrier(barrier_id);
    }
  }

  // Barrier has already been passed, return previous result immediately.
  if (barrier->passed) {
    // Special hook for shutdown barrier to disconnect task.
    if (barrier_id == shutdown_barrier_id_) {
      Status s = DisconnectTask(task);
      // Return any errors from the disconnect attempt, otherwise return the
      // barrier status outside of this hook.
      if (!s.ok()) {
        done(s);
        return;
      }
    }

    done(barrier->result);
    return;
  }

  // Add pending callbacks.
  barrier->done_callbacks.push_back(done);

  // Check if caller task is participating in the barrier.
  if (!barrier->tasks_at_barrier.contains(task)) {
    // Unexpected barrier call from a task not participating in the barrier.
    Status error = MakeCoordinationError(errors::InvalidArgument(
        absl::StrCat("A non-participating task (", GetTaskName(task),
                     ") called the barrier: ", barrier_id)));
    PassBarrier(barrier_id, error, barrier);
    return;
  }

  // Check if task args are specified consistently across barrier calls.
  if (!ValidateTaskArgs(participating_tasks, barrier->tasks_at_barrier,
                        cluster_state_.size())) {
    Status error = MakeCoordinationError(errors::InvalidArgument(absl::StrCat(
        "Conflicting tasks specified for the same barrier: ", barrier_id)));
    PassBarrier(barrier_id, error, barrier);
    return;
  }

  // Remove pending task.
  // We need to check if task made a repeated call after reaching the barrier.
  if (!barrier->tasks_at_barrier[task]) {
    barrier->tasks_at_barrier[task] = true;
    --barrier->num_pending_tasks;

    if (barrier->num_pending_tasks == 0) {
      PassBarrier(barrier_id, Status::OK(), barrier);
      return;
    }
  }
}

Status CoordinationServiceStandaloneImpl::CancelBarrier(
    const std::string& barrier_id, const CoordinatedTask& task) {
  mutex_lock l(state_mu_);
  auto it = barriers_.find(barrier_id);
  // Barrier not found.
  if (it == barriers_.end()) {
    return MakeCoordinationError(errors::NotFound(
        absl::StrCat("Barrier not found. Barrier Id: ", barrier_id)));
  }
  auto* barrier = &it->second;
  // Barrier has already been passed.
  if (barrier->passed) {
    return MakeCoordinationError(errors::FailedPrecondition(absl::StrCat(
        "Barrier (", barrier_id, ") has already been passed with status code: ",
        barrier->result.code())));
  }

  // Cancel barrier.
  Status cancelled = MakeCoordinationError(errors::Cancelled(absl::StrCat(
      "Barrier (", barrier_id, ") is cancelled by task: ", GetTaskName(task))));
  PassBarrier(barrier_id, cancelled, barrier);

  return Status::OK();
}

// Mark barrier as passed.
void CoordinationServiceStandaloneImpl::PassBarrier(
    absl::string_view barrier_id, Status result, BarrierState* barrier) {
  barrier->passed = true;
  barrier->result = result;
  // Special hook for device propagation barrier to set global device ids.
  if (barrier_id == device_propagation_barrier_id_) {
    SetXlaGlobalDeviceIds();
  }
  for (const auto& task_at_barrier : barrier->tasks_at_barrier) {
    // Clean up task state (used as error hooks).
    const CoordinatedTask& task = task_at_barrier.first;
    cluster_state_[GetTaskName(task)]->ExitBarrier(barrier_id);
  }

  // Special hook for shutdown barrier to disconnect tasks at the barrier.
  if (barrier_id == shutdown_barrier_id_) {
    if (result.ok()) {
      LOG(INFO) << "Shutdown barrier has passed.";
    } else {
      LOG(ERROR) << "Shutdown barrier failed: " << result
                 << ". This suggests that at least one worker did not complete "
                    "its job, or was too slow/hanging in its execution.";
    }
    Status shutdown_error = MakeCoordinationError(errors::Internal(
        absl::StrCat("Shutdown barrier has been passed with status: '",
                     barrier->result.ToString(),
                     "', but this task is not at the barrier yet.")));
    for (const std::pair<const CoordinatedTask, bool>& task_at_barrier :
         barrier->tasks_at_barrier) {
      const CoordinatedTask& task = task_at_barrier.first;
      bool at_barrier = task_at_barrier.second;
      if (at_barrier) {
        // Disconnect tasks that reached the barrier.
        Status disconnect_status = DisconnectTask(task);
        if (!disconnect_status.ok()) {
          LOG(ERROR) << disconnect_status;
        }
      } else {
        // Propagate errors to straggling tasks that have not reached the
        // barrier. The barrier must have failed if any task did not reach the
        // barrier.
        ReportServiceErrorToTaskAsync(task, shutdown_error);
      }
    }
  }
  barrier->tasks_at_barrier.clear();
  ongoing_barriers_.erase(barrier_id);
  // Note: barrier_id shouldn't be referenced after this line as its lifetime
  // may be tied to one of the callbacks.
  // Propagate results to participating tasks.
  for (const auto& callback : barrier->done_callbacks) {
    callback(result);
  }
  barrier->done_callbacks.clear();
}

bool CoordinationServiceStandaloneImpl::ValidateTaskArgs(

    const std::vector<CoordinatedTask>& tasks_args,
    const absl::flat_hash_map<CoordinatedTask, bool, CoordinatedTaskHash,
                              CoordinatedTaskEqual>& tasks_at_barrier,
    int64_t cluster_size) {
  if (tasks_args.empty()) {
    return tasks_at_barrier.size() == cluster_size;
  } else if (tasks_at_barrier.size() != tasks_args.size()) {
    return false;
  } else {
    for (const auto& task : tasks_args) {
      if (!tasks_at_barrier.contains(task)) {
        return false;
      }
    }
  }
  return true;
}

void CoordinationServiceStandaloneImpl::SetXlaGlobalDeviceIds() {
  // No-op if TF devices are specified.
  if (cluster_devices_.has_xla()) {
    int global_id = 0;
    for (xla::LocalTopologyProto& local_topology :
         *cluster_devices_.mutable_xla()->mutable_devices()->mutable_nodes()) {
      for (xla::DeviceProto& device : *local_topology.mutable_devices()) {
        device.set_global_device_id(global_id);
        ++global_id;
      }
    }
  }
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
