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

#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/random.h"
#include "tsl/platform/status.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinatedTaskState;
using tensorflow::CoordinatedTaskStateInfo;
using tensorflow::CoordinationServiceConfig;
using tensorflow::CoordinationServiceError;
using tensorflow::DeviceInfo;
using tensorflow::KeyValueEntry;

constexpr absl::Duration kDevicePropagationTimeout = absl::Hours(1);
constexpr int kDefaultHeartbeatTimeoutMs = 10 * 1000;  // 10 seconds
constexpr int kServiceToClientTimeoutMs = 10 * 1000;   // 10 seconds
constexpr size_t kOngoingBarriersSoftLimit = 20;
constexpr char kHealthCheckThread[] = "CoordinationServiceHealthCheck";
constexpr int kPendingTaskLogLimit = 20;
constexpr int kPendingStragglerLogLimit = 3;

std::string GetTaskName(std::string_view job_name, int task_id) {
  return absl::StrCat("/job:", job_name, "/replica:", 0, "/task:", task_id);
}

std::string GetTaskName(const CoordinatedTask& task) {
  return GetTaskName(task.job_name(), task.task_id());
}

CoordinatedTask GetTaskFromName(std::string_view task_name) {
  DeviceNameUtils::ParsedName parsed;
  DeviceNameUtils::ParseFullName(task_name, &parsed);
  CoordinatedTask task;
  task.set_job_name(parsed.job);
  task.set_task_id(parsed.task);
  return task;
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
      Env* env, const CoordinationServiceConfig& config,
      std::unique_ptr<CoordinationClientCache> client_cache);
  ~CoordinationServiceStandaloneImpl() override {
    absl::MutexLock lock(&state_mu_);
    Stop();
  }

  void SetDeviceAggregationFunction(
      std::function<DeviceInfo(const DeviceInfo& devices)>
          post_aggregate_device_fn) override;

  void LogConnectStatusLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  absl::Status RegisterTask(const CoordinatedTask& task,
                            uint64_t incarnation) override;
  void WaitForAllTasks(const CoordinatedTask& task, const DeviceInfo& devices,
                       StatusCallback done) override;
  void ShutdownTaskAsync(const CoordinatedTask& task,
                         StatusCallback done) override;
  absl::Status ResetTask(const CoordinatedTask& task) override;
  absl::Status RecordHeartbeat(const CoordinatedTask& task,
                               uint64_t incarnation) override;
  absl::Status ReportTaskError(const CoordinatedTask& task,
                               absl::Status error) override;
  std::vector<CoordinatedTaskStateInfo> GetTaskState(
      const std::vector<CoordinatedTask>& task) override;
  absl::Status InsertKeyValue(std::string_view key,
                              std::string_view value) override;
  absl::Status InsertKeyValue(std::string_view key, std::string_view value,
                              bool allow_overwrite) override;
  void GetKeyValueAsync(std::string_view key,
                        StatusOrValueCallback done) override;
  absl::StatusOr<std::string> TryGetKeyValue(std::string_view key) override;
  std::vector<KeyValueEntry> GetKeyValueDir(
      std::string_view directory_key) override;
  absl::Status DeleteKeyValue(std::string_view key) override;
  void BarrierAsync(std::string_view barrier_id, absl::Duration timeout,
                    const CoordinatedTask& task,
                    const std::vector<CoordinatedTask>& participating_tasks,
                    StatusCallback done) override;
  absl::Status CancelBarrier(std::string_view barrier_id,
                             const CoordinatedTask& task) override;
  void PollForErrorAsync(const CoordinatedTask& task,
                         StatusCallback done) override;

 private:
  const DeviceInfo& ListClusterDevices() override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  uint64_t GetServiceIncarnation() override;
  // Checks if any task has stopped sending heartbeats.
  void CheckHeartbeatTimeout();
  // Checks if any barrier has timed out.
  void CheckBarrierTimeout();
  // Checks both heartbeat and barrier timeouts. Use a single function so they
  // can be run in the same thread as threads are a constrained resource.
  void CheckStaleness();
  // Starts a thread to check staleness.
  void StartCheckStaleness();
  void Stop() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool ServiceHasStopped() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Report service error to a specified task.
  void ReportServiceErrorToTaskAsync(const CoordinatedTask& destination_task,
                                     absl::Status error);
  // Report error from a task to all other connected tasks if the task is not
  // recoverable.
  // Note: SetTaskError() must be called before propagating its error.
  void PropagateError(absl::Status error,
                      std::optional<CoordinatedTask> source_task = std::nullopt,
                      bool is_reported_by_task = false)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void SetTaskError(std::string_view task_name, absl::Status error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  absl::Status DisconnectTask(const CoordinatedTask& task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  struct BarrierState {
    bool passed = false;
    absl::Status result = absl::UnknownError(
        "Invalid barrier result.");  // Only valid if `passed` is true.
    uint64_t deadline_in_micros = 0;
    int num_pending_tasks = 0;
    // Specifies which tasks have called the barrier so far.
    absl::flat_hash_map<CoordinatedTask, bool, CoordinatedTaskHash,
                        CoordinatedTaskEqual>
        tasks_at_barrier;
    std::vector<StatusCallback> done_callbacks;
    // Specifies the task that initiated the barrier (the first task to call the
    // barrier).
    CoordinatedTask initiating_task;
  };
  // Validates that the barrier is invoked with the right args. Returns false if
  // the barrier should fail immediately.
  bool ValidateBarrierArgs(
      std::string_view barrier_id, absl::Duration timeout,
      const CoordinatedTask& task,
      const std::vector<CoordinatedTask>& participating_tasks,
      StatusCallback done);
  // Initializes a new barrier. Returns false if the barrier should fail
  // immediately.
  bool InitializeBarrier(
      BarrierState* barrier, std::string_view barrier_id,
      absl::Duration timeout, const CoordinatedTask& task,
      const std::vector<CoordinatedTask>& participating_tasks,
      StatusCallback done) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void PassBarrier(std::string_view barrier_id, absl::Status result,
                   BarrierState* barrier)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Post-barrier hook to aggregate device info.
  void AggregateClusterDevices() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Post-shutdown barrier hook to disconnect tasks that acked and propagate
  // errors to those that have not.
  void CompleteShutdownAfterBarrier(absl::Status result, BarrierState* barrier)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Check if participating tasks are specified correctly across barrier calls.
  bool ValidateTaskArgs(
      const std::vector<CoordinatedTask>& tasks_args,
      const absl::flat_hash_map<CoordinatedTask, bool, CoordinatedTaskHash,
                                CoordinatedTaskEqual>& tasks_at_barrier,
      int64_t cluster_size);
  bool isRecoverableJob(std::string_view task_name) const;
  // Sends responses to error polling requests when an error is encountered.
  void SendErrorPollingResponse(const absl::Status& error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Responds to error polling or stops the service when an error is
  // encountered. Should only be called when there is no service to client
  // connection. Returns true if the service stops, otherwise returns false.
  bool SendErrorPollingResponseOrStopService(const absl::Status& error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Returns whether the clients are polling for error from the service. If the
  // clients are not polling for error from the service, the service should stop
  // when there is an error. Otherwise, the service should not stop.
  bool IsClientPollingForError() const;

  class ErrorPollingState {
   public:
    // Returns whether the error polling requests have been responded.
    bool Responded() const { return responded_; }
    // Sets the error and executes the status callbacks.
    void SetError(const absl::Status& error);
    // Gets the error that is propagated to the agents.
    const absl::Status& GetError() const { return error_; }
    // Returns true if the task has sent request to poll for error from the
    // service.
    bool IsTaskPolling(absl::string_view task_name) const {
      return polling_task_names_.contains(task_name);
    }
    // Adds a task to the error polling state.
    void AddTask(const CoordinatedTask& task, StatusCallback&& done);

   private:
    bool responded_ = false;
    absl::Status error_ = absl::OkStatus();
    std::vector<StatusCallback> done_callbacks_;
    absl::flat_hash_set<std::string> polling_task_names_;
  };

  class TaskState {
   public:
    // Task state maintained on the coordination service side.
    // State transition:
    //                Register           Heartbeat
    //   DISCONNECTED -------> CONNECTED --------> ERROR (timeout)
    //                              |   ReportError
    //                              +--------------> ERROR
    //
    // When task state becomes ERROR, propagate this status to other CONNECTED
    // tasks in the cluster.

    CoordinatedTaskState GetState() { return state_; }
    absl::Status GetStatus() { return status_; }
    uint64_t GetTaskIncarnation() { return task_incarnation_; }
    void SetConnected(uint64_t task_incarnation);
    void Disconnect(uint64_t grace_period_duration_us);
    absl::Status RecordHeartbeat(uint64_t task_incarnation);
    int64_t TimeSinceLastHeartbeatMs();
    void SetError(absl::Status status);
    DeviceInfo GetDeviceInfo() { return devices_; }
    void CollectDeviceInfo(const DeviceInfo& devices) { devices_ = devices; }
    // Checks if task has called WaitForAllTasks() previously, which gathers the
    // local device info.
    bool DeviceInfoIsCollected() { return devices_.device_size() != 0; }

    absl::flat_hash_set<std::string> GetOngoingBarriers();
    void JoinBarrier(std::string_view barrier_id);
    void ExitBarrier(std::string_view barrier_id);
    // Returns true if the task has been disconnected beyond the grace period
    // and no further agent requests are expected. Note that the grace period
    // accounts for the lag time between the service recording the state change
    // and the agent stopping heartbeats/error polling.
    bool IsDisconnectedBeyondGracePeriod();

   private:
    // Incarnation ID for CPU:0 on remote task.
    uint64_t task_incarnation_ = 0;

    CoordinatedTaskState state_ = CoordinatedTaskState::TASKSTATE_DISCONNECTED;
    absl::Status status_;
    absl::Mutex last_heartbeat_mu_;
    uint64_t last_heartbeat_us_ ABSL_GUARDED_BY(last_heartbeat_mu_);
    // This denotes the deadline after which we stop accepting heartbeats or
    // error polling requests from a disconnected task. This grace period
    // accounts for the lag time between the service recording the state change
    // and the agent stopping heartbeats/error polling.
    uint64_t disconnect_grace_period_us_ = 0;
    DeviceInfo devices_;
    // For now, we assume there won't be many simultaneous barriers so we simply
    // use a set.
    absl::flat_hash_set<std::string> ongoing_barriers_for_task_;
  };

  std::unique_ptr<CoordinationClientCache> client_cache_;
  Env& env_;
  const uint64_t service_incarnation_ = random::New64();
  const uint64_t heartbeat_timeout_ms_;
  const absl::Duration shutdown_barrier_timeout_;
  // If a task restarts with a new incarnation, we may allow it to reconnect
  // silently if configured. This is useful when we know that a task can
  // immediately resume work upon re-connecting to the service.
  bool allow_new_incarnation_to_reconnect_ = false;
  // Whether the agents are polling for error from the service. It will be set
  // to true when the service sees the first error polling request. Once set to
  // true, the value will never change back to false, so no mutex is needed.
  bool client_polling_for_error_ = false;
  std::function<DeviceInfo(const DeviceInfo& devices)>
      post_aggregate_device_fn_;

  const std::string device_propagation_barrier_id_ =
      absl::StrCat("WaitForAllTasks::", std::to_string(service_incarnation_));
  const std::string shutdown_barrier_id_ =
      absl::StrCat("Shutdown::", std::to_string(service_incarnation_));

  absl::Mutex state_mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<TaskState>> cluster_state_
      ABSL_GUARDED_BY(state_mu_);
  DeviceInfo cluster_devices_ ABSL_GUARDED_BY(state_mu_);

  absl::Mutex kv_mu_;
  // Ordered map to store config key-values
  std::map<std::string, std::string> kv_store_ ABSL_GUARDED_BY(kv_mu_);
  absl::flat_hash_map<std::string, std::vector<StatusOrValueCallback>> get_cb_
      ABSL_GUARDED_BY(kv_mu_);

  absl::CondVar check_staleness_thread_cv_;
  bool shutting_down_ ABSL_GUARDED_BY(state_mu_) = false;
  std::unique_ptr<Thread> check_staleness_thread_;

  absl::flat_hash_map<std::string, BarrierState> barriers_
      ABSL_GUARDED_BY(state_mu_);
  // For now, we assume there won't be many simultaneous barriers so we simply
  // use a set.
  absl::flat_hash_set<std::string> ongoing_barriers_ ABSL_GUARDED_BY(state_mu_);

  absl::flat_hash_set<std::string> recoverable_jobs_;

  ErrorPollingState error_polling_state_ ABSL_GUARDED_BY(state_mu_);

  CoordinationServiceStandaloneImpl(const CoordinationServiceStandaloneImpl&) =
      delete;
  void operator=(const CoordinationServiceStandaloneImpl&) = delete;
};

void CoordinationServiceStandaloneImpl::ErrorPollingState::SetError(
    const absl::Status& error) {
  if (responded_) return;
  responded_ = true;
  error_ = error;
  for (auto& done_cb : done_callbacks_) {
    done_cb(error_);
  }
  done_callbacks_.clear();
}

void CoordinationServiceStandaloneImpl::ErrorPollingState::AddTask(
    const CoordinatedTask& task, StatusCallback&& done) {
  // Do not allow to insert a task if the service has already responded.
  if (Responded()) return;
  polling_task_names_.insert(GetTaskName(task));
  done_callbacks_.emplace_back(done);
}

void CoordinationServiceStandaloneImpl::TaskState::SetConnected(
    uint64_t task_incarnation) {
  state_ = CoordinatedTaskState::TASKSTATE_CONNECTED;
  status_ = absl::OkStatus();
  task_incarnation_ = task_incarnation;
  absl::MutexLock l(&last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
}

void CoordinationServiceStandaloneImpl::TaskState::Disconnect(
    uint64_t grace_period_duration_us) {
  disconnect_grace_period_us_ =
      Env::Default()->NowMicros() + grace_period_duration_us;
  state_ = CoordinatedTaskState::TASKSTATE_DISCONNECTED;
  status_ = absl::OkStatus();
}

void CoordinationServiceStandaloneImpl::TaskState::SetError(
    const absl::Status status) {
  if (state_ == CoordinatedTaskState::TASKSTATE_ERROR) return;
  state_ = CoordinatedTaskState::TASKSTATE_ERROR;
  status_ = status;
}

absl::Status CoordinationServiceStandaloneImpl::TaskState::RecordHeartbeat(
    uint64_t task_incarnation) {
  if (!status_.ok()) return status_;
  if (task_incarnation != task_incarnation_) {
    return MakeCoordinationError(absl::AbortedError(absl::StrCat(
        "Incarnation ID mismatch: expecting ", task_incarnation_, " but got ",
        task_incarnation, ". This means the remote task has restarted.")));
  }
  absl::MutexLock l(&last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
  return absl::OkStatus();
}

int64_t
CoordinationServiceStandaloneImpl::TaskState::TimeSinceLastHeartbeatMs() {
  absl::MutexLock l(&last_heartbeat_mu_);
  return (Env::Default()->NowMicros() - last_heartbeat_us_) / 1000;
}

absl::flat_hash_set<std::string>
CoordinationServiceStandaloneImpl::TaskState::GetOngoingBarriers() {
  return ongoing_barriers_for_task_;
}

void CoordinationServiceStandaloneImpl::TaskState::JoinBarrier(
    std::string_view barrier_id) {
  ongoing_barriers_for_task_.emplace(barrier_id);
}

void CoordinationServiceStandaloneImpl::TaskState::ExitBarrier(
    std::string_view barrier_id) {
  ongoing_barriers_for_task_.erase(barrier_id);
}

bool CoordinationServiceStandaloneImpl::TaskState::
    IsDisconnectedBeyondGracePeriod() {
  return GetState() == CoordinatedTaskState::TASKSTATE_DISCONNECTED &&
         Env::Default()->NowMicros() > disconnect_grace_period_us_;
}

void CoordinationServiceStandaloneImpl::SetDeviceAggregationFunction(
    std::function<DeviceInfo(const DeviceInfo& devices)>
        post_aggregate_device_fn) {
  post_aggregate_device_fn_ = std::move(post_aggregate_device_fn);
}

CoordinationServiceStandaloneImpl::CoordinationServiceStandaloneImpl(
    Env* env, const CoordinationServiceConfig& config,
    std::unique_ptr<CoordinationClientCache> client_cache)
    : client_cache_(std::move(client_cache)),
      env_(*env),
      heartbeat_timeout_ms_([&config]() -> uint64_t {
        return config.heartbeat_timeout_in_ms() > 0
                   ? config.heartbeat_timeout_in_ms()
                   : kDefaultHeartbeatTimeoutMs;
      }()),
      shutdown_barrier_timeout_(
          absl::Milliseconds(config.shutdown_barrier_timeout_in_ms())),
      allow_new_incarnation_to_reconnect_(
          config.allow_new_incarnation_to_reconnect()) {
  LOG(INFO) << "Initializing CoordinationService";
  recoverable_jobs_ = absl::flat_hash_set<std::string>(
      config.recoverable_jobs().cbegin(), config.recoverable_jobs().cend());
  for (const auto& job : config.coordinated_job_list()) {
    for (int i = 0; i < job.num_tasks(); ++i) {
      const std::string task_name = GetTaskName(job.name(), i);
      cluster_state_.emplace(task_name, std::make_unique<TaskState>());
    }
  }
  StartCheckStaleness();
}

void CoordinationServiceStandaloneImpl::CheckHeartbeatTimeout() {
  absl::Status status = absl::OkStatus();
  std::vector<std::string_view> stale_task_names;
  absl::MutexLock l(&state_mu_);
  for (const auto& [task_name, task_state] : cluster_state_) {
    // Skip tasks that are not registered or in error state.
    if (task_state->GetState() != CoordinatedTaskState::TASKSTATE_CONNECTED) {
      continue;
    }
    const bool is_stale =
        task_state->TimeSinceLastHeartbeatMs() > heartbeat_timeout_ms_;
    VLOG(10) << "Checking staleness for " << task_name
             << " stale?=" << is_stale;
    if (is_stale) {
      stale_task_names.push_back(task_name);
      status = MakeCoordinationError(absl::UnavailableError(
          absl::StrCat("Task ", task_name,
                       " heartbeat timeout. This indicates that the "
                       "remote task has failed, got preempted, or "
                       "crashed unexpectedly. Check the task logs "
                       "for an earlier error to debug further.")));
      SetTaskError(task_name, status);
      if (ServiceHasStopped()) {
        // Setting the task to error may cause service to stop (e.g. task is
        // waiting for shutdown barrier). In this case, all the state is invalid
        // and we should exit immediately.
        return;
      }
    }
  }
  // Propagate heartbeat timeout errors to other connected tasks.
  if (!stale_task_names.empty()) {
    absl::Status heartbeat_timeout_error =
        MakeCoordinationError(absl::UnavailableError(
            absl::StrCat("The following tasks are unhealthy (stopped sending "
                         "heartbeats):\n",
                         absl::StrJoin(stale_task_names, "\n"),
                         "\nCheck the task logs for an earlier error to debug "
                         "further.")));
    PropagateError(heartbeat_timeout_error);
  }
}

void CoordinationServiceStandaloneImpl::CheckBarrierTimeout() {
  absl::flat_hash_map<std::string, BarrierState*> expired_barriers;
  uint64_t current_time_micros = Env::Default()->NowMicros();
  {
    absl::MutexLock l(&state_mu_);
    // Gather barriers which have timed out.
    for (std::string_view barrier_id : ongoing_barriers_) {
      auto* barrier = &barriers_[barrier_id];
      if (current_time_micros > barrier->deadline_in_micros) {
        expired_barriers[barrier_id] = barrier;
      }
    }
    // Pass these barriers with the time out error.
    for (const auto& [barrier_id, barrier] : expired_barriers) {
      std::string pending_tasks;
      int pending_task_count = 0;
      // Count and track pending tasks that have not reached the barrier.
      for (const auto& [task, at_barrier] : barrier->tasks_at_barrier) {
        if (at_barrier) {
          continue;
        }
        ++pending_task_count;
        if (pending_task_count < kPendingTaskLogLimit) {
          absl::StrAppend(&pending_tasks, GetTaskName(task), "\n");
        }
      }
      const int64_t tasks_at_barrier =
          barrier->tasks_at_barrier.size() - pending_task_count;
      std::string error_message = absl::StrFormat(
          "Barrier timed out. Id: %s. This usually happens because a task "
          "triggered the barrier too early or too slowly. Please look at the "
          "task logs (both timed out and first task) to debug further.\n"
          "# of tasks that reached the barrier: %d/%d.\nThe first "
          "task at the barrier: %s. Some timed out task names:\n%s",
          barrier_id, tasks_at_barrier, barrier->tasks_at_barrier.size(),
          GetTaskName(barrier->initiating_task), pending_tasks);
      const absl::Status error =
          MakeCoordinationError(absl::DeadlineExceededError(error_message));
      PassBarrier(barrier_id, error, barrier);
    }
  }
}

void CoordinationServiceStandaloneImpl::CheckStaleness() {
  // Used to store stale tasks and barriers.
  while (true) {
    {
      absl::MutexLock l(&state_mu_);
      check_staleness_thread_cv_.WaitWithTimeout(&state_mu_, absl::Seconds(1));
      if (shutting_down_) {
        return;
      }
    }
    CheckHeartbeatTimeout();
    CheckBarrierTimeout();
  }
}

void CoordinationServiceStandaloneImpl::StartCheckStaleness() {
  check_staleness_thread_.reset(env_.StartThread(
      {}, kHealthCheckThread,
      absl::bind_front(&CoordinationServiceStandaloneImpl::CheckStaleness,
                       this)));
}

void CoordinationServiceStandaloneImpl::Stop() {
  {
    absl::MutexLock l(&kv_mu_);
    for (const auto& [key, get_kv_callbacks] : get_cb_) {
      for (const auto& get_kv_callback : get_kv_callbacks) {
        get_kv_callback(absl::CancelledError(
            absl::StrCat("Coordination service is shutting down. Cancelling "
                         "GetKeyValue() for key: ",
                         key)));
      }
    }
    get_cb_.clear();
  }
  // Indicate that the service is shutting down and stop accepting new RPCs.
  shutting_down_ = true;
  // Stop the heartbeat thread.
  check_staleness_thread_cv_.SignalAll();
  // Fail all ongoing barriers.
  for (auto& [barrier_id, barrier] : barriers_) {
    if (!barrier.passed) {
      absl::Status error =
          MakeCoordinationError(absl::AbortedError(absl::StrCat(
              "Barrier failed because service is shutting down. Barrier_id: ",
              barrier_id)));
      PassBarrier(barrier_id, error, &barrier);
    }
  }
  barriers_.clear();
  // Erase cluster state.
  // Note: sequence matters here, this must happen after barrier clean-up as
  // the state is used in `PassBarrier`.
  cluster_state_.clear();
  // Cancel all pending PollForErrorAsync() calls.
  if (IsClientPollingForError()) {
    SendErrorPollingResponse(
        absl::CancelledError("Coordination service is shutting down. "
                             "Cancelling PollForErrorAsync()"));
  }
}

bool CoordinationServiceStandaloneImpl::ServiceHasStopped() const {
  return shutting_down_;
}

// Helper to log progress to having waited for all tasks.
void CoordinationServiceStandaloneImpl::LogConnectStatusLocked() const {
  const int num_tasks = cluster_state_.size();
  int pending_tasks = 0;
  std::vector<std::string> task_names;
  for (const auto& [task_name, task_state] : cluster_state_) {
    if (task_state->GetState() != CoordinatedTaskState::TASKSTATE_CONNECTED) {
      pending_tasks++;
      if (task_names.size() < kPendingStragglerLogLimit) {
        task_names.push_back(task_name);
      }
    }
  }
  LOG(INFO) << "Waiting for " << pending_tasks << "/" << num_tasks
            << " tasks to connect.";
  if (!task_names.empty()) {
    LOG(INFO) << "Example stragglers:\n" << absl::StrJoin(task_names, "\n");
  }
}

absl::Status CoordinationServiceStandaloneImpl::RegisterTask(
    const CoordinatedTask& task, uint64_t incarnation) {
  const std::string task_name = GetTaskName(task);

  std::string error_message;
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    return MakeCoordinationError(absl::InternalError(absl::StrCat(
        "Coordination service has stopped. RegisterTask() from task: ",
        task_name,
        " failed. This usually implies an earlier error that caused "
        "coordination service to shut down before the workers disconnect "
        "gracefully. Check the task leader's logs for an earlier error to "
        "debug the root cause.")));
  }
  if (!cluster_state_.contains(task_name)) {
    // Note: return early here as unexpected task register errors should not
    // be propagated to other tasks.
    return MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Unexpected task registered with task_name=", task_name)));
  }

  auto* task_cluster_state = cluster_state_[task_name].get();
  const auto task_state = task_cluster_state->GetState();
  const auto task_status = task_cluster_state->GetStatus();

  if (task_state == CoordinatedTaskState::TASKSTATE_DISCONNECTED ||
      (allow_new_incarnation_to_reconnect_ &&
       (absl::IsUnavailable(task_status) &&
        task_status.GetPayload(CoordinationErrorPayloadKey())))) {
    // The task is allowed to register itself if:
    // - this task is currently disconnected (registering for the first time
    //   or has called ResetTask() previously).
    // - this task has lost connection previously which caused it to have
    //   an unavailable error state, but has now restarted (possibly with
    //   a new incarnation). This is only allowed if configured with
    //   `allow_new_incarnation_to_reconnect`.
    task_cluster_state->SetConnected(incarnation);
    LOG(INFO) << task_name
              << " has connected to coordination service. Incarnation: "
              << incarnation;
    LogConnectStatusLocked();
    return absl::OkStatus();
  } else if (task_state == CoordinatedTaskState::TASKSTATE_CONNECTED) {
    // This may happen if the service processes the initial RegisterTask(),
    // but the agent did not receive the response so the agent retries again.
    if (task_cluster_state->GetTaskIncarnation() == incarnation) {
      // This should be a no-op, but we update the last heartbeat timestamp
      // to give a longer grace period for the agent to start sending
      // heartbeats.
      task_cluster_state->SetConnected(incarnation);
      LOG(INFO) << task_name
                << " has connected to coordination service with the same "
                << "incarnation again: " << incarnation;
      LogConnectStatusLocked();
      return absl::OkStatus();
    } else {
      error_message =
          absl::StrCat(task_name,
                       " unexpectedly tried to connect with a different "
                       "incarnation. It has likely restarted.");
    }
  } else {
    // This task is connected or already in error, which implies it has
    // registered previously.
    error_message =
        absl::StrCat(task_name,
                     " unexpectedly tried to connect while it is already in "
                     "error. ResetTask() should be called before a "
                     "subsequent connect attempt.");
  }
  LOG(ERROR) << error_message;
  absl::Status error =
      MakeCoordinationError(absl::AbortedError(error_message), task);
  SetTaskError(task_name, error);
  PropagateError(error, task);
  return error;
}

void CoordinationServiceStandaloneImpl::WaitForAllTasks(
    const CoordinatedTask& task, const DeviceInfo& devices,
    StatusCallback done) {
  {
    absl::MutexLock l(&state_mu_);
    if (ServiceHasStopped()) {
      done(MakeCoordinationError(absl::InternalError(
          "Coordination service has stopped. WaitForAllTasks() failed.")));
      return;
    }
    const auto& task_state = cluster_state_.find(GetTaskName(task));
    // Collect task device info for the first time that task
    // has called WaitForAllTasks(). This will be aggregated when the barrier
    // passes.
    if (task_state != cluster_state_.end() &&
        !task_state->second->DeviceInfoIsCollected()) {
      task_state->second->CollectDeviceInfo(devices);
    }
  }
  BarrierAsync(device_propagation_barrier_id_, kDevicePropagationTimeout, task,
               {}, std::move(done));
}

void CoordinationServiceStandaloneImpl::ShutdownTaskAsync(
    const CoordinatedTask& task, StatusCallback done) {
  VLOG(3) << "Task " << GetTaskName(task) << " invoked ShutdownTaskAsync()";
  if (shutdown_barrier_timeout_ > absl::ZeroDuration()) {
    // Impose shutdown barrier so that all tasks can disconnect together.
    BarrierAsync(shutdown_barrier_id_, shutdown_barrier_timeout_, task, {},
                 done);
  } else {
    absl::Status status;
    {
      absl::MutexLock l(&state_mu_);
      if (ServiceHasStopped()) {
        status = MakeCoordinationError(absl::InternalError(
            "Coordination service has stopped. ShutdownTaskAsync() failed."));
      } else {
        // Disconnect task from service individually.
        status = DisconnectTask(task);
      }
    }
    done(status);
  }
}

absl::Status CoordinationServiceStandaloneImpl::ResetTask(
    const CoordinatedTask& task) {
  absl::MutexLock l(&state_mu_);
  return DisconnectTask(task);
}

absl::Status CoordinationServiceStandaloneImpl::DisconnectTask(
    const CoordinatedTask& task) {
  const std::string task_name = GetTaskName(task);
  // Check if task is valid and not already disconnected.
  if (ServiceHasStopped()) {
    return MakeCoordinationError(absl::InternalError(
        absl::StrCat("Coordination service has stopped. DisconnectTask() "
                     "failed for task_name=",
                     task_name)));
  } else if (!cluster_state_.contains(task_name)) {
    return MakeCoordinationError(absl::InvalidArgumentError(absl::StrCat(
        "Unexpected disconnect request with task_name=", task_name)));
  } else if (cluster_state_[task_name]->GetState() ==
             CoordinatedTaskState::TASKSTATE_DISCONNECTED) {
    return MakeCoordinationError(absl::FailedPreconditionError(
        absl::StrCat("The task is already disconnected: ", task_name)));
  }

  // Disconnect task and fail any ongoing barriers.
  cluster_state_[task_name]->Disconnect(
      /*grace_period_duration_us=*/heartbeat_timeout_ms_ * 1000);
  for (const auto& barrier_id :
       cluster_state_[task_name]->GetOngoingBarriers()) {
    absl::Status error = MakeCoordinationError(absl::InternalError(absl::StrCat(
        "Barrier failed because a task has disconnected. Barrier Id: ",
        barrier_id, ", Task: ", task_name)));
    PassBarrier(barrier_id, error, &barriers_[barrier_id]);
  }

  LOG(INFO) << task_name << " has disconnected from coordination service.";
  return absl::OkStatus();
}

const DeviceInfo& CoordinationServiceStandaloneImpl::ListClusterDevices() {
  return cluster_devices_;
}

uint64_t CoordinationServiceStandaloneImpl::GetServiceIncarnation() {
  return service_incarnation_;
}

absl::Status CoordinationServiceStandaloneImpl::ReportTaskError(
    const CoordinatedTask& task, absl::Status error) {
  const std::string task_name = GetTaskName(task);
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    return MakeCoordinationError(absl::InternalError(
        "Coordination service has stopped. ReportTaskError() failed."));
  } else if (!cluster_state_.contains(task_name)) {
    return MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Unexpected request from task ", task_name)));
  } else if (cluster_state_[task_name]->GetState() !=
             CoordinatedTaskState::TASKSTATE_CONNECTED) {
    return MakeCoordinationError(absl::FailedPreconditionError(
        "The task is not connected or already has an error."));
  }
  SetTaskError(task_name, error);
  PropagateError(error, task, /*is_reported_by_task=*/true);
  return absl::OkStatus();
}

std::vector<CoordinatedTaskStateInfo>
CoordinationServiceStandaloneImpl::GetTaskState(
    const std::vector<CoordinatedTask>& tasks) {
  std::vector<CoordinatedTaskStateInfo> states_info;
  for (const auto& task : tasks) {
    const std::string task_name = GetTaskName(task);
    auto& state_info = states_info.emplace_back();
    absl::Status error;
    {
      absl::MutexLock l(&state_mu_);
      state_info.set_state(cluster_state_[task_name]->GetState());
      error = cluster_state_[task_name]->GetStatus();
    }
    *state_info.mutable_task() = task;
    state_info.set_error_code(error.raw_code());
    state_info.set_error_message(std::string(error.message()));
    if (!error.ok()) {
      *state_info.mutable_error_payload()->mutable_source_task() = task;
      state_info.mutable_error_payload()->set_is_reported_error(false);
    }
  }
  return states_info;
}

absl::Status CoordinationServiceStandaloneImpl::RecordHeartbeat(
    const CoordinatedTask& task, uint64_t incarnation) {
  const std::string task_name = GetTaskName(task);
  absl::Status s = absl::OkStatus();
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    return MakeCoordinationError(absl::InternalError(absl::StrCat(
        "Coordination service has stopped. RecordHeartbeat() from task: ",
        task_name,
        " failed. This usually implies an earlier error that caused "
        "coordination service to shut down before the workers disconnect "
        "gracefully. Check the task leader's logs for an earlier error to "
        "debug the root cause.")));
  } else if (!cluster_state_.contains(task_name)) {
    return MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Unexpected heartbeat request from task: ", task_name,
                     ". This usually implies a configuration error.")));
  }
  if (!cluster_state_[task_name]->GetStatus().ok()) {
    return cluster_state_[task_name]->GetStatus();
  } else if (cluster_state_[task_name]->IsDisconnectedBeyondGracePeriod()) {
    // We accept heartbeats for a short grace period to account for the lag
    // time between the service recording the state change and the agent
    // stopping heartbeats.
    return MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Task with task_name=", task_name,
                     " must be registered before sending heartbeat messages")));
  }
  VLOG(10) << "Record heartbeat from task: " << task_name
           << "at incarnation: " << incarnation << "at " << absl::Now();
  s = cluster_state_[task_name]->RecordHeartbeat(incarnation);

  // Set and propagate any heartbeat errors.
  if (!s.ok()) {
    SetTaskError(task_name, s);
    PropagateError(s, task);
  }

  return s;
}

void CoordinationServiceStandaloneImpl::ReportServiceErrorToTaskAsync(
    const CoordinatedTask& destination_task, absl::Status error) {
  assert(!error.ok());

  // Don't report error if there is no service-to-client connection.
  if (client_cache_ == nullptr) {
    LOG(ERROR) << error;
    return;
  }

  auto request = std::make_shared<ReportErrorToTaskRequest>();
  auto response = std::make_shared<ReportErrorToTaskResponse>();
  request->set_error_code(error.raw_code());
  request->set_error_message(std::string(error.message()));
  CoordinatedTask* error_source =
      request->mutable_error_payload()->mutable_source_task();
  error_source->set_job_name("coordination_service");
  auto call_opts = std::make_shared<CallOptions>();
  call_opts->SetTimeout(kServiceToClientTimeoutMs);

  const std::string task_name = GetTaskName(destination_task);
  CoordinationClient* client = client_cache_->GetClient(task_name);
  client->ReportErrorToTaskAsync(
      call_opts.get(), request.get(), response.get(),
      [request, response, task_name, call_opts](absl::Status s) {
        if (!s.ok()) {
          LOG(ERROR) << "Encountered another error while reporting to "
                     << task_name << ": " << s;
        }
      });
}

void CoordinationServiceStandaloneImpl::PropagateError(
    absl::Status error, std::optional<CoordinatedTask> source_task,
    bool is_reported_by_task) {
  VLOG(3) << "PropagateError(): " << error;
  assert(!error.ok());
  // If there is no service-to-client connection, use error polling or stop
  // the service.
  if (client_cache_ == nullptr) {
    SendErrorPollingResponseOrStopService(error);
    return;
  }

  ReportErrorToTaskRequest request;
  request.set_error_code(error.raw_code());
  request.set_error_message(std::string(error.message()));
  CoordinationServiceError* payload = request.mutable_error_payload();
  payload->set_is_reported_error(is_reported_by_task);
  CallOptions call_opts;
  call_opts.SetTimeout(kServiceToClientTimeoutMs);
  if (source_task.has_value()) {
    // If the error task is recoverable, do not propagate the error to other
    // connected tasks.
    if (isRecoverableJob(source_task->job_name())) return;
    *payload->mutable_source_task() = *source_task;
  }

  std::vector<std::shared_ptr<absl::Notification>> notifications;

  for (const auto& pair : cluster_state_) {
    // Propagate error only to tasks that are connected
    if (pair.second->GetState() != CoordinatedTaskState::TASKSTATE_CONNECTED) {
      continue;
    }
    std::string task = pair.first;

    CoordinationClient* client = client_cache_->GetClient(task);
    auto response = std::make_shared<ReportErrorToTaskResponse>();
    auto n = std::make_shared<absl::Notification>();
    client->ReportErrorToTaskAsync(
        &call_opts, &request, response.get(),
        [response, n, task](absl::Status s) {
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
std::string NormalizeKey(std::string_view orig_key) {
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

absl::Status CoordinationServiceStandaloneImpl::InsertKeyValue(
    std::string_view key, std::string_view value) {
  return InsertKeyValue(key, value, /*allow_overwrite=*/false);
}

absl::Status CoordinationServiceStandaloneImpl::InsertKeyValue(
    std::string_view key, std::string_view value, bool allow_overwrite) {
  VLOG(3) << "InsertKeyValue(): " << key << ": " << value
          << " allow_overwrite: " << allow_overwrite;
  const std::string norm_key = NormalizeKey(key);
  absl::MutexLock l(&kv_mu_);
  if (!allow_overwrite && kv_store_.find(norm_key) != kv_store_.end()) {
    return MakeCoordinationError(absl::AlreadyExistsError(
        absl::StrCat("Config key ", key, " already exists.")));
  }
  kv_store_.insert_or_assign(norm_key, value);
  auto iter = get_cb_.find(norm_key);
  if (iter != get_cb_.end()) {
    for (const auto& cb : iter->second) {
      cb(value);
    }
    get_cb_.erase(iter);
  }
  return absl::OkStatus();
}

void CoordinationServiceStandaloneImpl::GetKeyValueAsync(
    std::string_view key, StatusOrValueCallback done) {
  VLOG(3) << "GetKeyValue(): " << key;
  const std::string norm_key = NormalizeKey(key);
  absl::MutexLock l(&kv_mu_);
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

absl::StatusOr<std::string> CoordinationServiceStandaloneImpl::TryGetKeyValue(
    std::string_view key) {
  VLOG(3) << "TryGetKeyValue(): " << key;
  const std::string norm_key = NormalizeKey(key);
  absl::MutexLock l(&kv_mu_);
  const auto& iter = kv_store_.find(norm_key);
  if (iter == kv_store_.end()) {
    return absl::NotFoundError(absl::StrCat("Config key ", key, " not found."));
  }
  return iter->second;
}

std::vector<KeyValueEntry> CoordinationServiceStandaloneImpl::GetKeyValueDir(
    std::string_view directory_key) {
  VLOG(3) << "TryGetKeyValueDir(): " << directory_key;
  std::vector<KeyValueEntry> kvs_in_directory;
  const std::string norm_key = NormalizeKey(directory_key);
  const std::string dir = absl::StrCat(norm_key, "/");

  absl::MutexLock l(&kv_mu_);
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

absl::Status CoordinationServiceStandaloneImpl::DeleteKeyValue(
    std::string_view key) {
  VLOG(3) << "DeleteKeyValue(): " << key;
  const std::string norm_key = NormalizeKey(key);
  absl::MutexLock l(&kv_mu_);
  // Delete directory: find key range that match directory prefix
  const std::string dir = absl::StrCat(norm_key, "/");
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
  return absl::OkStatus();
}

void CoordinationServiceStandaloneImpl::SetTaskError(std::string_view task_name,
                                                     absl::Status error) {
  cluster_state_[task_name]->SetError(error);
  LOG(ERROR) << task_name
             << " has been set to ERROR in coordination service: " << error;
  for (const auto& barrier_id :
       cluster_state_[task_name]->GetOngoingBarriers()) {
    absl::Status barrier_error =
        MakeCoordinationError(absl::InternalError(absl::StrCat(
            "Barrier failed beacuse a task is in error. Barrier Id: ",
            barrier_id, ", Task: ", task_name, "Error: ", error.message())));
    PassBarrier(barrier_id, barrier_error, &barriers_[barrier_id]);
  }
}

void CoordinationServiceStandaloneImpl::PollForErrorAsync(
    const CoordinatedTask& task, StatusCallback done) {
  const std::string task_name = GetTaskName(task);
  VLOG(3) << "Task " << task_name << " invoked PollForErrorAsync().";

  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    done(MakeCoordinationError(absl::InternalError(
        "PollForError requested after coordination service has shut down.")));
    return;
  }

  if (client_cache_ != nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Should not use error polling from service when "
                            "there is service to client connection.")));
    return;
  }

  client_polling_for_error_ = true;

  if (!cluster_state_.contains(task_name)) {
    done(MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Unexpected task (", task_name,
                     ") that is not in the cluster polling for errors."))));
    return;
  }

  // On the agent side, the error polling thread will only be started when the
  // task is connected, but by the time the request is processed by the service,
  // the task state may have changed due to actions by the service or the main
  // thread on the agent. As a way to handle this, we accept error polling for a
  // short grace period. After the grace period, the service will return an
  // error to the task.
  if (cluster_state_[task_name]->IsDisconnectedBeyondGracePeriod()) {
    done(MakeCoordinationError(absl::FailedPreconditionError(
        absl::StrCat("Task (", task_name,
                     ") that has not been registered or has disconnected "
                     "polling for errors."))));
    return;
  }

  if (cluster_state_[task_name]->GetState() ==
      CoordinatedTaskState::TASKSTATE_ERROR) {
    done(MakeCoordinationError(absl::FailedPreconditionError(absl::StrCat(
        "Task (", task_name,
        ") that is already in error state polling for errors. Current error: ",
        cluster_state_[task_name]->GetStatus().ToString()))));
    return;
  }

  if (error_polling_state_.Responded()) {
    done(error_polling_state_.GetError());
    return;
  }

  error_polling_state_.AddTask(task, std::move(done));
}

// Validates that the barrier is invoked with the right args. Returns false if
// the barrier should fail immediately.
bool CoordinationServiceStandaloneImpl::ValidateBarrierArgs(
    std::string_view barrier_id, absl::Duration timeout,
    const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    StatusCallback done) {
  // Check if caller task is participating in the barrier. If not, update
  // `barriers_` to cause subsequent calls from the same task and other tasks
  // that have already called this instance of the barrier to fail.
  const std::string source_task_name = GetTaskName(task);

  bool among_participating_tasks =
      std::find_if(participating_tasks.begin(), participating_tasks.end(),
                   [&](const CoordinatedTask& task) {
                     return GetTaskName(task) == source_task_name;
                   }) != participating_tasks.end();

  if (!participating_tasks.empty() && !among_participating_tasks) {
    const std::string task_name = GetTaskName(task);
    absl::Status error = MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("A non-participating task (", GetTaskName(task),
                     ") called the barrier: ", barrier_id)));
    {
      absl::MutexLock l(&state_mu_);
      // Check if coordination service has stopped. If so, return an error
      // immediately.
      if (ServiceHasStopped()) {
        done(MakeCoordinationError(absl::InternalError(
            "Barrier requested after coordination service has shut down.")));
        return false;
      }
      auto pair = barriers_.try_emplace(barrier_id);
      auto it = pair.first;
      auto* barrier = &it->second;
      // Make sure subsequent calls fail and existing waiting tasks receive the
      // error.
      PassBarrier(barrier_id, error, barrier);
    }
    done(error);
    return false;
  }
  return true;
};

// Initializes a new barrier. Returns false if the barrier should fail
// immediately.
bool CoordinationServiceStandaloneImpl::InitializeBarrier(
    BarrierState* barrier, std::string_view barrier_id, absl::Duration timeout,
    const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    StatusCallback done) {
  // Initialize barrier state.
  barrier->passed = false;
  barrier->initiating_task = task;
  // Assume barrier is for entire cluster if no tasks are specified.
  if (participating_tasks.empty()) {
    for (const auto& task_state : cluster_state_) {
      std::string_view task_name = task_state.first;
      barrier->tasks_at_barrier[GetTaskFromName(task_name)] = false;
    }
  } else {
    for (const auto& task : participating_tasks) {
      // Fail the barrier immediately if unexpected task is included in the
      // barrier.
      const std::string task_name = GetTaskName(task);
      if (!cluster_state_.contains(task_name)) {
        absl::Status error = MakeCoordinationError(absl::InvalidArgumentError(
            absl::StrCat("Unexpected task (", task_name,
                         ") that is not in the cluster called the barrier. "
                         "Barrier Id: ",
                         barrier_id)));
        PassBarrier(barrier_id, error, barrier);
        done(error);
        return false;
      }
      barrier->tasks_at_barrier[task] = false;
    }
  }
  barrier->num_pending_tasks = barrier->tasks_at_barrier.size();

  // Fail the barrier immediately if any tasks are already in error.
  for (const auto& pending_task : barrier->tasks_at_barrier) {
    const std::string task_name = GetTaskName(pending_task.first);
    if (cluster_state_[task_name]->GetState() ==
        CoordinatedTaskState::TASKSTATE_ERROR) {
      absl::Status error = MakeCoordinationError(absl::InternalError(
          absl::StrCat("Task (", task_name,
                       ") is already in error before the barrier "
                       "was called. Barrier Id: ",
                       barrier_id)));
      PassBarrier(barrier_id, error, barrier);
      done(error);
      return false;
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
  return true;
}

void CoordinationServiceStandaloneImpl::BarrierAsync(
    std::string_view barrier_id, absl::Duration timeout,
    const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    StatusCallback done) {
  VLOG(3) << "Task " << GetTaskName(task) << " invoked BarrierAsync("
          << barrier_id << ").";

  if (!ValidateBarrierArgs(barrier_id, timeout, task, participating_tasks,
                           done)) {
    return;  // Exit early if args are wrong.
  }

  absl::MutexLock l(&state_mu_);
  // Check if coordination service has stopped. If so, return an error
  // immediately.
  if (ServiceHasStopped()) {
    done(MakeCoordinationError(absl::InternalError(
        "Barrier requested after coordination service has shut down.")));
    return;
  }

  auto pair = barriers_.try_emplace(barrier_id);
  auto it = pair.first;
  bool inserted = pair.second;
  auto* barrier = &it->second;

  // Create barrier for the first time.
  if (inserted) {
    if (!InitializeBarrier(barrier, barrier_id, timeout, task,
                           participating_tasks, done)) {
      return;  // Exit early if barrier init failed.
    }
  }

  // Barrier has already been passed, return previous result immediately.
  if (barrier->passed) {
    // Special hook for shutdown barrier to disconnect task.
    if (barrier_id == shutdown_barrier_id_) {
      absl::Status s = DisconnectTask(task);
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

  // Check if task args are specified consistently across barrier calls.
  if (!ValidateTaskArgs(participating_tasks, barrier->tasks_at_barrier,
                        cluster_state_.size())) {
    absl::Status error =
        MakeCoordinationError(absl::InvalidArgumentError(absl::StrCat(
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
      PassBarrier(barrier_id, absl::OkStatus(), barrier);
      return;
    }
  }
}

absl::Status CoordinationServiceStandaloneImpl::CancelBarrier(
    std::string_view barrier_id, const CoordinatedTask& task) {
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    return MakeCoordinationError(absl::InternalError(
        "Coordination service has stopped. CancelBarrier() failed."));
  }
  auto [it, inserted] = barriers_.try_emplace(barrier_id);
  auto* barrier = &it->second;
  if (inserted) {
    LOG(WARNING) << "Barrier (" << barrier_id
                 << ") is cancelled before being created by task: "
                 << GetTaskName(task);
  }
  // Barrier has already been passed.
  if (barrier->passed) {
    return MakeCoordinationError(absl::FailedPreconditionError(absl::StrCat(
        "Barrier (", barrier_id, ") has already been passed with status code: ",
        barrier->result.code())));
  }

  // Cancel barrier.
  absl::Status cancelled = MakeCoordinationError(absl::CancelledError(
      absl::StrCat("Barrier (", barrier_id,
                   ") is cancelled by task: ", GetTaskName(task))));
  PassBarrier(barrier_id, cancelled, barrier);

  VLOG(3) << "Barrier (" << barrier_id << ") is cancelled.";
  return absl::OkStatus();
}

// Mark barrier as passed.
void CoordinationServiceStandaloneImpl::PassBarrier(std::string_view barrier_id,
                                                    absl::Status result,
                                                    BarrierState* barrier) {
  barrier->passed = true;
  barrier->result = result;
  VLOG(3) << "Barrier(" << barrier_id << ") has passed with status: " << result;
  // Special hook for device propagation barrier to set global device ids.
  if (barrier_id == device_propagation_barrier_id_) {
    AggregateClusterDevices();
  }
  for (const auto& task_at_barrier : barrier->tasks_at_barrier) {
    // Clean up task state (used as error hooks).
    const CoordinatedTask& task = task_at_barrier.first;
    cluster_state_[GetTaskName(task)]->ExitBarrier(barrier_id);
  }
  barrier->tasks_at_barrier.clear();
  ongoing_barriers_.erase(barrier_id);
  // Note: barrier_id shouldn't be referenced after this line as its lifetime
  // may be tied to one of the callbacks.
  // Propagate results to participating tasks.
  for (const auto& callback : barrier->done_callbacks) {
    callback(result);
  }
  // Special hook for shutdown barrier to disconnect tasks at the barrier and
  // propagate errors to those that have not.
  if (barrier_id == shutdown_barrier_id_) {
    CompleteShutdownAfterBarrier(result, barrier);
  }
  // It is possible that CompleteShutdownAfterBarrier() stops the service. In
  // this case, the barrier state is forcibly erased and we should not illegally
  // access it.
  if (!ServiceHasStopped()) {
    barrier->done_callbacks.clear();
  }
}

void CoordinationServiceStandaloneImpl::SendErrorPollingResponse(
    const absl::Status& error) {
  CHECK(IsClientPollingForError())
      << "`SendErrorPollingResponse` should only be called after agents poll "
         "errors from the service.";
  if (error_polling_state_.Responded()) {
    return;
  }
  if (!absl::IsCancelled(error)) {
    VLOG(2) << "An error is encountered. Sending the error as a response to "
               "all error polling requests: "
            << error;
  }
  std::vector<std::string> missing_tasks;
  missing_tasks.reserve(cluster_state_.size());
  for (const auto& [task_name, task_state] : cluster_state_) {
    if (!error_polling_state_.IsTaskPolling(task_name)) {
      missing_tasks.push_back(task_name);
    }
  }
  error_polling_state_.SetError(error);
  if (!missing_tasks.empty()) {
    LOG(ERROR) << absl::StrFormat(
        "The following %d tasks in the cluster has not sent request to poll "
        "for error. Error will not be propagated to these tasks: %s",
        missing_tasks.size(), absl::StrJoin(missing_tasks, ","));
  }
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

void CoordinationServiceStandaloneImpl::AggregateClusterDevices() {
  assert(cluster_devices_.device_size() == 0);
  std::vector<CoordinatedTask> ordered_tasks;
  // Sort by task name to set deterministic order for cluster devices.
  ordered_tasks.reserve(cluster_state_.size());
  for (const auto& task : cluster_state_) {
    ordered_tasks.push_back(GetTaskFromName(task.first));
  }
  std::sort(ordered_tasks.begin(), ordered_tasks.end(),
            [](const CoordinatedTask& task1, const CoordinatedTask& task2) {
              if (task1.job_name() != task2.job_name()) {
                return task1.job_name() < task2.job_name();
              }
              return task1.task_id() < task2.task_id();
            });

  // Aggregate to global device list.
  for (const auto& task : ordered_tasks) {
    cluster_devices_.MergeFrom(
        cluster_state_[GetTaskName(task)]->GetDeviceInfo());
  }

  if (post_aggregate_device_fn_ != nullptr) {
    cluster_devices_ = post_aggregate_device_fn_(cluster_devices_);
  }
}

void CoordinationServiceStandaloneImpl::CompleteShutdownAfterBarrier(
    absl::Status result, BarrierState* barrier) {
  if (result.ok()) {
    LOG(INFO) << "Shutdown barrier in coordination service has passed.";
  } else {
    LOG(ERROR) << "Shutdown barrier in coordination service has failed:\n"
               << result
               << "\nThis suggests that the workers are out of sync. Either "
                  "at least one worker is too fast in its execution / "
                  "crashed early or too slow / hanging. Check the logs for "
                  "an earlier error to identify the root cause.";
    absl::Status shutdown_error = MakeCoordinationError(absl::InternalError(
        absl::StrCat("Shutdown barrier has failed, but this task is not at the "
                     "barrier yet.\nBarrier result: '",
                     barrier->result.message())));
    // Propagate error to all tasks before disconnecting them.
    PropagateError(shutdown_error);
  }
  // It is possible that PropagateError() stops the service. In this case, the
  // task state is forcibly erased and disconnecting the task is not
  // necessary.
  if (ServiceHasStopped()) {
    return;
  }
  for (const auto& [task, at_barrier] : barrier->tasks_at_barrier) {
    if (at_barrier) {
      // Disconnect tasks that reached the barrier.
      absl::Status disconnect_status = DisconnectTask(task);
      if (!disconnect_status.ok()) {
        LOG(ERROR) << disconnect_status;
      }
    }
  }
}
}  // namespace

std::unique_ptr<CoordinationServiceInterface> EnableCoordinationService(
    Env* env, const CoordinationServiceConfig& config,
    std::unique_ptr<CoordinationClientCache> cache) {
  return std::make_unique<CoordinationServiceStandaloneImpl>(env, config,
                                                             std::move(cache));
}

bool CoordinationServiceStandaloneImpl::isRecoverableJob(
    const std::string_view task_name) const {
  return recoverable_jobs_.find(task_name) != recoverable_jobs_.end();
}

bool CoordinationServiceStandaloneImpl::SendErrorPollingResponseOrStopService(
    const absl::Status& error) {
  CHECK(!error.ok()) << "SendErrorPollingResponseOrStopService called with OK "
                        "status. Should always return an error.";
  // Should be called only when there is no service-to-client connection.
  assert(client_cache_ == nullptr);
  if (IsClientPollingForError()) {
    LOG(ERROR)
        << "Use error polling to propagate the following error to all tasks: "
        << error;
    SendErrorPollingResponse(error);
    return false;
  }

  LOG(ERROR) << "Stopping coordination service as there is no "
                "service-to-client connection, but we encountered an error: "
             << error;
  Stop();
  return true;
}

bool CoordinationServiceStandaloneImpl::IsClientPollingForError() const {
  return client_polling_for_error_;
}

// Register standalone coordination service implementation.
REGISTER_COORDINATION_SERVICE("standalone", EnableCoordinationService);

}  // namespace tsl
