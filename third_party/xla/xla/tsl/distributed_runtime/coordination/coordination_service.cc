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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
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
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/tsl/util/device_name_utils.h"

ABSL_FLAG(bool, leave_barriers_on_recoverable_agent_restart, false,
          "If true, allow the recoverable agent to leave ongoing barriers on "
          "restart.");

namespace tsl {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinatedTaskState;
using tensorflow::CoordinatedTaskStateInfo;
using tensorflow::CoordinationServiceConfig;
using tensorflow::CoordinationServiceError;
using tensorflow::DeviceInfo;
using tensorflow::KeyValueEntry;

constexpr char kClusterRegisterBarrierId[] =
    "[Init]Wait_for_all_tasks_to_register";
constexpr absl::Duration kDevicePropagationTimeout = absl::Hours(1);
constexpr int kDefaultHeartbeatTimeoutMs = 10 * 1000;  // 10 seconds
constexpr int kServiceToClientTimeoutMs = 10 * 1000;   // 10 seconds
constexpr size_t kOngoingBarriersSoftLimit = 20;
constexpr char kHealthCheckThread[] = "CoordinationServiceHealthCheck";
// Limit the number of stragglers we log to avoid `RESOURCE_EXHAUSTED` errors in
// the RPC layer from sending overly verbose errors.
constexpr int kPendingStragglerLogLimit = 3;
constexpr int kUniqueBarrierCounter = 0;

std::string GetTaskName(absl::string_view job_name, int task_id) {
  return absl::StrCat("/job:", job_name, "/replica:", 0, "/task:", task_id);
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

absl::Status MakeShutdownBarrierError(const absl::Status& error) {
  return MakeCoordinationError(absl::InternalError(absl::StrCat(
      "Shutdown barrier has failed.\nBarrier result: '", error.ToString())));
}

}  // namespace

void CoordinationService::ErrorPollingState::SetError(
    const absl::Status& error) {
  if (responded_) return;
  responded_ = true;
  error_ = error;
  for (auto& [_, done_cb] : done_callbacks_) {
    done_cb(error_);
  }
  done_callbacks_.clear();
}

void CoordinationService::ErrorPollingState::RemoveTask(
    const CoordinatedTask& task, absl::string_view reason) {
  if (done_callbacks_.contains(task)) {
    done_callbacks_[task](MakeCoordinationError(absl::CancelledError(
        absl::StrCat("Cancelled error polling because: ", reason))));
  }
  done_callbacks_.erase(task);
}

void CoordinationService::ErrorPollingState::AddTask(
    const CoordinatedTask& task, StatusCallback&& done) {
  // Do not allow to insert a task if the service has already responded.
  if (Responded()) return;
  polling_task_names_.insert(GetTaskName(task));
  RemoveTask(task, "new request from the same task");
  done_callbacks_[task] = done;
}

void CoordinationService::TaskState::SetConnected(uint64_t task_incarnation) {
  state_ = CoordinatedTaskState::TASKSTATE_CONNECTED;
  status_ = absl::OkStatus();
  task_incarnation_ = task_incarnation;
  absl::MutexLock l(&last_heartbeat_mu_);
  last_heartbeat_us_ = Env::Default()->NowMicros();
}

void CoordinationService::TaskState::Disconnect(
    uint64_t grace_period_duration_us) {
  disconnect_grace_period_us_ =
      Env::Default()->NowMicros() + grace_period_duration_us;
  state_ = CoordinatedTaskState::TASKSTATE_DISCONNECTED;
  status_ = absl::OkStatus();
}

bool CoordinationService::TaskState::SetError(const absl::Status& status) {
  if (state_ == CoordinatedTaskState::TASKSTATE_ERROR) return false;
  state_ = CoordinatedTaskState::TASKSTATE_ERROR;
  status_ = status;
  return true;
}

absl::Status CoordinationService::TaskState::RecordHeartbeat(
    uint64_t task_incarnation) {
  if (!status_.ok()) return status_;
  // Record heartbeat.
  if (task_incarnation_ == task_incarnation) {
    absl::MutexLock l(&last_heartbeat_mu_);
    last_heartbeat_us_ = Env::Default()->NowMicros();
    return absl::OkStatus();
  }
  // Task incarnation mismatch!
  if (IsRecoverable()) {
    return absl::OkStatus();  // Ignore, but don't record new heartbeat.
  } else {
    return MakeCoordinationError(absl::AbortedError(absl::StrCat(
        task_name_, " Heartbeat: Incarnation ID mismatch: expecting ",
        task_incarnation_, " but got ", task_incarnation,
        ". The task has restarted and likely crashed earlier - check for any "
        "earlier errors or any scheduler events (e.g. preemption, eviction) to "
        "debug further.")));
  }
}

int64_t CoordinationService::TaskState::TimeSinceLastHeartbeatMs() {
  absl::MutexLock l(&last_heartbeat_mu_);
  return (Env::Default()->NowMicros() - last_heartbeat_us_) / 1000;
}

absl::flat_hash_set<std::string>
CoordinationService::TaskState::GetOngoingBarriers() {
  return ongoing_barriers_for_task_;
}

void CoordinationService::TaskState::JoinBarrier(absl::string_view barrier_id) {
  ongoing_barriers_for_task_.emplace(barrier_id);
}

void CoordinationService::TaskState::ExitBarrier(absl::string_view barrier_id) {
  ongoing_barriers_for_task_.erase(barrier_id);
}

bool CoordinationService::TaskState::IsDisconnectedBeyondGracePeriod() {
  return GetState() == CoordinatedTaskState::TASKSTATE_DISCONNECTED &&
         Env::Default()->NowMicros() > disconnect_grace_period_us_;
}

void CoordinationService::SetDeviceAggregationFunction(
    std::function<DeviceInfo(const DeviceInfo& devices)>
        post_aggregate_device_fn) {
  post_aggregate_device_fn_ = std::move(post_aggregate_device_fn);
}

CoordinationService::CoordinationService(
    Env* env, const CoordinationServiceConfig& config,
    std::unique_ptr<CoordinationClientCache> client_cache)
    : client_cache_(std::move(client_cache)),
      env_(*env),
      heartbeat_timeout_ms_([&config]() -> uint64_t {
        return config.heartbeat_timeout_in_ms() > 0
                   ? config.heartbeat_timeout_in_ms()
                   : kDefaultHeartbeatTimeoutMs;
      }()),
      cluster_register_with_barrier_(config.cluster_register_with_barrier()),
      cluster_register_timeout_(
          absl::Milliseconds(config.cluster_register_timeout_in_ms())),
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
      cluster_state_.emplace(task_name, std::make_unique<TaskState>(task_name));
    }
  }
  StartCheckStaleness();
}

void CoordinationService::CheckHeartbeatTimeout() {
  absl::Status status = absl::OkStatus();
  std::vector<absl::string_view> stale_task_names;
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
                       "for an earlier error or scheduler events (e.g. "
                       "preemption, eviction) to debug further.")));

      SetTaskError(task_name, status);
    }
  }
  // Propagate heartbeat timeout errors to other connected tasks.
  if (!stale_task_names.empty()) {
    absl::Status heartbeat_timeout_error =
        MakeCoordinationError(absl::UnavailableError(
            absl::StrCat("The following tasks are unhealthy (stopped sending "
                         "heartbeats):\n",
                         absl::StrJoin(stale_task_names, "\n"),
                         "\nThe tasks have crashed. Check the task logs for an "
                         "earlier error, or scheduler events (e.g. preemption, "
                         "eviction) to debug further.")));
    PropagateError(heartbeat_timeout_error, stale_task_names);
  }
}

void CoordinationService::CheckBarrierStatusWithRecoverableTasks() {
  absl::MutexLock l(&state_mu_);
  if (!absl::GetFlag(FLAGS_leave_barriers_on_recoverable_agent_restart)) {
    return;
  }
  // Gather barriers which are ready to pass except for the recoverable tasks
  // reconnected during the barrier. When the flag
  // leave_barriers_on_recoverable_agent_restart is set, the recoverable tasks
  // will be removed from the barrier and the barrier will be passed.
  absl::flat_hash_set<BarrierState*> passing_barriers;
  for (absl::string_view barrier_id : ongoing_barriers_) {
    auto* barrier = &barriers_[barrier_id];
    if (barrier->num_pending_tasks ==
        barrier->recoverable_tasks_restarted_during_barrier.size()) {
      LOG(INFO) << "Barrier " << barrier_id << " has no pending tasks, this "
                << "might be because recoverable tasks have disconnected/"
                << "restarted and were removed from the barrier.";
      passing_barriers.insert(barrier);
    }
  }
  for (auto* barrier : passing_barriers) {
    for (auto& task : barrier->recoverable_tasks_restarted_during_barrier) {
      const std::string task_name = GetTaskName(task);
      const std::unique_ptr<TaskState>& task_state = cluster_state_[task_name];
      if (!barrier->tasks_at_barrier[task]) {
        --barrier->num_pending_tasks;
      }
      barrier->done_callbacks.erase(task);
      task_state->ExitBarrier(barrier->id);
      LOG(INFO) << "Removed the recoverable task " << task_name
                << " from the barrier: " << barrier->id;
    }
    PassBarrier(barrier, absl::OkStatus());
    barrier->recoverable_tasks_restarted_during_barrier.clear();
  }
}

void CoordinationService::CheckBarrierTimeout() {
  absl::flat_hash_map<std::string, BarrierState*> expired_barriers;
  uint64_t current_time_micros = Env::Default()->NowMicros();
  absl::MutexLock l(&state_mu_);
  // Gather barriers which have timed out.
  for (absl::string_view barrier_id : ongoing_barriers_) {
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
      if (pending_task_count < kPendingStragglerLogLimit) {
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
        BarrierName(*barrier), tasks_at_barrier,
        barrier->tasks_at_barrier.size(), GetTaskName(barrier->initiating_task),
        pending_tasks);
    const absl::Status error =
        MakeBarrierError(absl::DeadlineExceededError(error_message),
                         barrier->id, barrier->counter);
    PassBarrier(barrier, error);
  }
}

void CoordinationService::CheckStaleness() {
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
    CheckBarrierStatusWithRecoverableTasks();
    CheckBarrierTimeout();
  }
}

void CoordinationService::StartCheckStaleness() {
  check_staleness_thread_.reset(
      env_.StartThread({}, kHealthCheckThread, [this]() { CheckStaleness(); }));
}

void CoordinationService::Stop() {
  // Prevent recursion.
  if (shutting_down_) {
    return;
  }
  // Indicate that the service is shutting down and stop accepting new RPCs.
  shutting_down_ = true;
  // Stop the heartbeat thread.
  check_staleness_thread_cv_.SignalAll();
  // Fail all ongoing barriers.
  for (auto& [barrier_id, barrier] : barriers_) {
    if (!barrier.passed) {
      absl::Status error = MakeBarrierError(
          absl::AbortedError(absl::StrCat(
              "Barrier failed because service is shutting down. Barrier_id: ",
              BarrierName(barrier))),
          barrier.id, barrier.counter);
      PassBarrier(&barrier, error);
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

bool CoordinationService::ServiceHasStopped() const { return shutting_down_; }

// Helper to log progress to having waited for all tasks.
void CoordinationService::LogConnectStatusLocked() const {
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

absl::Status CoordinationService::RegisterTask(const CoordinatedTask& task,
                                               uint64_t incarnation) {
  absl::Notification done;
  absl::Status status;
  RegisterTaskAsync(task, incarnation, [&](absl::Status s) {
    status = s;
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

CoordinationService::BarrierCallback
CoordinationService::ConnectAfterBarrierPasses(absl::string_view task_name,
                                               uint64_t incarnation,
                                               StatusCallback done) {
  return [this, task = std::string(task_name), incarnation,
          done = std::move(done)](absl::Status s,
                                  int64_t unused_counter) mutable {
    state_mu_.AssertHeld();
    if (s.ok() && incarnation == cluster_state_[task]->GetTaskIncarnation()) {
      // Connect task to service.
      cluster_state_[task]->Connect();
      done(absl::OkStatus());
    } else if (s.ok() || absl::IsCancelled(s)) {
      // Avoid using `AbortedError` which typically has retry semantics.
      done(MakeCoordinationError(
          absl::AlreadyExistsError("Aborted connect attempt as there is a "
                                   "request from a newer incarnation.")));
    } else {
      // Non-cancellation error.
      done(s);
    }
  };
}

void CoordinationService::ConnectTask(const CoordinatedTask& task,
                                      uint64_t incarnation) {
  const std::string task_name = GetTaskName(task);
  const std::unique_ptr<TaskState>& task_state = cluster_state_[task_name];

  task_state->SetTaskIncarnation(incarnation);
  task_state->Connect();
  if (task_state->IsRecoverable()) {
    LeaveOngoingBarriers(task, "recoverable task silently connected again");
  }
}

void CoordinationService::RegisterTaskAsync(const CoordinatedTask& task,
                                            uint64_t incarnation,
                                            StatusCallback done) {
  const std::string task_name = GetTaskName(task);

  std::string error_message;
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    done(MakeCoordinationError(absl::InternalError(absl::StrCat(
        "Coordination service has stopped. RegisterTask() from task: ",
        task_name,
        " failed. This usually implies an earlier error that caused "
        "coordination service to shut down before the workers disconnect "
        "gracefully. Check the task leader's logs for an earlier error or "
        "scheduler events (e.g. preemption, eviction) to debug the root "
        "cause."))));
    return;
  }
  if (!cluster_state_.contains(task_name)) {
    // Note: return early here as unexpected task register errors should not
    // be propagated to other tasks.
    done(MakeCoordinationError(absl::InvalidArgumentError(absl::StrCat(
        "Unexpected task registered with task_name=", task_name))));
    return;
  }

  const std::unique_ptr<TaskState>& task_cluster_state =
      cluster_state_[task_name];
  task_cluster_state->SetRecoverable(task.recoverable());
  const auto task_state = task_cluster_state->GetState();
  const auto task_status = task_cluster_state->GetStatus();

  if (task_state == CoordinatedTaskState::TASKSTATE_DISCONNECTED ||
      ((allow_new_incarnation_to_reconnect_ ||
        task_cluster_state->IsRecoverable()) &&
       (absl::IsUnavailable(task_status) &&
        task_status.GetPayload(CoordinationErrorPayloadKey())))) {
    // The task is allowed to register itself if:
    // - this task is currently disconnected (registering for the first time
    //   or has called ResetTask() previously).
    // - this task has lost connection previously which caused it to have
    //   an unavailable error state, but has now restarted (possibly with
    //   a new incarnation). This is only allowed if configured with
    //   `allow_new_incarnation_to_reconnect`.
    if (cluster_register_with_barrier_) {
      // Impose barrier so that all tasks can register together.
      // Note: it is possible that the same task restarts multiple times and
      // registers itself with new incarnations.
      // That is okay; in this code branch, the tasks are not connected yet,
      // and the barrier has not succeeded yet.
      // There is no state that needs to be cleaned up.
      task_cluster_state->SetTaskIncarnation(incarnation);
      BarrierAsyncLocked(
          kClusterRegisterBarrierId, kUniqueBarrierCounter,
          cluster_register_timeout_, task, {},
          ConnectAfterBarrierPasses(task_name, incarnation, std::move(done)));
      return;
    }
    ConnectTask(task, incarnation);
    // TODO(b/369222279): Think about the barrier case - may need periodic
    // reporting of stragglers.
    LogConnectStatusLocked();
    done(absl::OkStatus());
    return;
  } else if (task_state == CoordinatedTaskState::TASKSTATE_CONNECTED) {
    // This may happen if the service processes the initial RegisterTask(),
    // but the agent did not receive the response so the agent retries again.
    if (task_cluster_state->GetTaskIncarnation() == incarnation ||
        task_cluster_state->IsRecoverable()) {
      // This should be a no-op, but we update the last heartbeat timestamp
      // to give a longer grace period for the agent to start sending
      // heartbeats.
      /// For recoverable tasks, we also leave any previously ongoing barriers.
      ConnectTask(task, incarnation);
      LogConnectStatusLocked();
      done(absl::OkStatus());
      return;
    } else {
      error_message =
          absl::StrCat(task_name,
                       " unexpectedly tried to connect with a different "
                       "incarnation. It has likely restarted.");
    }
  } else {
    // This task is already in error, which implies it has registered
    // previously.
    error_message =
        absl::StrCat(task_name,
                     " unexpectedly tried to connect while it is already in "
                     "error. ResetTask() should be called before a "
                     "subsequent connect attempt. Existing error: ",
                     task_status.ToString());
  }
  LOG(ERROR) << error_message;
  absl::Status error =
      MakeCoordinationError(absl::AbortedError(error_message), task);
  SetTaskError(task_name, error);
  PropagateError(error, {task});
  done(error);
}

void CoordinationService::WaitForAllTasks(const CoordinatedTask& task,
                                          const DeviceInfo& devices,
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
  BarrierAsync(device_propagation_barrier_id_, kUniqueBarrierCounter,
               kDevicePropagationTimeout, task, {},
               [done = std::move(done)](const absl::Status& s,
                                        int64_t unused_counter) { done(s); });
}

void CoordinationService::ShutdownTaskAsync(const CoordinatedTask& task,
                                            StatusCallback done) {
  VLOG(3) << "Task " << GetTaskName(task) << " invoked ShutdownTaskAsync()";
  if (shutdown_barrier_timeout_ > absl::ZeroDuration() && !task.recoverable()) {
    // Impose shutdown barrier so that all (non-recoverable) tasks can
    // disconnect together.
    // Notes:
    // 1. Recoverable tasks may disconnect and connect multiple times, thus we
    //    should not impose a barrier on them. Otherwise, the shutdown barrier
    //    will fail upon the first recoverable task restart.
    // 2. But a shutdown barrier across non-recoverable tasks is still needed
    //    to propagate any early shutdown errors from a non-recoverable task.
    // 3. Users should invoke their own barrier to signify the end of a
    //    workload so that there will still be a synchronized shutdown across
    //    all tasks.
    auto shutdown_tasks = GetTasksForShutdownBarrier();
    BarrierAsync(shutdown_barrier_id_, kUniqueBarrierCounter,
                 shutdown_barrier_timeout_, task, shutdown_tasks,
                 [done = std::move(done)](const absl::Status& s,
                                          int64_t unused_counter) {
                   if (s.ok()) {
                     done(absl::OkStatus());
                   } else {
                     done(MakeShutdownBarrierError(s));
                   }
                 });
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

absl::Status CoordinationService::ResetTask(const CoordinatedTask& task) {
  absl::MutexLock l(&state_mu_);
  return DisconnectTask(task);
}

absl::Status CoordinationService::DisconnectTask(const CoordinatedTask& task) {
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
  }
  const std::unique_ptr<TaskState>& task_state = cluster_state_[task_name];

  if (task_state->GetState() == CoordinatedTaskState::TASKSTATE_DISCONNECTED) {
    return MakeCoordinationError(absl::FailedPreconditionError(
        absl::StrCat("The task is already disconnected: ", task_name)));
  }

  // Disconnect task.
  task_state->Disconnect(
      /*grace_period_duration_us=*/heartbeat_timeout_ms_ * 1000);
  LeaveOngoingBarriers(task, "task disconnected");
  RefreshAliveness();
  error_polling_state_.RemoveTask(task, "task has disconnected.");
  LOG(INFO) << task_name << " has disconnected from coordination service.";
  return absl::OkStatus();
}

const DeviceInfo& CoordinationService::ListClusterDevices() {
  return cluster_devices_;
}

uint64_t CoordinationService::GetServiceIncarnation() {
  return service_incarnation_;
}

absl::Status CoordinationService::ReportTaskError(const CoordinatedTask& task,
                                                  const absl::Status& error) {
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
  PropagateError(error, {task}, /*is_reported_by_task=*/true);
  return absl::OkStatus();
}

CoordinatedTaskStateInfo CoordinationService::CreateTaskStateInfo(
    const CoordinatedTask& task, const TaskState& state) {
  CoordinatedTaskStateInfo info;
  info.set_state(state.GetState());
  info.set_incarnation(state.GetTaskIncarnation());
  absl::Status error = state.GetStatus();
  *info.mutable_task() = task;
  info.set_error_code(error.raw_code());
  info.set_error_message(std::string(error.message()));
  if (!error.ok()) {
    *info.mutable_error_payload()->mutable_source_task() = task;
    info.mutable_error_payload()->set_is_reported_error(false);
  }
  return info;
}

std::vector<CoordinatedTaskStateInfo> CoordinationService::GetTaskState(
    const std::vector<CoordinatedTask>& tasks) {
  std::vector<CoordinatedTaskStateInfo> states_info;
  states_info.reserve(tasks.size());

  absl::MutexLock l(&state_mu_);
  for (const auto& task : tasks) {
    states_info.push_back(
        CreateTaskStateInfo(task, *cluster_state_[GetTaskName(task)]));
  }
  return states_info;
}

std::vector<CoordinatedTaskStateInfo> CoordinationService::GetJobState(
    absl::string_view job_name) {
  absl::MutexLock l(&state_mu_);
  std::vector<CoordinatedTaskStateInfo> states_info;
  for (const auto& [name, task_state] : cluster_state_) {
    const CoordinatedTask task = GetTaskFromName(name);
    if (task.job_name() != job_name) {
      continue;
    }
    states_info.push_back(CreateTaskStateInfo(task, *cluster_state_[name]));
  }
  return states_info;
}

absl::Status CoordinationService::RecordHeartbeat(const CoordinatedTask& task,
                                                  uint64_t incarnation) {
  const std::string task_name = GetTaskName(task);
  absl::Status s = absl::OkStatus();
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    return MakeCoordinationError(absl::InternalError(absl::StrCat(
        "Coordination service has stopped. RecordHeartbeat() from task: ",
        task_name,
        " failed. This usually implies an earlier error that caused "
        "coordination service to shut down before the workers disconnect "
        "gracefully. Check the task leader's logs for an earlier error or "
        "scheduler events (e.g. preemption, eviction) to debug the root "
        "cause.")));
  } else if (!cluster_state_.contains(task_name)) {
    return MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Unexpected heartbeat request from task: ", task_name,
                     ". This usually implies a configuration error.")));
  }
  const std::unique_ptr<TaskState>& task_state = cluster_state_[task_name];
  if (!task_state->GetStatus().ok()) {
    return MakeCoordinationError(absl::AbortedError(absl::StrCat(
        "Unexpected heartbeat request from an already-in-error task: ",
        task_name,
        " with existing error: ", task_state->GetStatus().ToString())));
  } else if (task_state->IsDisconnectedBeyondGracePeriod()) {
    // We accept heartbeats for a short grace period to account for the lag
    // time between the service recording the state change and the agent
    // stopping heartbeats.
    return MakeCoordinationError(absl::InvalidArgumentError(
        absl::StrCat("Task with task_name=", task_name,
                     " must be registered before sending heartbeat messages. "
                     "The service might have restarted, please restart / reset "
                     "and register again.")));
  }
  VLOG(10) << "Record heartbeat from task: " << task_name
           << "at incarnation: " << incarnation << "at " << absl::Now();
  s = task_state->RecordHeartbeat(incarnation);

  // Set and propagate any heartbeat errors.
  if (!s.ok()) {
    SetTaskError(task_name, s);
    PropagateError(s, {task});
  }

  return s;
}

bool CoordinationService::AllTasksAreRecoverable(
    const std::vector<CoordinatedTask>& tasks) {
  for (const auto& task : tasks) {
    if (!cluster_state_[GetTaskName(task)]->IsRecoverable() &&
        !isRecoverableJob(task.job_name())) {
      return false;
    }
  }
  return true;
}

void CoordinationService::PropagateError(
    const absl::Status& error,
    const std::vector<absl::string_view>& source_task_names,
    bool is_reported_by_task) {
  std::vector<CoordinatedTask> source_tasks;
  source_tasks.reserve(source_task_names.size());
  for (const auto& task : source_task_names) {
    source_tasks.push_back(GetTaskFromName(task));
  }
  return PropagateError(error, source_tasks, is_reported_by_task);
}

void CoordinationService::PropagateError(
    const absl::Status& error, const std::vector<CoordinatedTask>& source_tasks,
    bool is_reported_by_task) {
  VLOG(3) << "PropagateError(): " << error;
  assert(!error.ok());
  assert(!source_tasks.empty());
  if (AllTasksAreRecoverable(source_tasks)) {
    VLOG(3) << "All tasks are recoverable, skip propagating error.";
    return;
  }
  // If there is no service-to-client connection, use error polling or stop
  // the service.
  if (client_cache_ == nullptr) {
    SendErrorPollingResponseOrFailAllTasks(error);
    return;
  }

  ReportErrorToTaskRequest request;
  request.set_error_code(error.raw_code());
  request.set_error_message(std::string(error.message()));
  CoordinationServiceError* payload = request.mutable_error_payload();
  payload->set_is_reported_error(is_reported_by_task);
  CallOptions call_opts;
  call_opts.SetTimeout(kServiceToClientTimeoutMs);
  // TODO(b/369222279): This logic will be removed shortly, so we don't bother
  // adding the full list of source tasks.
  if (!source_tasks.empty()) {
    *payload->mutable_source_task() = source_tasks[0];
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
        [response, n, task](const absl::Status& s) {
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
std::string NormalizeKey(absl::string_view orig_key) {
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

absl::Status CoordinationService::InsertKeyValue(absl::string_view key,
                                                 absl::string_view value) {
  VLOG(3) << "CoordinationService::InsertKeyValue(key=" << key
          << ", value=" << value << ")";
  return store_.Put(NormalizeKey(key), value, /*allow_overwrite=*/false);
}

absl::Status CoordinationService::InsertKeyValue(absl::string_view key,
                                                 absl::string_view value,
                                                 bool allow_overwrite) {
  VLOG(3) << "CoordinationService::InsertKeyValue(key=" << key
          << ", value=" << value << ", allow_overwrite=" << allow_overwrite
          << ")";
  return store_.Put(NormalizeKey(key), value, allow_overwrite);
}

void CoordinationService::GetKeyValueAsync(absl::string_view key,
                                           StatusOrValueCallback done) {
  VLOG(3) << "CoordinationService::GetKeyValueAsync(key=" << key << ")";
  store_.AddCallbackForKey(NormalizeKey(key), done);
}

absl::StatusOr<std::string> CoordinationService::TryGetKeyValue(
    absl::string_view key) {
  VLOG(3) << "CoordinationService::TryGetKeyValue(key=" << key << ")";
  std::optional<std::string> s = store_.Get(NormalizeKey(key));
  if (!s.has_value()) {
    return absl::NotFoundError(absl::StrCat("Config key ", key, " not found."));
  }
  return *std::move(s);
}

std::vector<KeyValueEntry> CoordinationService::GetKeyValueDir(
    absl::string_view directory_key) {
  VLOG(3) << "CoordinationService::GetKeyValueDir(directory_key="
          << directory_key << ")";
  return store_.GetPrefix(NormalizeKey(directory_key) + "/");
}

absl::Status CoordinationService::DeleteKeyValue(absl::string_view key) {
  VLOG(3) << "CoordinationService::DeleteKeyValue(key=" << key << ")";
  const std::string normalized = NormalizeKey(key);
  store_.Delete(normalized);
  store_.DeletePrefix(normalized + "/");
  return absl::OkStatus();
}

void CoordinationService::SetAllTasksError(const absl::Status& error) {
  for (const auto& task_state : cluster_state_) {
    SetTaskError(task_state.first, error);
  }
}

void CoordinationService::SetTaskError(absl::string_view task_name,
                                       const absl::Status& error) {
  const CoordinatedTask task = GetTaskFromName(task_name);
  const std::unique_ptr<TaskState>& task_state = cluster_state_[task_name];
  if (task_state->SetError(error)) {
    LeaveOngoingBarriers(
        task, absl::StrCat("task is set to ERROR: ", error.ToString()));
    RefreshAliveness();
  }
}

void CoordinationService::PollForErrorAsync(const CoordinatedTask& task,
                                            StatusCallback done) {
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

// Initializes a new barrier. Returns false if the barrier should fail
// immediately.
bool CoordinationService::InitializeBarrier(
    BarrierState* barrier, absl::string_view barrier_id, int64_t counter,
    absl::Duration timeout, const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    BarrierCallback done) {
  // Initialize barrier state.
  barrier->id = barrier_id;
  barrier->counter = counter;
  barrier->passed = false;
  barrier->result = absl::UnknownError("Invalid barrier result.");
  barrier->initiating_task = task;
  barrier->done_callbacks.clear();
  if (!InitializeTasksAtBarrier(barrier, participating_tasks, done)) {
    return false;
  }

  barrier->num_pending_tasks = barrier->tasks_at_barrier.size();

  // Fail the barrier immediately if any tasks are already in error.
  for (const auto& pending_task : barrier->tasks_at_barrier) {
    const std::string task_name = GetTaskName(pending_task.first);
    const std::unique_ptr<TaskState>& task_cluster_state =
        cluster_state_[task_name];
    if (!task_cluster_state->IsRecoverable() &&
        task_cluster_state->GetState() ==
            CoordinatedTaskState::TASKSTATE_ERROR) {
      absl::Status error = MakeBarrierError(
          absl::InternalError(absl::StrCat(
              "Task (", task_name,
              ") is already in error before the barrier "
              "was called. Barrier Id: ",
              BarrierName(*barrier),
              " Task error: ", task_cluster_state->GetStatus().ToString())),
          barrier->id, barrier->counter);
      PassBarrier(barrier, error);
      done(error, counter);
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

bool CoordinationService::InitializeTasksAtBarrier(
    BarrierState* barrier,
    const std::vector<CoordinatedTask>& participating_tasks,
    BarrierCallback done) {
  // Tasks were already specified for a previous barrier counter. Reset state
  // and return early.
  if (!barrier->tasks_at_barrier.empty()) {
    for (auto& it : barrier->tasks_at_barrier) {
      it.second = false;
    }
    return true;
  }
  if (participating_tasks.empty()) {
    // Assume barrier is for entire cluster if no tasks are specified.
    for (const auto& task_state : cluster_state_) {
      absl::string_view task_name = task_state.first;
      barrier->tasks_at_barrier[GetTaskFromName(task_name)] = false;
    }
    return true;
  }
  // Rely on the caller-specified task args.
  for (const auto& task : participating_tasks) {
    // Fail the barrier immediately if unexpected task is included in the
    // barrier.
    const std::string task_name = GetTaskName(task);
    if (!cluster_state_.contains(task_name)) {
      absl::Status error = MakeBarrierError(
          absl::InvalidArgumentError(
              absl::StrCat("Unexpected task (", task_name,
                           ") that is not in the cluster called the barrier. "
                           "Barrier Id: ",
                           BarrierName(*barrier))),
          barrier->id, barrier->counter);
      AddBarrierCallback(barrier, task, std::move(done));
      PassBarrier(barrier, error);
      return false;
    }
    barrier->tasks_at_barrier[task] = false;
  }
  return true;
}

void CoordinationService::AddBarrierCallback(BarrierState* barrier,
                                             const CoordinatedTask& task,
                                             BarrierCallback done) {
  auto it = barrier->done_callbacks.find(task);
  if (it != barrier->done_callbacks.end()) {
    it->second(absl::CancelledError(
                   absl::StrCat("Cancelled because there's a more recent "
                                "barrier call for ",
                                BarrierName(*barrier), " from the same task.")),
               barrier->counter);
  }
  barrier->done_callbacks[task] = std::move(done);
}

void CoordinationService::BarrierAsync(
    // Note: `barrier_id` uses a `std::string` instead of `string_view` as the
    // RPC may end (i.e. done callback is invoked) before this handler
    // completes, which would invalidate the `string_view`.
    std::string barrier_id, int64_t counter, absl::Duration timeout,
    const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    BarrierCallback done) {
  absl::MutexLock l(&state_mu_);
  return BarrierAsyncLocked(barrier_id, counter, timeout, task,
                            participating_tasks, std::move(done));
};

void CoordinationService::BarrierAsyncLocked(
    absl::string_view barrier_id, int64_t counter, absl::Duration timeout,
    const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& participating_tasks,
    BarrierCallback done) {
  VLOG(3) << "Task " << GetTaskName(task) << " invoked BarrierAsync("
          << BarrierName(barrier_id, counter) << ").";

  // Check if coordination service has stopped. If so, return an error
  // immediately.
  if (ServiceHasStopped()) {
    done(MakeBarrierError(
             absl::InternalError(absl::StrCat(
                 "Barrier ", BarrierName(barrier_id, counter),
                 " requested after coordination service has shut down.")),
             barrier_id, counter),
         counter);
    return;
  }

  // For the first barrier counter of a unique id, this will create an empty
  // struct.
  auto* barrier = &barriers_[barrier_id];

  // Maybe create new barrier instance.
  bool should_initialize_new_instance =
      // First barrier for this id.
      (BarrierIsUninitialized(*barrier) && (counter == 0)) ||
      // Previous barrier instance has passed. New barrier instance
      // requested.
      (barrier->passed && counter == (barrier->counter + 1));

  // Task just restarted and is starting a new barrier (since previous
  // barrier passed).
  if (  // Barrier has been passed before.
      barrier->passed &&
      // Task has likely just restarted.
      task.recoverable() && counter == 0 &&
      // Not a special once-only barrier.
      barrier_id != kClusterRegisterBarrierId &&
      barrier_id != device_propagation_barrier_id_ &&
      barrier_id != shutdown_barrier_id_) {
    should_initialize_new_instance = true;
    // Use the service's counter to initialize the new barrier.
    counter = barrier->counter + 1;
  }

  if (should_initialize_new_instance) {
    // Initialize new barrier instance state.
    if (!InitializeBarrier(barrier, barrier_id, counter, timeout, task,
                           participating_tasks, done)) {
      return;  // Exit early if barrier init failed.
    }
  }

  // Add pending callbacks.
  AddBarrierCallback(barrier, task, std::move(done));

  if (!should_initialize_new_instance && counter != barrier->counter &&
      // Recoverable tasks are allowed to use different counters and rejoin the
      // barrier.
      !(task.recoverable() && counter == 0)) {
    // Counter mismatch! This is likely due to a restart.
    FailBarrierWithCounterMismatch(barrier, counter);
    return;
  }

  // Check if task args are specified consistently across barrier calls, and if
  // caller is involved in the barrier.
  if (!ValidateTaskArgs(barrier, task, participating_tasks)) {
    return;
  }

  if (barrier->passed && counter == barrier->counter) {
    // Same counter, but barrier has already passed: return previous result.
    // This will rarely happen, except if the RPC layer retries the RPC
    // multiple times somehow.
    RepeatBarrierResult(barrier, task);
    return;
  }

  // Task has reached the barrier.
  ReachBarrier(barrier, task);
}

void CoordinationService::FailBarrierWithCounterMismatch(BarrierState* barrier,
                                                         int64_t counter) {
  std::string reason;
  if (counter == 0 || barrier->counter == 0) {
    reason =
        "The service or task probably restarted, check the earlier logs to "
        "debug further. Usually, a restart on the stale tasks will resolve "
        "the issue.";
  } else {
    reason =
        "One task is probably way too quick / slow in its execution. For "
        "example, one task started a barrier early, which timed out, and "
        "started a second barrier before the late task even joined once.";
  }

  // Counter mismatch! This is likely due to a restart.
  absl::Status error = MakeBarrierError(
      absl::InternalError(absl::StrCat(
          "Barrier (", BarrierName(barrier->id, counter),
          ") is invoked by the task, but the service thinks it is the ",
          barrier->counter, "-th instance. ", reason)),
      barrier->id, barrier->counter);
  PassBarrier(barrier, error);
}

absl::Status CoordinationService::CancelBarrier(
    // Note: `barrier_id` uses a `std::string` instead of `string_view` as the
    // RPC may end (i.e. done callback is invoked) before this handler
    // completes, which would invalidate the `string_view`.
    std::string barrier_id, int64_t counter, const CoordinatedTask& task) {
  std::string barrier_name = BarrierName(barrier_id, counter);
  absl::MutexLock l(&state_mu_);
  if (ServiceHasStopped()) {
    return MakeBarrierError(
        absl::InternalError(absl::StrCat(
            "Coordination service has stopped. CancelBarrier() for ",
            barrier_name, " failed.")),
        barrier_id, counter);
  }
  auto [it, inserted] = barriers_.try_emplace(barrier_id);
  auto* barrier = &it->second;
  if (inserted) {
    LOG(WARNING) << "Barrier (" << barrier_name
                 << ") is cancelled before being created by task: "
                 << GetTaskName(task);
  }
  // Cancelling stale barrier instance.
  if (barrier->counter != counter) {
    return MakeBarrierError(
        absl::FailedPreconditionError(
            absl::StrCat("Barrier (", barrier_name,
                         ") is cancelled by task, but the service thinks "
                         "it is the ",
                         barrier->counter,
                         "-th instance. This is likely due to a restart in the "
                         "task or service.")),
        barrier_id, barrier->counter);
  }
  // Barrier has already been passed.
  if (barrier->passed) {
    return MakeBarrierError(absl::FailedPreconditionError(absl::StrCat(
                                "Barrier (", barrier_name,
                                ") has already been passed with status code: ",
                                barrier->result.code())),
                            barrier_id, barrier->counter);
  }

  // Cancel barrier.
  absl::Status cancelled =
      MakeBarrierError(absl::CancelledError(absl::StrCat(
                           "Barrier (", barrier_name,
                           ") is cancelled by task: ", GetTaskName(task))),
                       barrier_id, barrier->counter);
  PassBarrier(barrier, cancelled);

  VLOG(3) << "Barrier (" << barrier_name << ") is cancelled.";
  return absl::OkStatus();
}

// Mark barrier as passed.
void CoordinationService::PassBarrier(BarrierState* barrier,
                                      const absl::Status& result) {
  barrier->passed = true;
  barrier->result = result;
  VLOG(3) << "Barrier(" << BarrierName(*barrier)
          << ") has passed with status: " << result;
  // Special hook for device propagation barrier to set global device ids.
  if (barrier->id == device_propagation_barrier_id_) {
    AggregateClusterDevices();
  }
  for (const auto& task_at_barrier : barrier->tasks_at_barrier) {
    // Clean up task state (used as error hooks).
    const CoordinatedTask& task = task_at_barrier.first;
    cluster_state_[GetTaskName(task)]->ExitBarrier(barrier->id);
  }
  ongoing_barriers_.erase(barrier->id);
  // Propagate results to participating tasks.
  for (const auto& [_, callback] : barrier->done_callbacks) {
    callback(result, barrier->counter);
  }
  barrier->done_callbacks.clear();
  if (barrier->id == kClusterRegisterBarrierId && !result.ok()) {
    // Set all tasks to error.
    absl::Status register_error =
        MakeCoordinationError(absl::InternalError(absl::StrCat(
            "Cluster registration failed with error: ", result.ToString())));
    SetAllTasksError(register_error);
    LOG(ERROR)
        << "Stopping coordination service as cluster registration failed. This "
           "may be due to 1) some tasks crashed earlier before connecting, 2) "
           "some tasks were never scheduled, or 3) scheduling delays. Consider "
           "setting a longer initialization timeout if such delays are "
           "expected, the timeout is currently set to: "
        << cluster_register_timeout_ << ".\n\nOriginal error: " << result;
    return;
  }
  // Special hook for shutdown barrier to disconnect tasks at the barrier and
  // propagate errors to those that have not.
  if (barrier->id == shutdown_barrier_id_) {
    CompleteShutdownAfterBarrier(result, barrier);
  }
  if (ServiceHasStopped()) {
    return;
  }
}

// Returns true if x is a (non-strict) subset of y.
bool TaskSetSubset(const CoordinationService::CoordinatedTaskSet& x,
                   const CoordinationService::CoordinatedTaskSet& y) {
  return std::all_of(x.begin(), x.end(), [&y](const CoordinatedTask& task) {
    return y.contains(task);
  });
}

// Returns true if sets x and y are equal.
//
// Note that the default equality operator (==) on absl::flat_hash_set invokes
// the equal operator on the underlying elements in the sets, but the equal
// operator is not defined on protos. Thus, we have to implement our own
// equality function.
bool TaskSetEqual(const CoordinationService::CoordinatedTaskSet& x,
                  const CoordinationService::CoordinatedTaskSet& y) {
  return x.size() == y.size() && TaskSetSubset(x, y);
}

CoordinationService::CoordinatedTaskSet CoordinationService::AliveTasks(
    const CoordinatedTaskSet& tasks) const {
  CoordinatedTaskSet alive_tasks;
  for (const CoordinatedTask& task : tasks) {
    auto it = cluster_state_.find(GetTaskName(task));
    if (it != cluster_state_.end() &&
        it->second->GetState() == CoordinatedTaskState::TASKSTATE_CONNECTED) {
      // We consider a task alive if it is CONNECTED.
      alive_tasks.insert(task);
    }
  }
  return alive_tasks;
}

void CoordinationService::RefreshAliveness() {
  // Try to finish every pending GetAliveTasks call.
  auto it = aliveness_states_.begin();
  while (it != aliveness_states_.end()) {
    CoordinatedTaskSet alive_tasks = AliveTasks(it->tasks);
    if (TaskSetSubset(alive_tasks, it->in_barrier)) {
      // Every alive task is in the barrier, so the barrier is satisfied. Return
      // the same set of alive tasks (alive_tasks) to every task in the barrier.
      std::vector<CoordinatedTask> v{alive_tasks.begin(), alive_tasks.end()};
      for (const GetAliveTasksCallback& done : it->dones) {
        done(absl::OkStatus(), v);
      }

      // Remove the pending GetAliveTasks call because it is no longer pending.
      it = aliveness_states_.erase(it);
    } else {
      // The pending GetAliveTasks call is still pending.
      ++it;
    }
  }
}

void CoordinationService::GetAliveTasksAsync(
    const tensorflow::CoordinatedTask& requesting_task,
    const std::vector<tensorflow::CoordinatedTask>& tasks,
    GetAliveTasksCallback done) {
  // TODO(mwhittaker): Figure out good timeout semantics and add timeouts.

  // Validate that the requesting task is a member of tasks.
  CoordinatedTaskSet task_set{tasks.begin(), tasks.end()};
  if (!task_set.contains(requesting_task)) {
    // TODO(mwhittaker): Consider relaxing the requirement that the requesting
    // task is one of the specified tasks.
    absl::Status err = absl::InvalidArgumentError(absl::StrCat(
        "Requesting task ", GetTaskName(requesting_task),
        " is not one of the tasks specified in a GetAliveTasks request."));
    done(err, {});
    return;
  }

  // Find the corresponding AlivenessState, creating a new one if needed.
  absl::MutexLock l(&state_mu_);
  auto it = std::find_if(aliveness_states_.begin(), aliveness_states_.end(),
                         [&task_set](const AlivenessState& state) {
                           return TaskSetEqual(state.tasks, task_set);
                         });
  if (it == aliveness_states_.end()) {
    aliveness_states_.push_back(AlivenessState{task_set});
    it = std::prev(aliveness_states_.end());
  }

  // Enter the requesting task into the barrier.
  it->in_barrier.insert(requesting_task);
  it->dones.push_back(std::move(done));

  // Finish the barrier, if possible.
  CoordinatedTaskSet alive_tasks = AliveTasks(task_set);
  if (TaskSetSubset(alive_tasks, it->in_barrier)) {
    std::vector<CoordinatedTask> v{alive_tasks.begin(), alive_tasks.end()};
    for (const GetAliveTasksCallback& done : it->dones) {
      done(absl::OkStatus(), v);
    }
    aliveness_states_.erase(it);
  }
}

void CoordinationService::SendErrorPollingResponse(const absl::Status& error) {
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

bool CoordinationService::ValidateTaskArgs(
    BarrierState* barrier, const CoordinatedTask& task,
    const std::vector<CoordinatedTask>& tasks_args) {
  // Assume all tasks are participating if no task is specified.
  if (tasks_args.empty()) {
    if (barrier->tasks_at_barrier.size() != cluster_state_.size()) {
      absl::Status error = MakeBarrierError(
          absl::InvalidArgumentError(absl::StrCat(
              BarrierName(*barrier),
              ": No tasks were specified by the client, which implies "
              "that the entire cluster is participating in the "
              "barrier. However, the service disagrees. Check that "
              "the callers are invoking the barrier with "
              "the same arguments.")),
          barrier->id, barrier->counter);
      PassBarrier(barrier, error);
      return false;
    }
    return true;
  }
  // Otherwise, check that the specified tasks are correct.
  if (barrier->tasks_at_barrier.size() != tasks_args.size() ||
      absl::c_any_of(tasks_args, [&](const CoordinatedTask& task) {
        return !barrier->tasks_at_barrier.contains(task);
      })) {
    absl::Status error = MakeBarrierError(
        absl::InvalidArgumentError(absl::StrCat(
            "Conflicting tasks specified by different processes for the same "
            "barrier: ",
            BarrierName(*barrier),
            ". Check that the callers are invoking the barrier with the same "
            "arguments.")),
        barrier->id, barrier->counter);
    PassBarrier(barrier, error);
    return false;
  }
  // Check if the caller task is included in the list of participating tasks.
  if (!barrier->tasks_at_barrier.contains(task)) {
    absl::Status error =
        MakeBarrierError(absl::InvalidArgumentError(absl::StrCat(
                             "A non-participating task (", GetTaskName(task),
                             ") called the barrier: ", BarrierName(*barrier))),
                         barrier->id, barrier->counter);
    PassBarrier(barrier, error);
    return false;
  }
  return true;
}

void CoordinationService::RepeatBarrierResult(BarrierState* barrier,
                                              const CoordinatedTask& task) {
  BarrierCallback done = barrier->done_callbacks[task];
  barrier->done_callbacks.erase(task);
  // Special hook for shutdown barrier to disconnect task.
  if (barrier->id == shutdown_barrier_id_) {
    absl::Status s = DisconnectTask(task);
    // Return any errors from the disconnect attempt, otherwise return the
    // barrier status outside of this hook.
    if (!s.ok()) {
      done(s, barrier->counter);
      return;
    }
  }
  done(barrier->result, barrier->counter);
}

void CoordinationService::LeaveOngoingBarriers(const CoordinatedTask& task,
                                               absl::string_view reason) {
  const std::string task_name = GetTaskName(task);
  const std::unique_ptr<TaskState>& task_state = cluster_state_[task_name];
  // Unregister recoverable task from ongoing barriers.
  if (task_state->IsRecoverable()) {
    for (const auto& barrier_id : task_state->GetOngoingBarriers()) {
      BarrierState* barrier = &barriers_[barrier_id];
      // Unregister task from barrier.
      if (absl::GetFlag(FLAGS_leave_barriers_on_recoverable_agent_restart)) {
        if (barrier->tasks_at_barrier.contains(task)) {
          // Remove task from barrier.
          barrier->recoverable_tasks_restarted_during_barrier.insert(task);
        }
      } else {
        if (barrier->tasks_at_barrier[task]) {
          barrier->tasks_at_barrier[task] = false;
          ++barrier->num_pending_tasks;
        }
      }

      // Cancel any pending callbacks.
      auto it = barrier->done_callbacks.find(task);
      if (it != barrier->done_callbacks.end()) {
        it->second(
            MakeBarrierError(
                absl::CancelledError(absl::StrCat(
                    "Stale barrier invocation is cancelled because:  ", reason,
                    ". Barrier: ", BarrierName(*barrier), " Task ", task_name)),
                barrier->id, barrier->counter),
            barrier->counter);
        barrier->done_callbacks.erase(it);
      }
    }
    return;
  }
  // Fail ongoing barriers since task is not recoverable.
  for (const auto& barrier_id : task_state->GetOngoingBarriers()) {
    BarrierState* barrier = &barriers_[barrier_id];
    absl::Status error = MakeBarrierError(
        absl::InternalError(absl::StrCat(
            "Barrier failed because: ", reason,
            ". Barrier Id: ", BarrierName(*barrier), ", Task: ", task_name)),
        barrier->id, barrier->counter);
    PassBarrier(barrier, error);
  }
}

void CoordinationService::ReachBarrier(BarrierState* barrier,
                                       const CoordinatedTask& task) {
  // Remove pending task.
  // We need to check if task made a repeated call after reaching the
  // barrier.
  if (!barrier->tasks_at_barrier[task]) {
    barrier->tasks_at_barrier[task] = true;
    --barrier->num_pending_tasks;

    if (barrier->num_pending_tasks == 0) {
      // Everyone has reached the barrier!
      PassBarrier(barrier, absl::OkStatus());
      return;
    }
  }
};

void CoordinationService::AggregateClusterDevices() {
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

void CoordinationService::DisconnectAllNonRecoverableTasks() {
  for (const auto& [task_name, state] : cluster_state_) {
    if (state->IsRecoverable()) {
      // Recoverable tasks will disconnect independently without the
      // barrier.
      continue;
    }
    auto s = DisconnectTask(GetTaskFromName(task_name));
    if (!s.ok()) {
      LOG(ERROR) << "Failed to disconnect task " << task_name << ": " << s;
    }
  }
}

std::vector<CoordinatedTask> CoordinationService::GetTasksForShutdownBarrier() {
  absl::MutexLock l(&state_mu_);
  if (shutdown_barrier_tasks_.empty()) {
    for (const auto& [task_name, task_state] : cluster_state_) {
      if (!task_state->IsRecoverable()) {
        shutdown_barrier_tasks_.push_back(GetTaskFromName(task_name));
      }
    }
  }
  return shutdown_barrier_tasks_;
}

void CoordinationService::CompleteShutdownAfterBarrier(
    const absl::Status& result, BarrierState* barrier) {
  if (result.ok()) {
    LOG(INFO) << "Shutdown barrier in coordination service has passed.";
    DisconnectAllNonRecoverableTasks();
  } else {
    LOG(ERROR) << "Shutdown barrier in coordination service has failed:\n"
               << result
               << "\nThis suggests that the workers are out of sync. Either at "
                  "least one worker (a) crashed early due to program error or "
                  "scheduler events (e.g. preemption, eviction), (b) was "
                  "too fast in its execution, or (c) too slow / hanging. Check "
                  "the logs (both the program and scheduler events) for an "
                  "earlier error to identify the root cause.";
    absl::Status shutdown_error = MakeShutdownBarrierError(result);
    // Propagate error to all tasks.
    // 1. Identify all straggling tasks.
    std::vector<CoordinatedTask> straggling_tasks;
    for (const auto& [task, at_barrier] : barrier->tasks_at_barrier) {
      if (!at_barrier) {
        straggling_tasks.push_back(task);
      }
    }
    // 2. Propagate error to everybody.
    PropagateError(shutdown_error, straggling_tasks);
    // 3. Set all tasks to error, to prevent unexpected connections from
    // restarted tasks.
    // This is done after (2) so that all tasks get the shutdown error.
    // (already-errored tasks don't receive new notifications)
    // For restarted tasks that hit this error, they should retry until
    // there is a new service instance.
    SetAllTasksError(shutdown_error);
  }
}

bool CoordinationService::isRecoverableJob(
    const absl::string_view task_name) const {
  return recoverable_jobs_.find(task_name) != recoverable_jobs_.end();
}

void CoordinationService::SendErrorPollingResponseOrFailAllTasks(
    const absl::Status& error) {
  CHECK(!error.ok()) << "SendErrorPollingResponseOrFailAllTasks called with OK "
                        "status. Should always return an error.";
  // Should be called only when there is no service-to-client connection.
  assert(client_cache_ == nullptr);
  if (IsClientPollingForError()) {
    LOG(ERROR)
        << "Use error polling to propagate the following error to all tasks: "
        << error;
    SendErrorPollingResponse(error);
  } else {
    absl::Status unheard_error =
        MakeCoordinationError(absl::InternalError(absl::StrCat(
            "All tasks were set to error because coordination service "
            "encountered an error, but was unable to inform clients. Error: ",
            error.ToString())));
    LOG(ERROR) << unheard_error;
    SetAllTasksError(unheard_error);
  }
}

bool CoordinationService::IsClientPollingForError() const {
  return client_polling_for_error_;
}

}  // namespace tsl
