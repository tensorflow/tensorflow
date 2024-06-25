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

#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include "xla/tsl/framework/cancellation.h"
#include "tsl/lib/monitoring/gauge.h"
#include "tsl/platform/env.h"
#include "tsl/platform/random.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/protobuf/coordination_config.pb.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace tsl {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinatedTaskState;
using tensorflow::CoordinatedTaskStateInfo;
using tensorflow::CoordinationServiceConfig;
using tensorflow::DeviceInfo;
using tensorflow::KeyValueEntry;

namespace {
auto* enabled_usage_metric =
    monitoring::Gauge<bool, 0>::New("/coordination_service/agent/enabled",
                                    "Tracks usage of coordination service.");

constexpr absl::Duration kDefaultClusterRegisterTimeout = absl::Hours(1);
constexpr absl::Duration kDefaultHeartbeatTimeout = absl::Seconds(10);
constexpr absl::Duration kDefaultShutdownTimeout = absl::Seconds(10);
constexpr char kHeartbeatThread[] = "CoordinationServiceHeartbeatLoop";

class CoordinationServiceAgentImpl : public CoordinationServiceAgent {
 public:
  CoordinationServiceAgentImpl() = default;
  ~CoordinationServiceAgentImpl() override {
    absl::Status s = ShutdownInternal();
    VLOG(3) << "Coordination agent dtor failed with status: " << s;
  }
  absl::Status Initialize(Env* env, std::string_view job_name, int task_id,
                          const CoordinationServiceConfig& configs,
                          std::unique_ptr<CoordinationClient> leader_client,
                          StatusCallback error_fn) override;
  absl::Status Initialize(Env* env, const CoordinatedTask& task,
                          const CoordinationServiceConfig& configs,
                          std::unique_ptr<CoordinationClient> leader_client,
                          StatusCallback error_fn) override;
  bool IsInitialized() override;
  bool IsConnected() override;
  bool IsError() override;

  absl::Status Connect() override;
  absl::Status WaitForAllTasks(const DeviceInfo& local_devices) override;
  const DeviceInfo& GetClusterDeviceInfo() override;
  absl::StatusOr<CoordinatedTask> GetOwnTask() override;
  absl::StatusOr<std::vector<CoordinatedTaskStateInfo>> GetTaskState(
      const std::vector<CoordinatedTask>& task) override;
  absl::Status ReportError(const absl::Status& error) override;
  absl::Status Shutdown() override;
  absl::Status Reset() override;

  absl::StatusOr<std::string> GetKeyValue(std::string_view key) override;
  absl::StatusOr<std::string> GetKeyValue(std::string_view key,
                                          absl::Duration timeout) override;
  std::shared_ptr<CallOptions> GetKeyValueAsync(
      std::string_view key, StatusOrValueCallback done) override;
  absl::StatusOr<std::string> TryGetKeyValue(std::string_view key) override;
  absl::StatusOr<std::vector<KeyValueEntry>> GetKeyValueDir(
      std::string_view key) override;
  void GetKeyValueDirAsync(std::string_view key,
                           StatusOrValueDirCallback done) override;
  absl::Status InsertKeyValue(std::string_view key,
                              std::string_view value) override;
  absl::Status InsertKeyValue(std::string_view key, std::string_view value,
                              bool allow_overwrite) override;
  absl::Status DeleteKeyValue(std::string_view key) override;
  absl::Status UpdateKeyValue(std::string_view key,
                              std::string_view value) override;

  absl::Status StartWatchKey(std::string_view key,
                             ChangedKeyValuesCallback on_change) override;
  absl::Status StopWatchKey(std::string_view key) override;
  absl::Status WaitAtBarrier(
      std::string_view barrier_id, absl::Duration timeout,
      const std::vector<CoordinatedTask>& tasks) override;
  void WaitAtBarrierAsync(std::string_view barrier_id, absl::Duration timeout,
                          const std::vector<CoordinatedTask>& tasks,
                          StatusCallback done) override;
  absl::Status CancelBarrier(std::string_view barrier_id) override;
  void CancelBarrierAsync(std::string_view barrier_id,
                          StatusCallback done) override;

  absl::StatusOr<Env*> GetEnv() override;

 protected:
  void SetError(const absl::Status& error) override;
  absl::Status ActivateWatch(
      std::string_view key, const std::map<std::string, std::string>&) override;
  // Returns an error if agent is not running. If `allow_disconnected` is true,
  // returns OK even if the agent is in DISCONNECTED state.
  absl::Status ValidateRunningAgent(bool allow_disconnected = false);
  void StopHeartbeat();

 private:
  absl::Status ShutdownInternal();

  Env* env_ = nullptr;  // Not owned.
  const uint64_t incarnation_id_ = random::New64();
  CoordinatedTask task_;
  CoordinationServiceConfig configs_;
  StatusCallback error_fn_;

  mutable absl::Mutex state_mu_;
  CoordinatedTaskState state_ TF_GUARDED_BY(state_mu_) =
      CoordinatedTaskState::TASKSTATE_UNINITIALIZED;
  absl::Status status_ TF_GUARDED_BY(state_mu_) = absl::OkStatus();
  // Note: this set grows without bounds. For now, this is okay as most users
  // require < 100 barriers. If there is a use case that requires many barriers,
  // consider using a monotonic sequence number to track instead.
  absl::flat_hash_set<std::string> used_barrier_ids_ TF_GUARDED_BY(state_mu_);

  uint64_t leader_incarnation_ = 0;
  DeviceInfo cluster_devices_;

  absl::Mutex heartbeat_thread_shutdown_mu_;
  absl::CondVar heartbeat_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(heartbeat_thread_shutdown_mu_) = false;
  std::unique_ptr<Thread> heartbeat_thread_;
  // Must outlive coordination client which may need to access it within
  // GetKeyValueAsync() callbacks.
  CancellationManager cancellation_manager_;
  std::unique_ptr<CoordinationClient> leader_client_;

  CoordinationServiceAgentImpl(const CoordinationServiceAgentImpl&) = delete;
  void operator=(const CoordinationServiceAgentImpl&) = delete;
};

absl::Status CoordinationServiceAgentImpl::Initialize(
    Env* env, std::string_view job_name, int task_id,
    const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
  CoordinatedTask task;
  task.set_job_name(std::string(job_name));
  task.set_task_id(task_id);
  return Initialize(env, task, configs, std::move(leader_client), error_fn);
}

absl::Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const CoordinatedTask& task,
    const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
  enabled_usage_metric->GetCell()->Set(true);
  absl::MutexLock l(&state_mu_);
  if (state_ != CoordinatedTaskState::TASKSTATE_UNINITIALIZED) {
    return MakeCoordinationError(absl::FailedPreconditionError(
        "Coordination service agent has already been initialized."));
  }

  env_ = env;
  task_ = task;
  configs_ = configs;
  if (configs_.service_leader().empty()) {
    return MakeCoordinationError(absl::InvalidArgumentError(
        "CoordinationServiceAgent must be initialized with a valid leader."));
  }
  leader_client_ = std::move(leader_client);
  if (leader_client_ == nullptr) {
    return MakeCoordinationError(absl::InvalidArgumentError(
        "CoordinationServiceAgent must have a valid leader client."));
  }
  error_fn_ = error_fn;
  state_ = CoordinatedTaskState::TASKSTATE_DISCONNECTED;
  return absl::OkStatus();
}

bool CoordinationServiceAgentImpl::IsInitialized() {
  absl::MutexLock l(&state_mu_);
  return state_ != CoordinatedTaskState::TASKSTATE_UNINITIALIZED;
}

bool CoordinationServiceAgentImpl::IsConnected() {
  absl::MutexLock l(&state_mu_);
  return state_ == CoordinatedTaskState::TASKSTATE_CONNECTED;
}

bool CoordinationServiceAgentImpl::IsError() {
  absl::MutexLock l(&state_mu_);
  return state_ == CoordinatedTaskState::TASKSTATE_ERROR;
}

void CoordinationServiceAgentImpl::StopHeartbeat() {
  {
    absl::MutexLock l(&heartbeat_thread_shutdown_mu_);
    shutting_down_ = true;
    heartbeat_thread_cv_.SignalAll();
  }
  heartbeat_thread_ = nullptr;
}

absl::Status CoordinationServiceAgentImpl::Connect() {
  VLOG(3) << "Agent has started trying to Connect().";
  {
    absl::MutexLock l(&state_mu_);
    if (state_ != CoordinatedTaskState::TASKSTATE_DISCONNECTED) {
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Coordination service agent is not in DISCONNECTED state."));
    }
  }
  absl::Status connect_status =
      absl::UnknownError("Connection not attempted yet.");
  RegisterTaskRequest request;
  *request.mutable_source_task() = task_;
  request.set_incarnation(incarnation_id_);
  RegisterTaskResponse response;

  const int64_t register_timeout =
      configs_.cluster_register_timeout_in_ms() > 0
          ? configs_.cluster_register_timeout_in_ms()
          : absl::ToInt64Milliseconds(kDefaultClusterRegisterTimeout);
  const absl::Time deadline =
      absl::Now() + absl::Milliseconds(register_timeout);
  int attempt = 0;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  do {
    ++attempt;
    CallOptions call_opts;
    call_opts.SetTimeout(absl::ToInt64Milliseconds(deadline - absl::Now()));
    absl::Notification n;
    leader_client_->RegisterTaskAsync(
        &call_opts, &request, &response, [&](absl::Status s) {
          if (s.ok()) {
            leader_incarnation_ = response.leader_incarnation();
            {
              absl::MutexLock l(&state_mu_);
              state_ = CoordinatedTaskState::TASKSTATE_CONNECTED;
            }
          }
          connect_status = s;
          n.Notify();
        });
    n.WaitForNotification();

    if (!connect_status.ok()) {
      // Exponential backoff with jitter. Note we will retry for `init_timeout`
      // time in total; the `14` here corresponds to an ~16s maximum interval
      // between connection attempts.
      const int backoff = 1 << std::min(14, attempt);
      absl::SleepFor(absl::Milliseconds(backoff * distribution(generator)));
    }
  } while (!connect_status.ok() && absl::Now() < deadline &&
           // Retries are attempted for:
           // 1. RPC errors.
           // 2. aborted duplicate task registration error - this means that
           // this task restarted and is trying to reconnect but the service
           // has not restarted yet.
           // 3. service has not been enabled - this could happen in the single
           // client scenario, where the server has been started but the service
           // cannot be used yet (nullptr). Presumably the service is in the
           // process of being enabled.
           (connect_status.GetPayload(CoordinationErrorPayloadKey()) ==
                std::nullopt ||
            absl::IsAborted(connect_status) ||
            absl::IsInternal(connect_status)));
  if (!connect_status.ok()) {
    SetError(connect_status);
    return connect_status;
  }

  LOG(INFO) << "Coordination agent has successfully connected.";
  heartbeat_thread_.reset(
      env_->StartThread(ThreadOptions(), kHeartbeatThread, [this]() -> void {
        HeartbeatRequest request;
        *request.mutable_source_task() = task_;
        request.set_incarnation(incarnation_id_);
        HeartbeatResponse response;
        const int64_t heartbeat_interval_ms =
            configs_.heartbeat_timeout_in_ms() > 0
                ? configs_.heartbeat_timeout_in_ms() / 2
                : absl::ToInt64Milliseconds(kDefaultHeartbeatTimeout) / 2;
        CallOptions call_opts;
        call_opts.SetTimeout(heartbeat_interval_ms);

        while (true) {
          absl::Status status;
          absl::Notification n;
          // Heartbeat RPC implementation automatically retries to tolerate
          // transient network failures.
          VLOG(10) << "HeartbeatRequest: " << request.DebugString();
          leader_client_->HeartbeatAsync(&call_opts, &request, &response,
                                         [&](absl::Status s) {
                                           status = s;
                                           n.Notify();
                                         });
          n.WaitForNotification();
          VLOG(10) << "HeartbeatResponse: " << status;
          if (!status.ok()) {
            // Ignore heartbeat errors and exit thread if shutting down. For
            // example, the agent may send a heartbeat right after Shutdown()
            // started, but before StopHeartbeat() and end of Shutdown(). This
            // results in an unexpected heartbeat error.
            // Waiting for a second allows us to identify if errors are due to
            // inflight heartbeats sent during shutdown and can be ignored.
            absl::SleepFor(absl::Seconds(1));
            {
              absl::MutexLock l(&heartbeat_thread_shutdown_mu_);

              if (shutting_down_) {
                return;
              }
            }
            SetError(status);
          } else if (response.leader_incarnation() != leader_incarnation_) {
            SetError(MakeCoordinationError(
                absl::AbortedError("Leader incarnation ID mismatch: the "
                                   "coordination leader has restarted.")));
          }
          // Send next heartbeat after an interval.
          {
            absl::MutexLock l(&heartbeat_thread_shutdown_mu_);
            heartbeat_thread_cv_.WaitWithTimeout(
                &heartbeat_thread_shutdown_mu_,
                absl::Milliseconds(heartbeat_interval_ms));
            if (shutting_down_) {
              return;
            }
          }
        }
      }));
  return absl::OkStatus();
}

absl::Status CoordinationServiceAgentImpl::WaitForAllTasks(
    const DeviceInfo& local_devices) {
  absl::Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    return agent_running_status;
  }
  WaitForAllTasksRequest request;
  *request.mutable_source_task() = task_;
  *request.mutable_device_info() = local_devices;
  VLOG(3) << "WaitForAllTasksRequest: " << request.DebugString();
  WaitForAllTasksResponse response;
  absl::Status status;
  absl::Notification n;
  leader_client_->WaitForAllTasksAsync(&request, &response,
                                       [&](absl::Status s) {
                                         status = s;
                                         n.Notify();
                                       });
  n.WaitForNotification();
  if (!status.ok()) {
    VLOG(3) << "WaitForAllTasksResponse: " << status;
    SetError(status);
    return status;
  }
  VLOG(3) << "WaitForAllTasksResponse: " << response.DebugString();
  cluster_devices_ = response.device_info();
  return absl::OkStatus();
}

const DeviceInfo& CoordinationServiceAgentImpl::GetClusterDeviceInfo() {
  return cluster_devices_;
}

absl::StatusOr<CoordinatedTask> CoordinationServiceAgentImpl::GetOwnTask() {
  if (!IsInitialized()) {
    return MakeCoordinationError(absl::FailedPreconditionError(
        "Agent has not been initialized; we do not "
        "know the associated task yet."));
  }
  return task_;
}

absl::StatusOr<std::vector<CoordinatedTaskStateInfo>>
CoordinationServiceAgentImpl::GetTaskState(
    const std::vector<CoordinatedTask>& tasks) {
  GetTaskStateRequest request;
  *request.mutable_source_task() = {tasks.begin(), tasks.end()};
  GetTaskStateResponse response;
  absl::Notification n;
  absl::StatusOr<std::vector<CoordinatedTaskStateInfo>> result;
  leader_client_->GetTaskStateAsync(
      &request, &response, [&](const absl::Status& s) {
        if (s.ok()) {
          result = std::vector<CoordinatedTaskStateInfo>(
              std::make_move_iterator(response.task_state().begin()),
              std::make_move_iterator(response.task_state().end()));
        } else {
          result = s;
        }
        n.Notify();
      });
  n.WaitForNotification();
  return result;
}

absl::Status CoordinationServiceAgentImpl::ReportError(
    const absl::Status& error) {
  {
    absl::MutexLock l(&state_mu_);
    if (state_ == CoordinatedTaskState::TASKSTATE_UNINITIALIZED) {
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Coordination service agent must be initialized first before "
          "reporting error."));
    } else if (state_ == CoordinatedTaskState::TASKSTATE_ERROR) {
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Coordination service agent is already in error state."));
    }
  }
  SetError(MakeCoordinationError(error, task_,
                                 /*is_reported_error=*/true));
  LOG(INFO) << "Reporting error to coordination service: " << error;
  ReportErrorToServiceRequest request;
  request.set_error_code(error.raw_code());
  request.set_error_message(std::string(error.message()));
  *request.mutable_error_origin() = task_;
  VLOG(5) << "ReportErrorToServiceRequest: " << request.DebugString();
  ReportErrorToServiceResponse response;

  absl::Notification n;
  leader_client_->ReportErrorToServiceAsync(
      &request, &response, [&](absl::Status s) {
        VLOG(5) << "ReportErrorToServiceResponse: " << s;
        if (!s.ok()) {
          LOG(ERROR)
              << "Encountered another error when reporting error to "
                 "coordination service: "
              << s
              << "\nThis is usually caused by an earlier error during "
                 "execution. Check the logs (this task or the leader) for "
                 "an earlier error to debug further.";
        }
        n.Notify();
      });
  n.WaitForNotification();
  return absl::OkStatus();
}

absl::Status CoordinationServiceAgentImpl::Shutdown() {
  return ShutdownInternal();
}

absl::Status CoordinationServiceAgentImpl::ShutdownInternal() {
  absl::Status status = absl::OkStatus();
  bool is_connected = false;
  {
    absl::MutexLock l(&state_mu_);
    is_connected = state_ == CoordinatedTaskState::TASKSTATE_CONNECTED;
  }
  // Disconnect agent from service.
  if (!configs_.agent_destruction_without_shutdown() && is_connected) {
    LOG(INFO) << "Coordination agent has initiated Shutdown().";
    ShutdownTaskRequest request;
    *request.mutable_source_task() = task_;
    ShutdownTaskResponse response;
    CallOptions call_opts;
    const int64_t shutdown_timeout =
        configs_.shutdown_barrier_timeout_in_ms() > 0
            ? configs_.shutdown_barrier_timeout_in_ms()
            : absl::ToInt64Milliseconds(kDefaultShutdownTimeout);
    call_opts.SetTimeout(shutdown_timeout);

    absl::Notification n;
    leader_client_->ShutdownTaskAsync(&call_opts, &request, &response,
                                      [&status, &n](absl::Status s) {
                                        status = s;
                                        n.Notify();
                                      });
    n.WaitForNotification();
    if (status.ok()) {
      LOG(INFO) << "Coordination agent has successfully shut down.";
    } else {
      LOG(ERROR)
          << "Failed to disconnect from coordination service with status: "
          << status
          << "\nProceeding with agent shutdown anyway. This is usually caused "
             "by an earlier error during execution. Check the logs (this task "
             "or the leader) for an earlier error to debug further.";
    }
  }

  // Tear down agent.
  StopHeartbeat();
  {
    absl::MutexLock l(&state_mu_);
    if (state_ == CoordinatedTaskState::TASKSTATE_ERROR) {
      const std::string status_message = absl::StrCat(
          "Shutdown() was called while coordination agent is in error state, "
          "implying that distributed execution failed. Note: agent will still "
          "shutdown anyway. Agent status: ",
          status_.ToString(),
          "\nThis is usually caused by an earlier error during execution. "
          "Check the logs (this task or the leader) for an earlier error to "
          "debug further.");
      status =
          MakeCoordinationError(absl::FailedPreconditionError(status_message));
      LOG(ERROR) << status_message;
    }
    state_ = CoordinatedTaskState::TASKSTATE_DISCONNECTED;
  }

  // Cancel all pending GetKeyValue() RPC calls.
  cancellation_manager_.StartCancel();
  return status;
}

absl::Status CoordinationServiceAgentImpl::Reset() {
  {
    absl::MutexLock l(&state_mu_);
    if (state_ != CoordinatedTaskState::TASKSTATE_ERROR) {
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Reset() failed: coordination service agent is not in ERROR state."));
    }
  }

  ResetTaskRequest request;
  *request.mutable_source_task() = task_;
  VLOG(3) << "ResetTaskRequest: " << request.DebugString();
  ResetTaskResponse response;

  absl::Status status;
  absl::Notification n;
  leader_client_->ResetTaskAsync(&request, &response,
                                 [&status, &n](absl::Status s) {
                                   status = s;
                                   n.Notify();
                                 });
  n.WaitForNotification();
  VLOG(3) << "ResetTaskResponse: " << status;
  if (!status.ok()) {
    return status;
  }

  // Reset agent state.
  StopHeartbeat();
  {
    absl::MutexLock l(&state_mu_);
    state_ = CoordinatedTaskState::TASKSTATE_DISCONNECTED;
  }
  {
    absl::MutexLock l(&heartbeat_thread_shutdown_mu_);
    shutting_down_ = false;
  }

  LOG(INFO) << "Coordination agent has been reset.";
  return status;
}

absl::StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
    std::string_view key) {
  return GetKeyValue(key, /*timeout=*/absl::InfiniteDuration());
}

absl::StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
    std::string_view key, absl::Duration timeout) {
  auto n = std::make_shared<absl::Notification>();
  auto result = std::make_shared<absl::StatusOr<std::string>>();
  GetKeyValueAsync(
      key, [n, result](const absl::StatusOr<std::string>& status_or_value) {
        *result = status_or_value;
        n->Notify();
      });
  bool call_completed_before_timeout =
      n->WaitForNotificationWithTimeout(timeout);
  if (!call_completed_before_timeout) {
    VLOG(3) << "GetKeyValue(" << key << ") timed out after " << timeout;
    return MakeCoordinationError(absl::DeadlineExceededError(absl::Substitute(
        "GetKeyValue() timed out with key: $0 and duration: $1", key,
        absl::FormatDuration(timeout))));
  }
  return *result;
}

std::shared_ptr<CallOptions> CoordinationServiceAgentImpl::GetKeyValueAsync(
    std::string_view key, StatusOrValueCallback done) {
  auto request = std::make_shared<GetKeyValueRequest>();
  request->set_key(key.data(), key.size());
  VLOG(3) << "GetKeyValueRequest: " << request->DebugString();
  auto response = std::make_shared<GetKeyValueResponse>();
  auto call_opts = std::make_shared<CallOptions>();

  const CancellationToken token =
      cancellation_manager_.get_cancellation_token();
  const bool already_cancelled = !cancellation_manager_.RegisterCallback(
      token, [call_opts]() { call_opts->StartCancel(); });
  if (already_cancelled) {
    done(absl::CancelledError("GetKeyValueAsync() was cancelled."));
    return call_opts;
  }
  leader_client_->GetKeyValueAsync(
      call_opts.get(), request.get(), response.get(),
      [call_opts, request, response, done = std::move(done),
       &cm = cancellation_manager_, token](const absl::Status& s) {
        // RPC call has completed (no longer needs to be cancelled if agent is
        // destroyed).
        cm.TryDeregisterCallback(token);

        // Retrieve server response.
        if (!s.ok()) {
          done(s);
          VLOG(3) << "GetKeyValueResponse: " << s;
        } else {
          done(response->kv().value());
          VLOG(3) << "GetKeyValueResponse: " << response->DebugString();
        }
      });
  return call_opts;
}

absl::StatusOr<std::string> CoordinationServiceAgentImpl::TryGetKeyValue(
    std::string_view key) {
  absl::Notification n;
  absl::StatusOr<std::string> result;
  TryGetKeyValueRequest request;
  request.set_key(key.data(), key.size());
  VLOG(3) << "TryGetKeyValueRequest: " << request.DebugString();
  TryGetKeyValueResponse response;
  leader_client_->TryGetKeyValueAsync(
      &request, &response, [&](const absl::Status& s) {
        if (s.ok()) {
          result = response.kv().value();
          VLOG(3) << "TryGetKeyValueResponse: " << result.value();
        } else {
          result = s;
          VLOG(3) << "TryGetKeyValueResponse: " << s;
        }
        n.Notify();
      });
  n.WaitForNotification();

  return result;
}

absl::StatusOr<std::vector<KeyValueEntry>>
CoordinationServiceAgentImpl::GetKeyValueDir(std::string_view key) {
  absl::Notification n;
  absl::StatusOr<std::vector<KeyValueEntry>> result;
  GetKeyValueDirAsync(
      key, [&n, &result](
               absl::StatusOr<std::vector<KeyValueEntry>> status_or_value) {
        result = std::move(status_or_value);
        n.Notify();
      });

  n.WaitForNotification();
  return result;
}

void CoordinationServiceAgentImpl::GetKeyValueDirAsync(
    std::string_view key, StatusOrValueDirCallback done) {
  auto request = std::make_shared<GetKeyValueDirRequest>();
  request->set_directory_key(key.data(), key.size());
  VLOG(3) << "GetKeyValueDirRequest: " << request->DebugString();
  auto response = std::make_shared<GetKeyValueDirResponse>();
  leader_client_->GetKeyValueDirAsync(
      request.get(), response.get(),
      [request, response, done = std::move(done)](const absl::Status& s) {
        if (!s.ok()) {
          done(s);
          VLOG(3) << "GetKeyValueDirResponse: " << s;
        } else {
          VLOG(3) << "GetKeyValueDirResponse: " << response->DebugString();
          std::vector<KeyValueEntry> kv_in_directory = {
              std::make_move_iterator(response->kv().begin()),
              std::make_move_iterator(response->kv().end())};
          done(kv_in_directory);
        }
      });
}

absl::Status CoordinationServiceAgentImpl::InsertKeyValue(
    std::string_view key, std::string_view value) {
  return InsertKeyValue(key, value, /*allow_overwrite=*/false);
}

absl::Status CoordinationServiceAgentImpl::InsertKeyValue(
    std::string_view key, std::string_view value, bool allow_overwrite) {
  InsertKeyValueRequest request;
  request.mutable_kv()->set_key(key.data(), key.size());
  request.mutable_kv()->set_value(value.data(), value.size());
  request.set_allow_overwrite(allow_overwrite);
  VLOG(3) << "InsertKeyValueRequest: " << request.DebugString();
  InsertKeyValueResponse response;

  absl::Status status;
  absl::Notification n;
  leader_client_->InsertKeyValueAsync(&request, &response, [&](absl::Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  VLOG(3) << "InsertKeyValueResponse: " << status;
  return status;
}

absl::Status CoordinationServiceAgentImpl::DeleteKeyValue(
    std::string_view key) {
  DeleteKeyValueRequest request;
  request.set_key(key.data(), key.size());
  request.set_is_directory(true);
  VLOG(3) << "DeleteKeyValueRequest: " << request.DebugString();
  DeleteKeyValueResponse response;

  absl::Status status;
  absl::Notification n;
  leader_client_->DeleteKeyValueAsync(&request, &response, [&](absl::Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  VLOG(3) << "DeleteKeyValueResponse " << status;
  return absl::OkStatus();
}

absl::Status CoordinationServiceAgentImpl::UpdateKeyValue(
    std::string_view key, std::string_view value) {
  return MakeCoordinationError(absl::UnimplementedError(
      "CoordinationServiceAgent::UpdateKeyValue is not implemented."));
}

absl::Status CoordinationServiceAgentImpl::StartWatchKey(
    std::string_view key,
    CoordinationServiceAgentImpl::ChangedKeyValuesCallback on_change) {
  return MakeCoordinationError(absl::UnimplementedError(
      "CoordinationServiceAgent::StartWatchKey is not implemented."));
}

absl::Status CoordinationServiceAgentImpl::StopWatchKey(std::string_view key) {
  return MakeCoordinationError(absl::UnimplementedError(
      "CoordinationServiceAgent::StopWatchKey is not implemented."));
}

void CoordinationServiceAgentImpl::SetError(const absl::Status& error) {
  assert(!error.ok());
  absl::MutexLock l(&state_mu_);
  if (state_ == CoordinatedTaskState::TASKSTATE_ERROR) return;

  LOG(ERROR) << "Coordination agent is set to ERROR: " << error;
  state_ = CoordinatedTaskState::TASKSTATE_ERROR;
  status_ = error;
  error_fn_(error);
}

absl::Status CoordinationServiceAgentImpl::ActivateWatch(
    std::string_view key, const std::map<std::string, std::string>& kvs) {
  return MakeCoordinationError(absl::UnimplementedError(
      "CoordinationServiceAgent::ActivateWatch is not implemented."));
}

absl::Status CoordinationServiceAgentImpl::WaitAtBarrier(
    std::string_view barrier_id, absl::Duration timeout,
    const std::vector<CoordinatedTask>& tasks) {
  absl::Status status;
  absl::Notification n;
  WaitAtBarrierAsync(barrier_id, timeout, tasks, [&](absl::Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

void CoordinationServiceAgentImpl::WaitAtBarrierAsync(
    std::string_view barrier_id, absl::Duration timeout,
    const std::vector<CoordinatedTask>& tasks, StatusCallback done) {
  absl::Status agent_running_status =
      ValidateRunningAgent(/*allow_disconnected=*/true);
  if (!agent_running_status.ok()) {
    done(agent_running_status);
    return;
  }
  {
    absl::MutexLock l(&state_mu_);
    auto [it, inserted] = used_barrier_ids_.insert(std::string(barrier_id));
    if (!inserted) {
      done(absl::FailedPreconditionError(absl::StrCat(
          "WaitAtBarrier() should not be called with the same id more than "
          "once. Barrier id: ",
          barrier_id)));
      return;
    }
  }
  auto request = std::make_shared<BarrierRequest>();
  auto response = std::make_shared<BarrierResponse>();
  request->set_barrier_id(std::string(barrier_id));
  request->set_barrier_timeout_in_ms(timeout / absl::Milliseconds(1));
  *request->mutable_source_task() = task_;
  *request->mutable_tasks() = {tasks.begin(), tasks.end()};
  VLOG(3) << "WaitAtBarrierRequest: " << request->DebugString();
  leader_client_->BarrierAsync(
      request.get(), response.get(),
      [request, response, done = std::move(done)](const absl::Status& s) {
        done(s);
        VLOG(3) << "WaitAtBarrierResponse: " << s;
      });
}

absl::Status CoordinationServiceAgentImpl::CancelBarrier(
    std::string_view barrier_id) {
  absl::Status status;
  absl::Notification n;
  CancelBarrierAsync(barrier_id, [&](const absl::Status& s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

void CoordinationServiceAgentImpl::CancelBarrierAsync(
    std::string_view barrier_id, StatusCallback done) {
  absl::Status agent_running_status =
      ValidateRunningAgent(/*allow_disconnected=*/true);
  if (!agent_running_status.ok()) {
    done(agent_running_status);
    return;
  }
  auto request = std::make_shared<CancelBarrierRequest>();
  auto response = std::make_shared<CancelBarrierResponse>();
  request->set_barrier_id(std::string(barrier_id));
  *request->mutable_source_task() = task_;
  VLOG(3) << "CancelBarrierRequest: " << request->DebugString();
  leader_client_->CancelBarrierAsync(
      request.get(), response.get(),
      [request, response, done = std::move(done)](const absl::Status& s) {
        done(s);
        VLOG(3) << "CancelBarrierResponse: " << s;
      });
}

// Returns an error if agent is not running.
absl::Status CoordinationServiceAgentImpl::ValidateRunningAgent(
    bool allow_disconnected) {
  absl::MutexLock l(&state_mu_);
  switch (state_) {
    case CoordinatedTaskState::TASKSTATE_CONNECTED:
      return absl::OkStatus();

    case CoordinatedTaskState::TASKSTATE_UNINITIALIZED:
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Agent must be in CONNECTED state. It is currently UNINITIALIZED."));

    case CoordinatedTaskState::TASKSTATE_DISCONNECTED:
      if (allow_disconnected) return absl::OkStatus();
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Agent must be in CONNECTED state. It is currently DISCONNECTED."));

    case CoordinatedTaskState::TASKSTATE_ERROR:
      return MakeCoordinationError(absl::FailedPreconditionError(
          "Agent must be in CONNECTED state. It is currently in ERROR."));

    default:
      return MakeCoordinationError(absl::FailedPreconditionError(absl::StrCat(
          "Agent is not in CONNECTED state. Current state: ", state_)));
  }
}

absl::StatusOr<Env*> CoordinationServiceAgentImpl::GetEnv() {
  if (!IsInitialized()) {
    return MakeCoordinationError(absl::FailedPreconditionError(
        "Coordination service agent has not been initialized."));
  }
  if (env_ == nullptr) {
    return MakeCoordinationError(
        absl::FailedPreconditionError("Coordination service agent was not "
                                      "initialized with a valid Env* object."));
  }
  return env_;
}

}  // namespace

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent() {
  return std::make_unique<CoordinationServiceAgentImpl>();
}

}  // namespace tsl
