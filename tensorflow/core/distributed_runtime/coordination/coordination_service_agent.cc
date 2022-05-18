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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

constexpr absl::Duration kDefaultClusterRegisterTimeout = absl::Hours(1);
constexpr absl::Duration kDefaultHeartbeatTimeout = absl::Seconds(10);
constexpr absl::Duration kDefaultShutdownTimeout = absl::Seconds(10);
constexpr char kHeartbeatThread[] = "CoordinationServiceHeartbeatLoop";

class CoordinationServiceAgentImpl : public CoordinationServiceAgent {
 public:
  CoordinationServiceAgentImpl() = default;
  ~CoordinationServiceAgentImpl() override {
    Status s = Shutdown();
    if (!s.ok()) {
      LOG(ERROR) << "Agent shutdown failed with status: " << s;
    }
    // Cancel all pending GetKeyValue() RPC calls.
    cancellation_manager_.StartCancel();
  }
  Status Initialize(Env* env, const ServerDef& server_def,
                    std::unique_ptr<CoordinationClientCache> client_cache,
                    StatusCallback error_fn) override;
  Status Initialize(Env* env, const std::string& job_name, int task_id,
                    const CoordinationServiceConfig& configs,
                    std::unique_ptr<CoordinationClient> leader_client,
                    StatusCallback error_fn) override;
  Status Initialize(Env* env, const CoordinatedTask& task,
                    const CoordinationServiceConfig& configs,
                    std::unique_ptr<CoordinationClient> leader_client,
                    StatusCallback error_fn) override;
  bool IsInitialized() override;

  Status Connect() override;
  Status WaitForAllTasks(
      const CoordinationServiceDeviceInfo& local_devices) override;
  const CoordinationServiceDeviceInfo& GetClusterDeviceInfo() override;
  StatusOr<CoordinatedTask> GetOwnTask() override;
  StatusOr<TaskState> GetTaskStatus(const CoordinatedTask& task) override;
  Status ReportError(const Status& error) override;
  Status Shutdown() override;
  Status Reset() override;

  StatusOr<std::string> GetKeyValue(const std::string& key) override;
  StatusOr<std::string> GetKeyValue(const std::string& key,
                                    absl::Duration timeout) override;
  void GetKeyValueAsync(const std::string& key,
                        StatusOrValueCallback done) override;
  StatusOr<std::vector<KeyValueEntry>> GetKeyValueDir(
      const std::string& key) override;
  void GetKeyValueDirAsync(const std::string& key,
                           StatusOrValueDirCallback done) override;
  Status InsertKeyValue(const std::string& key,
                        const std::string& value) override;
  Status DeleteKeyValue(const std::string& key) override;
  Status UpdateKeyValue(const std::string& key,
                        const std::string& value) override;

  Status StartWatchKey(const std::string& key,
                       ChangedKeyValuesCallback on_change) override;
  Status StopWatchKey(const std::string& key) override;
  Status WaitAtBarrier(const std::string& barrier_id, absl::Duration timeout,
                       const std::vector<CoordinatedTask>& tasks) override;
  void WaitAtBarrierAsync(const std::string& barrier_id, absl::Duration timeout,
                          const std::vector<CoordinatedTask>& tasks,
                          StatusCallback done) override;
  Status CancelBarrier(const std::string& barrier_id) override;

 protected:
  void SetError(const Status& error) override;
  Status ActivateWatch(const std::string& key,
                       const std::map<std::string, std::string>&) override;
  // Returns an error if agent is not running.
  Status ValidateRunningAgent();
  void StopHeartbeat();

 private:
  Env* env_ = nullptr;  // Not owned.
  const uint64_t incarnation_id_ = random::New64();
  CoordinatedTask task_;
  CoordinationServiceConfig configs_;
  std::unique_ptr<CoordinationClient> leader_client_;
  StatusCallback error_fn_;

  enum class State {
    UNINITIALIZED,
    DISCONNECTED,
    RUNNING,
    ERROR,
    SHUTDOWN,
  };
  mutable mutex state_mu_;
  State state_ TF_GUARDED_BY(state_mu_) = State::UNINITIALIZED;
  Status status_ TF_GUARDED_BY(state_mu_) = Status::OK();

  uint64_t leader_incarnation_ = 0;
  CoordinationServiceDeviceInfo cluster_devices_;

  mutex heartbeat_thread_shutdown_mu_;
  condition_variable heartbeat_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(heartbeat_thread_shutdown_mu_) = false;
  std::unique_ptr<Thread> heartbeat_thread_;
  CancellationManager cancellation_manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(CoordinationServiceAgentImpl);
};

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const ServerDef& server_def,
    std::unique_ptr<CoordinationClientCache> client_cache,
    StatusCallback error_fn) {
  CoordinationServiceConfig configs =
      server_def.default_session_config().experimental().coordination_config();
  if (configs.service_leader().empty()) {
    const std::string& collective_leader = server_def.default_session_config()
                                               .experimental()
                                               .collective_group_leader();
    if (!collective_leader.empty()) {
      configs.set_service_leader(collective_leader);
      LOG(INFO) << "No coordination leader is set, using the collective leader "
                << collective_leader;
    } else {
      const std::string& default_leader =
          strings::StrCat("/job:", server_def.job_name(), "/replica:0/task:0");
      configs.set_service_leader(default_leader);
      LOG(INFO) << "No coordination leader is set, using the default leader "
                << default_leader;
    }
  }
  return Initialize(
      env, server_def.job_name(), server_def.task_index(), configs,
      client_cache->GetOwnedClient(configs.service_leader()), error_fn);
}

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const std::string& job_name, int task_id,
    const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
  CoordinatedTask task;
  task.set_job_name(job_name);
  task.set_task_id(task_id);
  return Initialize(env, task, configs, std::move(leader_client), error_fn);
}

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const CoordinatedTask& task,
    const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
  mutex_lock l(state_mu_);
  if (state_ != State::UNINITIALIZED) {
    return MakeCoordinationError(errors::FailedPrecondition(
        "Coordination service agent has already been initialized."));
  }

  env_ = env;
  task_ = task;
  configs_ = configs;
  if (configs_.service_leader().empty()) {
    return MakeCoordinationError(errors::InvalidArgument(
        "CoordinationServiceAgent must be initialized with a valid leader."));
  }
  leader_client_ = std::move(leader_client);
  if (leader_client_ == nullptr) {
    return MakeCoordinationError(errors::InvalidArgument(
        "CoordinationServiceAgent must have a valid leader client."));
  }
  error_fn_ = error_fn;
  state_ = State::DISCONNECTED;
  return Status::OK();
}

bool CoordinationServiceAgentImpl::IsInitialized() {
  mutex_lock l(state_mu_);
  return state_ != State::UNINITIALIZED;
}

void CoordinationServiceAgentImpl::StopHeartbeat() {
  {
    mutex_lock l(heartbeat_thread_shutdown_mu_);
    shutting_down_ = true;
    heartbeat_thread_cv_.notify_all();
  }
  heartbeat_thread_.reset();
}

Status CoordinationServiceAgentImpl::Connect() {
  {
    mutex_lock l(state_mu_);
    if (state_ != State::DISCONNECTED) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Coordination service agent is not in DISCONNECTED state."));
    }
  }
  RegisterTaskRequest request;
  *request.mutable_source_task() = task_;
  request.set_incarnation(incarnation_id_);
  RegisterTaskResponse response;
  absl::Notification n;

  // Block until the remote service is up and the task is registered.
  CallOptions call_opts;
  const int64_t register_timeout =
      configs_.cluster_register_timeout_in_ms() > 0
          ? configs_.cluster_register_timeout_in_ms()
          : absl::ToInt64Milliseconds(kDefaultClusterRegisterTimeout);
  call_opts.SetTimeout(register_timeout);
  leader_client_->RegisterTaskAsync(
      &call_opts, &request, &response, [&](Status s) {
        if (!s.ok()) {
          SetError(s);
        } else {
          leader_incarnation_ = response.leader_incarnation();
          {
            mutex_lock l(state_mu_);
            state_ = State::RUNNING;
          }
        }
        n.Notify();
      });
  n.WaitForNotification();
  {
    mutex_lock l(state_mu_);
    if (state_ == State::ERROR) {
      return status_;
    }
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
          {
            mutex_lock l(heartbeat_thread_shutdown_mu_);
            heartbeat_thread_cv_.wait_for(
                l, std::chrono::milliseconds(heartbeat_interval_ms));
            if (shutting_down_) {
              return;
            }
          }
          Status status;
          absl::Notification n;
          // Heartbeat RPC implementation automatically retries to tolerate
          // transient network failures.
          leader_client_->HeartbeatAsync(&call_opts, &request, &response,
                                         [&](Status s) {
                                           status = s;
                                           n.Notify();
                                         });
          n.WaitForNotification();
          if (!status.ok()) {
            SetError(status);
          } else if (response.leader_incarnation() != leader_incarnation_) {
            SetError(MakeCoordinationError(
                errors::Aborted("Leader incarnation ID mismatch: the "
                                "coordination leader has restarted.")));
          }
        }
      }));
  return Status::OK();
}

Status CoordinationServiceAgentImpl::WaitForAllTasks(
    const CoordinationServiceDeviceInfo& local_devices) {
  Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    return agent_running_status;
  }
  WaitForAllTasksRequest request;
  *request.mutable_source_task() = task_;
  *request.mutable_local_device_info() = local_devices;
  WaitForAllTasksResponse response;
  Status status;
  absl::Notification n;
  leader_client_->WaitForAllTasksAsync(&request, &response, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  if (!status.ok()) {
    SetError(status);
    return status;
  }
  cluster_devices_.MergeFrom(response.cluster_device_info());
  return Status::OK();
}

const CoordinationServiceDeviceInfo&
CoordinationServiceAgentImpl::GetClusterDeviceInfo() {
  return cluster_devices_;
}

StatusOr<CoordinatedTask> CoordinationServiceAgentImpl::GetOwnTask() {
  if (!IsInitialized()) {
    return MakeCoordinationError(
        errors::FailedPrecondition("Agent has not been initialized; we do not "
                                   "know the associated task yet."));
  }
  return task_;
}

StatusOr<CoordinationServiceAgentImpl::TaskState>
CoordinationServiceAgentImpl::GetTaskStatus(const CoordinatedTask& task) {
  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgentImpl::GetTaskStatus is not implemented."));
}

Status CoordinationServiceAgentImpl::ReportError(const Status& error) {
  {
    mutex_lock l(state_mu_);
    if (state_ == State::UNINITIALIZED) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Coordination service agent must be initialized first before "
          "reporting error."));
    } else if (state_ == State::ERROR) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Coordination service agent is already in error state."));
    }
  }
  SetError(MakeCoordinationError(error, task_,
                                 /*is_reported_error=*/true));
  LOG(INFO) << "Reporting error to coordination service: " << error;
  ReportErrorToServiceRequest request;
  request.set_error_code(error.code());
  request.set_error_message(error.error_message());
  *request.mutable_error_origin() = task_;
  ReportErrorToServiceResponse response;

  absl::Notification n;
  leader_client_->ReportErrorToServiceAsync(&request, &response, [&](Status s) {
    if (!s.ok()) {
      LOG(ERROR) << "Encountered another error when reporting error to "
                    "coordination service: "
                 << s;
    }
    n.Notify();
  });
  n.WaitForNotification();
  return Status::OK();
}

Status CoordinationServiceAgentImpl::Shutdown() {
  Status status = Status::OK();
  bool is_connected = false;
  {
    mutex_lock l(state_mu_);
    is_connected = state_ == State::RUNNING;
  }
  // Disconnect agent from service.
  if (!configs_.agent_destruction_without_shutdown() && is_connected) {
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
                                      [&status, &n](Status s) {
                                        status = s;
                                        n.Notify();
                                      });
    n.WaitForNotification();
    if (status.ok()) {
      LOG(INFO) << "Coordination agent has successfully shut down.";
    } else {
      LOG(ERROR)
          << "Failed to disconnect from coordination service with status: "
          << status << ". Proceeding with agent shutdown anyway.";
    }
  }

  // Tear down agent.
  StopHeartbeat();
  {
    mutex_lock l(state_mu_);
    if (state_ == State::ERROR) {
      status = MakeCoordinationError(errors::FailedPrecondition(absl::StrCat(
          "Shutdown() was called while agent is in error state, implying that "
          "distributed execution failed. Note: agent will still shutdown "
          "anyway. Agent status: ",
          status_.ToString())));
    }
    state_ = State::SHUTDOWN;
  }
  return status;
}

Status CoordinationServiceAgentImpl::Reset() {
  {
    mutex_lock l(state_mu_);
    if (state_ != State::ERROR) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Reset() failed: coordination service agent is not in ERROR state."));
    }
  }

  ResetTaskRequest request;
  *request.mutable_source_task() = task_;
  ResetTaskResponse response;

  Status status;
  absl::Notification n;
  leader_client_->ResetTaskAsync(&request, &response, [&status, &n](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  if (!status.ok()) {
    return status;
  }

  // Reset agent state.
  StopHeartbeat();
  {
    mutex_lock l(state_mu_);
    state_ = State::DISCONNECTED;
  }
  {
    mutex_lock l(heartbeat_thread_shutdown_mu_);
    shutting_down_ = false;
  }

  LOG(INFO) << "Coordination agent has been reset.";
  return status;
}

StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
    const std::string& key) {
  return GetKeyValue(key, /*timeout=*/absl::InfiniteDuration());
}

StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
    const std::string& key, absl::Duration timeout) {
  auto n = std::make_shared<absl::Notification>();
  auto result = std::make_shared<StatusOr<std::string>>();
  GetKeyValueAsync(key,
                   [n, result](const StatusOr<std::string>& status_or_value) {
                     *result = status_or_value;
                     n->Notify();
                   });
  bool call_completed_before_timeout =
      n->WaitForNotificationWithTimeout(timeout);
  if (!call_completed_before_timeout) {
    return MakeCoordinationError(errors::DeadlineExceeded(absl::Substitute(
        "GetKeyValue() timed out with key: $0 and duration: $1", key,
        absl::FormatDuration(timeout))));
  }
  return *result;
}

StatusOr<std::vector<KeyValueEntry>>
CoordinationServiceAgentImpl::GetKeyValueDir(const std::string& key) {
  absl::Notification n;
  StatusOr<std::vector<KeyValueEntry>> result;
  GetKeyValueDirAsync(
      key, [&n, &result](StatusOr<std::vector<KeyValueEntry>> status_or_value) {
        result = std::move(status_or_value);
        n.Notify();
      });

  n.WaitForNotification();
  return result;
}

void CoordinationServiceAgentImpl::GetKeyValueDirAsync(
    const std::string& key, StatusOrValueDirCallback done) {
  auto request = std::make_shared<GetKeyValueDirRequest>();
  request->set_directory_key(key);
  auto response = std::make_shared<GetKeyValueDirResponse>();
  leader_client_->GetKeyValueDirAsync(
      request.get(), response.get(),
      [request, response, done = std::move(done)](const Status& s) {
        if (!s.ok()) {
          done(s);
        } else {
          std::vector<KeyValueEntry> kv_in_directory = {
              std::make_move_iterator(response->kv().begin()),
              std::make_move_iterator(response->kv().end())};
          done(kv_in_directory);
        }
      });
}

Status CoordinationServiceAgentImpl::InsertKeyValue(const std::string& key,
                                                    const std::string& value) {
  InsertKeyValueRequest request;
  request.mutable_kv()->set_key(key.data(), key.size());
  request.mutable_kv()->set_value(value.data(), value.size());
  InsertKeyValueResponse response;

  Status status;
  absl::Notification n;
  leader_client_->InsertKeyValueAsync(&request, &response, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

void CoordinationServiceAgentImpl::GetKeyValueAsync(
    const std::string& key, StatusOrValueCallback done) {
  auto request = std::make_shared<GetKeyValueRequest>();
  request->set_key(key);
  auto response = std::make_shared<GetKeyValueResponse>();
  auto call_opts = std::make_shared<CallOptions>();

  const CancellationToken token =
      cancellation_manager_.get_cancellation_token();
  const bool already_cancelled = !cancellation_manager_.RegisterCallback(
      token, [call_opts]() { call_opts->StartCancel(); });
  if (already_cancelled) {
    done(errors::Cancelled("GetKeyValueAsync() was cancelled."));
    return;
  }
  leader_client_->GetKeyValueAsync(
      call_opts.get(), request.get(), response.get(),
      [call_opts, request, response, done = std::move(done),
       &cm = cancellation_manager_, token](const Status& s) {
        // RPC call has completed (no longer needs to be cancelled if agent is
        // destroyed).
        cm.TryDeregisterCallback(token);

        // Retrieve server response.
        if (!s.ok()) {
          done(s);
        } else {
          done(response->kv().value());
        }
      });
}

Status CoordinationServiceAgentImpl::DeleteKeyValue(const std::string& key) {
  DeleteKeyValueRequest request;
  request.set_key(key);
  request.set_is_directory(true);
  DeleteKeyValueResponse response;

  Status status;
  absl::Notification n;
  leader_client_->DeleteKeyValueAsync(&request, &response, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return Status::OK();
}

Status CoordinationServiceAgentImpl::UpdateKeyValue(const std::string& key,
                                                    const std::string& value) {
  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::UpdateKeyValue is not implemented."));
}

Status CoordinationServiceAgentImpl::StartWatchKey(
    const std::string& key,
    CoordinationServiceAgentImpl::ChangedKeyValuesCallback on_change) {
  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::StartWatchKey is not implemented."));
}

Status CoordinationServiceAgentImpl::StopWatchKey(const std::string& key) {
  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::StopWatchKey is not implemented."));
}

void CoordinationServiceAgentImpl::SetError(const Status& error) {
  assert(!error.ok());
  mutex_lock l(state_mu_);
  if (state_ == State::ERROR) return;

  LOG(ERROR) << "Coordination agent is in ERROR: " << error;
  state_ = State::ERROR;
  status_ = error;
  error_fn_(error);
}

Status CoordinationServiceAgentImpl::ActivateWatch(
    const std::string& key, const std::map<std::string, std::string>& kvs) {
  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::ActivateWatch is not implemented."));
}

Status CoordinationServiceAgentImpl::WaitAtBarrier(
    const std::string& barrier_id, absl::Duration timeout,
    const std::vector<CoordinatedTask>& tasks) {
  Status status;
  absl::Notification n;
  WaitAtBarrierAsync(barrier_id, timeout, tasks, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

void CoordinationServiceAgentImpl::WaitAtBarrierAsync(
    const std::string& barrier_id, absl::Duration timeout,
    const std::vector<CoordinatedTask>& tasks, StatusCallback done) {
  Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    done(agent_running_status);
    return;
  }
  auto request = std::make_shared<BarrierRequest>();
  auto response = std::make_shared<BarrierResponse>();
  request->set_barrier_id(barrier_id);
  request->set_barrier_timeout_in_ms(timeout / absl::Milliseconds(1));
  *request->mutable_source_task() = task_;
  *request->mutable_tasks() = {tasks.begin(), tasks.end()};
  leader_client_->BarrierAsync(request.get(), response.get(),
                               [request, response, done = std::move(done)](
                                   const Status& s) { done(s); });
}

Status CoordinationServiceAgentImpl::CancelBarrier(
    const std::string& barrier_id) {
  Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    return agent_running_status;
  }
  CancelBarrierRequest request;
  CancelBarrierResponse response;
  request.set_barrier_id(barrier_id);
  *request.mutable_source_task() = task_;

  Status status;
  absl::Notification n;
  leader_client_->CancelBarrierAsync(&request, &response, [&](const Status& s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

// Returns an error if agent is not running.
Status CoordinationServiceAgentImpl::ValidateRunningAgent() {
  mutex_lock l(state_mu_);
  switch (state_) {
    case State::RUNNING:
      return Status::OK();

    case State::UNINITIALIZED:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently UNINITIALIZED."));

    case State::DISCONNECTED:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently DISCONNECTED."));

    case State::ERROR:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently in ERROR."));

    case State::SHUTDOWN:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently in SHUTDOWN."));

    default:
      return MakeCoordinationError(errors::FailedPrecondition(absl::StrCat(
          "Agent is not in RUNNING state. Current state: ", state_)));
  }
}

}  // namespace

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent() {
  return std::make_unique<CoordinationServiceAgentImpl>();
}

}  // namespace tensorflow
