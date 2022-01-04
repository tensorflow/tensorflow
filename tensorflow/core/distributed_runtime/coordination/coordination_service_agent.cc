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

#include <string>
#include <utility>

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

constexpr int kDefaultClusterRegisterTimeoutMs = 3600 * 1000;  // 3600 seconds
constexpr int kDefaultHeartbeatTimeoutMs = 10 * 1000;          // 10 seconds
constexpr char kHeartbeatThread[] = "CoordinationServiceHeartbeatLoop";

class CoordinationServiceAgentImpl : public CoordinationServiceAgent {
 public:
  CoordinationServiceAgentImpl() = default;
  ~CoordinationServiceAgentImpl() override { Stop(); }
  Status Initialize(Env* env, const DeviceMgr* device_mgr,
                    const ServerDef& server_def,
                    std::unique_ptr<CoordinationClientCache> client_cache,
                    StatusCallback error_fn) override;
  Status Initialize(Env* env, const DeviceMgr* device_mgr,
                    const std::string& job_name, int task_id,
                    const CoordinationServiceConfig& configs,
                    std::unique_ptr<CoordinationClient> leader_client,
                    StatusCallback error_fn) override;
  bool IsInitialized() override;

  Status Connect() override;
  Status WaitForAllTasks() override;
  const std::vector<DeviceAttributes>& GetClusterDeviceAttributes() override;
  StatusOr<TaskState> GetTaskStatus(const std::string& job_name,
                                    const int task_id) override;
  Status ReportError(const Status& error) override;
  Status Reset() override;

  StatusOr<std::string> GetKeyValue(const std::string& key) override;
  void GetKeyValueAsync(const std::string& key,
                        StatusOrValueCallback done) override;
  Status InsertKeyValue(const std::string& key,
                        const std::string& value) override;
  Status DeleteKeyValue(const std::string& key) override;
  Status UpdateKeyValue(const std::string& key,
                        const std::string& value) override;

  Status StartWatchKey(const std::string& key,
                       ChangedKeyValuesCallback on_change) override;
  Status StopWatchKey(const std::string& key) override;

 protected:
  void SetError(const Status& error) override;
  Status ActivateWatch(const std::string& key,
                       const std::map<std::string, std::string>&) override;
  void Stop();

 private:
  Env* env_;                     // Not owned.
  const DeviceMgr* device_mgr_;  // Not owned.
  const int64_t incarnation_id_ = random::New64();
  std::string job_name_;
  int task_id_;
  CoordinationServiceConfig configs_;
  std::unique_ptr<CoordinationClient> leader_client_;
  StatusCallback error_fn_;

  enum class State {
    UNINITIALIZED,
    DISCONNECTED,
    RUNNING,
    ERROR,
  };
  mutable mutex state_mu_;
  State state_ TF_GUARDED_BY(state_mu_) = State::UNINITIALIZED;
  Status status_ TF_GUARDED_BY(state_mu_) = Status::OK();

  uint64 leader_incarnation_;
  std::vector<DeviceAttributes> cluster_devices_;

  mutex heartbeat_thread_shutdown_mu_;
  condition_variable heartbeat_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(heartbeat_thread_shutdown_mu_) = false;
  std::unique_ptr<Thread> heartbeat_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(CoordinationServiceAgentImpl);
};

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const DeviceMgr* device_mgr, const ServerDef& server_def,
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
      env, device_mgr, server_def.job_name(), server_def.task_index(), configs,
      client_cache->GetOwnedClient(configs.service_leader()), error_fn);
}

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const DeviceMgr* device_mgr, const std::string& job_name,
    int task_id, const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
  mutex_lock l(state_mu_);
  if (state_ != State::UNINITIALIZED) {
    return errors::FailedPrecondition(
        "Coordination service agent has already been initialized.");
  }

  env_ = env;
  device_mgr_ = device_mgr;
  job_name_ = job_name;
  task_id_ = task_id;
  configs_ = configs;
  if (configs_.service_leader().empty()) {
    return errors::InvalidArgument(
        "CoordinationServiceAgent must be initialized with a valid leader.");
  }
  leader_client_ = std::move(leader_client);
  if (leader_client_ == nullptr) {
    return errors::InvalidArgument(
        "CoordinationServiceAgent must have a valid leader client.");
  }
  error_fn_ = error_fn;
  state_ = State::DISCONNECTED;
  return Status::OK();
}

bool CoordinationServiceAgentImpl::IsInitialized() {
  mutex_lock l(state_mu_);
  return state_ != State::UNINITIALIZED;
}

void CoordinationServiceAgentImpl::Stop() {
  {
    mutex_lock l(state_mu_);
    state_ = State::DISCONNECTED;
  }
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
      return errors::FailedPrecondition(
          "Coordination service agent is not in DISCONNECTED state.");
    }
  }
  RegisterWorkerRequest request;
  request.set_job(job_name_);
  request.set_task(task_id_);
  request.set_incarnation(incarnation_id_);
  RegisterWorkerResponse response;
  absl::Notification n;

  // Block until the remote service is up and the task is registered.
  CallOptions call_opts;
  const uint64 register_timeout =
      configs_.cluster_register_timeout_in_ms() > 0
          ? configs_.cluster_register_timeout_in_ms()
          : kDefaultClusterRegisterTimeoutMs;
  call_opts.SetTimeout(register_timeout);
  leader_client_->RegisterWorkerAsync(
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

  heartbeat_thread_.reset(
      env_->StartThread(ThreadOptions(), kHeartbeatThread, [this]() -> void {
        HeartbeatRequest request;
        request.set_job(job_name_);
        request.set_task(task_id_);
        request.set_incarnation(incarnation_id_);
        HeartbeatResponse response;
        const uint64 heartbeat_interval =
            configs_.heartbeat_timeout_in_ms() > 0
                ? configs_.heartbeat_timeout_in_ms() / 2
                : kDefaultHeartbeatTimeoutMs / 2;

        while (true) {
          {
            mutex_lock l(heartbeat_thread_shutdown_mu_);
            heartbeat_thread_cv_.wait_for(
                l, std::chrono::milliseconds(heartbeat_interval));
            if (shutting_down_) {
              return;
            }
          }
          Status status;
          absl::Notification n;
          // Heartbeat RPC implementation automatically retries to tolerate
          // transient network failures.
          leader_client_->HeartbeatAsync(&request, &response, [&](Status s) {
            status = s;
            n.Notify();
          });
          n.WaitForNotification();
          if (!status.ok()) {
            SetError(status);
          } else if (response.leader_incarnation() != leader_incarnation_) {
            SetError(
                errors::Aborted("Leader incarnation ID mismatch: the "
                                "coordination leader has restarted."));
          }
        }
      }));
  return Status::OK();
}

Status CoordinationServiceAgentImpl::WaitForAllTasks() {
  {
    mutex_lock l(state_mu_);
    if (state_ != State::RUNNING) {
      return errors::FailedPrecondition(
          "CoordinationServiceAgentImpl::WaitForAllTasks must be called when "
          "the coordination service agent is in RUNNING state.");
    }
  }
  WaitForAllTasksRequest request;
  request.set_job(job_name_);
  request.set_task(task_id_);
  std::vector<DeviceAttributes> devices;
  device_mgr_->ListDeviceAttributes(&devices);
  for (auto& d : devices) {
    request.add_local_device_attributes()->Swap(&d);
  }
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
  for (const auto& da : response.cluster_device_attributes()) {
    cluster_devices_.emplace_back(da);
  }
  return Status::OK();
}

const std::vector<DeviceAttributes>&
CoordinationServiceAgentImpl::GetClusterDeviceAttributes() {
  return cluster_devices_;
}

StatusOr<CoordinationServiceAgentImpl::TaskState>
CoordinationServiceAgentImpl::GetTaskStatus(const std::string& job_name,
                                            const int task_id) {
  return errors::Unimplemented(
      "CoordinationServiceAgentImpl::GetTaskStatus is not implemented.");
}

Status CoordinationServiceAgentImpl::ReportError(const Status& error) {
  {
    mutex_lock l(state_mu_);
    if (state_ == State::UNINITIALIZED) {
      return errors::FailedPrecondition(
          "Coordination service agent must be initialized first before "
          "reporting error.");
    } else if (state_ == State::ERROR) {
      return errors::FailedPrecondition(
          "Coordination service agent is already in error state.");
    }
  }
  SetError(error);
  LOG(INFO) << "Reporting error to coordination service: " << error;
  ReportErrorToServiceRequest request;
  request.set_error_code(error.code());
  request.set_error_message(error.error_message());
  request.set_source_job(job_name_);
  request.set_source_task(task_id_);
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

Status CoordinationServiceAgentImpl::Reset() {
  return errors::Unimplemented(
      "CoordinationServiceAgentImpl::Reset is not implemented.");
}

StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
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
  leader_client_->GetKeyValueAsync(
      request.get(), response.get(),
      [request, response, done = std::move(done)](const Status& s) {
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
  return errors::Unimplemented(
      "CoordinationServviceAgent::UpdateKeyValue is not implemented.");
}

Status CoordinationServiceAgentImpl::StartWatchKey(
    const std::string& key,
    CoordinationServiceAgentImpl::ChangedKeyValuesCallback on_change) {
  return errors::Unimplemented(
      "CoordinationServviceAgent::StartWatchKey is not implemented.");
}

Status CoordinationServiceAgentImpl::StopWatchKey(const std::string& key) {
  return errors::Unimplemented(
      "CoordinationServviceAgent::StopWatchKey is not implemented.");
}

void CoordinationServiceAgentImpl::SetError(const Status& error) {
  assert(!error.ok());
  mutex_lock l(state_mu_);
  if (state_ == State::ERROR) return;
  state_ = State::ERROR;
  status_ = error;
  error_fn_(error);
}

Status CoordinationServiceAgentImpl::ActivateWatch(
    const std::string& key, const std::map<std::string, std::string>& kvs) {
  return errors::Unimplemented(
      "CoordinationServviceAgent::ActivateWatch is not implemented.");
}

}  // namespace

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent() {
  return std::make_unique<CoordinationServiceAgentImpl>();
}

}  // namespace tensorflow
