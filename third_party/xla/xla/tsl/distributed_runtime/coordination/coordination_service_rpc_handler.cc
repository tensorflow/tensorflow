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

#include "xla/tsl/distributed_runtime/coordination/coordination_service_rpc_handler.h"

#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinationServiceError;
using tensorflow::KeyValueEntry;
}  // namespace

void CoordinationServiceRpcHandler::SetAgentInstance(
    CoordinationServiceAgent* agent) {
  mutex_lock l(mu_);
  agent_ = agent;
}

void CoordinationServiceRpcHandler::SetServiceInstance(
    CoordinationServiceInterface* service) {
  mutex_lock l(mu_);
  service_ = service;
}

void CoordinationServiceRpcHandler::RegisterTaskAsync(
    const RegisterTaskRequest* request, RegisterTaskResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service_->GetServiceIncarnation();
  response->set_leader_incarnation(leader_incarnation);
  done(service_->RegisterTask(task, incarnation));
}

void CoordinationServiceRpcHandler::HeartbeatAsync(
    const HeartbeatRequest* request, HeartbeatResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service_->GetServiceIncarnation();
  absl::Status s = service_->RecordHeartbeat(task, incarnation);
  if (!s.ok()) {
    done(s);
    return;
  }
  response->set_leader_incarnation(leader_incarnation);
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::WaitForAllTasksAsync(
    const WaitForAllTasksRequest* request, WaitForAllTasksResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  service_->WaitForAllTasks(
      request->source_task(), request->device_info(),
      [response, service = service_, done = std::move(done)](absl::Status s) {
        if (s.ok()) {
          *response->mutable_device_info() = service->ListClusterDevices();
        }
        done(s);
      });
}

void CoordinationServiceRpcHandler::ShutdownTaskAsync(
    const ShutdownTaskRequest* request, ShutdownTaskResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  service_->ShutdownTaskAsync(request->source_task(),
                              [done](absl::Status s) { done(s); });
}

void CoordinationServiceRpcHandler::ResetTaskAsync(
    const ResetTaskRequest* request, ResetTaskResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service_->ResetTask(request->source_task()));
}

void CoordinationServiceRpcHandler::ReportErrorToTaskAsync(
    const ReportErrorToTaskRequest* request,
    ReportErrorToTaskResponse* response, StatusCallback done) {
  tf_shared_lock l(mu_);
  if (agent_ == nullptr) {
    done(MakeCoordinationError(errors::Internal(
        "CoordinationServiceAgent is uninitialized or has already shutdown.")));
    return;
  }
  const CoordinationServiceError& error_payload = request->error_payload();
  absl::Status error(
      static_cast<absl::StatusCode>(request->error_code()),
      strings::StrCat(
          "Error reported from /job:", error_payload.source_task().job_name(),
          "/task:", error_payload.source_task().task_id(), ": ",
          request->error_message()));
  error = MakeCoordinationError(error, error_payload);
  agent_->SetError(error);
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::ReportErrorToServiceAsync(
    const ReportErrorToServiceRequest* request,
    ReportErrorToServiceResponse* response, StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service_->ReportTaskError(
      request->error_origin(),
      MakeCoordinationError(
          absl::Status{static_cast<absl::StatusCode>(request->error_code()),
                       request->error_message()},
          request->error_origin(),
          /*is_reported_error=*/true)));
}

void CoordinationServiceRpcHandler::GetTaskStateAsync(
    const GetTaskStateRequest* request, GetTaskStateResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  auto result = service_->GetTaskState(
      {request->source_task().begin(), request->source_task().end()});
  absl::c_move(result,
               RepeatedFieldBackInserter(response->mutable_task_state()));
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::InsertKeyValueAsync(
    const InsertKeyValueRequest* request, InsertKeyValueResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service_->InsertKeyValue(request->kv().key(), request->kv().value()));
}

void CoordinationServiceRpcHandler::GetKeyValueAsync(
    const GetKeyValueRequest* request, GetKeyValueResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  response->mutable_kv()->set_key(request->key());
  service_->GetKeyValueAsync(
      request->key(), [response, done = std::move(done)](
                          const absl::StatusOr<std::string>& status_or_value) {
        if (status_or_value.ok()) {
          response->mutable_kv()->set_value(status_or_value.value());
        }
        done(status_or_value.status());
      });
}

void CoordinationServiceRpcHandler::TryGetKeyValueAsync(
    const TryGetKeyValueRequest* request, TryGetKeyValueResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  auto result = service_->TryGetKeyValue(request->key());
  if (!result.ok()) {
    done(MakeCoordinationError(result.status()));
    return;
  }
  response->mutable_kv()->set_key(request->key());
  response->mutable_kv()->set_value(result.value());
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::GetKeyValueDirAsync(
    const GetKeyValueDirRequest* request, GetKeyValueDirResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  std::vector<KeyValueEntry> results =
      service_->GetKeyValueDir(request->directory_key());
  *response->mutable_kv() = {std::make_move_iterator(results.begin()),
                             std::make_move_iterator(results.end())};
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::DeleteKeyValueAsync(
    const DeleteKeyValueRequest* request, DeleteKeyValueResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service_->DeleteKeyValue(request->key()));
}

void CoordinationServiceRpcHandler::BarrierAsync(const BarrierRequest* request,
                                                 BarrierResponse* response,
                                                 StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  std::vector<CoordinatedTask> tasks = {request->tasks().begin(),
                                        request->tasks().end()};
  service_->BarrierAsync(
      request->barrier_id(),
      absl::Milliseconds(request->barrier_timeout_in_ms()),
      request->source_task(), tasks,
      [done = std::move(done)](const absl::Status& status) { done(status); });
}

void CoordinationServiceRpcHandler::CancelBarrierAsync(
    const CancelBarrierRequest* request, CancelBarrierResponse* response,
    StatusCallback done) {
  tf_shared_lock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service_->CancelBarrier(request->barrier_id(), request->source_task()));
}

}  // namespace tsl
