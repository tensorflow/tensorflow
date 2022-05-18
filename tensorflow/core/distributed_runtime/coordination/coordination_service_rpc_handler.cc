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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.h"

#include <string>
#include <utility>

#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {

void CoordinationServiceRpcHandler::SetAgentInstance(
    CoordinationServiceAgent* agent) {
  mutex_lock l(agent_mu_);
  agent_ = agent;
}

void CoordinationServiceRpcHandler::RegisterTaskAsync(
    const RegisterTaskRequest* request, RegisterTaskResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service->GetServiceIncarnation();
  response->set_leader_incarnation(leader_incarnation);
  done(service->RegisterTask(task, incarnation));
}

void CoordinationServiceRpcHandler::HeartbeatAsync(
    const HeartbeatRequest* request, HeartbeatResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service->GetServiceIncarnation();
  Status s = service->RecordHeartbeat(task, incarnation);
  if (!s.ok()) {
    done(s);
    return;
  }
  response->set_leader_incarnation(leader_incarnation);
  done(Status::OK());
}

void CoordinationServiceRpcHandler::WaitForAllTasksAsync(
    const WaitForAllTasksRequest* request, WaitForAllTasksResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  service->WaitForAllTasks(
      request->source_task(), request->local_device_info(),
      [response, service, done = std::move(done)](Status s) {
        if (s.ok()) {
          *response->mutable_cluster_device_info() =
              service->ListClusterDevices();
        }
        done(s);
      });
}

void CoordinationServiceRpcHandler::ShutdownTaskAsync(
    const ShutdownTaskRequest* request, ShutdownTaskResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  service->ShutdownTaskAsync(request->source_task(),
                             [done](Status s) { done(s); });
}

void CoordinationServiceRpcHandler::ResetTaskAsync(
    const ResetTaskRequest* request, ResetTaskResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->ResetTask(request->source_task()));
}

void CoordinationServiceRpcHandler::ReportErrorToTaskAsync(
    const ReportErrorToTaskRequest* request,
    ReportErrorToTaskResponse* response, StatusCallback done) {
  tf_shared_lock l(agent_mu_);
  if (agent_ == nullptr) {
    done(MakeCoordinationError(errors::Internal(
        "CoordinationServiceAgent is uninitialized or has already shutdown.")));
    return;
  }
  const CoordinationServiceError& error_payload = request->error_payload();
  Status error(static_cast<error::Code>(request->error_code()),
               strings::StrCat("Error reported from /job:",
                               error_payload.source_task().job_name(),
                               "/task:", error_payload.source_task().task_id(),
                               ": ", request->error_message()));
  error = MakeCoordinationError(error, error_payload);
  agent_->SetError(error);
  done(Status::OK());
}

void CoordinationServiceRpcHandler::ReportErrorToServiceAsync(
    const ReportErrorToServiceRequest* request,
    ReportErrorToServiceResponse* response, StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->ReportTaskError(
      request->error_origin(),
      MakeCoordinationError(
          Status{static_cast<error::Code>(request->error_code()),
                 request->error_message()},
          request->error_origin(),
          /*is_reported_error=*/true)));
}

void CoordinationServiceRpcHandler::InsertKeyValueAsync(
    const InsertKeyValueRequest* request, InsertKeyValueResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->InsertKeyValue(request->kv().key(), request->kv().value()));
}

void CoordinationServiceRpcHandler::GetKeyValueAsync(
    const GetKeyValueRequest* request, GetKeyValueResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  response->mutable_kv()->set_key(request->key());
  service->GetKeyValueAsync(
      request->key(), [response, done = std::move(done)](
                          const StatusOr<std::string>& status_or_value) {
        if (status_or_value.ok()) {
          response->mutable_kv()->set_value(status_or_value.ValueOrDie());
        }
        done(status_or_value.status());
      });
}

void CoordinationServiceRpcHandler::GetKeyValueDirAsync(
    const GetKeyValueDirRequest* request, GetKeyValueDirResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  std::vector<KeyValueEntry> results =
      service->GetKeyValueDir(request->directory_key());
  *response->mutable_kv() = {std::make_move_iterator(results.begin()),
                             std::make_move_iterator(results.end())};
  done(Status::OK());
}

void CoordinationServiceRpcHandler::DeleteKeyValueAsync(
    const DeleteKeyValueRequest* request, DeleteKeyValueResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->DeleteKeyValue(request->key()));
}

void CoordinationServiceRpcHandler::BarrierAsync(const BarrierRequest* request,
                                                 BarrierResponse* response,
                                                 StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  std::vector<CoordinatedTask> tasks = {request->tasks().begin(),
                                        request->tasks().end()};
  service->BarrierAsync(
      request->barrier_id(),
      absl::Milliseconds(request->barrier_timeout_in_ms()),
      request->source_task(), tasks,
      [done = std::move(done)](const Status& status) { done(status); });
}

void CoordinationServiceRpcHandler::CancelBarrierAsync(
    const CancelBarrierRequest* request, CancelBarrierResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->CancelBarrier(request->barrier_id(), request->source_task()));
}

}  // namespace tensorflow
