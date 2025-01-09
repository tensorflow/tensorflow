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

#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinationServiceError;
using tensorflow::KeyValueEntry;
}  // namespace

void CoordinationServiceRpcHandler::SetAgentInstance(
    CoordinationServiceAgent* agent) {
  absl::MutexLock l(&mu_);
  agent_ = agent;
}

void CoordinationServiceRpcHandler::SetServiceInstance(
    CoordinationServiceInterface* service) {
  absl::MutexLock l(&mu_);
  service_ = service;
}

void CoordinationServiceRpcHandler::RegisterTaskAsync(
    const tensorflow::RegisterTaskRequest* request,
    tensorflow::RegisterTaskResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service_->GetServiceIncarnation();
  response->set_leader_incarnation(leader_incarnation);
  service_->RegisterTaskAsync(task, incarnation, done);
}

void CoordinationServiceRpcHandler::HeartbeatAsync(
    const tensorflow::HeartbeatRequest* request,
    tensorflow::HeartbeatResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
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
    const tensorflow::WaitForAllTasksRequest* request,
    tensorflow::WaitForAllTasksResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
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
    const tensorflow::ShutdownTaskRequest* request,
    tensorflow::ShutdownTaskResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  service_->ShutdownTaskAsync(request->source_task(),
                              [done](absl::Status s) { done(s); });
}

void CoordinationServiceRpcHandler::ResetTaskAsync(
    const tensorflow::ResetTaskRequest* request,
    tensorflow::ResetTaskResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->ResetTask(request->source_task()));
}

void CoordinationServiceRpcHandler::ReportErrorToTaskAsync(
    const tensorflow::ReportErrorToTaskRequest* request,
    tensorflow::ReportErrorToTaskResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (agent_ == nullptr) {
    done(MakeCoordinationError(absl::InternalError(
        "CoordinationServiceAgent is uninitialized or has already shutdown.")));
    return;
  }
  const CoordinationServiceError& error_payload = request->error_payload();
  absl::Status error(
      static_cast<absl::StatusCode>(request->error_code()),
      absl::StrCat(
          "Error reported from /job:", error_payload.source_task().job_name(),
          "/task:", error_payload.source_task().task_id(), ": ",
          request->error_message()));
  error = MakeCoordinationError(error, error_payload);
  agent_->SetError(error);
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::ReportErrorToServiceAsync(
    const tensorflow::ReportErrorToServiceRequest* request,
    tensorflow::ReportErrorToServiceResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
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
    const tensorflow::GetTaskStateRequest* request,
    tensorflow::GetTaskStateResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  auto result = service_->GetTaskState(
      {request->source_task().begin(), request->source_task().end()});
  absl::c_move(result, tsl::protobuf::RepeatedFieldBackInserter(
                           response->mutable_task_state()));
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::InsertKeyValueAsync(
    const tensorflow::InsertKeyValueRequest* request,
    tensorflow::InsertKeyValueResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->InsertKeyValue(request->kv().key(), request->kv().value(),
                                request->allow_overwrite()));
}

void CoordinationServiceRpcHandler::GetKeyValueAsync(
    const tensorflow::GetKeyValueRequest* request,
    tensorflow::GetKeyValueResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  response->mutable_kv()->set_key(request->key());
  service_->GetKeyValueAsync(
      request->key(),
      [response, done = std::move(done)](
          const absl::StatusOr<std::string_view>& status_or_value) {
        if (status_or_value.ok()) {
          auto value = status_or_value.value();
          response->mutable_kv()->set_value(value.data(), value.size());
        }
        done(status_or_value.status());
      });
}

void CoordinationServiceRpcHandler::TryGetKeyValueAsync(
    const tensorflow::TryGetKeyValueRequest* request,
    tensorflow::TryGetKeyValueResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
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
    const tensorflow::GetKeyValueDirRequest* request,
    tensorflow::GetKeyValueDirResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  std::vector<KeyValueEntry> results =
      service_->GetKeyValueDir(request->directory_key());
  *response->mutable_kv() = {std::make_move_iterator(results.begin()),
                             std::make_move_iterator(results.end())};
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::DeleteKeyValueAsync(
    const tensorflow::DeleteKeyValueRequest* request,
    tensorflow::DeleteKeyValueResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->DeleteKeyValue(request->key()));
}

void CoordinationServiceRpcHandler::BarrierAsync(
    const tensorflow::BarrierRequest* request,
    tensorflow::BarrierResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  std::vector<CoordinatedTask> tasks = {request->tasks().begin(),
                                        request->tasks().end()};
  service_->BarrierAsync(request->barrier_id(), request->counter(),
                         absl::Milliseconds(request->barrier_timeout_in_ms()),
                         request->source_task(), tasks,
                         [done = std::move(done), response](
                             const absl::Status& status, int64_t counter) {
                           response->set_counter(counter);
                           done(status);
                         });
}

void CoordinationServiceRpcHandler::CancelBarrierAsync(
    const tensorflow::CancelBarrierRequest* request,
    tensorflow::CancelBarrierResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->CancelBarrier(request->barrier_id(), request->counter(),
                               request->source_task()));
}

void CoordinationServiceRpcHandler::GetAliveTasksAsync(
    const tensorflow::GetAliveTasksRequest* request,
    tensorflow::GetAliveTasksResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }

  std::vector<CoordinatedTask> tasks = {request->tasks().begin(),
                                        request->tasks().end()};
  service_->GetAliveTasksAsync(
      request->requesting_task(), tasks,
      [done = std::move(done), response](
          const absl::Status& status,
          const std::vector<tensorflow::CoordinatedTask>& alive_tasks) {
        *response->mutable_alive_tasks() = {alive_tasks.begin(),
                                            alive_tasks.end()};
        done(status);
      });
}

void CoordinationServiceRpcHandler::PollForErrorAsync(
    const tensorflow::PollForErrorRequest* request,
    tensorflow::PollForErrorResponse* response, StatusCallback done) {
  absl::ReaderMutexLock l(&mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  service_->PollForErrorAsync(
      request->source_task(),
      [done = std::move(done)](const absl::Status& status) { done(status); });
}

}  // namespace tsl
