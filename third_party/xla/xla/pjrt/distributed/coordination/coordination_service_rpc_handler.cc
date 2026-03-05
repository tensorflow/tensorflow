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

#include "xla/pjrt/distributed/coordination/coordination_service_rpc_handler.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/pjrt/distributed/coordination/coordination_service.h"
#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"
#include "xla/pjrt/distributed/coordination/coordination_service_error_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinationServiceError;
using tensorflow::KeyValueEntry;
}  // namespace

void CoordinationServiceRpcHandler::SetAgentInstance(
    CoordinationServiceAgent* agent) {
  absl::MutexLock l(mu_);
  agent_ = agent;
}

void CoordinationServiceRpcHandler::SetServiceInstance(
    CoordinationService* service) {
  absl::MutexLock l(mu_);
  service_ = service;
}

void CoordinationServiceRpcHandler::RegisterTaskAsync(
    const tensorflow::RegisterTaskRequest* request,
    tensorflow::RegisterTaskResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const IncarnationId incarnation(request->incarnation());
  const IncarnationId leader_incarnation = service_->GetServiceIncarnation();
  response->set_leader_incarnation(leader_incarnation.value());
  service_->RegisterTaskAsync(task.task_id(), incarnation, done);
}

void CoordinationServiceRpcHandler::HeartbeatAsync(
    const tensorflow::HeartbeatRequest* request,
    tensorflow::HeartbeatResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const IncarnationId incarnation(request->incarnation());
  const IncarnationId leader_incarnation = service_->GetServiceIncarnation();
  absl::Status s = service_->RecordHeartbeat(task.task_id(), incarnation);
  if (!s.ok()) {
    done(s);
    return;
  }
  response->set_leader_incarnation(leader_incarnation.value());
  done(absl::OkStatus());
}

void CoordinationServiceRpcHandler::ShutdownTaskAsync(
    const tensorflow::ShutdownTaskRequest* request,
    tensorflow::ShutdownTaskResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  service_->ShutdownTaskAsync(request->source_task().task_id(),
                              [done](absl::Status s) { done(s); });
}

void CoordinationServiceRpcHandler::ResetTaskAsync(
    const tensorflow::ResetTaskRequest* request,
    tensorflow::ResetTaskResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->ResetTask(request->source_task().task_id()));
}

void CoordinationServiceRpcHandler::WatchJobStateAsync(
    const tensorflow::WatchJobStateRequest* request,
    tensorflow::WatchJobStateResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }

  std::optional<int64_t> version_number;
  if (request->version_number() >= 0) {
    version_number.emplace(request->version_number());
  }
  service_->WatchJobState(
      version_number,
      [response, done](std::vector<tensorflow::CoordinatedTaskStateInfo> info,
                       int64_t version_number) {
        absl::c_move(info, tsl::protobuf::RepeatedFieldBackInserter(
                               response->mutable_task_state()));
        response->set_version_number(version_number);
        done(absl::OkStatus());
      });
}

void CoordinationServiceRpcHandler::InsertKeyValueAsync(
    const tensorflow::InsertKeyValueRequest* request,
    tensorflow::InsertKeyValueResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
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
    tensorflow::GetKeyValueResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  response->mutable_kv()->set_key(request->key());
  service_->GetKeyValueAsync(
      request->key(),
      [response, done = std::move(done)](
          const absl::StatusOr<absl::string_view>& status_or_value) {
        if (status_or_value.ok()) {
          auto value = status_or_value.value();
          response->mutable_kv()->set_value(value.data(), value.size());
        }
        done(status_or_value.status());
      });
}

void CoordinationServiceRpcHandler::TryGetKeyValueAsync(
    const tensorflow::TryGetKeyValueRequest* request,
    tensorflow::TryGetKeyValueResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
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

void CoordinationServiceRpcHandler::IncrementKeyValueAsync(
    const tensorflow::IncrementKeyValueRequest* request,
    tensorflow::IncrementKeyValueResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  auto result =
      service_->IncrementKeyValue(request->key(), request->increment());
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
    tensorflow::GetKeyValueDirResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
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
    tensorflow::DeleteKeyValueResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->DeleteKeyValue(request->key()));
}

void CoordinationServiceRpcHandler::BarrierAsync(
    const tensorflow::BarrierRequest* request,
    tensorflow::BarrierResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  std::vector<CoordinationService::TaskId> tasks;
  for (const tensorflow::CoordinatedTask& task : request->tasks()) {
    tasks.push_back(task.task_id());
  }
  service_->BarrierAsync(request->barrier_id(), request->counter(),
                         absl::Milliseconds(request->barrier_timeout_in_ms()),
                         request->source_task().task_id(), tasks,
                         [done = std::move(done), response](
                             const absl::Status& status, int64_t counter) {
                           response->set_counter(counter);
                           done(status);
                         });
}

void CoordinationServiceRpcHandler::CancelBarrierAsync(
    const tensorflow::CancelBarrierRequest* request,
    tensorflow::CancelBarrierResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  done(service_->CancelBarrier(request->barrier_id(), request->counter(),
                               request->source_task().task_id()));
}

void CoordinationServiceRpcHandler::GetAliveTasksAsync(
    const tensorflow::GetAliveTasksRequest* request,
    tensorflow::GetAliveTasksResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }

  std::vector<CoordinationService::TaskId> tasks;
  for (const tensorflow::CoordinatedTask& task : request->tasks()) {
    tasks.push_back(task.task_id());
  }
  service_->GetAliveTasksAsync(
      request->requesting_task().task_id(), tasks,
      [done = std::move(done), response](
          const absl::Status& status,
          const std::vector<CoordinationService::TaskId>& alive_tasks,
          const std::vector<IncarnationId>& incarnations) {
        for (const CoordinationService::TaskId task : alive_tasks) {
          response->add_alive_tasks()->set_task_id(task);
        }
        for (IncarnationId id : incarnations) {
          response->add_incarnations(id.value());
        }
        done(status);
      });
}

void CoordinationServiceRpcHandler::PollForErrorAsync(
    const tensorflow::PollForErrorRequest* request,
    tensorflow::PollForErrorResponse* response, tsl::StatusCallback done) {
  absl::ReaderMutexLock l(mu_);
  if (service_ == nullptr) {
    done(MakeCoordinationError(
        absl::InternalError("Coordination service is not enabled.")));
    return;
  }
  service_->PollForErrorAsync(
      request->source_task().task_id(),
      [done = std::move(done)](const absl::Status& status) { done(status); });
}

}  // namespace xla
