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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {

void CoordinationServiceRpcHandler::SetAgentInstance(
    CoordinationServiceAgent* agent) {
  agent_ = agent;
}

void CoordinationServiceRpcHandler::RegisterWorkerAsync(
    const RegisterWorkerRequest* request, RegisterWorkerResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(errors::Internal("Coordination service is not enabled."));
    return;
  }
  const std::string& job_name = request->job();
  const int task_id = request->task();
  const uint64 incarnation = request->incarnation();
  service->RegisterWorker(
      job_name, task_id, incarnation,
      [this, response, done = std::move(done)](Status s) {
        response->set_leader_incarnation(leader_incarnation_id_);
        done(s);
      });
}

void CoordinationServiceRpcHandler::HeartbeatAsync(
    const HeartbeatRequest* request, HeartbeatResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(errors::Internal("Coordination service is not enabled."));
    return;
  }
  const std::string& job_name = request->job();
  const int task_id = request->task();
  const uint64 incarnation = request->incarnation();
  Status s = service->RecordHeartbeat(job_name, task_id, incarnation);
  if (!s.ok()) {
    done(s);
    return;
  }
  response->set_leader_incarnation(leader_incarnation_id_);
  done(Status::OK());
}

void CoordinationServiceRpcHandler::WaitForAllTasksAsync(
    const WaitForAllTasksRequest* request, WaitForAllTasksResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(errors::Internal("Coordination service is not enabled."));
    return;
  }
  std::vector<DeviceAttributes> devices;
  for (const DeviceAttributes& da : request->local_device_attributes()) {
    devices.emplace_back(da);
  }
  service->WaitForAllTasks(
      request->job(), request->task(), std::move(devices),
      [response, service, done = std::move(done)](Status s) {
        if (s.ok()) {
          std::vector<DeviceAttributes> cluster_devices =
              service->ListClusterDevices();
          response->mutable_cluster_device_attributes()->Reserve(
              cluster_devices.size());
          for (auto& d : cluster_devices) {
            response->add_cluster_device_attributes()->Swap(&d);
          }
        }
        done(s);
      });
}

void CoordinationServiceRpcHandler::ReportErrorToAgentAsync(
    const ReportErrorToAgentRequest* request,
    ReportErrorToAgentResponse* response, StatusCallback done) {
  Status error(
      static_cast<error::Code>(request->error_code()),
      strings::StrCat("Error reported from /job:", request->source_job(),
                      "/task:", request->source_task(), ": ",
                      request->error_message()));
  agent_->SetError(error);
  done(Status::OK());
}

void CoordinationServiceRpcHandler::ReportErrorToServiceAsync(
    const ReportErrorToServiceRequest* request,
    ReportErrorToServiceResponse* response, StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(errors::Internal("Coordination service is not enabled."));
    return;
  }
  done(service->ReportTaskError(
      request->source_job(), request->source_task(),
      Status{static_cast<error::Code>(request->error_code()),
             request->error_message()}));
}

void CoordinationServiceRpcHandler::InsertKeyValueAsync(
    const InsertKeyValueRequest* request, InsertKeyValueResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(errors::Internal("Coordination service is not enabled."));
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
    done(errors::Internal("Coordination service is not enabled."));
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

void CoordinationServiceRpcHandler::DeleteKeyValueAsync(
    const DeleteKeyValueRequest* request, DeleteKeyValueResponse* response,
    StatusCallback done) {
  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(errors::Internal("Coordination service is not enabled."));
    return;
  }
  done(service->DeleteKeyValue(request->key()));
}

}  // namespace tensorflow
