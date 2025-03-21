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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
class CoordinationServiceRpcHandler {
 public:
  explicit CoordinationServiceRpcHandler() = default;

  void SetAgentInstance(CoordinationServiceAgent* agent);

  void SetServiceInstance(CoordinationServiceInterface* service);

  void RegisterTaskAsync(const tensorflow::RegisterTaskRequest* request,
                         tensorflow::RegisterTaskResponse* response,
                         StatusCallback done);

  void HeartbeatAsync(const tensorflow::HeartbeatRequest* request,
                      tensorflow::HeartbeatResponse* response,
                      StatusCallback done);

  void WaitForAllTasksAsync(const tensorflow::WaitForAllTasksRequest* request,
                            tensorflow::WaitForAllTasksResponse* response,
                            StatusCallback done);

  void ShutdownTaskAsync(const tensorflow::ShutdownTaskRequest* request,
                         tensorflow::ShutdownTaskResponse* response,
                         StatusCallback done);

  void ResetTaskAsync(const tensorflow::ResetTaskRequest* request,
                      tensorflow::ResetTaskResponse* response,
                      StatusCallback done);

  void ReportErrorToTaskAsync(
      const tensorflow::ReportErrorToTaskRequest* request,
      tensorflow::ReportErrorToTaskResponse* response, StatusCallback done);

  void ReportErrorToServiceAsync(
      const tensorflow::ReportErrorToServiceRequest* request,
      tensorflow::ReportErrorToServiceResponse* response, StatusCallback done);

  void GetTaskStateAsync(const tensorflow::GetTaskStateRequest* request,
                         tensorflow::GetTaskStateResponse* response,
                         StatusCallback done);

  void GetJobStateAsync(const tensorflow::GetJobStateRequest* request,
                        tensorflow::GetJobStateResponse* response,
                        StatusCallback done);

  void InsertKeyValueAsync(const tensorflow::InsertKeyValueRequest* request,
                           tensorflow::InsertKeyValueResponse* response,
                           StatusCallback done);

  void GetKeyValueAsync(const tensorflow::GetKeyValueRequest* request,
                        tensorflow::GetKeyValueResponse* response,
                        StatusCallback done);

  void TryGetKeyValueAsync(const tensorflow::TryGetKeyValueRequest* request,
                           tensorflow::TryGetKeyValueResponse* response,
                           StatusCallback done);

  void GetKeyValueDirAsync(const tensorflow::GetKeyValueDirRequest* request,
                           tensorflow::GetKeyValueDirResponse* response,
                           StatusCallback done);

  void DeleteKeyValueAsync(const tensorflow::DeleteKeyValueRequest* request,
                           tensorflow::DeleteKeyValueResponse* response,
                           StatusCallback done);

  void BarrierAsync(const tensorflow::BarrierRequest* request,
                    tensorflow::BarrierResponse* response, StatusCallback done);

  void CancelBarrierAsync(const tensorflow::CancelBarrierRequest* request,
                          tensorflow::CancelBarrierResponse* response,
                          StatusCallback done);

  void GetAliveTasksAsync(const tensorflow::GetAliveTasksRequest* request,
                          tensorflow::GetAliveTasksResponse* response,
                          StatusCallback done);

  void PollForErrorAsync(const tensorflow::PollForErrorRequest* request,
                         tensorflow::PollForErrorResponse* response,
                         StatusCallback done);

 private:
  absl::Mutex mu_;
  CoordinationServiceAgent* agent_ TF_GUARDED_BY(mu_) = nullptr;
  CoordinationServiceInterface* service_ TF_GUARDED_BY(mu_) = nullptr;
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
