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

#ifndef XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
#define XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_

#include "absl/synchronization/mutex.h"
#include "xla/pjrt/distributed/coordination/coordination_service.h"
#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/thread_annotations.h"

namespace xla {
class CoordinationServiceRpcHandler {
 public:
  explicit CoordinationServiceRpcHandler() = default;

  void SetAgentInstance(CoordinationServiceAgent* agent);

  void SetServiceInstance(CoordinationService* service);

  void RegisterTaskAsync(const tensorflow::RegisterTaskRequest* request,
                         tensorflow::RegisterTaskResponse* response,
                         tsl::StatusCallback done);

  void HeartbeatAsync(const tensorflow::HeartbeatRequest* request,
                      tensorflow::HeartbeatResponse* response,
                      tsl::StatusCallback done);

  void ShutdownTaskAsync(const tensorflow::ShutdownTaskRequest* request,
                         tensorflow::ShutdownTaskResponse* response,
                         tsl::StatusCallback done);

  void ResetTaskAsync(const tensorflow::ResetTaskRequest* request,
                      tensorflow::ResetTaskResponse* response,
                      tsl::StatusCallback done);

  void WatchJobStateAsync(const tensorflow::WatchJobStateRequest* request,
                          tensorflow::WatchJobStateResponse* response,
                          tsl::StatusCallback done);

  void InsertKeyValueAsync(const tensorflow::InsertKeyValueRequest* request,
                           tensorflow::InsertKeyValueResponse* response,
                           tsl::StatusCallback done);

  void GetKeyValueAsync(const tensorflow::GetKeyValueRequest* request,
                        tensorflow::GetKeyValueResponse* response,
                        tsl::StatusCallback done);

  void IncrementKeyValueAsync(
      const tensorflow::IncrementKeyValueRequest* request,
      tensorflow::IncrementKeyValueResponse* response,
      tsl::StatusCallback done);

  void TryGetKeyValueAsync(const tensorflow::TryGetKeyValueRequest* request,
                           tensorflow::TryGetKeyValueResponse* response,
                           tsl::StatusCallback done);

  void GetKeyValueDirAsync(const tensorflow::GetKeyValueDirRequest* request,
                           tensorflow::GetKeyValueDirResponse* response,
                           tsl::StatusCallback done);

  void DeleteKeyValueAsync(const tensorflow::DeleteKeyValueRequest* request,
                           tensorflow::DeleteKeyValueResponse* response,
                           tsl::StatusCallback done);

  void BarrierAsync(const tensorflow::BarrierRequest* request,
                    tensorflow::BarrierResponse* response,
                    tsl::StatusCallback done);

  void CancelBarrierAsync(const tensorflow::CancelBarrierRequest* request,
                          tensorflow::CancelBarrierResponse* response,
                          tsl::StatusCallback done);

  void GetAliveTasksAsync(const tensorflow::GetAliveTasksRequest* request,
                          tensorflow::GetAliveTasksResponse* response,
                          tsl::StatusCallback done);

  void PollForErrorAsync(const tensorflow::PollForErrorRequest* request,
                         tensorflow::PollForErrorResponse* response,
                         tsl::StatusCallback done);

 private:
  absl::Mutex mu_;
  CoordinationServiceAgent* agent_ TF_GUARDED_BY(mu_) = nullptr;
  CoordinationService* service_ TF_GUARDED_BY(mu_) = nullptr;
};

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
