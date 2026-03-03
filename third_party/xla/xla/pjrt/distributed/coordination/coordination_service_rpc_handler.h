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
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"

namespace xla {
class CoordinationServiceRpcHandler {
 public:
  explicit CoordinationServiceRpcHandler() = default;

  void SetAgentInstance(CoordinationServiceAgent* agent);

  void SetServiceInstance(CoordinationService* service);

  void RegisterTaskAsync(const xla::coordination::RegisterTaskRequest* request,
                         xla::coordination::RegisterTaskResponse* response,
                         tsl::StatusCallback done);

  void HeartbeatAsync(const xla::coordination::HeartbeatRequest* request,
                      xla::coordination::HeartbeatResponse* response,
                      tsl::StatusCallback done);

  void ShutdownTaskAsync(const xla::coordination::ShutdownTaskRequest* request,
                         xla::coordination::ShutdownTaskResponse* response,
                         tsl::StatusCallback done);

  void ResetTaskAsync(const xla::coordination::ResetTaskRequest* request,
                      xla::coordination::ResetTaskResponse* response,
                      tsl::StatusCallback done);

  void WatchJobStateAsync(
      const xla::coordination::WatchJobStateRequest* request,
      xla::coordination::WatchJobStateResponse* response,
      tsl::StatusCallback done);

  void InsertKeyValueAsync(
      const xla::coordination::InsertKeyValueRequest* request,
      xla::coordination::InsertKeyValueResponse* response,
      tsl::StatusCallback done);

  void GetKeyValueAsync(const xla::coordination::GetKeyValueRequest* request,
                        xla::coordination::GetKeyValueResponse* response,
                        tsl::StatusCallback done);

  void IncrementKeyValueAsync(
      const xla::coordination::IncrementKeyValueRequest* request,
      xla::coordination::IncrementKeyValueResponse* response,
      tsl::StatusCallback done);

  void TryGetKeyValueAsync(
      const xla::coordination::TryGetKeyValueRequest* request,
      xla::coordination::TryGetKeyValueResponse* response,
      tsl::StatusCallback done);

  void GetKeyValueDirAsync(
      const xla::coordination::GetKeyValueDirRequest* request,
      xla::coordination::GetKeyValueDirResponse* response,
      tsl::StatusCallback done);

  void DeleteKeyValueAsync(
      const xla::coordination::DeleteKeyValueRequest* request,
      xla::coordination::DeleteKeyValueResponse* response,
      tsl::StatusCallback done);

  void BarrierAsync(const xla::coordination::BarrierRequest* request,
                    xla::coordination::BarrierResponse* response,
                    tsl::StatusCallback done);

  void CancelBarrierAsync(
      const xla::coordination::CancelBarrierRequest* request,
      xla::coordination::CancelBarrierResponse* response,
      tsl::StatusCallback done);

  void GetAliveTasksAsync(
      const xla::coordination::GetAliveTasksRequest* request,
      xla::coordination::GetAliveTasksResponse* response,
      tsl::StatusCallback done);

  void PollForErrorAsync(const xla::coordination::PollForErrorRequest* request,
                         xla::coordination::PollForErrorResponse* response,
                         tsl::StatusCallback done);

 private:
  absl::Mutex mu_;
  CoordinationServiceAgent* agent_ TF_GUARDED_BY(mu_) = nullptr;
  CoordinationService* service_ TF_GUARDED_BY(mu_) = nullptr;
};

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
