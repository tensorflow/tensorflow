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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_CLIENT_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_CLIENT_H_

#include <memory>
#include <string>

#include "xla/tsl/distributed_runtime/call_options.h"
#include "tsl/platform/status.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace tsl {
using tensorflow::BarrierRequest;
using tensorflow::BarrierResponse;
using tensorflow::CancelBarrierRequest;
using tensorflow::CancelBarrierResponse;
using tensorflow::DeleteKeyValueRequest;
using tensorflow::DeleteKeyValueResponse;
using tensorflow::GetKeyValueDirRequest;
using tensorflow::GetKeyValueDirResponse;
using tensorflow::GetKeyValueRequest;
using tensorflow::GetKeyValueResponse;
using tensorflow::GetTaskStateRequest;
using tensorflow::GetTaskStateResponse;
using tensorflow::HeartbeatRequest;
using tensorflow::HeartbeatResponse;
using tensorflow::InsertKeyValueRequest;
using tensorflow::InsertKeyValueResponse;
using tensorflow::PollForErrorRequest;
using tensorflow::PollForErrorResponse;
using tensorflow::RegisterTaskRequest;
using tensorflow::RegisterTaskResponse;
using tensorflow::ReportErrorToServiceRequest;
using tensorflow::ReportErrorToServiceResponse;
using tensorflow::ReportErrorToTaskRequest;
using tensorflow::ReportErrorToTaskResponse;
using tensorflow::ResetTaskRequest;
using tensorflow::ResetTaskResponse;
using tensorflow::ShutdownTaskRequest;
using tensorflow::ShutdownTaskResponse;
using tensorflow::TryGetKeyValueRequest;
using tensorflow::TryGetKeyValueResponse;
using tensorflow::WaitForAllTasksRequest;
using tensorflow::WaitForAllTasksResponse;

// Base class of client interface for communicating with coordination service.
// Can be implemented by a variety of transports such as gRPC.
class CoordinationClient {
 public:
  virtual ~CoordinationClient() = default;

  virtual void RegisterTaskAsync(CallOptions* call_opts,
                                 const RegisterTaskRequest* request,
                                 RegisterTaskResponse* response,
                                 StatusCallback done) = 0;

  virtual void HeartbeatAsync(CallOptions* call_opts,
                              const HeartbeatRequest* request,
                              HeartbeatResponse* response,
                              StatusCallback done) = 0;

  virtual void WaitForAllTasksAsync(const WaitForAllTasksRequest* request,
                                    WaitForAllTasksResponse* response,
                                    StatusCallback done) = 0;

  virtual void ShutdownTaskAsync(CallOptions* call_opts,
                                 const ShutdownTaskRequest* request,
                                 ShutdownTaskResponse* response,
                                 StatusCallback done) = 0;

  virtual void ResetTaskAsync(const ResetTaskRequest* request,
                              ResetTaskResponse* response,
                              StatusCallback done) = 0;

  virtual void ReportErrorToTaskAsync(CallOptions* call_opts,
                                      const ReportErrorToTaskRequest* request,
                                      ReportErrorToTaskResponse* response,
                                      StatusCallback done) = 0;

  virtual void ReportErrorToServiceAsync(
      const ReportErrorToServiceRequest* request,
      ReportErrorToServiceResponse* response, StatusCallback done) = 0;

  virtual void GetTaskStateAsync(const GetTaskStateRequest* request,
                                 GetTaskStateResponse* response,
                                 StatusCallback done) = 0;

  virtual void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                                   InsertKeyValueResponse* response,
                                   StatusCallback done) = 0;

  virtual void GetKeyValueAsync(CallOptions* call_opts,
                                const GetKeyValueRequest* request,
                                GetKeyValueResponse* response,
                                StatusCallback done) = 0;

  virtual void TryGetKeyValueAsync(const TryGetKeyValueRequest* request,
                                   TryGetKeyValueResponse* response,
                                   StatusCallback done) = 0;

  virtual void GetKeyValueDirAsync(const GetKeyValueDirRequest* request,
                                   GetKeyValueDirResponse* response,
                                   StatusCallback done) = 0;

  virtual void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                                   DeleteKeyValueResponse* response,
                                   StatusCallback done) = 0;

  virtual void BarrierAsync(const BarrierRequest* request,
                            BarrierResponse* response, StatusCallback done) = 0;

  virtual void CancelBarrierAsync(const CancelBarrierRequest* request,
                                  CancelBarrierResponse* response,
                                  StatusCallback done) = 0;
  virtual void PollForErrorAsync(CallOptions* call_opts,
                                 const PollForErrorRequest* request,
                                 PollForErrorResponse* response,
                                 StatusCallback done) = 0;
};

// Simple wrapper class that can be used to retrieve CoordinationClients.
class CoordinationClientCache {
 public:
  virtual ~CoordinationClientCache() = default;

  // If the `target` names a remote task, returns a pointer of the
  // CoordinationClient object wrapping that channel to the remote task.
  virtual CoordinationClient* GetClient(const std::string& target) = 0;

  // If the `target` names a remote task, returns an owned pointer of the
  // CoordinationClient object wrapping that channel to the remote task.
  virtual std::unique_ptr<CoordinationClient> GetOwnedClient(
      const std::string& target) = 0;
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_CLIENT_H_
