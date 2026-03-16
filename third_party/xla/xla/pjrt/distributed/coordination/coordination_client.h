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

#ifndef XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_CLIENT_H_
#define XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_CLIENT_H_

#include <memory>
#include <string>

#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/platform/status.h"

namespace xla {
using xla::coordination::BarrierRequest;
using xla::coordination::BarrierResponse;
using xla::coordination::CancelBarrierRequest;
using xla::coordination::CancelBarrierResponse;
using xla::coordination::DeleteKeyValueRequest;
using xla::coordination::DeleteKeyValueResponse;
using xla::coordination::GetAliveTasksRequest;
using xla::coordination::GetAliveTasksResponse;
using xla::coordination::GetKeyValueDirRequest;
using xla::coordination::GetKeyValueDirResponse;
using xla::coordination::GetKeyValueRequest;
using xla::coordination::GetKeyValueResponse;
using xla::coordination::HeartbeatRequest;
using xla::coordination::HeartbeatResponse;
using xla::coordination::IncrementKeyValueRequest;
using xla::coordination::IncrementKeyValueResponse;
using xla::coordination::InsertKeyValueRequest;
using xla::coordination::InsertKeyValueResponse;
using xla::coordination::PollForErrorRequest;
using xla::coordination::PollForErrorResponse;
using xla::coordination::RegisterTaskRequest;
using xla::coordination::RegisterTaskResponse;
using xla::coordination::ResetTaskRequest;
using xla::coordination::ResetTaskResponse;
using xla::coordination::ShutdownTaskRequest;
using xla::coordination::ShutdownTaskResponse;
using xla::coordination::TryGetKeyValueRequest;
using xla::coordination::TryGetKeyValueResponse;
using xla::coordination::WatchJobStateRequest;
using xla::coordination::WatchJobStateResponse;

// Base class of client interface for communicating with coordination service.
// Can be implemented by a variety of transports such as gRPC.
class CoordinationClient {
 public:
  virtual ~CoordinationClient() = default;

  virtual void RegisterTaskAsync(tsl::CallOptions* call_opts,
                                 const RegisterTaskRequest* request,
                                 RegisterTaskResponse* response,
                                 tsl::StatusCallback done) = 0;

  virtual void HeartbeatAsync(tsl::CallOptions* call_opts,
                              const HeartbeatRequest* request,
                              HeartbeatResponse* response,
                              tsl::StatusCallback done) = 0;

  virtual void ShutdownTaskAsync(tsl::CallOptions* call_opts,
                                 const ShutdownTaskRequest* request,
                                 ShutdownTaskResponse* response,
                                 tsl::StatusCallback done) = 0;

  virtual void ResetTaskAsync(const ResetTaskRequest* request,
                              ResetTaskResponse* response,
                              tsl::StatusCallback done) = 0;

  virtual void WatchJobStateAsync(tsl::CallOptions* call_opts,
                                  const WatchJobStateRequest* request,
                                  WatchJobStateResponse* response,
                                  tsl::StatusCallback done) = 0;

  virtual void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                                   InsertKeyValueResponse* response,
                                   tsl::StatusCallback done) = 0;

  virtual void GetKeyValueAsync(tsl::CallOptions* call_opts,
                                const GetKeyValueRequest* request,
                                GetKeyValueResponse* response,
                                tsl::StatusCallback done) = 0;

  virtual void TryGetKeyValueAsync(const TryGetKeyValueRequest* request,
                                   TryGetKeyValueResponse* response,
                                   tsl::StatusCallback done) = 0;

  virtual void GetKeyValueDirAsync(const GetKeyValueDirRequest* request,
                                   GetKeyValueDirResponse* response,
                                   tsl::StatusCallback done) = 0;

  virtual void IncrementKeyValueAsync(const IncrementKeyValueRequest* request,
                                      IncrementKeyValueResponse* response,
                                      tsl::StatusCallback done) = 0;

  virtual void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                                   DeleteKeyValueResponse* response,
                                   tsl::StatusCallback done) = 0;

  virtual void BarrierAsync(tsl::CallOptions* call_opts,
                            const BarrierRequest* request,
                            BarrierResponse* response,
                            tsl::StatusCallback done) = 0;

  virtual void CancelBarrierAsync(const CancelBarrierRequest* request,
                                  CancelBarrierResponse* response,
                                  tsl::StatusCallback done) = 0;

  virtual void GetAliveTasksAsync(const GetAliveTasksRequest* request,
                                  GetAliveTasksResponse* response,
                                  tsl::StatusCallback done) = 0;

  virtual void PollForErrorAsync(tsl::CallOptions* call_opts,
                                 const PollForErrorRequest* request,
                                 PollForErrorResponse* response,
                                 tsl::StatusCallback done) = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_COORDINATION_COORDINATION_CLIENT_H_
