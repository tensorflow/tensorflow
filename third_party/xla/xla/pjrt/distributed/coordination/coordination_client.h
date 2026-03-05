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

#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"

namespace xla {
using tensorflow::BarrierRequest;
using tensorflow::BarrierResponse;
using tensorflow::CancelBarrierRequest;
using tensorflow::CancelBarrierResponse;
using tensorflow::DeleteKeyValueRequest;
using tensorflow::DeleteKeyValueResponse;
using tensorflow::GetAliveTasksRequest;
using tensorflow::GetAliveTasksResponse;
using tensorflow::GetKeyValueDirRequest;
using tensorflow::GetKeyValueDirResponse;
using tensorflow::GetKeyValueRequest;
using tensorflow::GetKeyValueResponse;
using tensorflow::HeartbeatRequest;
using tensorflow::HeartbeatResponse;
using tensorflow::IncrementKeyValueRequest;
using tensorflow::IncrementKeyValueResponse;
using tensorflow::InsertKeyValueRequest;
using tensorflow::InsertKeyValueResponse;
using tensorflow::PollForErrorRequest;
using tensorflow::PollForErrorResponse;
using tensorflow::RegisterTaskRequest;
using tensorflow::RegisterTaskResponse;
using tensorflow::ResetTaskRequest;
using tensorflow::ResetTaskResponse;
using tensorflow::ShutdownTaskRequest;
using tensorflow::ShutdownTaskResponse;
using tensorflow::TryGetKeyValueRequest;
using tensorflow::TryGetKeyValueResponse;
using tensorflow::WatchJobStateRequest;
using tensorflow::WatchJobStateResponse;

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
