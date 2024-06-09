/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_INTERFACE_H_
#define XLA_SERVICE_INTERFACE_H_

#include "absl/status/status.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Defines the interface for an XLA service on the client side. This service
// helps abstract around the actual implementation of a service - the service
// can be local (running in the same process), or remote - in which case an RPC
// stub is used as the implementation.
class ServiceInterface {
 public:
  ServiceInterface() {}
  virtual ~ServiceInterface() = default;

  // TODO(b/31824348): Convert to use absl::StatusOr.
  virtual absl::Status TransferToClient(const TransferToClientRequest* arg,
                                        TransferToClientResponse* result) = 0;

  virtual absl::Status TransferToServer(const TransferToServerRequest* arg,
                                        TransferToServerResponse* result) = 0;

  virtual absl::Status TransferToInfeed(const TransferToInfeedRequest* arg,
                                        TransferToInfeedResponse* result) = 0;

  virtual absl::Status TransferFromOutfeed(
      const TransferFromOutfeedRequest* arg,
      TransferFromOutfeedResponse* result) = 0;

  virtual absl::Status ResetDevice(const ResetDeviceRequest* arg,
                                   ResetDeviceResponse* result) = 0;

  virtual absl::Status Compile(const CompileRequest* arg,
                               CompileResponse* result) = 0;

  virtual absl::Status Execute(const ExecuteRequest* arg,
                               ExecuteResponse* result) = 0;

  virtual absl::Status ExecuteGraphParallel(
      const ExecuteGraphParallelRequest* arg,
      ExecuteParallelResponse* result) = 0;

  virtual absl::Status DeconstructTuple(const DeconstructTupleRequest* arg,
                                        DeconstructTupleResponse* result) = 0;

  virtual absl::Status GetComputationGraphStats(
      const ComputationGraphStatsRequest* arg,
      ComputationStatsResponse* result) = 0;

  virtual absl::Status GetShape(const GetShapeRequest* arg,
                                GetShapeResponse* result) = 0;

  virtual absl::Status CreateChannelHandle(
      const CreateChannelHandleRequest* arg,
      CreateChannelHandleResponse* result) = 0;

  virtual absl::Status GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                                        GetDeviceHandlesResponse* result) = 0;

  virtual absl::Status ComputeConstantGraph(
      const ComputeConstantGraphRequest* arg,
      ComputeConstantResponse* result) = 0;

  // Methods used by GlobalData.
  virtual absl::Status Unregister(const UnregisterRequest* arg,
                                  UnregisterResponse* result) = 0;
};

}  // namespace xla

#endif  // XLA_SERVICE_INTERFACE_H_
