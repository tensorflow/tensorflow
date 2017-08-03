/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INTERFACE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INTERFACE_H_

#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Defines the interface for an XLA service on the client side. This service
// helps abstract around the actual implementation of a service - the service
// can be local (running in the same process), or remote - in which case an RPC
// stub is used as the implementation.
class ServiceInterface {
 public:
  ServiceInterface() {}
  virtual ~ServiceInterface() = default;

  // TODO(b/31824348): Convert to use StatusOr.
  virtual tensorflow::Status TransferToClient(
      const TransferToClientRequest* arg, TransferToClientResponse* result) = 0;

  virtual tensorflow::Status TransferToServer(
      const TransferToServerRequest* arg, TransferToServerResponse* result) = 0;

  virtual tensorflow::Status TransferToInfeed(
      const TransferToInfeedRequest* arg, TransferToInfeedResponse* result) = 0;

  virtual tensorflow::Status TransferFromOutfeed(
      const TransferFromOutfeedRequest* arg,
      TransferFromOutfeedResponse* result) = 0;

  virtual tensorflow::Status ResetDevice(const ResetDeviceRequest* arg,
                                         ResetDeviceResponse* result) = 0;

  virtual tensorflow::Status LoadComputationSnapshot(
      const LoadComputationSnapshotRequest* request,
      LoadComputationSnapshotResponse* result) = 0;

  virtual tensorflow::Status Execute(const ExecuteRequest* arg,
                                     ExecuteResponse* result) = 0;

  virtual tensorflow::Status ExecuteParallel(
      const ExecuteParallelRequest* arg, ExecuteParallelResponse* result) = 0;

  virtual tensorflow::Status ExecuteAsync(const ExecuteAsyncRequest* arg,
                                          ExecuteAsyncResponse* result) = 0;

  virtual tensorflow::Status WaitForExecution(
      const WaitForExecutionRequest* arg, WaitForExecutionResponse* result) = 0;

  virtual tensorflow::Status DeconstructTuple(
      const DeconstructTupleRequest* arg, DeconstructTupleResponse* result) = 0;

  virtual tensorflow::Status GetComputationStats(
      const ComputationStatsRequest* arg, ComputationStatsResponse* result) = 0;

  virtual tensorflow::Status GetComputationShape(
      const GetComputationShapeRequest* arg,
      GetComputationShapeResponse* result) = 0;

  virtual tensorflow::Status GetShape(const GetShapeRequest* arg,
                                      GetShapeResponse* result) = 0;

  virtual tensorflow::Status CreateChannelHandle(
      const CreateChannelHandleRequest* arg,
      CreateChannelHandleResponse* result) = 0;

  virtual tensorflow::Status GetDeviceHandles(
      const GetDeviceHandlesRequest* arg, GetDeviceHandlesResponse* result) = 0;

  // Methods used by ComputationBuilder.
  virtual tensorflow::Status Computation(const ComputationRequest* arg,
                                         ComputationResponse* result) = 0;

  virtual tensorflow::Status Op(const OpRequest* arg, OpResponse* result) = 0;

  virtual tensorflow::Status GetLocalShape(const GetLocalShapeRequest* arg,
                                           GetLocalShapeResponse* result) = 0;

  virtual tensorflow::Status SetReturnValue(
      const SetReturnValueRequest* arg, SetReturnValueResponse* results) = 0;

  virtual tensorflow::Status IsConstant(const IsConstantRequest* arg,
                                        IsConstantResponse* result) = 0;

  virtual tensorflow::Status ComputeConstant(
      const ComputeConstantRequest* arg, ComputeConstantResponse* result) = 0;

  // Methods used by Computation.
  virtual tensorflow::Status SnapshotComputation(
      const SnapshotComputationRequest* ag,
      SnapshotComputationResponse* result) = 0;

  // Methods used by GlobalData.
  virtual tensorflow::Status Unregister(const UnregisterRequest* arg,
                                        UnregisterResponse* result) = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERFACE_H_
