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

#include "tensorflow/compiler/xla/client/client.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

Client::Client(ServiceInterface* stub) : stub_(stub) {}

Client::~Client() = default;

StatusOr<std::unique_ptr<Literal>> Client::Transfer(
    const GlobalData& data, const Shape* shape_with_layout) {
  TransferToClientRequest request;
  *request.mutable_data() = data.handle();
  if (shape_with_layout != nullptr) {
    *request.mutable_shape_with_layout() = *shape_with_layout;
  }
  TransferToClientResponse response;

  VLOG(1) << "making transfer request";
  VLOG(3) << "TransferToClientRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferToClient(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToClientResponse: {" << response.DebugString() << "}";

  if (!response.has_literal()) {
    return FailedPrecondition(
        "server provided response without a literal in "
        "TransferToClient request");
  }

  return WrapUnique(response.release_literal());
}

Status Client::TransferInProcess(const GlobalData& data, void* destination) {
  TransferToClientInProcessRequest request;
  *request.mutable_data() = data.handle();
  request.set_buffer(reinterpret_cast<uint64>(destination));
  TransferToClientInProcessResponse response;

  VLOG(1) << "making transfer in-process request";
  VLOG(3) << "TransferToClientInProcessRequest: {" << request.DebugString()
          << "}";
  Status s = stub_->TransferToClientInProcess(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToClientInProcessResponse: {" << response.DebugString()
          << "}";
  return Status::OK();
}

StatusOr<std::unique_ptr<GlobalData>> Client::TransferToServer(
    const Literal& literal, const DeviceHandle* device_handle) {
  TransferToServerRequest request;
  *request.mutable_literal() = literal;
  if (device_handle) {
    *request.mutable_device_handle() = *device_handle;
  }
  TransferToServerResponse response;

  VLOG(1) << "making transfer to server request";
  VLOG(3) << "TransferToServerRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferToServer(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToServerResponse: {" << response.DebugString() << "}";

  if (!response.has_data()) {
    return FailedPrecondition(
        "server provided response without a data handle in "
        "TransferToServer request");
  }

  return MakeUnique<GlobalData>(stub_, response.data());
}

Status Client::TransferToInfeed(const Literal& literal, int64 replica_id,
                                const DeviceHandle* device_handle) {
  TransferToInfeedRequest request;
  *request.mutable_literal() = literal;
  if (device_handle) {
    *request.mutable_device_handle() = *device_handle;
  }
  request.set_replica_id(replica_id);
  TransferToInfeedResponse response;

  VLOG(1) << "making transfer to infeed request";
  VLOG(3) << "TransferToInfeedRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferToInfeed(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToInfeedResponse: {" << response.DebugString() << "}";
  return Status::OK();
}

Status Client::ResetDevice() {
  ResetDeviceRequest request;
  ResetDeviceResponse response;

  VLOG(1) << "making reset device request";
  VLOG(3) << "ResetDeviceRequest: {" << request.DebugString() << "}";
  Status s = stub_->ResetDevice(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "ResetDeviceResponse: {" << response.DebugString() << "}";
  return Status::OK();
}

StatusOr<std::unique_ptr<Literal>> Client::ExecuteAndTransfer(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GlobalData> data,
      Execute(computation, arguments, execution_options, execution_profile));

  const Shape* shape_with_output_layout = nullptr;
  if (execution_options && execution_options->has_shape_with_output_layout()) {
    shape_with_output_layout = &execution_options->shape_with_output_layout();
  }
  return Transfer(*data, shape_with_output_layout);
}

StatusOr<std::unique_ptr<GlobalData>> Client::TransferToServerInProcess(
    const Shape& shape, const void* buffer) {
  TransferToServerInProcessRequest request;
  request.set_buffer(reinterpret_cast<uint64>(buffer));
  *request.mutable_shape() = shape;
  TransferToServerInProcessResponse response;

  VLOG(1) << "making transfer to server in-process request";
  VLOG(3) << "TransferToServerInProcessRequest: {" << request.DebugString()
          << "}";
  Status s = stub_->TransferToServerInProcess(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToServerInProcessResponse: {" << response.DebugString()
          << "}";

  if (!response.has_data()) {
    return FailedPrecondition(
        "server provided response without a data handle in "
        "TransferToServerInProcess request");
  }

  return MakeUnique<GlobalData>(stub_, response.data());
}

StatusOr<Computation> Client::LoadSnapshot(const SessionModule& module) {
  LoadComputationSnapshotRequest request;
  *request.mutable_module() = module;
  LoadComputationSnapshotResponse response;

  Status s = stub_->LoadComputationSnapshot(&request, &response);
  if (!s.ok()) {
    return s;
  }

  VLOG(1) << "load snapshot response: " << response.ShortDebugString();
  return Computation(stub_, response.computation());
}

StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
  ExecuteRequest request;
  *request.mutable_computation() = computation.handle();
  if (execution_options != nullptr) {
    *request.mutable_execution_options() = *execution_options;
  }
  for (GlobalData* argument : arguments) {
    *request.add_arguments() = argument->handle();
  }

  ExecuteResponse response;
  VLOG(1) << "making execute request: " << request.ShortDebugString();
  Status s = stub_->Execute(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  if (execution_profile != nullptr) {
    *execution_profile = response.profile();
    if (VLOG_IS_ON(1)) {
      TF_ASSIGN_OR_RETURN(
          auto execution_stats,
          ExecutionStatsAsString(computation, response.profile()));
      VLOG(1) << execution_stats;
    }
  }

  return MakeUnique<GlobalData>(stub_, response.output());
}

StatusOr<std::vector<std::unique_ptr<GlobalData>>> Client::ExecuteParallel(
    tensorflow::gtl::ArraySlice<ComputationInstance> computations) {
  ExecuteParallelRequest request;

  for (const ComputationInstance& computation : computations) {
    ExecuteRequest single_request;
    *single_request.mutable_computation() = computation.computation.handle();
    for (GlobalData* argument : computation.arguments) {
      *single_request.add_arguments() = argument->handle();
    }
    if (computation.device_handle != nullptr) {
      *single_request.mutable_device_handle() = *computation.device_handle;
    }
    *single_request.mutable_execution_options() = computation.execution_options;
    *request.add_requests() = single_request;
  }

  ExecuteParallelResponse response;
  VLOG(1) << "making execute-parallel request: " << request.ShortDebugString();
  tensorflow::Status s = stub_->ExecuteParallel(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  std::vector<std::unique_ptr<GlobalData>> outputs;
  for (int64 i = 0; i < computations.size(); ++i) {
    outputs.push_back(
        MakeUnique<GlobalData>(stub_, response.responses(i).output()));
    if (computations[i].execution_profile != nullptr) {
      *computations[i].execution_profile = response.responses(i).profile();
    }
  }

  return std::move(outputs);
}

StatusOr<std::vector<DeviceHandle>> Client::GetDeviceHandles(
    int64 device_count) {
  if (device_count < 1) {
    return InvalidArgument("device_count must be greater than 0");
  }
  GetDeviceHandlesRequest request;
  request.set_device_count(device_count);

  GetDeviceHandlesResponse response;
  VLOG(1) << "making get device request: " << request.ShortDebugString();
  tensorflow::Status s = stub_->GetDeviceHandles(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  std::vector<DeviceHandle> device_handles;
  for (const DeviceHandle& device_handle : response.device_handles()) {
    device_handles.push_back(device_handle);
  }

  return device_handles;
}

StatusOr<ExecutionHandle> Client::ExecuteAsync(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments,
    const ExecutionOptions* execution_options) {
  ExecuteAsyncRequest request;
  *request.mutable_computation() = computation.handle();
  for (GlobalData* argument : arguments) {
    *request.add_arguments() = argument->handle();
  }
  if (execution_options != nullptr) {
    *request.mutable_execution_options() = *execution_options;
  }

  ExecuteAsyncResponse response;
  VLOG(1) << "making execute async request: " << request.ShortDebugString();
  Status s = stub_->ExecuteAsync(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  return response.execution();
}

StatusOr<std::unique_ptr<GlobalData>> Client::WaitForExecution(
    const Computation& computation, const ExecutionHandle& execution,
    ExecutionProfile* execution_profile) {
  WaitForExecutionRequest request;
  *request.mutable_execution() = execution;

  WaitForExecutionResponse response;
  VLOG(1) << "making wait-for-execute request: " << request.ShortDebugString();
  Status s = stub_->WaitForExecution(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  if (execution_profile != nullptr) {
    *execution_profile = response.profile();
    if (VLOG_IS_ON(1)) {
      TF_ASSIGN_OR_RETURN(
          auto execution_stats,
          ExecutionStatsAsString(computation, response.profile()));
      VLOG(1) << execution_stats;
    }
  }

  return MakeUnique<GlobalData>(stub_, response.output());
}

Status Client::Unregister(const GlobalData& data) {
  UnregisterRequest request;
  *request.mutable_data() = data.handle();
  UnregisterResponse response;

  VLOG(1) << "making unregister request";
  Status s = stub_->Unregister(&request, &response);
  VLOG(1) << "done with request";

  return s;
}

StatusOr<std::vector<std::unique_ptr<GlobalData>>> Client::DeconstructTuple(
    const GlobalData& data) {
  DeconstructTupleRequest request;
  *request.mutable_tuple_handle() = data.handle();
  DeconstructTupleResponse response;

  VLOG(1) << "making DestructTuple request";
  Status s = stub_->DeconstructTuple(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  std::vector<std::unique_ptr<GlobalData>> handles;
  for (auto& handle : response.element_handles()) {
    handles.push_back(MakeUnique<GlobalData>(stub_, handle));
  }
  return std::move(handles);
}

StatusOr<ComputationStats> Client::GetComputationStats(
    const Computation& computation) const {
  ComputationStatsRequest request;
  *request.mutable_computation() = computation.handle();
  ComputationStatsResponse response;

  VLOG(1) << "making computation stats request";
  Status s = stub_->GetComputationStats(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  CHECK(response.has_stats());
  return response.stats();
}

StatusOr<std::unique_ptr<ProgramShape>> Client::GetComputationShape(
    const Computation& computation) {
  GetComputationShapeRequest request;
  *request.mutable_computation() = computation.handle();
  GetComputationShapeResponse response;

  VLOG(1) << "making get-computation-shape request";
  Status s = stub_->GetComputationShape(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  return WrapUnique(response.release_program_shape());
}

StatusOr<Shape> Client::GetShape(const GlobalData& data) {
  GetShapeRequest request;
  *request.mutable_data() = data.handle();
  GetShapeResponse response;

  VLOG(1) << "making get shape request";
  Status s = stub_->GetShape(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  return response.shape();
}

StatusOr<string> Client::ExecutionStatsAsString(
    const Computation& computation, const ExecutionProfile& profile) {
  TF_ASSIGN_OR_RETURN(auto computation_stats, GetComputationStats(computation));
  int64 total_flops =
      computation_stats.flop_count() + computation_stats.transcendental_count();
  if (profile.compute_time_ns() > 0) {
    int64 nanoseconds = profile.compute_time_ns();
    int64 cycle_count = profile.compute_cycle_count();
    double gflops = total_flops / nanoseconds;
    return tensorflow::strings::StrCat(
        "[Execution Statistics] flop count: ", computation_stats.flop_count(),
        ", transcendental count: ", computation_stats.transcendental_count(),
        ", compute execution time: ", nanoseconds, " nsec",
        ", compute cycles: ", cycle_count, ", performance: ", gflops,
        "gflop/s");
  }
  return string("[Execution Statistics] not available.");
}

StatusOr<ChannelHandle> Client::CreateChannelHandle() {
  CreateChannelHandleRequest request;
  CreateChannelHandleResponse response;

  VLOG(1) << "making create channel handle request";
  Status s = stub_->CreateChannelHandle(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  return response.channel();
}

}  // namespace xla
