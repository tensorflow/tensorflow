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

#include "xla/client/client.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/execution_options_util.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {

Client::Client(Service* stub) : stub_(stub) {}

Client::~Client() = default;

absl::StatusOr<Literal> Client::Transfer(const GlobalData& data,
                                         const Shape* shape_with_layout) {
  return stub_->TransferToClient(data, shape_with_layout);
}

absl::StatusOr<std::unique_ptr<GlobalData>> Client::TransferToServer(
    const LiteralSlice& literal, const DeviceHandle* device_handle) {
  return stub_->TransferToServer(literal, device_handle);
}

absl::Status Client::TransferToInfeed(const LiteralSlice& literal,
                                      int64_t replica_id,
                                      const DeviceHandle* device_handle) {
  return stub_->TransferToInfeed(literal, replica_id, device_handle);
}

absl::StatusOr<Literal> Client::TransferFromOutfeed(
    const Shape* shape_with_layout, int64_t replica_id,
    const DeviceHandle* device_handle) {
  return stub_->TransferFromOutfeed(shape_with_layout, replica_id,
                                    device_handle);
}

absl::Status Client::ResetDevice() { return stub_->ResetDevice(); }

absl::StatusOr<Literal> Client::ExecuteAndTransfer(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GlobalData> data,
      Execute(computation, arguments, execution_options, execution_profile));

  std::optional<Shape> shape_with_output_layout;
  if (execution_options && execution_options->has_shape_with_output_layout()) {
    shape_with_output_layout =
        Shape(execution_options->shape_with_output_layout());
  }
  return Transfer(*data, shape_with_output_layout.has_value()
                             ? &(*shape_with_output_layout)
                             : nullptr);
}

absl::StatusOr<Literal> Client::ComputeConstant(
    const XlaComputation& computation, const Layout* output_layout) const {
  return stub_->ComputeConstantGraph(computation, output_layout);
}

absl::StatusOr<XlaComputation> Client::LoadSnapshot(const HloSnapshot& module) {
  TF_RET_CHECK(module.has_hlo() && module.hlo().has_hlo_module());
  return XlaComputation(module.hlo().hlo_module());
}

absl::StatusOr<ExecutionHandle> Client::Compile(
    const XlaComputation& computation, absl::Span<const Shape> argument_shapes,
    const ExecutionOptions* execution_options) {
  std::optional<ExecutionOptions> opts;
  if (!execution_options) {
    opts = CreateDefaultExecutionOptions();
  }

  return stub_->Compile(computation, argument_shapes,
                        execution_options ? *execution_options : *opts);
}

absl::StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const ExecutionHandle& handle, absl::Span<GlobalData* const> arguments,
    ExecutionProfile* execution_profile) {
  return stub_->Execute(handle, arguments, execution_profile);
}

absl::StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
  // Create an ExecutionOptions if necessary, or set its DeviceHandles.
  std::optional<ExecutionOptions> options_storage;
  if (!execution_options || execution_options->device_handles().empty()) {
    if (execution_options) {
      options_storage.emplace(*execution_options);
    } else {
      options_storage.emplace(CreateDefaultExecutionOptions());
    }
    execution_options = &*options_storage;

    TF_ASSIGN_OR_RETURN(auto device_handles,
                        GetDeviceHandles(/*device_count=*/1));
    TF_RET_CHECK(!device_handles.empty());
    *options_storage->add_device_handles() = std::move(device_handles[0]);
  }

  std::vector<XlaComputationInstance> computation_instances = {
      XlaComputationInstance{
          computation,
          std::vector<GlobalData*>(arguments.begin(), arguments.end()),
          *execution_options, execution_profile}};

  // Instead of invoking Compile() and Execute(), invoke
  // Service::ExecuteParallel() to execute our one computation.  Compile()
  // caches the executable forever, which isn't what we want.
  VLOG(1) << "Making ExecuteParallel request: "
          << execution_options->DebugString();
  TF_ASSIGN_OR_RETURN(auto results, ExecuteParallel(computation_instances));
  VLOG(1) << "ExecuteParallel request done.";

  // The result selection is a bit hacky, but better than assuming it is
  // device 0.
  //
  // TODO(b/118493728): Allow Execute to return one result per computation.
  for (int64_t i = 0, end = results.size(); i < end; i++) {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(*results[i]));
    if (!ShapeUtil::IsEmptyTuple(shape)) {
      VLOG(3) << "Fetching result from device " << i << ": "
              << ShapeUtil::HumanString(shape);
      return std::move(results[i]);
    }
  }
  TF_RET_CHECK(!results.empty());
  VLOG(1) << "Defaulting to device 0 result";
  return std::move(results[0]);
}

absl::StatusOr<std::vector<std::unique_ptr<GlobalData>>>
Client::ExecuteParallel(absl::Span<const XlaComputationInstance> computations) {
  return stub_->ExecuteGraphParallel(computations);
}

absl::StatusOr<std::vector<DeviceHandle>> Client::GetDeviceHandles(
    int64_t device_count) {
  if (device_count < 1) {
    return InvalidArgument("device_count must be greater than 0");
  }

  return stub_->GetDeviceHandles(device_count);
}

absl::Status Client::Unregister(const GlobalData& data) {
  return stub_->Unregister(data.handle());
}

absl::StatusOr<std::vector<std::unique_ptr<GlobalData>>>
Client::DeconstructTuple(const GlobalData& data) {
  return stub_->DeconstructTuple(data);
}

absl::StatusOr<std::unique_ptr<ProgramShape>> Client::GetComputationShape(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(const auto& result, computation.GetProgramShape());
  return std::make_unique<ProgramShape>(result);
}

absl::StatusOr<Shape> Client::GetShape(const GlobalData& data) {
  return stub_->GetShape(data);
}

absl::StatusOr<ChannelHandle> Client::CreateChannelHandleByType(
    ChannelHandle::ChannelType type) {
  return stub_->CreateChannelHandle(type);
}

absl::StatusOr<ChannelHandle> Client::CreateChannelHandle() {
  return CreateChannelHandleByType(ChannelHandle::DEVICE_TO_DEVICE);
}

absl::StatusOr<ChannelHandle> Client::CreateHostToDeviceChannelHandle() {
  return CreateChannelHandleByType(ChannelHandle::HOST_TO_DEVICE);
}

absl::StatusOr<ChannelHandle> Client::CreateDeviceToHostChannelHandle() {
  return CreateChannelHandleByType(ChannelHandle::DEVICE_TO_HOST);
}

}  // namespace xla
