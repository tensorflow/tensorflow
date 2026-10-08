/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/backends/gpu/runtime/cudnn_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace xla {
namespace gpu {

CuDnnThunk::CuDnnThunk(std::string fingerprint, ThunkInfo thunk_info,
                       std::vector<ShapedSlice> args,
                       std::vector<bool> output_args, bool should_memzero,
                       std::optional<int64_t> sdpa_dropout_seed)
    : TracedCommand(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      args_(std::move(args)),
      output_args_(std::move(output_args)),
      should_memzero_(should_memzero),
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status CuDnnThunk::Initialize(const InitializeParams& params) {
  return graphs_.GetOrCreateAndInitialize(
      params.stream->parent()->device_ordinal(),
      [&](std::unique_ptr<se::dnn::DnnGraph>* graph) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(*graph, CreateGraph(params));
        if (sdpa_dropout_seed_.has_value()) {
          (*graph)->InitDropoutState(params.local_device_count,
                                     *sdpa_dropout_seed_, 16);
        }
        return absl::OkStatus();
      });
}

absl::StatusOr<std::unique_ptr<se::dnn::DnnGraph>> CuDnnThunk::CreateGraph(
    const InitializeParams& params) {
  se::dnn::DnnSupport* dnn = params.stream->parent()->AsDnn();
  if (dnn == nullptr) {
    return absl::InternalError(
        "Failed to initialize DNN support for CuDnnThunk");
  }
  auto serialized = params.src.dnn_compiled_graphs.find(fingerprint_);
  if (serialized == params.src.dnn_compiled_graphs.end()) {
    return Internal("No serialized cuDNN graph with fingerprint '%s'",
                    fingerprint_);
  }
  return dnn->DeserializeGraph(*params.stream, serialized->second);
}

absl::StatusOr<se::dnn::DnnGraph*> CuDnnThunk::GetGraph(
    const se::StreamExecutor* executor) const {
  std::unique_ptr<se::dnn::DnnGraph>* graph =
      graphs_.Find(executor->device_ordinal());
  if (graph == nullptr || *graph == nullptr) {
    return Internal(
        "cuDNN graph for device ordinal %d has not been initialized; "
        "CuDnnThunk::Initialize() must run on every device before use",
        executor->device_ordinal());
  }
  return graph->get();
}

absl::Status CuDnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  ABSL_ASSIGN_OR_RETURN(se::dnn::DnnGraph * graph,
                   GetGraph(params.stream->parent()));
  std::vector<se::DeviceAddressBase> buffer_args;
  buffer_args.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    auto addr = params.buffer_allocations->GetDeviceAddress(arg.slice);
    if (output_args_[buffer_args.size()]) {
      if (should_memzero_) {
        ABSL_RETURN_IF_ERROR(params.stream->MemZero(&addr, addr.size()));
      }
      tsl::profiler::MarkMemoryInitialized(
          addr.opaque(), addr.size(),
          static_cast<tsl::profiler::StreamHandle>(
              params.stream->platform_specific_handle().stream));
    }
    buffer_args.push_back(addr);
  }
  return graph->Execute(*params.stream,
                        absl::Span<se::DeviceAddressBase>(buffer_args),
                        params.collective_params->local_device_id.value());
}

absl::StatusOr<const se::CommandBuffer::Command*> CuDnnThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  ABSL_ASSIGN_OR_RETURN(se::dnn::DnnGraph * graph,
                   GetGraph(execute_params.stream->parent()));
  std::vector<se::DeviceAddressBase> operands;
  operands.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg.slice);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }

  ABSL_ASSIGN_OR_RETURN(const bool supports_explicit,
                   graph->SupportsExplicitCommandBufferConstruction());
  if (supports_explicit) {
    if (auto* create = std::get_if<RecordCreate>(&record_action)) {
      return command_buffer->CreateDnnGraphCommand(
          *graph, *execute_params.stream,
          absl::Span<se::DeviceAddressBase>(operands), create->dependencies);
    }
    if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
      ABSL_RETURN_IF_ERROR(command_buffer->UpdateDnnGraphCommand(
          update->command, *graph, *execute_params.stream,
          absl::Span<se::DeviceAddressBase>(operands)));
      return update->command;
    }
    return Internal("Invalid record action");
  }
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return graph->Execute(
            *stream, absl::Span<se::DeviceAddressBase>(operands),
            execute_params.collective_params->local_device_id.value());
      });
}

absl::StatusOr<ThunkProto> CuDnnThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_cudnn_thunk()->set_fingerprint(fingerprint_);

  for (const ShapedSlice& arg : args_) {
    ABSL_ASSIGN_OR_RETURN(*proto.mutable_cudnn_thunk()->add_args(), arg.ToProto());
  }
  for (const bool is_output : output_args_) {
    proto.mutable_cudnn_thunk()->add_output_args(is_output);
  }
  proto.mutable_cudnn_thunk()->set_should_memzero(should_memzero_);
  if (sdpa_dropout_seed_.has_value()) {
    proto.mutable_cudnn_thunk()->set_sdpa_dropout_seed(
        static_cast<int64_t>(*sdpa_dropout_seed_));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<CuDnnThunk>> CuDnnThunk::FromProto(
    ThunkInfo thunk_info, const CudnnThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<ShapedSlice> args;
  args.reserve(proto.args_size());
  for (const ShapedSliceProto& arg : proto.args()) {
    ABSL_ASSIGN_OR_RETURN(args.emplace_back(),
                     ShapedSlice::FromProto(arg, buffer_allocations));
  }
  std::vector<bool> output_args;
  output_args.reserve(proto.output_args_size());
  for (const bool output_arg : proto.output_args()) {
    output_args.push_back(output_arg);
  }
  std::optional<uint64_t> sdpa_dropout_seed;
  if (proto.has_sdpa_dropout_seed()) {
    sdpa_dropout_seed = static_cast<uint64_t>(proto.sdpa_dropout_seed());
  }
  return std::make_unique<CuDnnThunk>(
      proto.fingerprint(), std::move(thunk_info), std::move(args),
      std::move(output_args), proto.should_memzero(), sdpa_dropout_seed);
}

}  // namespace gpu
}  // namespace xla
