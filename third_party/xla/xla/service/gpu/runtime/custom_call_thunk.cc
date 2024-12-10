/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/custom_call_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

using xla::ffi::CallFrame;
using xla::ffi::CallFrameBuilder;
using xla::ffi::CallOptions;

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name, CustomCallTarget call_target,
    std::vector<std::optional<Slice>> operands,
    std::vector<std::optional<Slice>> results, const std::string& opaque) {
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(call_target),
      std::move(operands), std::move(results), opaque));
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name,
    XLA_FFI_Handler_Bundle bundle, std::vector<std::optional<Slice>> operands,
    std::vector<std::optional<Slice>> results, AttributesMap attributes,
    const HloComputation* called_computation) {
  auto execution_state = std::make_unique<ffi::ExecutionState>();

  // Initialize FFI handler state if it has an instantiate callback.
  if (bundle.instantiate) {
    // At FFI handler instantiation time, we don't have any arguments or
    // results or access to the underlying device (stream, etc.)
    CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);

    CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(attributes);

    builder.AddAttributes(attrs.Build());
    CallFrame call_frame = builder.Build();

    CallOptions options;
    options.execution_state = execution_state.get();
    TF_RETURN_IF_ERROR(Call(bundle.instantiate, call_frame, options,
                            XLA_FFI_ExecutionStage_INSTANTIATE));
  }

  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), bundle, std::move(operands),
      std::move(results), std::move(attributes), std::move(execution_state),
      called_computation));
}

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info, std::string target_name,
                                 CustomCallTarget call_target,
                                 std::vector<std::optional<Slice>> operands,
                                 std::vector<std::optional<Slice>> results,
                                 const std::string& opaque)
    : Thunk(Thunk::kCustomCall, thunk_info),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      call_target_(std::move(call_target)),
      opaque_(opaque) {}

CustomCallThunk::CustomCallThunk(
    ThunkInfo thunk_info, std::string target_name,
    XLA_FFI_Handler_Bundle bundle, std::vector<std::optional<Slice>> operands,
    std::vector<std::optional<Slice>> results, AttributesMap attributes,
    std::unique_ptr<ffi::ExecutionState> execution_state,
    const HloComputation* called_computation)
    : Thunk(Thunk::kCustomCall, thunk_info),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      bundle_(bundle),
      attributes_(std::move(attributes)),
      execution_state_(std::move(execution_state)),
      called_computation_(called_computation) {}

absl::Status CustomCallThunk::ExecuteCustomCall(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (!slice.has_value()) {
        buffers.push_back(nullptr);
        continue;
      }

      if (!slice->slice.allocation())
        return Internal("custom call input missing buffer allocation");

      buffers.push_back(
          params.buffer_allocations->GetDeviceAddress(slice->slice).opaque());
    }
  }

  XlaCustomCallStatus custom_call_status;
  call_target_(params.stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return Internal("CustomCall failed: %s", *message);
  } else {
    return absl::OkStatus();
  }
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    XLA_FFI_Handler* handler, XLA_FFI_ExecutionStage stage, se::Stream* stream,
    const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations) {
  if (handler == nullptr) {
    return absl::InternalError("FFI execute handler is not set");
  }
  if (stage != XLA_FFI_ExecutionStage_PREPARE &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is constructed.
  CallFrameBuilder builder(operands_.size(), results_.size());
  auto device_address =
      [buffer_allocations](
          BufferAllocation::Slice slice) -> se::DeviceMemoryBase {
    return buffer_allocations ? buffer_allocations->GetDeviceAddress(slice)
                              : se::DeviceMemoryBase{};
  };

  for (auto& operand : operands_) {
    if (!operand.has_value()) {
      builder.AddTokenArg();
      continue;
    }

    if (!operand->slice.allocation())
      return Internal("custom call argument missing buffer allocation");

    builder.AddBufferArg(device_address(operand->slice),
                         operand->shape.element_type(),
                         operand->shape.dimensions());
  }

  for (auto& result : results_) {
    if (!result.has_value()) {
      builder.AddTokenRet();
      continue;
    }

    if (!result->slice.allocation())
      return Internal("custom call result missing buffer allocation");

    builder.AddBufferRet(device_address(result->slice),
                         result->shape.element_type(),
                         result->shape.dimensions());
  }

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Append(attributes_);

  builder.AddAttributes(attrs.Build());
  CallFrame call_frame = builder.Build();

  int32_t device_ordinal = -1;
  se::DeviceMemoryAllocator* allocator = nullptr;
  if (stage != XLA_FFI_ExecutionStage_PREPARE) {
    device_ordinal = buffer_allocations->device_ordinal();
    allocator = buffer_allocations->memory_allocator();
  }

  CallOptions options = {
      device_ordinal, CallOptions::GpuOptions{stream, allocator},
      called_computation_, execution_context, execution_state_.get()};
  return Call(handler, call_frame, options, stage);
}

absl::Status CustomCallThunk::Prepare(const PrepareParams& params,
                                      ResourceRequests& resource_requests) {
  if (!bundle_ || !bundle_->prepare) {
    return absl::OkStatus();
  }

  return ExecuteFfiHandler(bundle_->prepare, XLA_FFI_ExecutionStage_PREPARE,
                           /*stream=*/nullptr,
                           /*execution_context=*/nullptr,
                           /*buffer_allocations=*/nullptr);
}

absl::Status CustomCallThunk::Initialize(const InitializeParams& params) {
  if (!bundle_ || !bundle_->initialize) {
    return absl::OkStatus();
  }

  return ExecuteFfiHandler(
      bundle_->initialize, XLA_FFI_ExecutionStage_INITIALIZE, params.stream,
      params.ffi_execution_context, params.buffer_allocations);
}

absl::Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (bundle_.has_value()) {
    return ExecuteFfiHandler(bundle_->execute, XLA_FFI_ExecutionStage_EXECUTE,
                             params.stream, params.ffi_execution_context,
                             params.buffer_allocations);
  }
  return ExecuteCustomCall(params);
}

}  // namespace gpu
}  // namespace xla
