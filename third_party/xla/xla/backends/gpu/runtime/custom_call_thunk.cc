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

#include "xla/backends/gpu/runtime/custom_call_thunk.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

using xla::ffi::CallFrame;
using xla::ffi::CallFrameBuilder;
using xla::ffi::CallOptions;

// Builds a call frame prototype for typed-FFI custom calls with dummy device
// memory addresses. This is called once when creating the CustomCall thunk,
// then the thunk will need to update the addresses at runtime.
static absl::StatusOr<ffi::CallFrame> BuildCallFramePrototype(
    absl::Span<const std::optional<CustomCallThunk::Slice>> operands,
    absl::Span<const std::optional<CustomCallThunk::Slice>> results,
    CustomCallThunk::AttributesMap attributes) {
  CallFrameBuilder builder(
      /*num_args=*/operands.size(),
      /*num_rets=*/results.size());

  // Add prototype input buffers with actual data types and shapes. Device
  // memory addresses will be updated at runtime.
  for (int i = 0; i < operands.size(); ++i) {
    auto& operand = operands[i];

    if (!operand.has_value()) {
      builder.AddTokenArg();
      continue;
    }

    auto elements = absl::c_accumulate(operand->shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(operand->shape.element_type());
    se::DeviceMemoryBase placeholder_arg(nullptr, elements * dtype_bytes);
    builder.AddBufferArg(placeholder_arg, operand->shape.element_type(),
                         operand->shape.dimensions());
  }

  // Add prototype output buffers with actual data types and shapes. Device
  // memory addresses will be updated at runtime.
  for (int i = 0; i < results.size(); ++i) {
    auto& result = results[i];

    if (!result.has_value()) {
      builder.AddTokenRet();
      continue;
    }

    auto elements = absl::c_accumulate(result->shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(result->shape.element_type());
    se::DeviceMemoryBase placeholder_ret(nullptr, elements * dtype_bytes);
    builder.AddBufferRet(placeholder_ret, result->shape.element_type(),
                         result->shape.dimensions());
  }

  // Add attributes if any.
  if (!attributes.empty()) {
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));
    builder.AddAttributes(attrs.Build());
  }

  return builder.Build();
}

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

  TF_ASSIGN_OR_RETURN(
      CallFrame call_frame,
      BuildCallFramePrototype(operands, results, std::move(attributes)));

  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), bundle, std::move(operands),
      std::move(results), std::move(call_frame), std::move(execution_state),
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
    std::vector<std::optional<Slice>> results, CallFrame call_frame,
    std::unique_ptr<ffi::ExecutionState> execution_state,
    const HloComputation* called_computation)
    : Thunk(Thunk::kCustomCall, thunk_info),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      bundle_(bundle),
      call_frame_(std::move(call_frame)),
      call_frames_([this] { return call_frame_->Copy(); }),
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

      if (!slice->slice.allocation()) {
        return Internal("custom call input missing buffer allocation");
      }

      buffers.push_back(
          params.buffer_allocations->GetDeviceAddress(slice->slice).opaque());
    }
  }

  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  XlaCustomCallStatus custom_call_status;
  call_target_(stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return Internal("CustomCall failed: %s", *message);
  } else {
    return absl::OkStatus();
  }
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    RunId run_id, XLA_FFI_Handler* handler, XLA_FFI_ExecutionStage stage,
    se::Stream* stream, const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations) {
  if (handler == nullptr) {
    return absl::InternalError("FFI execute handler is not set");
  }
  if (stage != XLA_FFI_ExecutionStage_PREPARE &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  auto device_memory = [&](BufferAllocation::Slice slice) {
    return buffer_allocations ? buffer_allocations->GetDeviceAddress(slice)
                              : se::DeviceMemoryBase{};
  };

  // Collect arguments buffers.
  absl::InlinedVector<se::DeviceMemoryBase, 8> arguments;
  arguments.reserve(operands_.size());
  for (auto& operand : operands_) {
    if (!operand.has_value()) {
      arguments.push_back(se::DeviceMemoryBase{});
    } else {
      arguments.push_back(device_memory(operand->slice));
    }
  }

  // Collect results buffers.
  absl::InlinedVector<se::DeviceMemoryBase, 4> results;
  results.reserve(results_.size());
  for (auto& result : results_) {
    if (!result.has_value()) {
      results.push_back(se::DeviceMemoryBase{});
    } else {
      results.push_back(device_memory(result->slice));
    }
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));

  int32_t device_ordinal = -1;
  se::DeviceMemoryAllocator* allocator = nullptr;
  if (stage != XLA_FFI_ExecutionStage_PREPARE) {
    device_ordinal = buffer_allocations->device_ordinal();
    allocator = buffer_allocations->memory_allocator();
  }

  CallOptions options = {run_id,
                         device_ordinal,
                         CallOptions::GpuOptions{stream, allocator},
                         called_computation_,
                         execution_context,
                         execution_state_.get()};
  return Call(handler, *call_frame, options, stage);
}

absl::Status CustomCallThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  if (!bundle_ || !bundle_->prepare) {
    return absl::OkStatus();
  }

  return ExecuteFfiHandler(
      params.collective_params ? params.collective_params->run_id : RunId{-1},
      bundle_->prepare, XLA_FFI_ExecutionStage_PREPARE,
      /*stream=*/nullptr,
      /*execution_context=*/nullptr,
      /*buffer_allocations=*/nullptr);
}

absl::Status CustomCallThunk::Initialize(const InitializeParams& params) {
  if (!bundle_ || !bundle_->initialize) {
    return absl::OkStatus();
  }

  return ExecuteFfiHandler(
      params.collective_params ? params.collective_params->run_id : RunId{-1},
      bundle_->initialize, XLA_FFI_ExecutionStage_INITIALIZE, params.stream,
      params.ffi_execution_context, params.buffer_allocations);
}

absl::Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  if (bundle_.has_value()) {
    return ExecuteFfiHandler(
        params.collective_params ? params.collective_params->run_id : RunId{-1},
        bundle_->execute, XLA_FFI_ExecutionStage_EXECUTE, stream,
        params.ffi_execution_context, params.buffer_allocations);
  }
  return ExecuteCustomCall(params);
}

}  // namespace gpu
}  // namespace xla
