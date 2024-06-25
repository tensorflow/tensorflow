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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

using xla::ffi::CallFrame;
using xla::ffi::CallFrameBuilder;
using xla::ffi::CallOptions;

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info,
                                 CustomCallTarget call_target,
                                 std::vector<std::optional<Slice>> operands,
                                 std::vector<std::optional<Slice>> results,
                                 const std::string& opaque)
    : Thunk(Thunk::kCustomCall, thunk_info),
      operands_(std::move(operands)),
      results_(std::move(results)),
      call_target_(std::move(call_target)),
      opaque_(opaque) {}

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info,
                                 XLA_FFI_Handler_Bundle bundle,
                                 std::vector<std::optional<Slice>> operands,
                                 std::vector<std::optional<Slice>> results,
                                 AttributesMap attributes,
                                 const HloComputation* called_computation)
    : Thunk(Thunk::kCustomCall, thunk_info),
      operands_(std::move(operands)),
      results_(std::move(results)),
      bundle_(bundle),
      attributes_(std::move(attributes)),
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  XlaCustomCallStatus custom_call_status;
  call_target_(gpu_stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return Internal("CustomCall failed: %s", *message);
  } else {
    return absl::OkStatus();
  }
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Custom calls on GPU are not supported in this configuration. Please "
      "build with --config=cuda or --config=rocm");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    XLA_FFI_Handler* handler, XLA_FFI_ExecutionStage stage,
    int32_t device_ordinal, se::Stream* stream,
    se::DeviceMemoryAllocator* allocator,
    const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations) {
  if (handler == nullptr) {
    return absl::InternalError("FFI execute handler is not set");
  }

  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is constructed.
  CallFrameBuilder builder;

  for (auto& operand : operands_) {
    if (!operand.has_value())
      return Internal("FFI handlers do not support tokens (yet)!");
    if (!operand->slice.allocation())
      return Internal("custom call argument missing buffer allocation");

    builder.AddBufferArg(buffer_allocations->GetDeviceAddress(operand->slice),
                         operand->shape.element_type(),
                         operand->shape.dimensions());
  }

  for (auto& result : results_) {
    if (!result.has_value())
      return Internal("FFI handlers do not support tokens (yet)!");
    if (!result->slice.allocation())
      return Internal("custom call result missing buffer allocation");

    builder.AddBufferRet(buffer_allocations->GetDeviceAddress(result->slice),
                         result->shape.element_type(),
                         result->shape.dimensions());
  }

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Append(attributes_);

  builder.AddAttributes(attrs.Build());
  CallFrame call_frame = builder.Build();

  CallOptions options = {device_ordinal, stream, allocator, called_computation_,
                         execution_context};
  return Call(bundle_->execute, call_frame, options, stage);
}

absl::Status CustomCallThunk::Prepare(const PrepareParams& params,
                                      ResourceRequests& resource_requests) {
  if (bundle_ && bundle_->prepare) {
    return absl::InternalError("FFI prepare stage is not yet supported");
  }
  return absl::OkStatus();
}

absl::Status CustomCallThunk::Initialize(const InitializeParams& params) {
  if (!bundle_ || !bundle_->initialize) {
    return absl::OkStatus();
  }

  return ExecuteFfiHandler(
      bundle_->initialize, XLA_FFI_ExecutionStage_INITIALIZE,
      params.buffer_allocations->device_ordinal(), params.stream,
      params.buffer_allocations->memory_allocator(),
      params.ffi_execution_context, params.buffer_allocations);
}

absl::Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (bundle_.has_value()) {
    return ExecuteFfiHandler(
        bundle_->execute, XLA_FFI_ExecutionStage_EXECUTE,
        params.buffer_allocations->device_ordinal(), params.stream,
        params.buffer_allocations->memory_allocator(),
        params.ffi_execution_context, params.buffer_allocations);
  }
  return ExecuteCustomCall(params);
}

}  // namespace gpu
}  // namespace xla
