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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/barrier_requests.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/custom_call_target.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/primitive_util.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/platform.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

using xla::ffi::CallFrame;
using xla::ffi::CallFrameBuilder;
using xla::ffi::InvokeContext;

// Builds a call frame prototype for typed-FFI custom calls with dummy device
// memory addresses. This is called once when creating the CustomCall thunk,
// then the thunk will need to update the addresses at runtime.
static absl::StatusOr<ffi::CallFrame> BuildCallFramePrototype(
    absl::Span<const NullableShapedSlice> operands,
    absl::Span<const NullableShapedSlice> results,
    ffi::AttributesMap attributes) {
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
    se::DeviceAddressBase placeholder_arg(nullptr, elements * dtype_bytes);
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
    se::DeviceAddressBase placeholder_ret(nullptr, elements * dtype_bytes);
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

static absl::StatusOr<CustomCallThunk::CustomCallTarget>
ResolveLegacyCustomCall(const CustomCallTargetRegistry& registry,
                        absl::string_view target_name,
                        absl::string_view platform_name,
                        CustomCallApiVersion api_version) {
  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      std::string(target_name), std::string(platform_name));

  // For information about this calling convention, see
  // xla/g3doc/custom_call.md.
  switch (api_version) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL: {
      constexpr absl::string_view kErrorMessage =
          "Custom call API version `API_VERSION_ORIGINAL` is not supported by "
          "XLA:GPU. Prefer https://docs.jax.dev/en/latest/ffi.html. It will be "
          "fully removed in November 2025.";
      if constexpr (tsl::kIsOpenSource) {
        LOG(ERROR) << kErrorMessage;
      } else {
        LOG(FATAL) << kErrorMessage;
      }

      return [call_target](stream_executor::Stream* stream, void** buffers,
                           const char* opaque, size_t opaque_len,
                           XlaCustomCallStatus*) {
        reinterpret_cast<CustomCallWithOpaqueStreamHandle>(call_target)(
            stream->platform_specific_handle().stream, buffers, opaque,
            opaque_len);
      };
      break;
    }
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      return [call_target](stream_executor::Stream* stream, void** buffers,
                           const char* opaque, size_t opaque_len,
                           XlaCustomCallStatus* status) {
        reinterpret_cast<CustomCallWithStatusAndOpaqueStreamHandle>(
            call_target)(stream->platform_specific_handle().stream, buffers,
                         opaque, opaque_len, status);
      };
      break;
    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      return absl::InvalidArgumentError(
          "Called ResolveLegacyCustomCall with API_VERSION_TYPED_FFI");
    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      api_version);
  }
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name, CustomCallTarget call_target,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, std::string opaque) {
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(operands),
      std::move(results), std::move(opaque), std::move(call_target),
      /*api_version=*/std::nullopt));
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, std::string opaque,
    CustomCallApiVersion api_version, absl::string_view platform_name) {
  if (api_version == CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    return absl::InvalidArgumentError(
        "Called overload of CustomCallThunk::Create that is intended for "
        "legacy custom calls with api_version=API_VERSION_TYPED_FFI");
  }

  TF_ASSIGN_OR_RETURN(
      CustomCallTarget call_target,
      ResolveLegacyCustomCall(*CustomCallTargetRegistry::Global(), target_name,
                              platform_name, api_version));

  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(operands),
      std::move(results), std::move(opaque), call_target, api_version));
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, ffi::AttributesMap attributes,
    const HloComputation* called_computation, absl::string_view platform_name,
    const se::GpuComputeCapability& gpu_compute_capability,
    std::unique_ptr<ffi::ExecutionState> execution_state) {
  TF_ASSIGN_OR_RETURN(ffi::HandlerRegistration registration,
                      ffi::FindHandler(target_name, platform_name));

  return Create(thunk_info, std::move(target_name),
                std::move(registration.bundle), std::move(operands),
                std::move(results), std::move(attributes), called_computation,
                gpu_compute_capability, std::move(execution_state));
}

static InvokeContext BuildInstantiateInvokeContext(
    ffi::ExecutionState* execution_state,
    const se::GpuComputeCapability* gpu_compute_capability) {
  InvokeContext context{};
  context.execution_state = execution_state;
  context.backend_context = InvokeContext::GpuContext{
      /*.stream=*/nullptr,
      /*.allocator=*/nullptr,
      /*.collective_params=*/nullptr,
      /*.collective_clique_requests=*/nullptr,
      /*.collective_memory_requests=*/nullptr,
      /*.barrier_requests=*/nullptr,
      /*.collective_cliques=*/nullptr,
      /*.collective_memory=*/nullptr,
      /*.gpu_target_config=*/gpu_compute_capability,
  };
  return context;
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name,
    XLA_FFI_Handler_Bundle bundle, std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, ffi::AttributesMap attributes,
    const HloComputation* called_computation,
    const se::GpuComputeCapability& gpu_compute_capability,
    std::unique_ptr<ffi::ExecutionState> execution_state) {
  // Initialize FFI handler state if it has an instantiate callback.
  if (execution_state == nullptr) {
    execution_state = std::make_unique<ffi::ExecutionState>();
    if (bundle.instantiate) {
      // At FFI handler instantiation time, we don't have any arguments or
      // results or access to the underlying device (stream, etc.)
      CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);

      CallFrameBuilder::AttributesBuilder attrs;
      attrs.Append(attributes);

      builder.AddAttributes(attrs.Build());
      CallFrame call_frame = builder.Build();

      InvokeContext call_options = BuildInstantiateInvokeContext(
          execution_state.get(), &gpu_compute_capability);
      RETURN_IF_ERROR(Invoke(ffi::GetXlaFfiApi(), bundle.instantiate,
                             call_frame, call_options,
                             XLA_FFI_ExecutionStage_INSTANTIATE));
    }
  }

  TF_ASSIGN_OR_RETURN(CallFrame call_frame,
                      BuildCallFramePrototype(operands, results, attributes));
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(bundle),
      std::move(operands), std::move(results), std::move(call_frame),
      std::move(attributes), std::move(execution_state), called_computation));
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name, OwnedHandlerBundle bundle,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results,
    xla::ffi::AttributesMap attributes,
    const HloComputation* called_computation,
    const se::GpuComputeCapability& gpu_compute_capability) {
  if (!bundle.execute) {
    return absl::InvalidArgumentError(
        "Execute handler is required for a CustomCallThunk");
  }

  auto execution_state = std::make_unique<ffi::ExecutionState>();

  // Initialize FFI handler state if it has an instantiate callback.
  if (bundle.instantiate) {
    // At FFI handler instantiation time, we don't have any arguments or
    // results or access to the underlying device (stream, etc.), however users
    // have access to some information about the target architecture like
    // `TargetGpuComputeCapability`.
    CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);

    CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(attributes);

    builder.AddAttributes(attrs.Build());
    CallFrame call_frame = builder.Build();

    InvokeContext context = BuildInstantiateInvokeContext(
        execution_state.get(), &gpu_compute_capability);
    TF_RETURN_IF_ERROR(Invoke(ffi::GetXlaFfiApi(), *bundle.instantiate,
                              call_frame, context,
                              xla::ffi::ExecutionStage::kInstantiate));
  }

  TF_ASSIGN_OR_RETURN(CallFrame call_frame,
                      BuildCallFramePrototype(operands, results, attributes));
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(bundle),
      std::move(operands), std::move(results), std::move(call_frame),
      std::move(attributes), std::move(execution_state), called_computation));
}

CustomCallThunk::CustomCallThunk(
    ThunkInfo thunk_info, std::string target_name,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, std::string opaque,
    CustomCallTarget call_target,
    const std::optional<CustomCallApiVersion>& api_version)
    : Thunk(Thunk::kCustomCall, thunk_info),
      api_version_(api_version),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      call_target_(std::move(call_target)),
      opaque_(std::move(opaque)) {}

CustomCallThunk::CustomCallThunk(
    ThunkInfo thunk_info, std::string target_name,
    std::variant<XLA_FFI_Handler_Bundle, OwnedHandlerBundle> bundle,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, CallFrame call_frame,
    ffi::AttributesMap attributes,
    std::unique_ptr<ffi::ExecutionState> execution_state,
    const HloComputation* called_computation)
    : Thunk(Thunk::kCustomCall, thunk_info),
      api_version_(CustomCallApiVersion::API_VERSION_TYPED_FFI),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      bundle_(std::move(bundle)),
      attributes_(std::move(attributes)),
      call_frame_(std::move(call_frame)),
      call_frames_([this] { return call_frame_->Copy(); }),
      execution_state_(std::move(execution_state)),
      called_computation_(called_computation) {}

absl::Status CustomCallThunk::ExecuteCustomCall(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<ShapedSlice>& slice : slices) {
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
  }
  return absl::OkStatus();
}

// Builds a call frame for the custom call.
//
// If `buffer_allocations` is provided, the call frame will contain the actual
// device memory addresses of the buffers. Otherwise, the call frame will
// contain placeholders - this should only be the case when calling Prepare()
// stage handler.
absl::StatusOr<ObjectPool<CallFrame>::BorrowedObject>
CustomCallThunk::BuildCallFrame(
    const BufferAllocations* absl_nullable buffer_allocations) {
  auto device_memory = [&](BufferAllocation::Slice slice) {
    return buffer_allocations ? buffer_allocations->GetDeviceAddress(slice)
                              : se::DeviceAddressBase{};
  };

  // Collect arguments buffers.
  absl::InlinedVector<se::DeviceAddressBase, 8> arguments;
  arguments.reserve(operands_.size());
  for (auto& operand : operands_) {
    if (!operand.has_value()) {
      arguments.push_back(se::DeviceAddressBase{});
    } else {
      arguments.push_back(device_memory(operand->slice));
    }
  }

  // Collect results buffers.
  absl::InlinedVector<se::DeviceAddressBase, 4> results;
  results.reserve(results_.size());
  for (auto& result : results_) {
    if (!result.has_value()) {
      results.push_back(se::DeviceAddressBase{});
    } else {
      results.push_back(device_memory(result->slice));
    }
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));
  return call_frame;
}

// Builds call options object for the custom call.
//
// `stream` and `buffer_allocations may only be non-null for options passed to
// Prepare()_stage handler.
InvokeContext CustomCallThunk::BuildInvokeContext(
    RunId run_id, se::Stream* absl_nullable stream,
    const BufferAllocations* absl_nullable buffer_allocations,
    const CollectiveParams* absl_nullable collective_params,
    CollectiveCliqueRequests* absl_nullable collective_clique_requests,
    CollectiveMemoryRequests* absl_nullable collective_memory_requests,
    BarrierRequests* absl_nullable barrier_requests,
    const CollectiveCliques* absl_nullable collective_cliques,
    const CollectiveMemory* absl_nullable collective_memory,
    const ffi::ExecutionContext* absl_nullable execution_context) {
  int32_t device_ordinal = -1;
  se::DeviceAddressAllocator* allocator = nullptr;
  if (buffer_allocations != nullptr) {
    device_ordinal = buffer_allocations->device_ordinal();
    allocator = buffer_allocations->memory_allocator();
  }

  const se::GpuComputeCapability* gpu_compute_capability = nullptr;
  if (stream != nullptr) {
    gpu_compute_capability =
        &stream->parent()->GetDeviceDescription().gpu_compute_capability();
  }

  return InvokeContext{
      run_id,
      device_ordinal,
      InvokeContext::GpuContext{
          stream, allocator, collective_params, collective_clique_requests,
          collective_memory_requests, barrier_requests, collective_cliques,
          collective_memory, gpu_compute_capability},
      called_computation_,
      execution_context,
      execution_state_.get()};
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    RunId run_id, XLA_FFI_Handler* handler, XLA_FFI_ExecutionStage stage,
    se::Stream* stream, const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations,
    const CollectiveParams* absl_nullable collective_params,
    CollectiveCliqueRequests* absl_nullable collective_clique_requests,
    CollectiveMemoryRequests* absl_nullable collective_memory_requests,
    BarrierRequests* absl_nullable barrier_requests,
    const CollectiveCliques* absl_nullable collective_cliques,
    const CollectiveMemory* absl_nullable collective_memory) {
  if (handler == nullptr) {
    return absl::InternalError("FFI execute handler is not set");
  }
  if (stage != XLA_FFI_ExecutionStage_PREPARE &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  TF_ASSIGN_OR_RETURN(auto call_frame, BuildCallFrame(buffer_allocations));
  InvokeContext context = BuildInvokeContext(
      run_id, stream, buffer_allocations, collective_params,
      collective_clique_requests, collective_memory_requests, barrier_requests,
      collective_cliques, collective_memory, execution_context);
  return Invoke(ffi::GetXlaFfiApi(), handler, *call_frame, context, stage);
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    RunId run_id, xla::ffi::Ffi& handler, xla::ffi::ExecutionStage stage,
    se::Stream* stream, const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations,
    const CollectiveParams* absl_nullable collective_params,
    CollectiveCliqueRequests* absl_nullable collective_clique_requests,
    CollectiveMemoryRequests* absl_nullable collective_memory_requests,
    BarrierRequests* absl_nullable barrier_requests,
    const CollectiveCliques* absl_nullable collective_cliques,
    const CollectiveMemory* absl_nullable collective_memory) {
  if (stage != xla::ffi::ExecutionStage::kPrepare &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  TF_ASSIGN_OR_RETURN(auto call_frame, BuildCallFrame(buffer_allocations));
  InvokeContext context = BuildInvokeContext(
      run_id, stream, buffer_allocations, collective_params,
      collective_clique_requests, collective_memory_requests, barrier_requests,
      collective_cliques, collective_memory, execution_context);
  return Invoke(ffi::GetXlaFfiApi(), handler, *call_frame, context, stage);
}

absl::Status CustomCallThunk::Prepare(const PrepareParams& params) {
  if (bundle_.has_value()) {
    const RunId run_id =
        params.collective_params ? params.collective_params->run_id : RunId{-1};

    if (const auto* c_bundle =
            std::get_if<XLA_FFI_Handler_Bundle>(&bundle_.value());
        c_bundle && c_bundle->prepare) {
      return ExecuteFfiHandler(
          run_id, c_bundle->prepare, XLA_FFI_ExecutionStage_PREPARE,
          /*stream=*/nullptr,
          /*execution_context=*/nullptr,
          /*buffer_allocations=*/params.buffer_allocations,
          /*collective_params=*/params.collective_params,
          /*collective_clique_requests=*/params.collective_clique_requests,
          /*collective_memory_requests=*/params.collective_memory_requests,
          /*barrier_requests=*/params.barrier_requests,
          /*collective_cliques=*/nullptr,
          /*collective_memory=*/nullptr);
    }
    if (const auto* owned_bundle =
            std::get_if<OwnedHandlerBundle>(&bundle_.value());
        owned_bundle && owned_bundle->prepare) {
      return ExecuteFfiHandler(
          run_id, *owned_bundle->prepare, xla::ffi::ExecutionStage::kPrepare,
          /*stream=*/nullptr,
          /*execution_context=*/nullptr,
          /*buffer_allocations=*/params.buffer_allocations,
          /*collective_params=*/params.collective_params,
          /*collective_clique_requests=*/params.collective_clique_requests,
          /*collective_memory_requests=*/params.collective_memory_requests,
          /*barrier_requests=*/params.barrier_requests,
          /*collective_cliques=*/nullptr,
          /*collective_memory=*/nullptr);
    }
  }

  return absl::OkStatus();
}

absl::Status CustomCallThunk::Initialize(const InitializeParams& params) {
  if (bundle_.has_value()) {
    const RunId run_id =
        params.collective_params ? params.collective_params->run_id : RunId{-1};

    if (const auto* c_bundle =
            std::get_if<XLA_FFI_Handler_Bundle>(&bundle_.value());
        c_bundle && c_bundle->initialize) {
      return ExecuteFfiHandler(
          run_id, *c_bundle->initialize, XLA_FFI_ExecutionStage_INITIALIZE,
          params.stream, params.ffi_execution_context,
          params.buffer_allocations, params.collective_params,
          /*collective_clique_requests=*/nullptr,
          /*collective_memory_requests=*/nullptr,
          /*barrier_requests=*/nullptr, params.collective_cliques,
          params.collective_memory);
    }
    if (const auto* owned_bundle =
            std::get_if<OwnedHandlerBundle>(&bundle_.value());
        owned_bundle && owned_bundle->initialize) {
      return ExecuteFfiHandler(
          run_id, *owned_bundle->initialize,
          xla::ffi::ExecutionStage::kInitialize, params.stream,
          params.ffi_execution_context, params.buffer_allocations,
          params.collective_params, /*collective_clique_requests=*/nullptr,
          /*collective_memory_requests=*/nullptr, /*barrier_requests=*/nullptr,
          params.collective_cliques, params.collective_memory);
    }
  }
  return absl::OkStatus();
}

absl::Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  if (bundle_.has_value()) {
    const RunId run_id =
        params.collective_params ? params.collective_params->run_id : RunId{-1};
    if (const auto* c_bundle =
            std::get_if<XLA_FFI_Handler_Bundle>(&bundle_.value());
        c_bundle) {
      return ExecuteFfiHandler(
          run_id, c_bundle->execute, XLA_FFI_ExecutionStage_EXECUTE, stream,
          params.ffi_execution_context, params.buffer_allocations,
          params.collective_params, /*collective_clique_requests=*/nullptr,
          /*collective_memory_requests=*/nullptr, /*barrier_requests=*/nullptr,
          params.collective_cliques, params.collective_memory);
    }
    if (const auto* owned_bundle =
            std::get_if<OwnedHandlerBundle>(&bundle_.value());
        owned_bundle) {
      if (!owned_bundle->execute) {
        return absl::InternalError("FFI execute handler is not set");
      }
      return ExecuteFfiHandler(
          run_id, *owned_bundle->execute, xla::ffi::ExecutionStage::kExecute,
          stream, params.ffi_execution_context, params.buffer_allocations,
          params.collective_params, /*collective_clique_requests=*/nullptr,
          /*collective_memory_requests=*/nullptr, /*barrier_requests=*/nullptr,
          params.collective_cliques, params.collective_memory);
    }
  }

  return ExecuteCustomCall(params);
}

absl::StatusOr<ThunkProto> CustomCallThunk::ToProto() const {
  if (!api_version_.has_value()) {
    return absl::FailedPreconditionError(
        "CustomCallThunk was created from a non-registered target and cannot "
        "be serialized to a proto");
  }

  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_custom_call_thunk()->set_target_name(target_name_);
  proto.mutable_custom_call_thunk()->set_opaque(opaque_);
  proto.mutable_custom_call_thunk()->set_api_version(api_version_.value());
  if (called_computation_ != nullptr) {
    proto.mutable_custom_call_thunk()->set_called_computation(
        called_computation_->name());
  }

  for (const NullableShapedSlice& operand : operands_) {
    TF_ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_operands(),
                        operand.ToProto());
  }

  for (const NullableShapedSlice& result : results_) {
    TF_ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_results(),
                        result.ToProto());
  }

  if (attributes_.has_value()) {
    *proto.mutable_custom_call_thunk()->mutable_attributes() =
        attributes_->ToProto();
  }

  if (execution_state_ && execution_state_->IsSerializable()) {
    TF_ASSIGN_OR_RETURN(
        *proto.mutable_custom_call_thunk()->mutable_execution_state(),
        execution_state_->ToProto());
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::FromProto(
    ThunkInfo thunk_info, const CustomCallThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const HloModule* absl_nullable hlo_module, absl::string_view platform_name,
    const se::GpuComputeCapability& gpu_compute_capability) {
  if (hlo_module == nullptr && proto.has_called_computation()) {
    return absl::InvalidArgumentError(
        "HloModule is required to deserialize a CustomCallThunk with a "
        "called computation");
  }

  std::vector<NullableShapedSlice> operands, results;
  for (const auto& operand_proto : proto.operands()) {
    TF_ASSIGN_OR_RETURN(
        NullableShapedSlice operand,
        NullableShapedSlice::FromProto(operand_proto, buffer_allocations));
    operands.push_back(std::move(operand));
  }
  for (const auto& result_proto : proto.results()) {
    TF_ASSIGN_OR_RETURN(
        NullableShapedSlice result,
        NullableShapedSlice::FromProto(result_proto, buffer_allocations));
    results.push_back(std::move(result));
  }

  if (proto.api_version() != CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    // Create a thunk that uses the legacy custom call registry.
    return CustomCallThunk::Create(
        std::move(thunk_info), proto.target_name(), std::move(operands),
        std::move(results), proto.opaque(), proto.api_version(), platform_name);
  }

  TF_ASSIGN_OR_RETURN(ffi::AttributesMap attributes,
                      ffi::AttributesMap::FromProto(proto.attributes()));

  HloComputation* called_computation = nullptr;
  if (proto.has_called_computation()) {
    CHECK(hlo_module != nullptr);  // This check is needed for static analysis.
    called_computation =
        hlo_module->GetComputationWithName(proto.called_computation());
    if (called_computation == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "HloComputation '", proto.called_computation(),
          "' not found in the HloModule with name '", hlo_module->name(), "'"));
    }
  }
  std::unique_ptr<ffi::ExecutionState> execution_state;
  if (proto.has_execution_state()) {
    TF_ASSIGN_OR_RETURN(
        auto state, ffi::ExecutionState::FromProto(proto.execution_state()));
    execution_state = std::make_unique<ffi::ExecutionState>(std::move(state));
  }

  return CustomCallThunk::Create(
      std::move(thunk_info), proto.target_name(), std::move(operands),
      std::move(results), std::move(attributes), called_computation,
      platform_name, gpu_compute_capability, std::move(execution_state));
}

}  // namespace gpu
}  // namespace xla
