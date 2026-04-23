/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/legacy_custom_call_thunk.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/custom_call_target.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/platform.h"

namespace xla::gpu {

static absl::StatusOr<LegacyCustomCallThunk::CustomCallTarget>
ResolveLegacyCustomCall(const CustomCallTargetRegistry& registry,
                        absl::string_view target_name,
                        absl::string_view platform_name,
                        CustomCallApiVersion api_version) {
  void* call_target =
      registry.Lookup(std::string(target_name), std::string(platform_name));

  if (call_target == nullptr) {
    return NotFound(
        "No registered implementation for custom call to %s for platform %s",
        target_name, platform_name);
  }

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

absl::StatusOr<std::unique_ptr<LegacyCustomCallThunk>>
LegacyCustomCallThunk::Create(ThunkInfo thunk_info, std::string target_name,
                              CustomCallTarget call_target,
                              std::vector<NullableShapedSlice> operands,
                              std::vector<NullableShapedSlice> results,
                              std::string opaque) {
  return absl::WrapUnique(new LegacyCustomCallThunk(
      thunk_info, std::move(target_name), std::move(operands),
      std::move(results), std::move(opaque), std::move(call_target),
      /*api_version=*/std::nullopt));
}

absl::StatusOr<std::unique_ptr<LegacyCustomCallThunk>>
LegacyCustomCallThunk::Create(ThunkInfo thunk_info, std::string target_name,
                              std::vector<NullableShapedSlice> operands,
                              std::vector<NullableShapedSlice> results,
                              std::string opaque,
                              CustomCallApiVersion api_version,
                              absl::string_view platform_name) {
  ASSIGN_OR_RETURN(
      CustomCallTarget call_target,
      ResolveLegacyCustomCall(*CustomCallTargetRegistry::Global(), target_name,
                              platform_name, api_version));

  return absl::WrapUnique(new LegacyCustomCallThunk(
      thunk_info, std::move(target_name), std::move(operands),
      std::move(results), std::move(opaque), call_target, api_version));
}

LegacyCustomCallThunk::LegacyCustomCallThunk(
    ThunkInfo thunk_info, std::string target_name,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, std::string opaque,
    CustomCallTarget call_target,
    const std::optional<CustomCallApiVersion>& api_version)
    : TracedCommand(CommandType::kCustomCallCmd, Thunk::kCustomCall,
                    thunk_info),
      api_version_(api_version),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      call_target_(std::move(call_target)),
      opaque_(std::move(opaque)) {}

absl::Status LegacyCustomCallThunk::ExecuteOnStream(
    const ExecuteParams& params) {
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

  XlaCustomCallStatus custom_call_status;
  call_target_(params.stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return Internal("CustomCall failed: %s", *message);
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> LegacyCustomCallThunk::ToProto() const {
  if (!api_version_.has_value()) {
    return absl::FailedPreconditionError(
        "LegacyCustomCallThunk was created from a non-registered target and "
        "cannot be serialized to a proto");
  }

  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_custom_call_thunk()->set_target_name(target_name_);
  proto.mutable_custom_call_thunk()->set_opaque(opaque_);
  proto.mutable_custom_call_thunk()->set_api_version(api_version_.value());

  for (const NullableShapedSlice& operand : operands_) {
    ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_operands(),
                     operand.ToProto());
  }

  for (const NullableShapedSlice& result : results_) {
    ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_results(),
                     result.ToProto());
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<LegacyCustomCallThunk>>
LegacyCustomCallThunk::FromProto(
    ThunkInfo thunk_info, const CustomCallThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    absl::string_view platform_name) {
  std::vector<NullableShapedSlice> operands, results;
  for (const auto& operand_proto : proto.operands()) {
    ASSIGN_OR_RETURN(
        NullableShapedSlice operand,
        NullableShapedSlice::FromProto(operand_proto, buffer_allocations));
    operands.push_back(std::move(operand));
  }
  for (const auto& result_proto : proto.results()) {
    ASSIGN_OR_RETURN(
        NullableShapedSlice result,
        NullableShapedSlice::FromProto(result_proto, buffer_allocations));
    results.push_back(std::move(result));
  }

  return LegacyCustomCallThunk::Create(
      std::move(thunk_info), proto.target_name(), std::move(operands),
      std::move(results), proto.opaque(), proto.api_version(), platform_name);
}

}  // namespace xla::gpu
