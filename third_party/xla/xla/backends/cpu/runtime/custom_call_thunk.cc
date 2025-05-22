/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/custom_call_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

using AttributesMap = ffi::CallFrameBuilder::AttributesMap;

static absl::StatusOr<AttributesMap> ParseAttributes(
    absl::string_view backend_config) {
  AttributesMap attributes;
  if (!backend_config.empty() && backend_config != "{}") {
    // Parse backend config into an MLIR dictionary.
    mlir::MLIRContext mlir_context;
    mlir::Attribute attr = mlir::parseAttribute(backend_config, &mlir_context);
    if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
      // Convert the MLIR dictionary to FFI attributes.
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    } else {
      return Internal(
          "Unsupported backend config. Expected a string parsable into "
          "dictionary attribute");
    }

    VLOG(3) << absl::StreamFormat("  attributes: %s", backend_config);
  }

  return attributes;
}

// Call `instantiate` callback if passed. This function needs its own copy of
// attributes, that's what AttributesBuilder expects, there's no way around it.
static absl::Status InstantiateHandlerState(
    ffi::HandlerRegistration& handler, ffi::ExecutionState* execution_state,
    AttributesMap attributes) {
  // Initialize FFI handler state if it has an instantiate callback.
  if (handler.bundle.instantiate) {
    // At FFI handler instantiation time, we don't have any arguments or results
    ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);

    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));

    builder.AddAttributes(attrs.Build());
    ffi::CallFrame instantiate_call_frame = builder.Build();

    ffi::CallOptions options;
    options.execution_state = execution_state;
    TF_RETURN_IF_ERROR(Call(handler.bundle.instantiate, instantiate_call_frame,
                            options, XLA_FFI_ExecutionStage_INSTANTIATE));
  }

  return absl::OkStatus();
}

// Builds a call frame prototype for typed-FFI custom calls with dummy device
// memory addresses. This is called once when creating the CustomCall thunk,
// then the thunk will need to update the addresses at runtime.
static absl::StatusOr<ffi::CallFrame> BuildCallFrameForTypedFFI(
    const CustomCallApiVersion version,
    const CustomCallThunk::OpBuffers& op_buffers,
    const absl::string_view backend_config, AttributesMap attributes) {
  ffi::CallFrameBuilder builder(
      /*num_args=*/op_buffers.arguments_buffers.size(),
      /*num_rets=*/op_buffers.results_buffers.size());

  // Add prototype input buffers with actual data types and shapes. Device
  // memory addresses will be updated at runtime.
  for (int i = 0; i < op_buffers.arguments_buffers.size(); ++i) {
    auto& shape = op_buffers.arguments_shapes[i];

    if (shape.IsToken()) {
      builder.AddTokenArg();
      continue;
    }

    auto elements = absl::c_accumulate(shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(shape.element_type());
    se::DeviceMemoryBase placeholder_arg(nullptr, elements * dtype_bytes);
    builder.AddBufferArg(placeholder_arg, shape.element_type(),
                         shape.dimensions());
  }

  // Add prototype output buffers with actual data types and shapes. Device
  // memory addresses will be updated at runtime.
  for (int i = 0; i < op_buffers.results_buffers.size(); ++i) {
    auto& shape = op_buffers.results_shapes[i];

    if (shape.IsToken()) {
      builder.AddTokenRet();
      continue;
    }

    auto elements = absl::c_accumulate(shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(shape.element_type());
    se::DeviceMemoryBase placeholder_ret(nullptr, elements * dtype_bytes);
    builder.AddBufferRet(placeholder_ret, shape.element_type(),
                         shape.dimensions());
  }

  // Add attributes if any.
  if (!attributes.empty()) {
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));
    builder.AddAttributes(attrs.Build());
  }

  return builder.Build();
}

static absl::StatusOr<CustomCallThunk::CustomCallTarget> ToCustomCallTarget(
    CustomCallApiVersion api_version, absl::string_view target_name,
    void* target) {
  if (target == nullptr) {
    return NotFound(
        "No registered implementation for untyped custom call to %s for Host",
        target_name);
  }

  switch (api_version) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
#ifdef PLATFORM_GOOGLE
      LOG(FATAL)
#else
      LOG(ERROR)
#endif
          << "Custom call API version `API_VERSION_ORIGINAL` is not supported "
             "by XLA:CPU. Prefer https://docs.jax.dev/en/latest/ffi.html. It "
             "will be fully removed in November 2025.";

      using v1_signature = void (*)(void* /*out*/, const void** /*in*/);
      return [target](void* out, const void** in, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
        auto fn = reinterpret_cast<v1_signature>(target);
        fn(out, in);
      };
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      using v2_signature = void (*)(void* /*out*/, const void** /*in*/,
                                    XlaCustomCallStatus* /*status*/);
      return [target](void* out, const void** in, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
        auto fn = reinterpret_cast<v2_signature>(target);
        fn(out, in, status);
      };
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      using v3_signature =
          void (*)(void* /*out*/, const void** /*in*/, const char* /*opaque*/,
                   size_t /*opaque_len*/, XlaCustomCallStatus* /*status*/);
      return reinterpret_cast<v3_signature>(target);
    default:
      return InvalidArgument(
          "Unknown custom-call API version enum value: %d (%s)", api_version,
          CustomCallApiVersion_Name(api_version));
  }
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    Info info, absl::string_view target_name, OpBuffers op_buffers,
    absl::string_view backend_config, CustomCallApiVersion api_version) {
  std::optional<ffi::CallFrame> call_frame;
  auto execution_state = std::make_unique<ffi::ExecutionState>();

  // Resolve custom call target name to the target callable.
  std::variant<CustomCallTarget, ffi::HandlerRegistration> target;

  if (api_version == CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    TF_ASSIGN_OR_RETURN(target, ffi::FindHandler(target_name, "Host"));

    TF_ASSIGN_OR_RETURN(AttributesMap attributes,
                        ParseAttributes(backend_config));

    TF_RETURN_IF_ERROR(InstantiateHandlerState(
        std::get<1>(target), execution_state.get(), attributes));

    TF_ASSIGN_OR_RETURN(call_frame, BuildCallFrameForTypedFFI(
                                        api_version, op_buffers, backend_config,
                                        std::move(attributes)));
  } else {
    auto* registry = CustomCallTargetRegistry::Global();
    TF_ASSIGN_OR_RETURN(
        target,
        ToCustomCallTarget(api_version, target_name,
                           registry->Lookup(std::string(target_name), "Host")));
  }

  return absl::WrapUnique(new CustomCallThunk(
      std::move(info), target_name, std::move(target), std::move(op_buffers),
      api_version, std::move(backend_config), std::move(call_frame),
      std::move(execution_state)));
}

CustomCallThunk::CustomCallThunk(
    Info info, absl::string_view target_name,
    std::variant<CustomCallTarget, ffi::HandlerRegistration> target,
    OpBuffers op_buffers, CustomCallApiVersion api_version,
    absl::string_view backend_config, std::optional<ffi::CallFrame> call_frame,
    std::unique_ptr<ffi::ExecutionState> execution_state)
    : Thunk(Kind::kCustomCall, std::move(info)),
      target_name_(target_name),
      target_(std::move(target)),
      op_buffers_(std::move(op_buffers)),
      api_version_(api_version),
      backend_config_(std::move(backend_config)),
      call_frame_(std::move(call_frame)),
      call_frames_([this] { return call_frame_->Copy(); }),
      execution_state_(std::move(execution_state)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CustomCallThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat(
      "CustomCall: %s, #arguments=%d, #results=%d", target_name_,
      op_buffers_.arguments_buffers.size(), op_buffers_.results_buffers.size());
  if (api_version_ == CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    return CallTypedFFI(params);
  }
  return CallUntypedAPI(params);
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CustomCallThunk::CallTypedFFI(
    const ExecuteParams& params) {
  if (params.custom_call_params == nullptr) {
    return Internal("CustomCallExecuteParams cannot be nullptr.");
  }

  // Collect argument buffers.
  absl::InlinedVector<se::DeviceMemoryBase, 8> arguments;
  arguments.reserve(op_buffers_.arguments_buffers.size());
  for (int i = 0; i < op_buffers_.arguments_buffers.size(); ++i) {
    BufferAllocation::Slice& slice = op_buffers_.arguments_buffers[i];
    TF_ASSIGN_OR_RETURN(arguments.emplace_back(),
                        params.buffer_allocations->GetDeviceAddress(slice));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(arguments[i].opaque(),
                                        arguments[i].size());
    VLOG(3) << absl::StreamFormat(
        "  arg: %s in slice %s (%p)",
        op_buffers_.arguments_shapes[i].ToString(true), slice.ToString(),
        arguments[i].opaque());
  }

  // Collect results buffers.
  absl::InlinedVector<se::DeviceMemoryBase, 4> results;
  results.reserve(op_buffers_.results_buffers.size());
  for (int i = 0; i < op_buffers_.results_buffers.size(); ++i) {
    BufferAllocation::Slice& slice = op_buffers_.results_buffers[i];
    TF_ASSIGN_OR_RETURN(results.emplace_back(),
                        params.buffer_allocations->GetDeviceAddress(slice));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(results[i].opaque(), results[i].size());
    VLOG(3) << absl::StreamFormat("  res: %s in slice %s (%p)",
                                  op_buffers_.results_shapes[i].ToString(true),
                                  slice.ToString(), results[i].opaque());
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_.GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));

  // Forward ExecutableRunOptions to the FFI handlers via the call options.
  CustomCallExecuteParams* custom_call_params = params.custom_call_params;
  ffi::CallOptions call_options = {
      custom_call_params->run_id,
      custom_call_params->device_ordinal,
      ffi::CallOptions::CpuOptions{custom_call_params->intra_op_thread_pool},
      /*called_computation=*/nullptr,
      custom_call_params->ffi_execution_context,
      execution_state_.get()};

  ffi::HandlerRegistration& handler = std::get<1>(target_);
  return ffi::CallAsync(handler.bundle.execute, *call_frame, call_options);
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CustomCallThunk::CallUntypedAPI(
    const ExecuteParams& params) {
  // Collect raw input pointers in an array.
  absl::InlinedVector<const void*, 8> arguments;
  arguments.reserve(op_buffers_.arguments_buffers.size());
  for (int i = 0; i < op_buffers_.arguments_buffers.size(); ++i) {
    auto& slice = op_buffers_.arguments_buffers[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase arg,
                        params.buffer_allocations->GetDeviceAddress(slice));
    arguments.push_back(arg.opaque());
    VLOG(3) << absl::StreamFormat(
        "  arg: %s in slice %s (%p)",
        op_buffers_.arguments_shapes[i].ToString(true), slice.ToString(),
        arg.opaque());
  }
  const void** in_ptrs = arguments.data();

  // Collect raw output pointers in another array.
  absl::InlinedVector<void*, 4> results;
  results.reserve(op_buffers_.results_buffers.size());
  for (int i = 0; i < op_buffers_.results_buffers.size(); ++i) {
    auto& slice = op_buffers_.results_buffers[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase res,
                        params.buffer_allocations->GetDeviceAddress(slice));
    results.push_back(res.opaque());
    VLOG(3) << absl::StreamFormat("  res: %s in slice %s (%p)",
                                  op_buffers_.results_shapes[i].ToString(true),
                                  slice.ToString(), res.opaque());
  }

  void* out_ptr = op_buffers_.is_tuple_result ? results.data() : results[0];

  // Call the function and check execution status.
  CustomCallTarget custom_call_target = std::get<0>(target_);

  XlaCustomCallStatus status;
  custom_call_target(out_ptr, in_ptrs, backend_config_.c_str(),
                     backend_config_.size(), &status);

  auto status_message = xla::CustomCallStatusGetMessage(&status);
  if (status_message.has_value()) {
    return Internal("%s", status_message.value());
  }
  return OkExecuteEvent();
}

CustomCallThunk::BufferUses CustomCallThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& argument : op_buffers_.arguments_buffers) {
    buffer_uses.emplace_back(argument, BufferUse::kRead);
  }
  for (const auto& result : op_buffers_.results_buffers) {
    buffer_uses.emplace_back(result, BufferUse::kWrite);
  }
  return buffer_uses;
}

}  // namespace xla::cpu
