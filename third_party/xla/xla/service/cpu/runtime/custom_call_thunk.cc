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

#include "xla/service/cpu/runtime/custom_call_thunk.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    Info info, absl::string_view target_name, OpBuffers op_buffers,
    absl::string_view backend_config, CustomCallApiVersion api_version) {
  return absl::WrapUnique(
      new CustomCallThunk(std::move(info), target_name, std::move(op_buffers),
                          std::move(backend_config), api_version));
}

CustomCallThunk::CustomCallThunk(Info info, absl::string_view target_name,
                                 OpBuffers op_buffers,
                                 absl::string_view backend_config,
                                 CustomCallApiVersion api_version)
    : Thunk(Kind::kCustomCall, std::move(info)),
      target_name_(target_name),
      op_buffers_(std::move(op_buffers)),
      backend_config_(std::move(backend_config)),
      api_version_(api_version) {}

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
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  // Find the registered FFI handler for this target.
  auto handler = ffi::FindHandler(target_name_, "Host");
  if (!handler.ok()) {
    // Overwrite the returned error code (kNotFound) to kInternal to match the
    // original CPU implementation.
    // TODO(penporn): Change this to kUnimplemented to match the GPU backend
    // when thunks is the only runtime for CPU.
    return Internal(
        "No registered implementation for FFI custom call to %s for Host",
        target_name_);
  }

  // Build the FFI call frame.
  ffi::CallFrameBuilder builder;

  // Add input buffers.
  for (int i = 0; i < op_buffers_.arguments_buffers.size(); ++i) {
    auto& slice = op_buffers_.arguments_buffers[i];
    auto& shape = op_buffers_.arguments_shapes[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase arg,
                        params.buffer_allocations->GetDeviceAddress(slice));
    builder.AddBufferArg(arg, shape.element_type(), shape.dimensions());
    VLOG(3) << absl::StreamFormat("  arg: %s in slice %s (%p)",
                                  shape.ToString(true), slice.ToString(),
                                  arg.opaque());
  }

  // Add output buffers.
  for (int i = 0; i < op_buffers_.results_buffers.size(); ++i) {
    auto& slice = op_buffers_.results_buffers[i];
    auto& shape = op_buffers_.results_shapes[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase res,
                        params.buffer_allocations->GetDeviceAddress(slice));
    builder.AddBufferRet(res, shape.element_type(), shape.dimensions());
    VLOG(3) << absl::StreamFormat("  res: %s in slice %s (%p)",
                                  shape.ToString(true), slice.ToString(),
                                  res.opaque());
  }

  // Add attributes.
  if (!backend_config_.empty()) {
    // Parse backend config into an MLIR dictionary.
    mlir::MLIRContext mlir_context;
    ffi::CallFrameBuilder::FlatAttributesMap attributes;
    mlir::Attribute attr = mlir::parseAttribute(backend_config_, &mlir_context);
    if (auto dict = attr.dyn_cast_or_null<mlir::DictionaryAttr>()) {
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    } else {
      return Internal(
          "Unsupported backend config. Expected a string parsable into "
          "dictionary attribute");
    }
    // Convert the MLIR dictionary to FFI attributes.
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));
    builder.AddAttributes(attrs.Build());
    VLOG(3) << absl::StreamFormat("  attributes: %s", backend_config_);
  }

  // Forward ExecutableRunOptions to the FFI handlers via the call options.
  CustomCallExecuteParams* custom_call_params = params.custom_call_params;
  ffi::CallOptions call_options = {custom_call_params->device_ordinal,
                                   custom_call_params->stream,
                                   custom_call_params->allocator,
                                   /*called_computation=*/nullptr,
                                   custom_call_params->ffi_execution_context};

  // Call the function and check execution status.
  ffi::CallFrame call_frame = builder.Build();
  auto status = ffi::Call(handler->bundle.execute, call_frame, call_options);
  if (!status.ok()) {
    // Overwrite the returned error code to kInternal to match the original CPU
    // implementation.
    // TODO(penporn): Use TF_RETURN_IF_ERROR when thunks is the only runtime.
    return Internal("%s", status.message());
  }
  return OkExecuteEvent();
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CustomCallThunk::CallUntypedAPI(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  // Find the corresponding call target.
  void* call_target =
      CustomCallTargetRegistry::Global()->Lookup(target_name_, "Host");
  if (!call_target) {
    // Use kInternal to match the original CPU implementation.
    // TODO(penporn): Change this to kUnimplemented to match the GPU backend
    // when thunks is the only runtime for CPU.
    return Internal(
        "No registered implementation for untyped custom call to %s for Host",
        target_name_);
  }

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
  void* out_ptr = results.size() == 1 ? results[0] : results.data();

  // Set up the correct function type for each API version.
  CustomCallTarget custom_call_target;
  switch (api_version_) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
      using v1_signature = void (*)(void* /*out*/, const void** /*in*/);
      custom_call_target = [call_target](void* out, const void** in,
                                         const char* opaque, size_t opaque_len,
                                         XlaCustomCallStatus* status) {
        auto fn = reinterpret_cast<v1_signature>(call_target);
        fn(out, in);
      };
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      using v2_signature = void (*)(void* /*out*/, const void** /*in*/,
                                    XlaCustomCallStatus* /*status*/);
      custom_call_target = [call_target](void* out, const void** in,
                                         const char* opaque, size_t opaque_len,
                                         XlaCustomCallStatus* status) {
        auto fn = reinterpret_cast<v2_signature>(call_target);
        fn(out, in, status);
      };
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      using v3_signature =
          void (*)(void* /*out*/, const void** /*in*/, const char* /*opaque*/,
                   size_t /*opaque_len*/, XlaCustomCallStatus* /*status*/);
      custom_call_target = reinterpret_cast<v3_signature>(call_target);
      break;
    default:
      return InvalidArgument(
          "Unknown custom-call API version enum value: %d (%s)", api_version_,
          CustomCallApiVersion_Name(api_version_));
  }

  // Call the function and check execution status.
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
