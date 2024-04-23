/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/ffi/ffi_api.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/call_frame.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"

//===----------------------------------------------------------------------===//
// XLA FFI C structs definition
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error {
  absl::Status status;
};

struct XLA_FFI_ExecutionContext {
  const xla::ServiceExecutableRunOptions* run_options;
  const xla::HloComputation* called_computation;
};

//===----------------------------------------------------------------------===//

namespace xla::ffi {

bool IsCommandBufferCompatible(XLA_FFI_Handler_Traits traits) {
  return traits & XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
}

//===----------------------------------------------------------------------===//
// Calling XLA FFI handlers
//===----------------------------------------------------------------------===//

// WARNING: These functions defined in `call_frame.h` as we need to make them
// available without having to depend on `ffi.h` header.

Status TakeStatus(XLA_FFI_Error* error) {
  if (error == nullptr) return absl::OkStatus();
  Status status = std::move(error->status);
  delete error;
  return status;
}

Status Call(Ffi& handler, CallFrame& call_frame, const CallOptions& options) {
  XLA_FFI_ExecutionContext ctx = {options.run_options,
                                  options.called_computation};
  XLA_FFI_CallFrame ffi_call_frame = call_frame.Build(GetXlaFfiApi(), &ctx);
  return TakeStatus(handler.Call(&ffi_call_frame));
}

Status Call(XLA_FFI_Handler* handler, CallFrame& call_frame,
            const CallOptions& options) {
  XLA_FFI_ExecutionContext ctx = {options.run_options,
                                  options.called_computation};
  XLA_FFI_CallFrame ffi_call_frame = call_frame.Build(GetXlaFfiApi(), &ctx);
  return TakeStatus((*handler)(&ffi_call_frame));
}

//===----------------------------------------------------------------------===//
// XLA FFI registry
//===----------------------------------------------------------------------===//

using HandlerKey = std::pair<std::string, std::string>;
using HandlerRegistry = absl::flat_hash_map<HandlerKey, HandlerRegistration>;

static HandlerKey MakeHandlerKey(std::string_view name,
                                 std::string_view platform) {
  return std::make_pair(std::string(name), absl::AsciiStrToLower(platform));
}

static HandlerRegistry& GetHandlerRegistry() {
  static auto* registry = new HandlerRegistry();
  return *registry;
}

static Status RegisterHandler(std::string_view name, std::string_view platform,
                              XLA_FFI_Handler* handler,
                              XLA_FFI_Handler_Traits traits) {
  auto emplaced = GetHandlerRegistry().try_emplace(
      MakeHandlerKey(name, platform), HandlerRegistration{handler, traits});
  if (!emplaced.second)
    return absl::InvalidArgumentError(
        absl::StrCat("Duplicate FFI handler registration for ", name,
                     " on a platform ", platform));
  return OkStatus();
}

absl::StatusOr<HandlerRegistration> FindHandler(std::string_view name,
                                                std::string_view platform) {
  auto it = GetHandlerRegistry().find(MakeHandlerKey(name, platform));
  if (it == GetHandlerRegistry().end())
    return absl::NotFoundError(absl::StrCat("No FFI handler registered for ",
                                            name, " on a platform ", platform));
  return it->second;
}

absl::flat_hash_map<std::string, HandlerRegistration> StaticRegisteredHandlers(
    std::string_view platform) {
  absl::flat_hash_map<std::string, HandlerRegistration> calls;
  for (const auto& [metadata, handler] : GetHandlerRegistry()) {
    if (absl::AsciiStrToLower(platform) == metadata.second) {
      calls[metadata.first] = handler;
    }
  }

  return calls;
}

//===----------------------------------------------------------------------===//
// XLA FFI Api Implementation
//===----------------------------------------------------------------------===//

static std::string StructSizeErrorMsg(std::string_view struct_name,
                                      size_t expected, size_t actual) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ", expected,
                      ", got ", actual, ". Check installed software versions. ",
                      "The framework XLA FFI API version is ",
                      XLA_FFI_API_MAJOR, ".", XLA_FFI_API_MINOR, ".");
}

static Status ActualStructSizeIsGreaterOrEqual(std::string_view struct_name,
                                               size_t expected, size_t actual) {
  if (actual < expected) {
    return absl::InvalidArgumentError(
        StructSizeErrorMsg(struct_name, expected, actual));
  }
  if (actual > expected) {
    VLOG(2) << StructSizeErrorMsg(struct_name, expected, actual);
  }
  return absl::OkStatus();
}

static absl::StatusCode ToStatusCode(XLA_FFI_Error_Code errc) {
  switch (errc) {
    case XLA_FFI_Error_Code_OK:
      return absl::StatusCode::kOk;
    case XLA_FFI_Error_Code_CANCELLED:
      return absl::StatusCode::kCancelled;
    case XLA_FFI_Error_Code_UNKNOWN:
      return absl::StatusCode::kUnknown;
    case XLA_FFI_Error_Code_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case XLA_FFI_Error_Code_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case XLA_FFI_Error_Code_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case XLA_FFI_Error_Code_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case XLA_FFI_Error_Code_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case XLA_FFI_Error_Code_RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case XLA_FFI_Error_Code_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case XLA_FFI_Error_Code_ABORTED:
      return absl::StatusCode::kAborted;
    case XLA_FFI_Error_Code_OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case XLA_FFI_Error_Code_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case XLA_FFI_Error_Code_INTERNAL:
      return absl::StatusCode::kInternal;
    case XLA_FFI_Error_Code_UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case XLA_FFI_Error_Code_DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    case XLA_FFI_Error_Code_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
  }
}

#define XLA_FFI_RETURN_IF_ERROR(expr)                                   \
  do {                                                                  \
    Status _status = (expr);                                            \
    if (!_status.ok()) {                                                \
      XLA_FFI_Error* _c_status = new XLA_FFI_Error{std::move(_status)}; \
      return _c_status;                                                 \
    }                                                                   \
  } while (false)

static XLA_FFI_Error* XLA_FFI_Error_Create(XLA_FFI_Error_Create_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Error_Create", XLA_FFI_Error_Create_Args_STRUCT_SIZE,
      args->struct_size));

  return new XLA_FFI_Error{Status(ToStatusCode(args->errc), args->message)};
}

static void XLA_FFI_Error_GetMessage(XLA_FFI_Error_GetMessage_Args* args) {
  Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Error_GetMessage", XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  // absl::Status owns error message in a std::string which guarantees that
  // we'll get a null terminated string.
  args->message = args->error->status.message().data();
}

static void XLA_FFI_Error_Destroy(XLA_FFI_Error_Destroy_Args* args) {
  Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Error_Destroy", XLA_FFI_Error_Destroy_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  delete args->error;
}

static XLA_FFI_Error* XLA_FFI_Handler_Register(
    XLA_FFI_Handler_Register_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Handler_Register", XLA_FFI_Handler_Register_Args_STRUCT_SIZE,
      args->struct_size));

  if (auto status = RegisterHandler(args->name, args->platform, args->handler,
                                    args->traits);
      !status.ok()) {
    return new XLA_FFI_Error{std::move(status)};
  }
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_Stream_Get(XLA_FFI_Stream_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Stream_Get", XLA_FFI_Stream_Get_Args_STRUCT_SIZE,
      args->struct_size));

  auto handle = args->ctx->run_options->stream()->platform_specific_handle();
  args->stream = handle.stream;

  return nullptr;
}

//===----------------------------------------------------------------------===//
// XLA FFI Internal Api Implementation
//===----------------------------------------------------------------------===//

static XLA_FFI_Error* XLA_FFI_INTERNAL_Error_Forward(void* status) {
  return new XLA_FFI_Error{std::move(*reinterpret_cast<Status*>(status))};
}

static void* XLA_FFI_INTERNAL_Stream_Get(XLA_FFI_ExecutionContext* ctx) {
  return ctx->run_options->stream();
}

static int32_t XLA_FFI_INTERNAL_DeviceOrdinal_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return ctx->run_options->device_ordinal();
}

static void* XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return ctx->run_options->allocator();
}

static void* XLA_FFI_INTERNAL_CalledComputation_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<HloComputation*>(ctx->called_computation);
}

//===----------------------------------------------------------------------===//
// XLA FFI Api access
//===----------------------------------------------------------------------===//

extern "C" const XLA_FFI_Api* XLA_FFI_GetApi() { return GetXlaFfiApi(); }

static XLA_FFI_InternalApi internal_api = {
    XLA_FFI_INTERNAL_Error_Forward,
    XLA_FFI_INTERNAL_Stream_Get,
    XLA_FFI_INTERNAL_DeviceOrdinal_Get,
    XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get,
    XLA_FFI_INTERNAL_CalledComputation_Get,
};

static XLA_FFI_Api api = {
    XLA_FFI_Api_STRUCT_SIZE,
    /*priv=*/nullptr,

    &internal_api,

    XLA_FFI_Error_Create,      // creates error
    XLA_FFI_Error_GetMessage,  // get error message
    XLA_FFI_Error_Destroy,     // frees error
    XLA_FFI_Handler_Register,  // registers handler
    XLA_FFI_Stream_Get,        // returns platform specific stream
};

const XLA_FFI_Api* GetXlaFfiApi() { return &api; }

}  // namespace xla::ffi
