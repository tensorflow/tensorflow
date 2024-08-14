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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/type_id_registry.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

//===----------------------------------------------------------------------===//
// XLA FFI C structs definition
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error {
  absl::Status status;
};

struct XLA_FFI_ExecutionContext {
  struct CpuContext {
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  };

  struct GpuContext {
    stream_executor::Stream* stream = nullptr;
    stream_executor::DeviceMemoryAllocator* allocator = nullptr;
  };

  using BackendContext = std::variant<std::monostate, CpuContext, GpuContext>;

  int32_t device_ordinal = -1;
  BackendContext backend_context = {};

  const xla::HloComputation* called_computation = nullptr;
  const xla::ffi::ExecutionContext* execution_context = nullptr;
  xla::ffi::ExecutionState* execution_state = nullptr;
};

//===----------------------------------------------------------------------===//

namespace xla::ffi {

bool IsCommandBufferCompatible(XLA_FFI_Handler_Traits traits) {
  return traits & XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
}

static XLA_FFI_ExecutionContext CreateExecutionContext(
    const CallOptions& options) {
  using BackendContext = XLA_FFI_ExecutionContext::BackendContext;

  // Converts CallOptions to corresponding backend context.
  struct BackendVisitor {
    BackendContext operator()(const std::monostate&) const {
      return std::monostate{};
    }

    BackendContext operator()(const CallOptions::CpuOptions& options) const {
      return XLA_FFI_ExecutionContext::CpuContext{options.intra_op_thread_pool};
    }

    BackendContext operator()(const CallOptions::GpuOptions& options) const {
      return XLA_FFI_ExecutionContext::GpuContext{options.stream,
                                                  options.allocator};
    }
  };

  return XLA_FFI_ExecutionContext{
      options.device_ordinal,
      std::visit(BackendVisitor{}, options.backend_options),
      options.called_computation,
      internal::ScopedExecutionContext::GetCallExecutionContext(options),
      options.execution_state,
  };
}

//===----------------------------------------------------------------------===//
// Calling XLA FFI handlers
//===----------------------------------------------------------------------===//

absl::Status TakeStatus(XLA_FFI_Error* error) {
  if (error == nullptr) return absl::OkStatus();
  absl::Status status = std::move(error->status);
  delete error;
  return status;
}

absl::Status CallWithApi(const XLA_FFI_Api* api, Ffi& handler,
                         CallFrame& call_frame, const CallOptions& options,
                         ExecutionStage stage) {
  XLA_FFI_ExecutionContext ctx = CreateExecutionContext(options);
  XLA_FFI_CallFrame ffi_call_frame =
      call_frame.Build(api, &ctx, static_cast<XLA_FFI_ExecutionStage>(stage));
  XLA_FFI_Error* status = nullptr;
  try {
    status = handler.Call(&ffi_call_frame);
  } catch (std::exception& e) {
    return Unknown("XLA FFI call failed: %s", e.what());
  }
  return TakeStatus(status);
}

absl::Status Call(Ffi& handler, CallFrame& call_frame,
                  const CallOptions& options, ExecutionStage stage) {
  return CallWithApi(GetXlaFfiApi(), handler, call_frame, options, stage);
}

absl::Status Call(XLA_FFI_Handler* handler, CallFrame& call_frame,
                  const CallOptions& options, XLA_FFI_ExecutionStage stage) {
  XLA_FFI_ExecutionContext ctx = CreateExecutionContext(options);
  XLA_FFI_CallFrame ffi_call_frame =
      call_frame.Build(GetXlaFfiApi(), &ctx, stage);
  XLA_FFI_Error* status = nullptr;
  try {
    status = (*handler)(&ffi_call_frame);
  } catch (std::exception& e) {
    return Unknown("XLA FFI call failed: %s", e.what());
  }
  return TakeStatus(status);
}

namespace internal {
static thread_local const ExecutionContext* scoped_execution_context = nullptr;

ScopedExecutionContext::ScopedExecutionContext(const ExecutionContext* context)
    : recover_(scoped_execution_context) {
  scoped_execution_context = context;
}

ScopedExecutionContext::~ScopedExecutionContext() {
  scoped_execution_context = recover_;
}

const ExecutionContext* ScopedExecutionContext::GetCallExecutionContext(
    const CallOptions& options) {
  if (scoped_execution_context != nullptr) {
    return scoped_execution_context;
  }
  return options.execution_context;
}
}  // namespace internal

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

static std::vector<std::string> GetHandlerStages(
    const XLA_FFI_Handler_Bundle& bundle) {
  std::vector<std::string> stages;
  if (bundle.instantiate != nullptr) stages.push_back("instantiate");
  if (bundle.prepare != nullptr) stages.push_back("prepare");
  if (bundle.initialize != nullptr) stages.push_back("initialize");
  if (bundle.execute != nullptr) stages.push_back("execute");
  return stages;
}

static absl::Status RegisterHandler(std::string_view name,
                                    std::string_view platform,
                                    XLA_FFI_Handler_Bundle bundle,
                                    XLA_FFI_Handler_Traits traits) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  VLOG(2) << absl::StreamFormat(
      "Register XLA FFI handler for '%s'; platform=%s (canonical=%s), "
      "stages=[%s], command_buffer_compatible=%v",
      name, platform, canonical_platform,
      absl::StrJoin(GetHandlerStages(bundle), ", "),
      IsCommandBufferCompatible(traits));

  if (bundle.execute == nullptr) {
    return InvalidArgument(
        "FFI handler for %s on a platform %s must provide an execute "
        "implementation",
        name, platform);
  }

  auto emplaced =
      GetHandlerRegistry().try_emplace(MakeHandlerKey(name, canonical_platform),
                                       HandlerRegistration{bundle, traits});
  if (!emplaced.second) {
    auto existing = emplaced.first->second;
    if (existing.traits != traits) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on a platform %s "
          "(canonical %s) with different traits",
          name, platform, canonical_platform);
    }
    if (existing.bundle.prepare != bundle.prepare ||
        existing.bundle.initialize != bundle.initialize ||
        existing.bundle.execute != bundle.execute) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on a platform %s "
          "(canonical %s) with different bundle addresses",
          name, platform, canonical_platform);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<HandlerRegistration> FindHandler(std::string_view name,
                                                std::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  auto it = GetHandlerRegistry().find(MakeHandlerKey(name, canonical_platform));
  if (it == GetHandlerRegistry().end()) {
    return NotFound(
        "No FFI handler registered for %s on a platform %s (canonical %s)",
        name, platform, canonical_platform);
  }
  return it->second;
}

absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(std::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  absl::flat_hash_map<std::string, HandlerRegistration> calls;
  for (const auto& [metadata, handler] : GetHandlerRegistry()) {
    if (canonical_platform == metadata.second) {
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

static absl::Status ActualStructSizeIsGreaterOrEqual(
    std::string_view struct_name, size_t expected, size_t actual) {
  if (actual < expected) {
    return InvalidArgument("%s",
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
    absl::Status _status = (expr);                                      \
    if (!_status.ok()) {                                                \
      XLA_FFI_Error* _c_status = new XLA_FFI_Error{std::move(_status)}; \
      return _c_status;                                                 \
    }                                                                   \
  } while (false)

static XLA_FFI_Error* XLA_FFI_Error_Create(XLA_FFI_Error_Create_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Error_Create", XLA_FFI_Error_Create_Args_STRUCT_SIZE,
      args->struct_size));

  return new XLA_FFI_Error{
      absl::Status(ToStatusCode(args->errc), args->message)};
}

static void XLA_FFI_Error_GetMessage(XLA_FFI_Error_GetMessage_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
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
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
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

  if (auto status = RegisterHandler(
          std::string_view(args->name.ptr, args->name.len),
          std::string_view(args->platform.ptr, args->platform.len),
          args->bundle, args->traits);
      !status.ok()) {
    return new XLA_FFI_Error{std::move(status)};
  }
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_Stream_Get(XLA_FFI_Stream_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Stream_Get", XLA_FFI_Stream_Get_Args_STRUCT_SIZE,
      args->struct_size));

  auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new XLA_FFI_Error{
        Unimplemented("XLA FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->stream == nullptr)) {
    return new XLA_FFI_Error{
        Unimplemented("XLA FFI GPU stream is not available")};
  }

  auto handle = gpu->stream->platform_specific_handle();
  args->stream = handle.stream;

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_TypeId_Register(
    XLA_FFI_TypeId_Register_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_ExecutionContext_Get_Args",
      XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  auto type_id = TypeIdRegistry::RegisterExternalTypeId(
      std::string_view(args->name.ptr, args->name.len));
  if (!type_id.ok()) {
    return new XLA_FFI_Error{std::move(type_id).status()};
  }

  args->type_id->type_id = type_id->value();
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_ExecutionContext_Get(
    XLA_FFI_ExecutionContext_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_ExecutionContext_Get_Args",
      XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  DCHECK(args->ctx->execution_context) << "ExecutionContext must be set";
  auto user_data = args->ctx->execution_context->Lookup(
      TypeIdRegistry::TypeId(args->type_id->type_id));
  if (!user_data.ok()) {
    return new XLA_FFI_Error{std::move(user_data).status()};
  }

  args->data = *user_data;
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_State_Set(XLA_FFI_State_Set_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_State_Set_Args", XLA_FFI_State_Set_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";
  absl::Status status = args->ctx->execution_state->Set(
      TypeIdRegistry::TypeId(args->type_id->type_id), args->state,
      [deleter = args->deleter](void* state) { deleter(state); });

  if (!status.ok()) {
    return new XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_State_Get(XLA_FFI_State_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_State_Get_Args", XLA_FFI_State_Get_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";
  absl::StatusOr<void*> state = args->ctx->execution_state->Get(
      TypeIdRegistry::TypeId(args->type_id->type_id));
  if (!state.ok()) {
    return new XLA_FFI_Error{std::move(state).status()};
  }

  args->state = *state;
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_DeviceMemory_Allocate(
    XLA_FFI_DeviceMemory_Allocate_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_DeviceMemory_Allocate_Args",
      XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE, args->struct_size));

  auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  // TODO(ezhulenev): Device memory allocation should be supported for all
  // backends, not just GPU, although for CPU it doesn't make much sense, as
  // plain `new` is sufficient.
  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new XLA_FFI_Error{
        InvalidArgument("XLA FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->allocator == nullptr)) {
    return new XLA_FFI_Error{
        Unimplemented("No device memory allocator available on this platform")};
  }

  // TODO(ezhulenev): We happen to have the same alignment requirement for
  // device memory on CPU and GPU backends, but instead of hardcoding it here
  // we should query it for the platform XLA FFI handler is registered with.
  static constexpr int64_t kMaxAlignment = 16;

  if (!absl::has_single_bit(args->alignment) ||
      args->alignment > kMaxAlignment) {
    return new XLA_FFI_Error{
        InvalidArgument("Unsupported alignment: %d", args->alignment)};
  }

  absl::StatusOr<stream_executor::OwningDeviceMemory> memory =
      gpu->allocator->Allocate(args->ctx->device_ordinal, args->size);
  if (!memory.ok()) {
    return new XLA_FFI_Error{std::move(memory).status()};
  }

  args->data = memory->Release().opaque();
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_DeviceMemory_Free(
    XLA_FFI_DeviceMemory_Free_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_DeviceMemory_Free_Args",
      XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE, args->struct_size));

  auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  // TODO(ezhulenev): Device memory allocation should be supported for all
  // backends, not just GPU, although for CPU it doesn't make much sense, as
  // plain `new` is sufficient.
  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new XLA_FFI_Error{
        Unimplemented("XLA FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->allocator == nullptr)) {
    return new XLA_FFI_Error{
        Unimplemented("No device memory allocator available on this platform")};
  }

  absl::Status status = gpu->allocator->Deallocate(
      args->ctx->device_ordinal,
      stream_executor::DeviceMemoryBase(args->data, args->size));
  if (!status.ok()) {
    return new XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// XLA FFI Internal Api Implementation
//===----------------------------------------------------------------------===//

static XLA_FFI_Error* XLA_FFI_INTERNAL_Error_Forward(void* status) {
  auto* absl_status = reinterpret_cast<absl::Status*>(status);
  if (ABSL_PREDICT_TRUE(absl_status->ok())) {
    return nullptr;
  }
  return new XLA_FFI_Error{std::move(*absl_status)};
}

static void* XLA_FFI_INTERNAL_Stream_Get(XLA_FFI_ExecutionContext* ctx) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return gpu->stream;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static int32_t XLA_FFI_INTERNAL_DeviceOrdinal_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return ctx->device_ordinal;
}

static void* XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(
    XLA_FFI_ExecutionContext* ctx) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return gpu->allocator;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static void* XLA_FFI_INTERNAL_CalledComputation_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<HloComputation*>(ctx->called_computation);
}

static void* XLA_FFI_INTERNAL_ExecutionContext_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<ffi::ExecutionContext*>(ctx->execution_context);
}

static void* XLA_FFI_INTERNAL_ExecutionState_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<ffi::ExecutionState*>(ctx->execution_state);
}

void* XLA_FFI_INTERNAL_IntraOpThreadPool_Get(XLA_FFI_ExecutionContext* ctx) {
  if (auto* cpu = std::get_if<XLA_FFI_ExecutionContext::CpuContext>(
          &ctx->backend_context)) {
    return const_cast<Eigen::ThreadPoolDevice*>(cpu->intra_op_thread_pool);
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI CPU context is not available")};
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
    XLA_FFI_INTERNAL_ExecutionContext_Get,
    XLA_FFI_INTERNAL_ExecutionState_Get,
    XLA_FFI_INTERNAL_IntraOpThreadPool_Get,
};

static XLA_FFI_Api api = {
    XLA_FFI_Api_STRUCT_SIZE,
    /*priv=*/nullptr,

    XLA_FFI_Api_Version{
        XLA_FFI_Api_Version_STRUCT_SIZE,
        /*priv=*/nullptr,
        XLA_FFI_API_MAJOR,
        XLA_FFI_API_MINOR,
    },

    &internal_api,

    XLA_FFI_Error_Create,
    XLA_FFI_Error_GetMessage,
    XLA_FFI_Error_Destroy,
    XLA_FFI_Handler_Register,
    XLA_FFI_Stream_Get,
    XLA_FFI_TypeId_Register,
    XLA_FFI_ExecutionContext_Get,
    XLA_FFI_State_Set,
    XLA_FFI_State_Get,
    XLA_FFI_DeviceMemory_Allocate,
    XLA_FFI_DeviceMemory_Free,
};

const XLA_FFI_Api* GetXlaFfiApi() { return &api; }

}  // namespace xla::ffi
