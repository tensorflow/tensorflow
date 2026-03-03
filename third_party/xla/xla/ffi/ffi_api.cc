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
#include <utility>
#include <variant>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_internal_api.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/ffi/invoke.h"
#include "xla/ffi/type_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/util.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::ffi {

// Forward declare XLA:FFI API access defined below. This API is available
// publicly via the `xla/ffi/ffi.h` or `xla/ffi/api/ffi.h` header. In this
// translation unit we implement it, and it is critical that this translation
// unit linked exactly one time into the main binary. It must not be linked
// into FFI handler implementations as it will lead to duplicate static
// registries in multiple object files.
const XLA_FFI_Api* GetXlaFfiApi();

// This is an implementation of XLA FFI API defined in `api/c_api.h` header. It
// should be linked statically into the "main" XLA binary, and third party FFI
// handlers can be linked and registered dynamically. When we build JAX PjRt
// plugins we essentially have a full copy of XLA in each plugin and each
// plugin has a separate FFI API implementation (and separate handler and type
// registries).
//
// FFI handlers built from the same XLA commit with the same toolchain can also
// use `api/c_api_internal.h` to get access to various internal data structures.

//===----------------------------------------------------------------------===//
// XLA FFI Api Implementation
//===----------------------------------------------------------------------===//

static std::string StructSizeErrorMsg(absl::string_view struct_name,
                                      size_t expected, size_t actual) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ", expected,
                      ", got ", actual, ". Check installed software versions. ",
                      "The framework XLA FFI API version is ",
                      XLA_FFI_API_MAJOR, ".", XLA_FFI_API_MINOR, ".");
}

static absl::Status ActualStructSizeIsGreaterOrEqual(
    absl::string_view struct_name, size_t expected, size_t actual) {
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

static XLA_FFI_Error* XLA_FFI_Future_Create(XLA_FFI_Future_Create_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Future_Create", XLA_FFI_Future_Create_Args_STRUCT_SIZE,
      args->struct_size));
  args->future =
      new XLA_FFI_Future{tsl::MakeConstructedAsyncValueRef<tsl::Chain>()};
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_Future_SetAvailable(
    XLA_FFI_Future_SetAvailable_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Future_SetAvailable",
      XLA_FFI_Future_SetAvailable_Args_STRUCT_SIZE, args->struct_size));
  args->future->async_value.SetStateConcrete();
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_Future_SetError(
    XLA_FFI_Future_SetError_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Future_SetError", XLA_FFI_Future_SetError_Args_STRUCT_SIZE,
      args->struct_size));

  if (args->error == nullptr || args->error->status.ok()) {
    return new XLA_FFI_Error{InvalidArgument("Error must not be null or OK")};
  }

  absl::Status error = TakeStatus(args->error);
  args->future->async_value.SetError(std::move(error));

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_Handler_Register(
    XLA_FFI_Handler_Register_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Handler_Register", XLA_FFI_Handler_Register_Args_STRUCT_SIZE,
      args->struct_size));

  if (auto status = RegisterHandler(
          GetXlaFfiApi(), absl::string_view(args->name.ptr, args->name.len),
          absl::string_view(args->platform.ptr, args->platform.len),
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

static XLA_FFI_Error* XLA_FFI_RunId_Get(XLA_FFI_RunId_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_RunId_Get", XLA_FFI_RunId_Get_Args_STRUCT_SIZE,
      args->struct_size));

  args->run_id = args->ctx->run_id.ToInt();

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_DeviceOrdinal_Get(
    XLA_FFI_DeviceOrdinal_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_DeviceOrdinal_Get", XLA_FFI_DeviceOrdinal_Get_Args_STRUCT_SIZE,
      args->struct_size));
  args->device_ordinal = args->ctx->device_ordinal;
  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_Type_Register(XLA_FFI_Type_Register_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_ExecutionContext_Get_Args",
      XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  absl::string_view type_name(args->name.ptr, args->name.len);
  TypeRegistry::TypeId type_id(args->type_id->type_id);
  TypeRegistry::TypeInfo type_info = {args->type_info->deleter};

  // If type_id is unknown, we are registering a new type and XLA will assign
  // a unique type id to it.
  if (type_id == TypeRegistry::kUnknownTypeId) {
    auto assigned_type_id = TypeRegistry::AssignTypeId(type_name, type_info);
    if (!assigned_type_id.ok()) {
      return new XLA_FFI_Error{std::move(assigned_type_id).status()};
    }

    args->type_id->type_id = assigned_type_id->value();
    return nullptr;
  }

  // If type_id is set, we are relying on the caller-provided unique type id.
  auto registered_type_id =
      TypeRegistry::RegisterTypeId(type_name, type_id, type_info);
  if (!registered_type_id.ok()) {
    return new XLA_FFI_Error{std::move(registered_type_id)};
  }

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_ExecutionContext_Get(
    XLA_FFI_ExecutionContext_Get_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_ExecutionContext_Get_Args",
      XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  DCHECK(args->ctx->execution_context) << "ExecutionContext must be set";
  auto user_data = args->ctx->execution_context->Lookup(
      TypeRegistry::TypeId(args->type_id->type_id));
  if (!user_data.ok()) {
    return new XLA_FFI_Error{std::move(user_data).status()};
  }

  args->data = *user_data;
  return nullptr;
}

static ExecutionState* GetExecutionState(XLA_FFI_ExecutionContext* ctx,
                                         XLA_FFI_ExecutionStage stage) {
  switch (stage) {
    case XLA_FFI_ExecutionStage_INSTANTIATE:
      return ctx->state_context.instantiate;
    case XLA_FFI_ExecutionStage_PREPARE:
      return ctx->state_context.prepare;
    case XLA_FFI_ExecutionStage_INITIALIZE:
      return ctx->state_context.initialize;
    case XLA_FFI_ExecutionStage_EXECUTE:
      DCHECK(false) << "Execution stage doesn't have a state";
      return nullptr;
  }
}

namespace {
// This is a struct declaration for `XLA_FFI_State_Set/Get_Args` in XLA:FFI
// version 0.2. We use this struct to detect older XLA:FFI clients for backward
// compatibility reasons. This can be removed 15 Feb 2027.
struct XLA_FFI_State_Args_V02 {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_TypeId* type_id;
  void* state;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_State_Args_V02, state);
}  // namespace

static XLA_FFI_Error* XLA_FFI_State_Set(XLA_FFI_State_Set_Args* args) {
  // If struct size matches the legacy struct layout, always assume that we set
  // the state for instantiation stage.
  if (args->struct_size == XLA_FFI_State_Args_V02_STRUCT_SIZE) {
    auto* v02 = reinterpret_cast<XLA_FFI_State_Args_V02*>(args);

    XLA_FFI_State_Set_Args compat = {XLA_FFI_State_Set_Args_STRUCT_SIZE};
    compat.ctx = v02->ctx;
    compat.stage = XLA_FFI_ExecutionStage_INSTANTIATE;
    compat.type_id = v02->type_id;
    compat.state = v02->state;

    return XLA_FFI_State_Set(&compat);
  }

  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_State_Set_Args", XLA_FFI_State_Set_Args_STRUCT_SIZE,
      args->struct_size));

  ExecutionState* execution_state = GetExecutionState(args->ctx, args->stage);
  DCHECK(execution_state) << "ExecutionState must be set";

  absl::Status status = execution_state->Set(
      TypeRegistry::TypeId(args->type_id->type_id), args->state);
  if (!status.ok()) {
    return new XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_State_Get(XLA_FFI_State_Get_Args* args) {
  // If struct size matches the legacy struct layout, always assume that we get
  // the state for instantiation stage.
  if (args->struct_size == XLA_FFI_State_Args_V02_STRUCT_SIZE) {
    auto* v02 = reinterpret_cast<XLA_FFI_State_Args_V02*>(args);

    XLA_FFI_State_Get_Args compat = {XLA_FFI_State_Set_Args_STRUCT_SIZE};
    compat.ctx = v02->ctx;
    compat.stage = XLA_FFI_ExecutionStage_INSTANTIATE;
    compat.type_id = v02->type_id;
    compat.state = v02->state;

    return XLA_FFI_State_Get(&compat);
  }

  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_State_Get_Args", XLA_FFI_State_Get_Args_STRUCT_SIZE,
      args->struct_size));

  ExecutionState* execution_state = GetExecutionState(args->ctx, args->stage);
  DCHECK(execution_state) << "ExecutionState must be set";

  absl::StatusOr<void*> state =
      execution_state->Get(TypeRegistry::TypeId(args->type_id->type_id));
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

  absl::StatusOr<stream_executor::ScopedDeviceAddress<uint8_t>> memory =
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
      stream_executor::DeviceAddressBase(args->data, args->size));
  if (!status.ok()) {
    return new XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

static absl::StatusOr<const Eigen::ThreadPoolDevice*> GetIntraOpThreadPool(
    const XLA_FFI_ExecutionContext* ctx) {
  auto* cpu =
      std::get_if<XLA_FFI_ExecutionContext::CpuContext>(&ctx->backend_context);

  if (ABSL_PREDICT_FALSE(cpu == nullptr)) {
    return Unimplemented("XLA FFI CPU context is not available");
  }

  if (ABSL_PREDICT_FALSE(cpu->intra_op_thread_pool == nullptr)) {
    return Unimplemented("No intra-op thread pool available on this platform");
  }

  return cpu->intra_op_thread_pool;
}

static XLA_FFI_Error* XLA_FFI_ThreadPool_Schedule(
    XLA_FFI_ThreadPool_Schedule_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_ThreadPool_Schedule_Args",
      XLA_FFI_ThreadPool_Schedule_Args_STRUCT_SIZE, args->struct_size));

  auto intra_op_thread_pool = GetIntraOpThreadPool(args->ctx);
  if (!intra_op_thread_pool.ok()) {
    return new XLA_FFI_Error{std::move(intra_op_thread_pool).status()};
  }

  (*intra_op_thread_pool)
      ->enqueueNoNotification(
          [task = args->task, data = args->data] { (*task)(data); });

  return nullptr;
}

static XLA_FFI_Error* XLA_FFI_ThreadPool_NumThreads(
    XLA_FFI_ThreadPool_NumThreads_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_ThreadPool_NumThreads_Args",
      XLA_FFI_ThreadPool_NumThreads_Args_STRUCT_SIZE, args->struct_size));

  auto intra_op_thread_pool = GetIntraOpThreadPool(args->ctx);
  if (!intra_op_thread_pool.ok()) {
    return new XLA_FFI_Error{std::move(intra_op_thread_pool).status()};
  }

  *args->num_threads = (*intra_op_thread_pool)->numThreadsInPool();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// XLA FFI Api access
//===----------------------------------------------------------------------===//

const XLA_FFI_Api* GetXlaFfiApi() {
  static XLA_FFI_Api api = {
      XLA_FFI_Api_STRUCT_SIZE,
      /*extension_start=*/nullptr,

      XLA_FFI_Api_Version{
          XLA_FFI_Api_Version_STRUCT_SIZE,
          /*extension_start=*/nullptr,
          XLA_FFI_API_MAJOR,
          XLA_FFI_API_MINOR,
      },

      internal::GetInternalApi(),

      XLA_FFI_Error_Create,
      XLA_FFI_Error_GetMessage,
      XLA_FFI_Error_Destroy,
      XLA_FFI_Handler_Register,
      XLA_FFI_Stream_Get,
      XLA_FFI_Type_Register,
      XLA_FFI_ExecutionContext_Get,
      XLA_FFI_State_Set,
      XLA_FFI_State_Get,
      XLA_FFI_DeviceMemory_Allocate,
      XLA_FFI_DeviceMemory_Free,
      XLA_FFI_ThreadPool_Schedule,
      XLA_FFI_ThreadPool_NumThreads,
      XLA_FFI_Future_Create,
      XLA_FFI_Future_SetAvailable,
      XLA_FFI_Future_SetError,
      XLA_FFI_RunId_Get,
      XLA_FFI_DeviceOrdinal_Get,
  };

  return &api;
}

extern "C" const XLA_FFI_Api* XLA_FFI_GetApi() { return GetXlaFfiApi(); }

}  // namespace xla::ffi
