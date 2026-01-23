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
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_internal_api.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/ffi/type_registry.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::ffi {

// The minimum XLA:FFI API version that XLA runtime supports.
static constexpr std::pair<int32_t, int32_t> kMinSupportedApiVersion = {
    /*major=*/0,
    /*minor=*/1,
};

// The maximum XLA:FFI API version that XLA runtime supports.
static constexpr std::pair<int32_t, int32_t> kMaxSupportedApiVersion = {
    XLA_FFI_API_MAJOR,
    XLA_FFI_API_MINOR,
};

static bool IsSupportedApiVersion(const XLA_FFI_Api_Version& api_version) {
  std::pair<int32_t, int32_t> version = {api_version.major_version,
                                         api_version.minor_version};
  return version >= kMinSupportedApiVersion &&
         version <= kMaxSupportedApiVersion;
}

bool IsCommandBufferCompatible(const XLA_FFI_Metadata& metadata) {
  return metadata.traits & XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
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
      return XLA_FFI_ExecutionContext::GpuContext{
          options.stream,
          options.allocator,
          options.collective_params,
          options.collective_clique_requests,
          options.collective_memory_requests,
          options.collective_cliques};
    }
  };

  return XLA_FFI_ExecutionContext{
      options.run_id,
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
  if (ABSL_PREDICT_TRUE(error == nullptr)) {
    return absl::OkStatus();
  }
  absl::Status status = std::move(error->status);
  delete error;
  return status;
}

tsl::AsyncValueRef<tsl::Chain> TakeFuture(XLA_FFI_Future* future) {
  // Non-reference-counted async value ref for synchronous FFI handlers.
  static tsl::AsyncValueOwningRef<tsl::Chain>* chain = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<tsl::Chain>();
    return new tsl::AsyncValueOwningRef<tsl::Chain>(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
  }();

  if (ABSL_PREDICT_TRUE(future == nullptr)) {
    return chain->AsRef();
  }

  // If the future is already completed, immediately return the underlying
  // async value and delete the XLA_FFI_Future.
  if (ABSL_PREDICT_TRUE(future->async_value.IsAvailable())) {
    tsl::AsyncValueRef<tsl::Chain> async_value = std::move(future->async_value);
    delete future;
    return async_value;
  }

  // If the future is not completed, return a copy of the underlying async
  // value and keep XLA_FFI_Future alive until it is completed.
  tsl::AsyncValueRef<tsl::Chain> async_value = future->async_value;
  async_value.AndThen([future] { delete future; });
  return async_value;
}

template <typename Handler>
static absl::StatusOr<XLA_FFI_Future*> Call(Handler& handler,
                                            CallFrame& call_frame,
                                            const CallOptions& options,
                                            ExecutionStage stage) {
  XLA_FFI_ExecutionContext ctx = CreateExecutionContext(options);
  XLA_FFI_CallFrame ffi_call_frame = call_frame.Build(
      GetXlaFfiApi(), &ctx, static_cast<XLA_FFI_ExecutionStage>(stage));

  XLA_FFI_Error* error = nullptr;

  // FFI handlers might be defined in external libraries and use exceptions, so
  // take extra care to catch them and convert to a status.
  try {
    if constexpr (std::is_same_v<Handler, Ffi>) {
      error = handler.Call(&ffi_call_frame);
    } else if constexpr (std::is_same_v<Handler, XLA_FFI_Handler*>) {
      error = (*handler)(&ffi_call_frame);
    } else {
      static_assert(sizeof(Handler) == 0, "Unsupported handler type");
    }
  } catch (std::exception& e) {
    return Unknown("XLA FFI call failed: %s", e.what());
  }

  // If FFI handler returned synchronous error, it must not launch any
  // asynchronous work that can also return an error.
  if (error != nullptr) {
    DCHECK_EQ(ffi_call_frame.future, nullptr)
        << "Error must not be used together with a future";
    return TakeStatus(error);
  }

  return ffi_call_frame.future;
}

static absl::Status BlockUntilReady(XLA_FFI_Future* future) {
  if (ABSL_PREDICT_TRUE(future == nullptr)) {
    return absl::OkStatus();
  }

  tsl::AsyncValueRef<tsl::Chain> av = TakeFuture(future);
  tsl::BlockUntilReady(av);
  return ABSL_PREDICT_FALSE(av.IsError()) ? av.GetError() : absl::OkStatus();
}

absl::Status Call(Ffi& handler, CallFrame& call_frame,
                  const CallOptions& options, ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(XLA_FFI_Future * future,
                      Call<Ffi>(handler, call_frame, options, stage));
  return BlockUntilReady(future);
}

absl::Status Call(XLA_FFI_Handler* handler, CallFrame& call_frame,
                  const CallOptions& options, XLA_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      XLA_FFI_Future * future,
      Call<XLA_FFI_Handler*>(handler, call_frame, options,
                             static_cast<ExecutionStage>(stage)));
  return BlockUntilReady(future);
}

tsl::AsyncValueRef<tsl::Chain> CallAsync(Ffi& handler, CallFrame& call_frame,
                                         const CallOptions& options,
                                         ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(XLA_FFI_Future * future,
                      Call<Ffi>(handler, call_frame, options, stage));
  return TakeFuture(future);
}

tsl::AsyncValueRef<tsl::Chain> CallAsync(XLA_FFI_Handler* handler,
                                         CallFrame& call_frame,
                                         const CallOptions& options,
                                         XLA_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      XLA_FFI_Future * future,
      Call<XLA_FFI_Handler*>(handler, call_frame, options,
                             static_cast<ExecutionStage>(stage)));
  return TakeFuture(future);
}

static XLA_FFI_Metadata PrepareMetadata() {
  return XLA_FFI_Metadata{XLA_FFI_Metadata_STRUCT_SIZE,
                          XLA_FFI_Api_Version{XLA_FFI_Api_Version_STRUCT_SIZE}};
}

static XLA_FFI_Metadata_Extension PrepareMetadataExtension(
    XLA_FFI_Metadata* metadata) {
  return XLA_FFI_Metadata_Extension{
      XLA_FFI_Extension_Base{XLA_FFI_Metadata_Extension_STRUCT_SIZE,
                             XLA_FFI_Extension_Metadata},
      metadata};
}

static XLA_FFI_CallFrame PrepareMetadataCallFrame(
    XLA_FFI_Metadata_Extension* extension) {
  return XLA_FFI_CallFrame{
      XLA_FFI_CallFrame_STRUCT_SIZE,
      &extension->extension_base,
      /*api=*/GetXlaFfiApi(),
      /*context=*/nullptr,
      /*stage=*/XLA_FFI_ExecutionStage_EXECUTE,
      /*args=*/XLA_FFI_Args{XLA_FFI_Args_STRUCT_SIZE},
      /*rets=*/XLA_FFI_Rets{XLA_FFI_Rets_STRUCT_SIZE},
      /*attrs=*/XLA_FFI_Attrs{XLA_FFI_Attrs_STRUCT_SIZE},
  };
}

absl::StatusOr<XLA_FFI_Metadata> GetMetadata(Ffi& handler) {
  XLA_FFI_Metadata metadata = PrepareMetadata();
  XLA_FFI_Metadata_Extension extension = PrepareMetadataExtension(&metadata);
  XLA_FFI_CallFrame call_frame = PrepareMetadataCallFrame(&extension);
  XLA_FFI_Error* error = nullptr;
  try {
    error = handler.Call(&call_frame);
  } catch (std::exception& e) {
    return Unknown("Fetching XLA FFI metadata failed: %s", e.what());
  }
  if (error != nullptr) {
    return TakeStatus(error);
  }
  return metadata;
}

absl::StatusOr<XLA_FFI_Metadata> GetMetadata(XLA_FFI_Handler* handler) {
  XLA_FFI_Metadata metadata = PrepareMetadata();
  XLA_FFI_Metadata_Extension extension = PrepareMetadataExtension(&metadata);
  XLA_FFI_CallFrame call_frame = PrepareMetadataCallFrame(&extension);
  XLA_FFI_Error* error = nullptr;
  try {
    error = (*handler)(&call_frame);
  } catch (std::exception& e) {
    return Unknown("Fetching XLA FFI metadata failed: %s", e.what());
  }
  if (error != nullptr) {
    return TakeStatus(error);
  }
  return metadata;
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

static HandlerKey MakeHandlerKey(absl::string_view name,
                                 absl::string_view platform) {
  return std::make_pair(std::string(name), absl::AsciiStrToLower(platform));
}

static HandlerRegistry& GetHandlerRegistry() {
  static auto* const registry = new HandlerRegistry();
  return *registry;
}

static std::vector<std::string> GetHandlerStages(
    const XLA_FFI_Handler_Bundle& bundle) {
  std::vector<std::string> stages;
  if (bundle.instantiate != nullptr) {
    stages.push_back("instantiate");
  }
  if (bundle.prepare != nullptr) {
    stages.push_back("prepare");
  }
  if (bundle.initialize != nullptr) {
    stages.push_back("initialize");
  }
  if (bundle.execute != nullptr) {
    stages.push_back("execute");
  }
  return stages;
}

static absl::Status RegisterHandler(absl::string_view name,
                                    absl::string_view platform,
                                    XLA_FFI_Handler_Bundle bundle,
                                    XLA_FFI_Handler_Traits traits) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  if (bundle.execute == nullptr) {
    return InvalidArgument(
        "FFI handler for %s on a platform %s must provide an execute "
        "implementation",
        name, platform);
  }

  // Check the API version that FFI handler was compiled with is supported.
  TF_ASSIGN_OR_RETURN(XLA_FFI_Metadata metadata, GetMetadata(bundle.execute));
  if (!IsSupportedApiVersion(metadata.api_version)) {
    return InvalidArgument(
        "XLA FFI handler registration for %s on platform %s (canonical %s) "
        "failed because the handler's API version (%d.%d) is incompatible "
        "with "
        "the framework's API version (%d.%d). Minimum supported API version "
        "is "
        "(%d.%d).",
        name, platform, canonical_platform, metadata.api_version.major_version,
        metadata.api_version.minor_version, kMaxSupportedApiVersion.first,
        kMaxSupportedApiVersion.second, kMinSupportedApiVersion.first,
        kMinSupportedApiVersion.second);
  }

  // Incorporate handler traits passed explicitly via handler registration API.
  metadata.traits |= traits;

  // Incorporate state type id from the instantiate implementation if present.
  if (bundle.instantiate) {
    TF_ASSIGN_OR_RETURN(XLA_FFI_Metadata instantiate_metadata,
                        GetMetadata(bundle.instantiate));
    metadata.state_type_id = instantiate_metadata.state_type_id;
  }

  VLOG(2) << absl::StreamFormat(
      "Register XLA FFI handler for '%s'; platform=%s (canonical=%s), "
      "stages=[%s], metadata=%v",
      name, platform, canonical_platform,
      absl::StrJoin(GetHandlerStages(bundle), ", "), metadata);

  HandlerRegistration registration{metadata, bundle};
  auto [it, emplaced] = GetHandlerRegistry().try_emplace(
      MakeHandlerKey(name, canonical_platform), registration);

  // We might accidentally link the same FFI library multiple times (because
  // linking shared libraries is hard), and we choose to ignore this problem as
  // long as we register exactly the same handler.
  if (!emplaced) {
    const HandlerRegistration& existing = it->second;
    if (existing.metadata != metadata) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on platform %s "
          "(canonical %s) with different metadata: %v vs %v",
          name, platform, canonical_platform, existing.metadata, metadata);
    }
    if (existing.bundle != bundle) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on platform %s "
          "(canonical %s) with different bundle addresses",
          name, platform, canonical_platform);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<HandlerRegistration> FindHandler(absl::string_view name,
                                                absl::string_view platform) {
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
StaticRegisteredHandlers(absl::string_view platform) {
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
          absl::string_view(args->name.ptr, args->name.len),
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
    auto assigned_type_id =
        TypeRegistry::AssignExternalTypeId(type_name, type_info);
    if (!assigned_type_id.ok()) {
      return new XLA_FFI_Error{std::move(assigned_type_id).status()};
    }

    args->type_id->type_id = assigned_type_id->value();
    return nullptr;
  }

  // If type_id is set, we are relying on the caller-provided unique type id.
  auto registered_type_id =
      TypeRegistry::RegisterExternalTypeId(type_name, type_id, type_info);
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

static XLA_FFI_Error* XLA_FFI_State_Set(XLA_FFI_State_Set_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_State_Set_Args", XLA_FFI_State_Set_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";

  absl::Status status = args->ctx->execution_state->Set(
      TypeRegistry::TypeId(args->type_id->type_id), args->state);
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
      TypeRegistry::TypeId(args->type_id->type_id));
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
