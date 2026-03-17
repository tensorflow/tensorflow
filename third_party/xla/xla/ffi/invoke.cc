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

#include "xla/ffi/invoke.h"

#include <exception>
#include <variant>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// Invoking XLA FFI handlers.
//===----------------------------------------------------------------------===//

namespace {
// Converts InvokeContext to corresponding XLA_FFI_ExecutionContext context.
struct BackendVisitor {
  XLA_FFI_ExecutionContext::BackendContext operator()(
      const std::monostate&) const {
    return std::monostate{};
  }

  XLA_FFI_ExecutionContext::BackendContext operator()(
      const InvokeContext::CpuContext& cpu) const {
    return XLA_FFI_ExecutionContext::CpuContext{cpu.intra_op_thread_pool};
  }

  XLA_FFI_ExecutionContext::BackendContext operator()(
      const InvokeContext::GpuContext& gpu) const {
    return XLA_FFI_ExecutionContext::GpuContext{
        gpu.stream,
        gpu.allocator,
        gpu.collective_params,
        gpu.collective_clique_requests,
        gpu.collective_memory_requests,
        gpu.collective_cliques,
        gpu.collective_memory,
        gpu.compute_capability,
    };
  }
};
}  // namespace

static XLA_FFI_ExecutionContext CreateExecutionContext(
    const InvokeContext& context) {
  return XLA_FFI_ExecutionContext{
      context.run_id,
      context.device_ordinal,
      std::visit(BackendVisitor{}, context.backend_context),
      XLA_FFI_ExecutionContext::StateContext{context.state_context.instantiate,
                                             context.state_context.prepare,
                                             context.state_context.initialize},
      context.called_computation,
      internal::ScopedExecutionContext::GetCallExecutionContext(context)};
}

template <typename Handler>
static absl::StatusOr<XLA_FFI_Future*> Invoke(const XLA_FFI_Api* api,
                                              Handler& handler,
                                              CallFrame& call_frame,
                                              const InvokeContext& context,
                                              ExecutionStage stage) {
  XLA_FFI_ExecutionContext ctx = CreateExecutionContext(context);
  XLA_FFI_CallFrame ffi_call_frame =
      call_frame.Build(api, &ctx, static_cast<XLA_FFI_ExecutionStage>(stage));

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

absl::Status Invoke(const XLA_FFI_Api* api, Ffi& handler, CallFrame& call_frame,
                    const InvokeContext& context, ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(XLA_FFI_Future * future,
                      Invoke<Ffi>(api, handler, call_frame, context, stage));
  return BlockUntilReady(future);
}

absl::Status Invoke(const XLA_FFI_Api* api, XLA_FFI_Handler* handler,
                    CallFrame& call_frame, const InvokeContext& context,
                    XLA_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      XLA_FFI_Future * future,
      Invoke<XLA_FFI_Handler*>(api, handler, call_frame, context,
                               static_cast<ExecutionStage>(stage)));
  return BlockUntilReady(future);
}

tsl::AsyncValueRef<tsl::Chain> InvokeAsync(const XLA_FFI_Api* api, Ffi& handler,
                                           CallFrame& call_frame,
                                           const InvokeContext& context,
                                           ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(XLA_FFI_Future * future,
                      Invoke<Ffi>(api, handler, call_frame, context, stage));
  return TakeFuture(future);
}

tsl::AsyncValueRef<tsl::Chain> InvokeAsync(const XLA_FFI_Api* api,
                                           XLA_FFI_Handler* handler,
                                           CallFrame& call_frame,
                                           const InvokeContext& context,
                                           XLA_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      XLA_FFI_Future * future,
      Invoke<XLA_FFI_Handler*>(api, handler, call_frame, context,
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
    const XLA_FFI_Api* api, XLA_FFI_Metadata_Extension* extension) {
  return XLA_FFI_CallFrame{
      XLA_FFI_CallFrame_STRUCT_SIZE,
      &extension->extension_base,
      /*api=*/api,
      /*context=*/nullptr,
      /*stage=*/XLA_FFI_ExecutionStage_EXECUTE,
      /*args=*/XLA_FFI_Args{XLA_FFI_Args_STRUCT_SIZE},
      /*rets=*/XLA_FFI_Rets{XLA_FFI_Rets_STRUCT_SIZE},
      /*attrs=*/XLA_FFI_Attrs{XLA_FFI_Attrs_STRUCT_SIZE},
  };
}

absl::StatusOr<XLA_FFI_Metadata> GetMetadata(const XLA_FFI_Api* api,
                                             Ffi& handler) {
  XLA_FFI_Metadata metadata = PrepareMetadata();
  XLA_FFI_Metadata_Extension extension = PrepareMetadataExtension(&metadata);
  XLA_FFI_CallFrame call_frame = PrepareMetadataCallFrame(api, &extension);
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

absl::StatusOr<XLA_FFI_Metadata> GetMetadata(const XLA_FFI_Api* api,
                                             XLA_FFI_Handler* handler) {
  XLA_FFI_Metadata metadata = PrepareMetadata();
  XLA_FFI_Metadata_Extension extension = PrepareMetadataExtension(&metadata);
  XLA_FFI_CallFrame call_frame = PrepareMetadataCallFrame(api, &extension);
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

//===----------------------------------------------------------------------===//
// ScopedExecutionContext implementation.
//===----------------------------------------------------------------------===//

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
    const InvokeContext& context) {
  return scoped_execution_context ? scoped_execution_context
                                  : context.execution_context;
}

}  // namespace internal
}  // namespace xla::ffi
