/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/ffi/ffi_internal_api.h"

#include <cstdint>
#include <utility>
#include <variant>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/ffi/type_registry.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"

namespace xla::ffi::internal {

//===----------------------------------------------------------------------===//
// Generic XLA internal APIs available on all XLA backends.
//===----------------------------------------------------------------------===//

static XLA_FFI_Error* XLA_FFI_INTERNAL_Error_Forward(void* status) {
  auto* absl_status = reinterpret_cast<absl::Status*>(status);
  if (ABSL_PREDICT_TRUE(absl_status->ok())) {
    return nullptr;
  }
  return new XLA_FFI_Error{std::move(*absl_status)};
}

static XLA_FFI_Future* XLA_FFI_INTERNAL_Future_Forward(void* async_value) {
  auto* tsl_async_value = reinterpret_cast<tsl::AsyncValue*>(async_value);
  DCHECK(tsl_async_value) << "Async value must not be null";

  return new XLA_FFI_Future{
      tsl::AsyncValueRef<tsl::Chain>(tsl::TakeRef(tsl_async_value))};
}

static void* XLA_FFI_Internal_HandlerRegistrationMap_Get() {
  return &internal::StaticHandlerRegistrationMap();
}

static void* XLA_FFI_Internal_TypeRegistrationMap_Get() {
  return &internal::StaticTypeRegistrationMap();
}

static int32_t XLA_FFI_INTERNAL_DeviceOrdinal_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return ctx->device_ordinal;
}

static int64_t XLA_FFI_INTERNAL_RunId_Get(XLA_FFI_ExecutionContext* ctx) {
  return ctx->run_id.ToInt();
}

static void* XLA_FFI_INTERNAL_CalledComputation_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<HloComputation*>(ctx->called_computation);  // NOLINT
}

static void* XLA_FFI_INTERNAL_ExecutionContext_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<ExecutionContext*>(ctx->execution_context);  // NOLINT
}

static void* XLA_FFI_INTERNAL_ExecutionState_Get(
    XLA_FFI_ExecutionContext* ctx) {
  return const_cast<ExecutionState*>(ctx->execution_state);  // NOLINT
}

//===----------------------------------------------------------------------===//
// XLA:CPU specific internal APIs.
//===----------------------------------------------------------------------===//

static XLA_FFI_Error* XLA_FFI_INTERNAL_IntraOpThreadPool_Get(
    XLA_FFI_ExecutionContext* ctx, void** thread_pool) {
  if (auto* cpu = std::get_if<XLA_FFI_ExecutionContext::CpuContext>(
          &ctx->backend_context)) {
    *thread_pool = const_cast<Eigen::ThreadPoolDevice*>(  // NOLINT
        cpu->intra_op_thread_pool);
    return nullptr;
  }

  // For GPU backend we don't have intra-op thread pool, but we didn't promise
  // to return one, so instead of an error we return a nullptr thread pool.
  if (auto* _ = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return nullptr;
  }

  return new XLA_FFI_Error{InvalidArgument("XLA FFI context is not available")};
}

//===----------------------------------------------------------------------===//
// XLA:GPU specific internal APIs.
//===----------------------------------------------------------------------===//

static XLA_FFI_Error* XLA_FFI_INTERNAL_Stream_Get(XLA_FFI_ExecutionContext* ctx,
                                                  void** stream) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *stream = gpu->stream;
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(
    XLA_FFI_ExecutionContext* ctx, void** allocator) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *allocator = gpu->allocator;
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_CollectiveParams_Get(
    XLA_FFI_ExecutionContext* ctx, void** collective_params) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *collective_params = const_cast<xla::gpu::CollectiveParams*>(  // NOLINT
        gpu->collective_params);
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_CollectiveCliqueRequests_Get(
    XLA_FFI_ExecutionContext* ctx, void** collective_clique_requests) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *collective_clique_requests = gpu->collective_clique_requests;
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_CollectiveMemoryRequests_Get(
    XLA_FFI_ExecutionContext* ctx, void** collective_memory_requests) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *collective_memory_requests = gpu->collective_memory_requests;
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_BarrierRequests_Get(
    XLA_FFI_ExecutionContext* ctx, void** barrier_requests) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *barrier_requests = gpu->barrier_requests;
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}
static XLA_FFI_Error* XLA_FFI_INTERNAL_CollectiveCliques_Get(
    XLA_FFI_ExecutionContext* ctx, void** collective_clique) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *collective_clique = const_cast<xla::gpu::CollectiveCliques*>(  // NOLINT
        gpu->collective_cliques);
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_CollectiveMemory_Get(
    XLA_FFI_ExecutionContext* ctx, void** collective_memory) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *collective_memory = const_cast<xla::gpu::CollectiveMemory*>(  // NOLINT
        gpu->collective_memory);
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static XLA_FFI_Error* XLA_FFI_INTERNAL_GpuComputeCapability_Get(
    XLA_FFI_ExecutionContext* ctx, void** gpu_compute_capability) {
  if (auto* gpu = std::get_if<XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    *gpu_compute_capability =
        const_cast<stream_executor::GpuComputeCapability*>(  // NOLINT
            gpu->gpu_compute_capability);
    return nullptr;
  }

  return new XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

const XLA_FFI_InternalApi* GetInternalApi() {
  static XLA_FFI_InternalApi internal_api = {
      // Generic XLA APIs available on all XLA backends.
      XLA_FFI_INTERNAL_Error_Forward,
      XLA_FFI_INTERNAL_Future_Forward,
      XLA_FFI_Internal_HandlerRegistrationMap_Get,
      XLA_FFI_Internal_TypeRegistrationMap_Get,
      XLA_FFI_INTERNAL_DeviceOrdinal_Get,
      XLA_FFI_INTERNAL_RunId_Get,
      XLA_FFI_INTERNAL_CalledComputation_Get,
      XLA_FFI_INTERNAL_ExecutionContext_Get,
      XLA_FFI_INTERNAL_ExecutionState_Get,

      // XLA:CPU specific APIs.
      XLA_FFI_INTERNAL_IntraOpThreadPool_Get,

      // XLA:GPU specific APIs.
      XLA_FFI_INTERNAL_Stream_Get,
      XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get,
      XLA_FFI_INTERNAL_CollectiveParams_Get,
      XLA_FFI_INTERNAL_CollectiveCliqueRequests_Get,
      XLA_FFI_INTERNAL_CollectiveMemoryRequests_Get,
      XLA_FFI_INTERNAL_BarrierRequests_Get,
      XLA_FFI_INTERNAL_CollectiveCliques_Get,
      XLA_FFI_INTERNAL_CollectiveMemory_Get,
      XLA_FFI_INTERNAL_GpuComputeCapability_Get,
  };

  return &internal_api;
}

}  // namespace xla::ffi::internal
