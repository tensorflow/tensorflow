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

#ifndef XLA_BACKENDS_GPU_FFI_H_
#define XLA_BACKENDS_GPU_FFI_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "absl/base/optimization.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/ffi.h"  // IWYU pragma: export
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// Type tags to bind parameters passed via execution context to FFI handler
//===----------------------------------------------------------------------===//

// Type tag binds to one of the following types defined by XLA:GPU runtime:
struct Stream {};                      //  se::Stream*
struct Allocator {};                   //  se::DeviceAddressAllocator*
struct ScratchAllocator {};            //  se::OwningScratchAllocator
struct CollectiveParams {};            //  const xla::gpu::CollectiveParams*
struct CollectiveCliqueRequests {};    //  xla::gpu::CollectiveCliqueRequests*
struct CollectiveMemoryRequests {};    //  xla::gpu::CollectiveMemoryRequests*
struct CollectiveCliques {};           //  xla::gpu::CollectiveCliques*
struct CollectiveMemory {};            //  xla::gpu::CollectiveMemory*
struct TargetGpuComputeCapability {};  //  const se::GpuComputeCapability*
struct CpuTargetMachineOptions {};     //  const xla::cpu::TargetMachineOptions*

// Parametrized type tags for binding additional streams. A single stream id
// binds as se::Stream*, while multiple ids bind as std::array<se::Stream*, N>.
template <size_t id, size_t... ids>
struct ComputationStream {};
template <size_t id, size_t... ids>
struct CommunicationStream {};

// Parametrized type tag for platform stream (binds as T, e.g. cudaStream_t).
template <typename T>
struct PlatformStream {};

//===----------------------------------------------------------------------===//
// Context decoding
//===----------------------------------------------------------------------===//

namespace internal {

template <size_t n>
std::optional<std::array<se::Stream*, n>> OptionalStreamsToArray(
    const std::array<std::optional<se::Stream*>, n>& decoded) {
  std::array<se::Stream*, n> streams;
  for (size_t i = 0; i < streams.size(); ++i) {
    if (!decoded[i].has_value()) {
      return std::nullopt;
    }
    streams[i] = *decoded[i];
  }
  return streams;
}

}  // namespace internal

template <>
struct CtxDecoding<Stream> {
  using Type = se::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic, api->internal_api->XLA_FFI_INTERNAL_Stream_Get,
        "stream");
  }
};

template <typename T>
struct CtxDecoding<PlatformStream<T>> {
  using Type = T;
  static_assert(std::is_pointer_v<T>, "platform stream type must be a pointer");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    if (auto stream = CtxDecoding<Stream>::Decode(api, ctx, diagnostic)) {
      return reinterpret_cast<Type>(
          stream.value()->platform_specific_handle().stream);
    }
    return std::nullopt;
  }
};

template <size_t id>
struct CtxDecoding<ComputationStream<id>> {
  using Type = se::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    void* result = nullptr;
    if (XLA_FFI_Error* error =
            api->internal_api->XLA_FFI_INTERNAL_ComputationStream_Get(
                ctx, static_cast<int64_t>(id), &result);
        ABSL_PREDICT_FALSE(error)) {
      diagnostic.Emit("Failed to get computation stream: ")
          << internal::GetErrorMessage(api, error);
      internal::DestroyError(api, error);
      return std::nullopt;
    }
    return reinterpret_cast<Type>(result);
  }
};

template <size_t id0, size_t id1, size_t... ids>
struct CtxDecoding<ComputationStream<id0, id1, ids...>> {
  using Type = std::array<se::Stream*, 2 + sizeof...(ids)>;

  template <size_t stream_id>
  using StreamDecoding = CtxDecoding<ComputationStream<stream_id>>;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::OptionalStreamsToArray<2 + sizeof...(ids)>({
        StreamDecoding<id0>::Decode(api, ctx, diagnostic),
        StreamDecoding<id1>::Decode(api, ctx, diagnostic),
        StreamDecoding<ids>::Decode(api, ctx, diagnostic)...,
    });
  }
};

template <size_t id>
struct CtxDecoding<CommunicationStream<id>> {
  using Type = se::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    void* result = nullptr;
    if (XLA_FFI_Error* error =
            api->internal_api->XLA_FFI_INTERNAL_CommunicationStream_Get(
                ctx, static_cast<int64_t>(id), &result);
        ABSL_PREDICT_FALSE(error)) {
      diagnostic.Emit("Failed to get communication stream: ")
          << internal::GetErrorMessage(api, error);
      internal::DestroyError(api, error);
      return std::nullopt;
    }
    return reinterpret_cast<Type>(result);
  }
};

template <size_t id0, size_t id1, size_t... ids>
struct CtxDecoding<CommunicationStream<id0, id1, ids...>> {
  using Type = std::array<se::Stream*, 2 + sizeof...(ids)>;

  template <size_t stream_id>
  using StreamDecoding = CtxDecoding<CommunicationStream<stream_id>>;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::OptionalStreamsToArray<2 + sizeof...(ids)>({
        StreamDecoding<id0>::Decode(api, ctx, diagnostic),
        StreamDecoding<id1>::Decode(api, ctx, diagnostic),
        StreamDecoding<ids>::Decode(api, ctx, diagnostic)...,
    });
  }
};

template <>
struct CtxDecoding<Allocator> {
  using Type = se::DeviceAddressAllocator*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get,
        "device memory allocator");
  }
};

template <>
struct CtxDecoding<ScratchAllocator> {
  using Type = se::OwningScratchAllocator<>;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    int32_t device_ordinal =
        api->internal_api->XLA_FFI_INTERNAL_DeviceOrdinal_Get(ctx);

    auto device_allocator =
        CtxDecoding<Allocator>::Decode(api, ctx, diagnostic);
    if (ABSL_PREDICT_FALSE(!device_allocator)) {
      return std::nullopt;
    }

    return se::OwningScratchAllocator<>(device_ordinal, *device_allocator);
  }
};

template <>
struct CtxDecoding<CollectiveParams> {
  using Type = const xla::gpu::CollectiveParams*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CollectiveParams_Get,
        "collective params");
  }
};

template <>
struct CtxDecoding<CollectiveCliqueRequests> {
  using Type = xla::gpu::CollectiveCliqueRequests*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CollectiveCliqueRequests_Get,
        "collective clique requests");
  }
};

template <>
struct CtxDecoding<CollectiveMemoryRequests> {
  using Type = xla::gpu::CollectiveMemoryRequests*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CollectiveMemoryRequests_Get,
        "collective memory requests");
  }
};

template <>
struct CtxDecoding<CollectiveCliques> {
  using Type = xla::gpu::CollectiveCliques*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CollectiveCliques_Get,
        "collective cliques");
  }
};

template <>
struct CtxDecoding<CollectiveMemory> {
  using Type = xla::gpu::CollectiveMemory*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CollectiveMemory_Get,
        "collective memory");
  }
};

template <>
struct CtxDecoding<TargetGpuComputeCapability> {
  using Type = const se::GpuComputeCapability*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_GpuComputeCapability_Get,
        "gpu compute capability");
  }
};

template <>
struct CtxDecoding<CpuTargetMachineOptions> {
  using Type = const xla::cpu::TargetMachineOptions*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CpuTargetMachineOptions_Get,
        "cpu target machine options");
  }
};

}  // namespace xla::ffi

#endif  // XLA_BACKENDS_GPU_FFI_H_
