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

#include <cstdint>
#include <optional>

#include "absl/base/optimization.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/ffi.h"  // IWYU pragma: export
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// Type tags to bind parameters passed via execution context to FFI handler
//===----------------------------------------------------------------------===//

// Type tag binds to one of the following types defined by XLA:GPU runtime:
struct Stream {};                    //  `se::Stream*`
struct Allocator {};                 //  `se::DeviceAddressAllocator*`
struct ScratchAllocator {};          //  `se::OwningScratchAllocator`
struct CollectiveParams {};          //  `const xla::gpu::CollectiveParams*`
struct CollectiveCliqueRequests {};  //  `xla::gpu::CollectiveCliqueRequests*`
struct CollectiveMemoryRequests {};  //  `xla::gpu::CollectiveMemoryRequests*`
struct CollectiveCliques {};         //  `const xla::gpu::CollectiveCliques*`

// Parametrized type tag for platform stream, e.g. `cudaStream_t`
template <typename T>
struct PlatformStream {};

//===----------------------------------------------------------------------===//
// Context decoding
//===----------------------------------------------------------------------===//

template <>
struct CtxDecoding<Stream> {
  using Type = stream_executor::Stream*;

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

template <>
struct CtxDecoding<Allocator> {
  using Type = stream_executor::DeviceAddressAllocator*;

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
  using Type = stream_executor::OwningScratchAllocator<>;

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

    return stream_executor::OwningScratchAllocator<>(device_ordinal,
                                                     *device_allocator);
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
  using Type = const xla::gpu::CollectiveCliques*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_CollectiveCliques_Get,
        "collective cliques");
  }
};

}  // namespace xla::ffi

#endif  // XLA_BACKENDS_GPU_FFI_H_
