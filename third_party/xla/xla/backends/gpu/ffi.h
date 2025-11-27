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
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/api/api.h"  // IWYU pragma: export
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// Type tags to bind parameters passed via execution context to FFI handler
//===----------------------------------------------------------------------===//

struct Stream {};            // binds `se::Stream*`
struct Allocator {};         // binds `se::DeviceMemoryAllocator*`
struct ScratchAllocator {};  // binds `se::OwningScratchAllocator`

template <typename T>
struct PlatformStream {};  // binds a platform stream, e.g. `cudaStream_t`

//===----------------------------------------------------------------------===//
// Context decoding
//===----------------------------------------------------------------------===//

template <>
struct CtxDecoding<Stream> {
  using Type = stream_executor::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    void* stream = nullptr;
    if (XLA_FFI_Error* error =
            api->internal_api->XLA_FFI_INTERNAL_Stream_Get(ctx, &stream);
        ABSL_PREDICT_FALSE(error)) {
      diagnostic.Emit("Failed to get stream: ")
          << internal::GetErrorMessage(api, error);
      internal::DestroyError(api, error);
      return std::nullopt;
    }
    return reinterpret_cast<Type>(stream);
  }
};

template <>
struct CtxDecoding<Allocator> {
  using Type = stream_executor::DeviceMemoryAllocator*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    void* device_allocator = nullptr;
    if (XLA_FFI_Error* error =
            api->internal_api->XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(
                ctx, &device_allocator);
        ABSL_PREDICT_FALSE(error)) {
      diagnostic.Emit("Failed to get device memory allocator: ")
          << internal::GetErrorMessage(api, error);
      internal::DestroyError(api, error);
      return std::nullopt;
    }
    return reinterpret_cast<Type>(device_allocator);
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

}  // namespace xla::ffi

#endif  // XLA_BACKENDS_GPU_FFI_H_
