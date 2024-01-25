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

#ifndef XLA_FFI_FFI_H_
#define XLA_FFI_FFI_H_

#ifdef TENSORFLOW_COMPILER_XLA_FFI_API_FFI_H_
#error Two different XLA FFI implementations cannot be included together
#endif  // XLA_FFI_API_FFI_H_

#include <cstdint>
#include <optional>

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::ffi {

// A tag to declare called computation argument in FFI handler.
struct CalledComputation {};

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

struct Buffer {
  PrimitiveType dtype;
  se::DeviceMemoryBase data;
  absl::Span<const int64_t> dimensions;

  // TODO(ezhulenev): Remove this implicit conversion once we'll migrate to FFI
  // handlers from runtime custom calls.
  operator runtime::MemrefView() {  // NOLINT
    return runtime::MemrefView{dtype, data.opaque(), dimensions};
  }
};

//===----------------------------------------------------------------------===//
// Arguments decoding
//===----------------------------------------------------------------------===//

template <>
struct ArgDecoding<Buffer> {
  static std::optional<Buffer> Decode(XLA_FFI_ArgType type, void* arg,
                                      DiagnosticEngine&) {
    if (type != XLA_FFI_ArgType_BUFFER) return std::nullopt;
    auto* buf = reinterpret_cast<XLA_FFI_Buffer*>(arg);

    Buffer buffer;
    buffer.dtype = PrimitiveType(buf->dtype);
    buffer.data = se::DeviceMemoryBase(buf->data);
    buffer.dimensions = absl::MakeConstSpan(buf->dims, buf->rank);
    return buffer;
  }
};

//===----------------------------------------------------------------------===//
// Context decoding
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We should remove `ServiceExecutableRunOptions` context and
// pass only se::Stream to FFI handlers.
template <>
struct CtxDecoding<ServiceExecutableRunOptions> {
  using Type = const ServiceExecutableRunOptions*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    void* ptr = api->internal_api->XLA_FFI_ServiceExecutableRunOptions_Get(ctx);
    return reinterpret_cast<Type>(ptr);
  }
};

template <>
struct CtxDecoding<CalledComputation> {
  using Type = const HloComputation*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    void* ptr = api->internal_api->XLA_FFI_CalledComputation_Get(ctx);
    return reinterpret_cast<Type>(ptr);
  }
};

//===----------------------------------------------------------------------===//
// Result encoding
//===----------------------------------------------------------------------===//

template <>
struct ResultEncoding<Status> {
  static XLA_FFI_Error* Encode(XLA_FFI_Api* api, Status status) {
    return api->internal_api->XLA_FFI_Error_Forward(&status);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_FFI_H_
