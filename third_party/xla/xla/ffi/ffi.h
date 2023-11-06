/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <string_view>

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/call_frame.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

struct Buffer {
  PrimitiveType primitive_type;
  se::DeviceMemoryBase data;
  absl::Span<const int64_t> dimensions;

  // TODO(ezhulenev): Remove this implicit conversion once we'll migrate to FFI
  // handlers from runtime custom calls.
  operator runtime::MemrefView() {  // NOLINT
    return runtime::MemrefView{primitive_type, data.opaque(), dimensions};
  }
};

//===----------------------------------------------------------------------===//
// Arguments decoding
//===----------------------------------------------------------------------===//

template <>
struct ArgDecoding<Buffer> {
  static std::optional<Buffer> Decode(XLA_FFI_ArgType type, void* arg) {
    if (type != XLA_FFI_ArgType_BUFFER) return std::nullopt;
    auto* buf = reinterpret_cast<XLA_FFI_Buffer*>(arg);

    Buffer buffer;
    buffer.primitive_type = PrimitiveType(buf->primitive_type);
    buffer.data = se::DeviceMemoryBase(buf->data);
    buffer.dimensions = absl::MakeConstSpan(buf->dims, buf->rank);
    return buffer;
  }
};

//===----------------------------------------------------------------------===//
// Context decoding
//===----------------------------------------------------------------------===//

template <>
struct CtxDecoding<ServiceExecutableRunOptions> {
  using Type = const ServiceExecutableRunOptions*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx) {
    void* ptr = api->internal_api->XLA_FFI_ServiceExecutableRunOptions_Get(ctx);
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

//===----------------------------------------------------------------------===//
// Result encoding
//===----------------------------------------------------------------------===//

// Takes ownership of the XLA FFI error and returns underlying status. Frees
// `error` if it's not nullptr; returns OK status otherwise.
Status TakeStatus(XLA_FFI_Error* error);

struct CallOptions {
  const ServiceExecutableRunOptions* run_options = nullptr;
};

Status Call(Ffi& handler, CallFrame& call_frame,
            const CallOptions& options = {});

Status Call(XLA_FFI_Handler* handler, CallFrame& call_frame,
            const CallOptions& options = {});

//===----------------------------------------------------------------------===//
// XLA FFI registry
//===----------------------------------------------------------------------===//

// Returns registered FFI handler for a given name, or an error if it's not
// found in the static registry.
StatusOr<XLA_FFI_Handler*> FindHandler(std::string_view name);

//===----------------------------------------------------------------------===//
// XLA FFI Api Implementation
//===----------------------------------------------------------------------===//

XLA_FFI_Api* GetXlaFfiApi();

}  // namespace xla::ffi

#endif  // XLA_FFI_FFI_H_
