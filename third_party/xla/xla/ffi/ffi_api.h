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

#ifndef XLA_FFI_FFI_API_H_
#define XLA_FFI_FFI_API_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::ffi {

// This is an implementation of XLA FFI API defined in `api/c_api.h` header. It
// should be linked statically into the "main" XLA binary, and third party FFI
// handlers can be linked and registered dynamically.
//
// FFI handlers registered statically (and built from the same XLA commit with
// the same toolchain) can also use `api/c_api_internal.h` to get access to
// various internal data structures.

//===----------------------------------------------------------------------===//
// Calling XLA FFI handlers
//===----------------------------------------------------------------------===//

struct CallOptions {
  int32_t device_ordinal = -1;

  se::Stream* stream = nullptr;
  se::DeviceMemoryAllocator* allocator = nullptr;

  const HloComputation* called_computation = nullptr;
  const ExecutionContext* execution_context = nullptr;
  ExecutionState* execution_state = nullptr;
};

// Takes ownership of the XLA FFI error and returns underlying status. Frees
// `error` if it's not nullptr; returns OK status otherwise.
absl::Status TakeStatus(XLA_FFI_Error* error);

absl::Status CallWithApi(const XLA_FFI_Api* api, Ffi& handler,
                         CallFrame& call_frame, const CallOptions& options = {},
                         ExecutionStage stage = ExecutionStage::kExecute);

absl::Status Call(Ffi& handler, CallFrame& call_frame,
                  const CallOptions& options = {},
                  ExecutionStage stage = ExecutionStage::kExecute);

absl::Status Call(
    XLA_FFI_Handler* handler, CallFrame& call_frame,
    const CallOptions& options = {},
    XLA_FFI_ExecutionStage stage = XLA_FFI_ExecutionStage_EXECUTE);

namespace internal {
// This is an internal workaround to override FFI execution context for FFI
// calls executed in the current thread with `context` in tests that use legacy
// xla::Client, xla::Service and xla::Backend APIs because it's not worth it to
// add proper execution context support throughout all abstraction layers
// (legacy client APIs should be eventually deleted instead). This workaround
// should not be used outside of tests.
class ScopedExecutionContext {
 public:
  explicit ScopedExecutionContext(const ExecutionContext* context);
  ~ScopedExecutionContext();

  ScopedExecutionContext(ScopedExecutionContext&&) = delete;
  ScopedExecutionContext& operator=(ScopedExecutionContext&&) = delete;

  // Returns an execution context that should be used for FFI calls based on the
  // call options and the current thread's execution context.
  static const ExecutionContext* GetCallExecutionContext(
      const CallOptions& options);

 private:
  const ExecutionContext* recover_;
};
}  // namespace internal

//===----------------------------------------------------------------------===//
// XLA FFI registry
//===----------------------------------------------------------------------===//

struct HandlerRegistration {
  XLA_FFI_Handler_Bundle bundle = {};
  XLA_FFI_Handler_Traits traits = {};
};

bool IsCommandBufferCompatible(XLA_FFI_Handler_Traits traits);

// Returns registered FFI handler for a given name and platform, or an error if
// it's not found in the static registry.
absl::StatusOr<HandlerRegistration> FindHandler(std::string_view name,
                                                std::string_view platform);

// Returns all registered calls in the static registry for a given platform.
absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(std::string_view platform);

//===----------------------------------------------------------------------===//
// XLA FFI Api Implementation
//===----------------------------------------------------------------------===//

const XLA_FFI_Api* GetXlaFfiApi();

}  // namespace xla::ffi

#endif  // XLA_FFI_FFI_API_H_
