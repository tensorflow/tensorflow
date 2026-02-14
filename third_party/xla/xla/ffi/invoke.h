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

#ifndef XLA_FFI_INVOKE_H_
#define XLA_FFI_INVOKE_H_

#include <cstdint>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/tsl/concurrency/chain.h"

// XLA FFI `Invoke` is an utility function that provides a way to call XLA FFI
// handlers with a given set of arguments and attributes packed into a call
// frame and with a given invocation context that contains pointers to the
// parameters that identify XLA program execution.

//===----------------------------------------------------------------------===//
// Forward declare backend-specific types.
//===----------------------------------------------------------------------===//

namespace xla {
class HloComputation;
}  // namespace xla

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace stream_executor {
class Stream;
class DeviceAddressAllocator;
class GpuComputeCapability;
}  // namespace stream_executor

namespace xla::gpu {
struct CollectiveParams;
class CollectiveCliqueRequests;
class CollectiveMemoryRequests;
class BarrierRequests;
class CollectiveCliques;
class CollectiveMemory;
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// Invoking XLA FFI handlers
//===----------------------------------------------------------------------===//

namespace xla::ffi {

// Context for invoking XLA FFI handlers. Backend specific context must be
// constructed from `xla::ExecuteRunOptions`, to give FFI handlers access to
// XLA runtime internals.
struct InvokeContext {
  struct CpuContext {
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  };

  struct GpuContext {
    stream_executor::Stream* stream = nullptr;
    stream_executor::DeviceAddressAllocator* allocator = nullptr;
    const gpu::CollectiveParams* collective_params = nullptr;
    gpu::CollectiveCliqueRequests* collective_clique_requests = nullptr;
    gpu::CollectiveMemoryRequests* collective_memory_requests = nullptr;
    gpu::BarrierRequests* barrier_requests = nullptr;
    const gpu::CollectiveCliques* collective_cliques = nullptr;
    const gpu::CollectiveMemory* collective_memory = nullptr;
    const stream_executor::GpuComputeCapability* compute_capability = nullptr;
  };

  using BackendContext = std::variant<std::monostate, CpuContext, GpuContext>;

  RunId run_id = RunId{-1};
  int32_t device_ordinal = -1;
  BackendContext backend_context;

  const HloComputation* called_computation = nullptr;
  const ExecutionContext* execution_context = nullptr;
  ExecutionState* execution_state = nullptr;
};

// Invokes an XLA FFI handler with the given call frame and context. This is a
// synchronous call and it might block the caller thread if the handler is
// asynchronous. It is unsafe to call if from a thread pool that runs tasks
// scheduled by the handler itself.
absl::Status Invoke(const XLA_FFI_Api* api, Ffi& handler, CallFrame& call_frame,
                    const InvokeContext& context = {},
                    ExecutionStage stage = ExecutionStage::kExecute);

absl::Status Invoke(
    const XLA_FFI_Api* api, XLA_FFI_Handler* handler, CallFrame& call_frame,
    const InvokeContext& context = {},
    XLA_FFI_ExecutionStage stage = XLA_FFI_ExecutionStage_EXECUTE);

// Invokes an XLA FFI handler with the given call frame and context. This is an
// asynchronous call and it will not block the caller thread. Returned async
// value will become available when the handler completes execution.
tsl::AsyncValueRef<tsl::Chain> InvokeAsync(
    const XLA_FFI_Api* api, Ffi& handler, CallFrame& call_frame,
    const InvokeContext& context = {},
    ExecutionStage stage = ExecutionStage::kExecute);

tsl::AsyncValueRef<tsl::Chain> InvokeAsync(
    const XLA_FFI_Api* api, XLA_FFI_Handler* handler, CallFrame& call_frame,
    const InvokeContext& context = {},
    XLA_FFI_ExecutionStage stage = XLA_FFI_ExecutionStage_EXECUTE);

// Gets metadata from the handler by invoking it with a special call frame.
absl::StatusOr<XLA_FFI_Metadata> GetMetadata(const XLA_FFI_Api* api,
                                             Ffi& handler);
absl::StatusOr<XLA_FFI_Metadata> GetMetadata(const XLA_FFI_Api* api,
                                             XLA_FFI_Handler* handler);

//===----------------------------------------------------------------------===//
// ScopedExecutionContext for tests and internal users.
//===----------------------------------------------------------------------===//

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
      const InvokeContext& context);

 private:
  const ExecutionContext* recover_;
};

}  // namespace internal
}  // namespace xla::ffi

#endif  // XLA_FFI_INVOKE_H_
