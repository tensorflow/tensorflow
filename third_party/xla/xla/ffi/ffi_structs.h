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

#ifndef XLA_FFI_FFI_STRUCTS_H_
#define XLA_FFI_FFI_STRUCTS_H_

#include <cstdint>
#include <variant>

#include "absl/status/status.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

//===----------------------------------------------------------------------===//
// Forward declare backend-specific types.
//===----------------------------------------------------------------------===//

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace stream_executor {
class Stream;
class DeviceAddressAllocator;
}  // namespace stream_executor

namespace xla::gpu {
struct CollectiveParams;
class CollectiveCliqueRequests;
class CollectiveMemoryRequests;
class CollectiveCliques;
class CollectiveMemory;
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// XLA FFI C structs definition
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error {
  absl::Status status;
};

struct XLA_FFI_Future {
  tsl::AsyncValueRef<tsl::Chain> async_value;
};

// This struct corresponds to `InvokeContext` available to XLA:FFI C++ clients,
// the the invoke context for documentation.
struct XLA_FFI_ExecutionContext {
  struct CpuContext {
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  };

  struct GpuContext {
    stream_executor::Stream* stream = nullptr;
    stream_executor::DeviceAddressAllocator* allocator = nullptr;
    const xla::gpu::CollectiveParams* collective_params = nullptr;
    xla::gpu::CollectiveCliqueRequests* collective_clique_requests = nullptr;
    xla::gpu::CollectiveMemoryRequests* collective_memory_requests = nullptr;
    const xla::gpu::CollectiveCliques* collective_cliques = nullptr;
    const xla::gpu::CollectiveMemory* collective_memory = nullptr;
    const stream_executor::GpuComputeCapability* gpu_compute_capability =
        nullptr;
  };

  using BackendContext = std::variant<std::monostate, CpuContext, GpuContext>;

  struct StateContext {
    xla::ffi::ExecutionState* instantiate = nullptr;
    xla::ffi::ExecutionState* prepare = nullptr;
    xla::ffi::ExecutionState* initialize = nullptr;
  };

  xla::RunId run_id = xla::RunId{0};
  int32_t device_ordinal = -1;

  BackendContext backend_context;
  StateContext state_context;

  const xla::HloComputation* called_computation = nullptr;
  const xla::ffi::ExecutionContext* execution_context = nullptr;
};

#endif  // XLA_FFI_FFI_STRUCTS_H_
