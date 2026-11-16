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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_TEST_UTILS_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_TEST_UTILS_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/custom_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// Helper for thunk unit tests that owns per-execution lifecycle state for a
// single simulated execution (such as `CommandBufferRequests`,
// `ExecutionScopedState`, and `ServiceExecutableRunOptions`) and builds
// `Thunk::PrepareParams`, `Thunk::InitializeParams`, and
// `Thunk::ExecuteParams` without repetitive `nullptr` arguments.
//
// Lifetime contract: `PrepareParams()`, `InitializeParams()`, and
// `ExecuteParams()` return parameter structs holding pointers into `*this` (as
// well as any caller-supplied pointers passed to `With*()`). The
// `TestThunkParams` instance must outlive all returned parameter structs and
// any `Prepare`, `Initialize`, `Record`, or `ExecuteOnStream` calls using them;
// rvalue overloads are deleted to reject temporary misuse at compile time.
class TestThunkParams {
 public:
  TestThunkParams() = default;
  TestThunkParams(se::StreamExecutor* executor,
                  const BufferAllocations* buffer_allocations)
      : executor_(executor), buffer_allocations_(buffer_allocations) {}
  TestThunkParams(se::StreamExecutor* executor,
                  const BufferAllocations* buffer_allocations,
                  se::Stream* stream)
      : executor_(executor),
        buffer_allocations_(buffer_allocations),
        stream_(stream) {}

  TestThunkParams(const TestThunkParams&) = delete;
  TestThunkParams& operator=(const TestThunkParams&) = delete;

  TestThunkParams& WithExecutor(se::StreamExecutor* executor) & {
    executor_ = executor;
    return *this;
  }
  // Sets the buffer allocations and resets the owned `CollectiveMemoryRequests`
  // when the pointer changes (invalidating any references or `PrepareParams`
  // previously obtained from the owned `CollectiveMemoryRequests`). Note:
  // `InitializeParams()` forwards `buffer_allocations` directly (allowing
  // `nullptr` when initialization does not require allocations), whereas
  // `PrepareParams()` and `ExecuteParams()` substitute an empty default
  // `BufferAllocations` when `nullptr` to satisfy their non-null contracts.
  TestThunkParams& WithBufferAllocations(
      const BufferAllocations* buffer_allocations) & {
    if (buffer_allocations_ != buffer_allocations) {
      buffer_allocations_ = buffer_allocations;
      owned_memory_requests_.reset();
    }
    return *this;
  }
  TestThunkParams& WithStream(se::Stream* stream) & {
    stream_ = stream;
    return *this;
  }
  TestThunkParams& WithCommandBufferTraceStream(
      se::Stream* command_buffer_trace_stream) & {
    command_buffer_trace_stream_ = command_buffer_trace_stream;
    return *this;
  }
  TestThunkParams& WithCollectiveParams(CollectiveParams* collective_params) & {
    collective_params_ = collective_params;
    return *this;
  }
  TestThunkParams& WithCollectiveCliqueRequests(
      CollectiveCliqueRequests* collective_clique_requests) & {
    collective_clique_requests_ = collective_clique_requests;
    return *this;
  }
  TestThunkParams& WithCollectiveMemoryRequests(
      CollectiveMemoryRequests* collective_memory_requests) & {
    memory_requests_override_ = collective_memory_requests;
    return *this;
  }
  TestThunkParams& WithCollectiveCliques(
      CollectiveCliques* collective_cliques) & {
    collective_cliques_ = collective_cliques;
    return *this;
  }
  TestThunkParams& WithCollectiveMemory(CollectiveMemory* collective_memory) & {
    collective_memory_ = collective_memory;
    return *this;
  }
  TestThunkParams& WithCustomOptions(const CustomOptions* custom_options) & {
    custom_options_ = custom_options;
    return *this;
  }
  TestThunkParams& WithFfiExecutionContext(
      const ffi::ExecutionContext* ffi_execution_context) & {
    ffi_execution_context_ = ffi_execution_context;
    return *this;
  }
  TestThunkParams& WithRunOptions(
      const ServiceExecutableRunOptions* run_options) & {
    run_options_ = run_options != nullptr ? run_options : &owned_run_options_;
    return *this;
  }
  // Note: `Thunk::ExecutableSource::text` and `binary` hold borrowed views
  // whose backing storage must outlive any `InitializeParams()` / `Initialize`
  // calls using them (`dnn_compiled_graphs` is owned).
  TestThunkParams& WithExecutableSource(Thunk::ExecutableSource src) & {
    src_ = std::move(src);
    return *this;
  }
  TestThunkParams& WithLocalDeviceCount(int local_device_count) & {
    local_device_count_ = local_device_count;
    return *this;
  }
  TestThunkParams& WithExecutionScopedState(
      Thunk::ExecutionScopedState* execution_scoped_state) & {
    execution_scoped_state_ = execution_scoped_state;
    return *this;
  }
  TestThunkParams& WithCommandBufferRequests(
      CommandBufferRequests* command_buffer_requests) & {
    CHECK_NE(command_buffer_requests, nullptr);
    command_buffer_requests_ = command_buffer_requests;
    return *this;
  }
  TestThunkParams& WithPersistentAllocIndices(
      absl::Span<const BufferAllocation::Index> persistent_alloc_indices) & {
    persistent_alloc_indices_.emplace(persistent_alloc_indices.begin(),
                                      persistent_alloc_indices.end());
    return *this;
  }
  TestThunkParams& WithAdditionalComputeStreams(
      std::vector<se::Stream*> additional_compute_streams) & {
    additional_compute_streams_ = std::move(additional_compute_streams);
    return *this;
  }
  TestThunkParams& WithLocalityDomainStreams(
      absl::flat_hash_map<LocalityDomainStreamId, se::Stream*>
          locality_domain_streams) & {
    locality_domain_streams_ = std::move(locality_domain_streams);
    return *this;
  }

  CommandBufferRequests& command_buffer_requests() & {
    return *command_buffer_requests_;
  }
  const CommandBufferRequests& command_buffer_requests() const& {
    return *command_buffer_requests_;
  }
  const CommandBufferRequests& command_buffer_requests() const&& = delete;
  CollectiveCliqueRequests& collective_clique_requests() & {
    CHECK_NE(collective_clique_requests_, nullptr);
    return *collective_clique_requests_;
  }
  CollectiveMemoryRequests& collective_memory_requests() &;
  Thunk::ExecutionScopedState& execution_scoped_state() & {
    CHECK_NE(execution_scoped_state_, nullptr);
    return *execution_scoped_state_;
  }
  ServiceExecutableRunOptions* mutable_run_options() & {
    CHECK_EQ(run_options_, &owned_run_options_);
    return &owned_run_options_;
  }

  Thunk::PrepareParams PrepareParams() &;
  Thunk::PrepareParams PrepareParams() && = delete;

  Thunk::InitializeParams InitializeParams() &;
  Thunk::InitializeParams InitializeParams() && = delete;

  Thunk::ExecuteParams ExecuteParams() &;
  Thunk::ExecuteParams ExecuteParams() && = delete;

 private:
  const BufferAllocations* effective_buffer_allocations() const {
    return buffer_allocations_ != nullptr ? buffer_allocations_
                                          : &default_buffer_allocations_;
  }
  CollectiveMemoryRequests* effective_collective_memory_requests();
  std::optional<absl::Span<const BufferAllocation::Index>>
  effective_persistent_alloc_indices() const;

  se::StreamExecutor* executor_ = nullptr;
  const BufferAllocations* buffer_allocations_ = nullptr;
  BufferAllocations default_buffer_allocations_{
      /*buffers=*/{}, /*device_ordinal=*/0, /*memory_allocator=*/nullptr};
  se::Stream* stream_ = nullptr;
  se::Stream* command_buffer_trace_stream_ = nullptr;

  CollectiveParams* collective_params_ = nullptr;
  CollectiveCliqueRequests owned_clique_requests_;
  CollectiveCliqueRequests* collective_clique_requests_ =
      &owned_clique_requests_;
  std::optional<CollectiveMemoryRequests> owned_memory_requests_;
  std::optional<CollectiveMemoryRequests*> memory_requests_override_;
  CollectiveCliques* collective_cliques_ = nullptr;
  CollectiveMemory* collective_memory_ = nullptr;

  std::optional<const CustomOptions*> custom_options_;
  std::optional<const ffi::ExecutionContext*> ffi_execution_context_;
  ServiceExecutableRunOptions owned_run_options_;
  const ServiceExecutableRunOptions* run_options_ = &owned_run_options_;
  Thunk::ExecutableSource src_;
  std::optional<int> local_device_count_;

  Thunk::ExecutionScopedState owned_execution_scoped_state_;
  Thunk::ExecutionScopedState* execution_scoped_state_ =
      &owned_execution_scoped_state_;
  CommandBufferRequests owned_command_buffer_requests_;
  CommandBufferRequests* command_buffer_requests_ =
      &owned_command_buffer_requests_;

  std::optional<std::vector<BufferAllocation::Index>> persistent_alloc_indices_;
  std::vector<se::Stream*> additional_compute_streams_;
  absl::flat_hash_map<LocalityDomainStreamId, se::Stream*>
      locality_domain_streams_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_TEST_UTILS_H_
