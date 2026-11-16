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

#include "xla/backends/gpu/runtime/thunk_test_utils.h"

#include <optional>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {

CollectiveMemoryRequests*
TestThunkParams::effective_collective_memory_requests() {
  if (memory_requests_override_.has_value()) {
    return *memory_requests_override_;
  }
  if (!owned_memory_requests_.has_value()) {
    owned_memory_requests_.emplace(*effective_buffer_allocations());
  }
  return &*owned_memory_requests_;
}

std::optional<absl::Span<const BufferAllocation::Index>>
TestThunkParams::effective_persistent_alloc_indices() const {
  if (!persistent_alloc_indices_.has_value()) {
    return std::nullopt;
  }
  return absl::MakeConstSpan(*persistent_alloc_indices_);
}

CollectiveMemoryRequests& TestThunkParams::collective_memory_requests() & {
  CollectiveMemoryRequests* reqs = effective_collective_memory_requests();
  CHECK_NE(reqs, nullptr);
  return *reqs;
}

Thunk::PrepareParams TestThunkParams::PrepareParams() & {
  CHECK_NE(executor_, nullptr);
  return Thunk::PrepareParams(
      collective_params_, collective_clique_requests_,
      effective_collective_memory_requests(), executor_,
      effective_buffer_allocations(), execution_scoped_state_,
      custom_options_.value_or(run_options_->run_options().custom_options()),
      command_buffer_requests_);
}

Thunk::InitializeParams TestThunkParams::InitializeParams() & {
  return Thunk::InitializeParams(
      executor_, src_, buffer_allocations_, stream_,
      command_buffer_trace_stream_, collective_params_, collective_cliques_,
      collective_memory_,
      custom_options_.value_or(run_options_->run_options().custom_options()),
      ffi_execution_context_.value_or(
          run_options_->run_options().ffi_execution_context()),
      local_device_count_.value_or(run_options_->local_device_count()),
      execution_scoped_state_, effective_persistent_alloc_indices(),
      command_buffer_requests_);
}

Thunk::ExecuteParams TestThunkParams::ExecuteParams() & {
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      *run_options_, *effective_buffer_allocations(), stream_,
      command_buffer_trace_stream_, collective_params_, collective_cliques_,
      collective_memory_, additional_compute_streams_, execution_scoped_state_,
      effective_persistent_alloc_indices(), locality_domain_streams_);
  if (custom_options_.has_value()) {
    params.custom_options = *custom_options_;
  }
  if (ffi_execution_context_.has_value()) {
    params.ffi_execution_context = *ffi_execution_context_;
  }
  return params;
}

}  // namespace xla::gpu
