/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CONDITIONAL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CONDITIONAL_THUNK_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// ConditionalThunk implements the conditional instruction on GPU by reading the
// predicate of the conditional and executing the true or the false computation
// depending on the value of the predicate.
//
// ConditionalThunk assumes that the buffers of the conditional result and the
// result of the true and false computations share the same allocation. Also,
// the buffers of the true operand of the conditional and that of the parameter
// instruction of the true computation share the same allocation. Similarly, the
// buffers of the false operand and that of the parameter instruction of the
// false computation share the same allocation.
class ConditionalThunk : public Thunk {
 public:
  ConditionalThunk(
      ThunkInfo thunk_info,
      const BufferAllocation::Slice& branch_index_buffer_index,
      std::vector<std::unique_ptr<SequentialThunk>>&& branch_thunks,
      bool branch_index_is_bool);

  ConditionalThunk(const ConditionalThunk&) = delete;
  ConditionalThunk& operator=(const ConditionalThunk&) = delete;

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::Span<const std::unique_ptr<SequentialThunk>> branch_thunks() const {
    return branch_thunks_;
  }

  const BufferAllocation::Slice& branch_index_buffer() const {
    return branch_index_buffer_index_;
  }

  void ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const override;
  bool branch_index_is_bool() const { return branch_index_is_bool_; }

  absl::StatusOr<ThunkProto> ToProto() const override;

  // Deserializes a ConditionalThunk from its proto representation.
  // Parameters:
  // - thunk_info: Metadata about the thunk
  // - thunk_proto: Serialized ConditionalThunk proto message.
  // - buffer_allocations: Buffer allocations available for use by the thunk.
  // - deserializer: Callable (e.g., lambda) for deserializing nested thunks.
  //
  // Returns a unique_ptr to a ConditionalThunk on success, or an error status
  // on failure.
  static absl::StatusOr<std::unique_ptr<ConditionalThunk>> FromProto(
      ThunkInfo thunk_info, const ConditionalThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      const Deserializer& deserializer);

 private:
  const BufferAllocation::Slice branch_index_buffer_index_;
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks_;
  bool branch_index_is_bool_;

  // Host memory pool for transferring predicate value from device to host.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<HostMemoryPool>>
      host_memory_pools_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_CONDITIONAL_THUNK_H_
