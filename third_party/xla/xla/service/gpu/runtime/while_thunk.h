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

#ifndef XLA_SERVICE_GPU_RUNTIME_WHILE_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_WHILE_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// WhileThunk implements the while instruction on GPU by invoking a thunk
// sequence for the while 'condition' computation, and (conditionally) another
// thunk sequence for the while 'body' computation. WhileThunk assumes that
// buffers for the following set of while-related instructions share the same
// allocation:
//   init, condition.parameter, body.parameter, body.root, while.result
//
// WhileThunk synchronizes the stream to test the result of the 'condition'
// computation.
//
// If `trip_count` is available it means that the while loop trip count is known
// statically and while loop is actually a for loop, and in this case at run
// time condition thunk might not be executed and instead body thunk will be
// executed for `trip_count` times.
class WhileThunk : public Thunk {
 public:
  // Constructs a WhileThunk to compute while instruction 'hlo'.
  WhileThunk(ThunkInfo thunk_info,
             const BufferAllocation::Slice& condition_result_buffer_index,
             std::unique_ptr<ThunkSequence> condition_thunk_sequence,
             std::unique_ptr<ThunkSequence> body_thunk_sequence,
             std::optional<int64_t> trip_count = std::nullopt);
  WhileThunk(const WhileThunk&) = delete;
  WhileThunk& operator=(const WhileThunk&) = delete;

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  SequentialThunk* condition_thunk_sequence() const {
    return condition_thunk_sequence_.get();
  }

  SequentialThunk* body_thunk_sequence() const {
    return body_thunk_sequence_.get();
  }

  const BufferAllocation::Slice& condition_result_buffer() const {
    return condition_result_buffer_index_;
  }

  // Returns the current loop iteration if the caller is inside a while loop(s).
  //
  // Implementation relies on thread local storage, be careful when call it from
  // code running on multiple threads.
  static absl::StatusOr<int64_t> CurrentLoopIteration(int64_t depth = 0);

 private:
  const BufferAllocation::Slice condition_result_buffer_index_;
  std::unique_ptr<SequentialThunk> condition_thunk_sequence_;
  std::unique_ptr<SequentialThunk> body_thunk_sequence_;
  std::optional<int64_t> trip_count_;

  // Pinned host memory for transfering predicate value from device to host.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      predicates_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_WHILE_THUNK_H_
