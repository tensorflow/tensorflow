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

#include "xla/service/gpu/runtime/while_thunk.h"

#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

static std::list<int64_t>& LoopCounters() {
  // TODO(b/343294327): Do not rely on thread-local storage.
  static thread_local std::list<int64_t> loop_counters;
  return loop_counters;
}

absl::StatusOr<int64_t> WhileThunk::CurrentLoopIteration(int64_t depth) {
  if (depth >= LoopCounters().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Loop depth %d is greater than the number of tracked loops %d", depth,
        LoopCounters().size()));
  }

  auto counter = LoopCounters().begin();
  std::advance(counter, depth);
  return *counter;
}

WhileThunk::WhileThunk(
    ThunkInfo thunk_info,
    const BufferAllocation::Slice& condition_result_buffer_index,
    std::unique_ptr<ThunkSequence> condition_thunk_sequence,
    std::unique_ptr<ThunkSequence> body_thunk_sequence,
    std::optional<int64_t> trip_count)
    : Thunk(Kind::kWhile, thunk_info),
      condition_result_buffer_index_(condition_result_buffer_index),
      condition_thunk_sequence_(std::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*condition_thunk_sequence))),
      body_thunk_sequence_(std::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*body_thunk_sequence))),
      trip_count_(trip_count) {}

absl::Status WhileThunk::Prepare(const PrepareParams& params,
                                 ResourceRequests& resource_requests) {
  TF_RETURN_IF_ERROR(
      condition_thunk_sequence_->Prepare(params, resource_requests));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Prepare(params, resource_requests));
  return absl::OkStatus();
}

absl::Status WhileThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(condition_thunk_sequence_->Initialize(params));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(params));

  absl::MutexLock lock(&mutex_);
  if (auto it = predicates_.find(params.executor); it == predicates_.end()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> allocation,
                        params.executor->HostMemoryAllocate(sizeof(bool)));
    predicates_.emplace(params.executor, std::move(allocation));
  }

  return absl::OkStatus();
}

absl::Status WhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;

  int64_t& iter = LoopCounters().emplace_front();
  absl::Cleanup cleanup = [&] { LoopCounters().pop_front(); };

  se::DeviceMemoryBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  if (trip_count_.has_value()) {
    VLOG(2) << "Executing WhileThunk for " << *trip_count_ << " iterations";
    for (iter = 0; iter < trip_count_; ++iter) {
      VLOG(3) << "Executing iteration # " << iter;
      TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
    }
    return absl::OkStatus();
  }

  // Get memory allocation for copying condition result from device.
  bool* condition_result = [&] {
    absl::MutexLock lock(&mutex_);
    return reinterpret_cast<bool*>(predicates_.at(stream.parent())->opaque());
  }();

  while (true) {
    VLOG(3) << "Executing WhileThunk condition computation; iter=" << iter;
    TF_RETURN_IF_ERROR(condition_thunk_sequence_->ExecuteOnStream(params));

    // Copy the result of condition computation and break the loop if 'false'.
    TF_RETURN_IF_ERROR(
        stream.Memcpy(condition_result, condition_result_data, sizeof(bool)));

    if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
      return absl::InternalError(absl::StrFormat(
          "Failed to complete all kernels launched on stream %p: %s", &stream,
          blocked.message()));
    }

    VLOG(3) << "condition_result = " << *condition_result;
    if (!*condition_result) {
      VLOG(3) << "Break WhileThunk loop; iter=" << iter;
      break;
    }

    VLOG(3) << "Executing WhileThunk body computation; iter=" << iter;
    TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
    ++iter;
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
