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

#include "xla/backends/gpu/runtime/while_thunk.h"

#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

struct RunningLoop {
  const HloInstruction* loop_instr;
  int64_t counter;
};

static std::list<RunningLoop>& RunningLoops() {
  // TODO(b/343294327): Do not rely on thread-local storage.
  static thread_local std::list<RunningLoop> loops;
  return loops;
}

absl::StatusOr<int64_t> WhileThunk::CurrentLoopIteration(int64_t depth) {
  if (depth >= RunningLoops().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Loop depth %d is greater than the number of tracked loops %d", depth,
        RunningLoops().size()));
  }

  auto loop = RunningLoops().begin();
  std::advance(loop, depth);
  return loop->counter;
}

absl::StatusOr<int64_t> WhileThunk::CurrentLoopIteration(
    const HloInstruction* while_instr) {
  for (const auto& loop : RunningLoops()) {
    if (loop.loop_instr == while_instr) {
      return loop.counter;
    }
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Loop %s is not currently running", while_instr->name()));
}

WhileThunk::WhileThunk(
    ThunkInfo thunk_info, const HloInstruction* loop,
    const BufferAllocation::Slice& condition_result_buffer_index,
    std::unique_ptr<SequentialThunk> condition_thunk_sequence,
    std::unique_ptr<SequentialThunk> body_thunk_sequence,
    std::optional<int64_t> trip_count)
    : Thunk(Kind::kWhile, thunk_info),
      loop_(loop),
      condition_result_buffer_index_(condition_result_buffer_index),
      condition_thunk_sequence_(std::move(condition_thunk_sequence)),
      body_thunk_sequence_(std::move(body_thunk_sequence)),
      trip_count_(trip_count) {}

absl::Status WhileThunk::Prepare(const PrepareParams& params,
                                 ResourceRequestsInterface& resource_requests) {
  TF_RETURN_IF_ERROR(
      condition_thunk_sequence_->Prepare(params, resource_requests));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Prepare(params, resource_requests));
  return absl::OkStatus();
}

absl::Status WhileThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(condition_thunk_sequence_->Initialize(params));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(params));

  absl::MutexLock lock(&mutex_);
  if (!host_memory_pools_.contains(params.executor)) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HostMemoryPool> pool,
        HostMemoryPool::Create(params.executor, PrimitiveType::PRED));
    host_memory_pools_[params.executor] = std::move(pool);
  }
  return absl::OkStatus();
}

absl::Status WhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;

  RunningLoop& loop = RunningLoops().emplace_front();
  loop.loop_instr = loop_;
  int64_t& iter = loop.counter;
  absl::Cleanup cleanup = [&] { RunningLoops().pop_front(); };

  if (trip_count_.has_value()) {
    VLOG(2) << "Executing WhileThunk for " << *trip_count_ << " iterations";
    for (iter = 0; iter < trip_count_; ++iter) {
      VLOG(3) << "Executing iteration # " << iter
              << " (Device: " << stream.parent()->device_ordinal() << ")";
      TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
    }
    return absl::OkStatus();
  }

  HostMemoryPool* pool;
  {
    absl::MutexLock lock(&mutex_);
    pool = host_memory_pools_.at(stream.parent()).get();
  }
  TF_ASSIGN_OR_RETURN(HostMemoryPool::Handle handle, pool->Acquire());
  bool* condition_result = handle.get<bool>();
  se::DeviceMemoryBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  while (true) {
    TraceMe trace(
        [&] { return TraceMeEncode("While", {{"iteration:", iter}}); });
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

    VLOG(3) << "Executing WhileThunk body computation; iter=" << iter
            << " (Device: " << stream.parent()->device_ordinal() << ")";
    TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
    ++iter;
  }
  return absl::OkStatus();
}

void WhileThunk::ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  condition_thunk_sequence_->ForAllThunks(fn);
  body_thunk_sequence_->ForAllThunks(fn);
}

std::string WhileThunk::ToString(int indent) const {
  std::string indent_str(indent * 2, ' ');
  std::string result;
  absl::StrAppend(&result, indent_str, "\ncondition:\n");
  absl::StrAppend(&result, condition_thunk_sequence_->ToString(indent + 1));
  absl::StrAppend(&result, indent_str, "body:\n");
  absl::StrAppend(&result, body_thunk_sequence_->ToString(indent + 1));
  return result;
}

absl::StatusOr<ThunkProto> WhileThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  auto* while_proto = proto.mutable_while_thunk();
  TF_ASSIGN_OR_RETURN(*while_proto->mutable_condition_result_buffer_index(),
                      condition_result_buffer_index_.ToProto());

  if (condition_thunk_sequence_) {
    TF_ASSIGN_OR_RETURN(ThunkProto thunk_proto,
                        condition_thunk_sequence_->ToProto());
    *while_proto->mutable_condition_thunk_sequence() =
        thunk_proto.sequential_thunk();
  }

  if (body_thunk_sequence_) {
    TF_ASSIGN_OR_RETURN(ThunkProto thunk_proto,
                        body_thunk_sequence_->ToProto());
    *while_proto->mutable_body_thunk_sequence() =
        thunk_proto.sequential_thunk();
  }

  if (trip_count_.has_value()) {
    while_proto->set_trip_count(*trip_count_);
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunk::FromProto(
    ThunkInfo thunk_info, const WhileThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const Deserializer& deserializer) {
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice condition_result_buffer_index,
      BufferAllocation::Slice::FromProto(
          thunk_proto.condition_result_buffer_index(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SequentialThunk> condition_thunk_sequence,
      SequentialThunk::FromProto(
          thunk_info, thunk_proto.condition_thunk_sequence(), deserializer));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SequentialThunk> body_thunk_sequence,
      SequentialThunk::FromProto(thunk_info, thunk_proto.body_thunk_sequence(),
                                 deserializer));
  std::optional<int64_t> trip_count;
  if (thunk_proto.has_trip_count()) {
    trip_count = thunk_proto.trip_count();
  }
  return std::make_unique<WhileThunk>(
      std::move(thunk_info), /*loop=*/nullptr, condition_result_buffer_index,
      std::move(condition_thunk_sequence), std::move(body_thunk_sequence),
      trip_count);
}

}  // namespace gpu
}  // namespace xla
