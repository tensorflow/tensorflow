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

#include "xla/backends/gpu/runtime/copy_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

DeviceToDeviceCopyThunk::DeviceToDeviceCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {}

absl::Status DeviceToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_);
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_);
  VLOG(3) << "Memcpy D2D of size " << mem_size_ << " from "
          << source_data.opaque() << " to " << destination_data.opaque();
  return params.stream->Memcpy(&destination_data, source_data, mem_size_);
}

//===----------------------------------------------------------------------===//
// CopyThunk
//===----------------------------------------------------------------------===//

CopyThunk::CopyThunk(ThunkInfo thunk_info,
                     const BufferAllocation::Slice& source_buffer,
                     const BufferAllocation::Slice& destination_buffer,
                     uint64_t mem_size)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {}

absl::Status CopyThunk::ExecuteOnStream(const ExecuteParams& params) {
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// CopyAsyncEvents
//===----------------------------------------------------------------------===//

// Emplace() will insert {key, event} pair into the hash map,
// and return the event in order to do RecordEvent() for async memcpy.
absl::Status CopyThunk::AsyncEvents::Emplace(se::StreamExecutor* executor,
                                             const HloInstruction* instr,
                                             std::unique_ptr<se::Event> event) {
  Key key = {executor, instr};
  absl::MutexLock lock(&mutex_);
  VLOG(3) << "Emplace event " << event.get();
  if (auto [it, inserted] = events_.try_emplace(key, std::move(event));
      inserted) {
    return absl::OkStatus();
  }
  return absl::InternalError("Async copy event already exists!");
}

// Retrieve a completion event started by copy-start instruction
// `instr`, and remove the event from the collection.
absl::StatusOr<std::unique_ptr<se::Event>> CopyThunk::AsyncEvents::Extract(
    se::StreamExecutor* executor, const HloInstruction* instr) {
  Key key = {executor, instr};
  absl::MutexLock lock(&mutex_);
  if (auto event = events_.extract(key)) {
    VLOG(3) << "Extract event " << event.mapped().get();
    return std::move(event.mapped());
  }
  return absl::InternalError("Async copy event was not found!");
}

//===----------------------------------------------------------------------===//
// DeviceToHostCopyThunk
//===----------------------------------------------------------------------===//
DeviceToHostCopyThunk::DeviceToHostCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size,
    std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    const HloInstruction* instr)
    : CopyThunk(std::move(thunk_info), source_buffer, destination_buffer,
                mem_size),
      async_events_(std::move(async_events)),
      instr_(instr) {}

absl::Status DeviceToHostCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination());
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source());
  void* cpu_dst = destination_data.opaque();
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  TF_RETURN_IF_ERROR(stream->Memcpy(cpu_dst, source_data, size_bytes()));
  if (stream == params.stream) {
    VLOG(2) << "Memcpy D2H from the main stream";
    return absl::OkStatus();
  }
  VLOG(2) << "Memcpy D2H from the other stream";
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto event, executor->CreateEvent());
  // Record memcpy operation completion.
  TF_RETURN_IF_ERROR(stream->RecordEvent(event.get()));
  VLOG(3) << "Emplace events: " << event.get()
          << " for instr: " << instr_->ToString();
  return async_events_->Emplace(executor, instr_, std::move(event));
}

//===----------------------------------------------------------------------===//
// HostToDeviceCopyThunk
//===----------------------------------------------------------------------===//
HostToDeviceCopyThunk::HostToDeviceCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size,
    std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    const HloInstruction* instr)
    : CopyThunk(std::move(thunk_info), source_buffer, destination_buffer,
                mem_size),
      async_events_(std::move(async_events)),
      instr_(instr) {}

absl::Status HostToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination());
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source());
  void* cpu_src = source_data.opaque();
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  TF_RETURN_IF_ERROR(stream->Memcpy(&destination_data, cpu_src, size_bytes()));
  if (stream == params.stream) {
    VLOG(2) << "Memcpy H2D from the main stream";
    return absl::OkStatus();
  }
  VLOG(2) << "Memcpy H2D from the other stream";
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto event, executor->CreateEvent());
  // Record memcpy operation completion.
  TF_RETURN_IF_ERROR(stream->RecordEvent(event.get()));
  VLOG(3) << "Emplace events: " << event.get()
          << " for instr: " << instr_->ToString();
  return async_events_->Emplace(executor, instr_, std::move(event));
}

//===----------------------------------------------------------------------===//
// CopyDoneThunk
//===----------------------------------------------------------------------===//

CopyDoneThunk::CopyDoneThunk(
    Thunk::Kind kind, ThunkInfo thunk_info,
    std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    const HloInstruction* copy_start_instr)
    : Thunk(kind, std::move(thunk_info)),
      async_events_(std::move(async_events)),
      copy_start_instr_(copy_start_instr) {}

absl::Status CopyDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "CopyDone thunk between a host and a device for: "
          << copy_start_instr_->ToString();
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Event> event,
                      async_events_->Extract(executor, copy_start_instr_));
  return params.stream->WaitFor(event.get());
}

//===----------------------------------------------------------------------===//
// DynamicMemcpyThunk
//===----------------------------------------------------------------------===//

namespace {

absl::StatusOr<int64_t> EvaluateDynamicOffsets(
    HloEvaluator& evaluator,
    absl::Span<const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset>
        offsets) {
  int64_t offset_sum = 0;
  for (const auto& offset : offsets) {
    TF_ASSIGN_OR_RETURN(
        auto config,
        offset.while_loop->backend_config<xla::WhileLoopBackendConfig>());

    TF_RET_CHECK(config.has_known_init_step());
    TF_ASSIGN_OR_RETURN(int64_t iteration,
                        WhileThunk::CurrentLoopIteration(offset.while_loop));
    int64_t induction_variable = config.known_init_step().init() +
                                 iteration * config.known_init_step().step();

    Literal induction_variable_literal(offset.induction_variable->shape());
    TF_RETURN_IF_ERROR(
        induction_variable_literal.SetIntegralAsS64({}, induction_variable));
    TF_ASSIGN_OR_RETURN(
        Literal array_index_literal,
        evaluator.EvaluateWithSubstitutions(
            offset.offset,
            {{offset.induction_variable, &induction_variable_literal}}, true));

    std::optional<int64_t> array_index =
        LiteralUtil::LiteralAsScalarInt64(array_index_literal);
    if (!array_index) {
      return absl::InternalError("Failed to evaluate offset");
    }

    int64_t clamped_index =
        std::max<int64_t>(0, std::min(*array_index, offset.dimension_size - 1));
    VLOG(3) << "Iteration index " << induction_variable
            << " resulted in array index " << *array_index << ".";
    offset_sum += clamped_index * offset.byte_stride;
  }
  return offset_sum;
}

}  // namespace

DynamicMemcpyThunk::DynamicMemcpyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size,
    DynamicMemcpyThunk::MemcpyDescriptor descriptor)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size),
      descriptor_(descriptor) {}

absl::Status DynamicMemcpyThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::DeviceMemoryBase src_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_);
  se::DeviceMemoryBase dst_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_);

  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  TF_ASSIGN_OR_RETURN(
      int64_t src_offset,
      EvaluateDynamicOffsets(evaluator, descriptor_.src_dynamic_offsets));
  src_offset += descriptor_.src_byte_static_offset;

  TF_ASSIGN_OR_RETURN(
      int64_t dst_offset,
      EvaluateDynamicOffsets(evaluator, descriptor_.dst_dynamic_offsets));
  dst_offset += descriptor_.dst_byte_static_offset;

  auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
  auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);
  VLOG(3) << "Memcpy of size " << mem_size_ << " from "
          << src_with_offset.opaque() << " (offset " << src_offset << ") to "
          << dst_with_offset.opaque() << " (offset " << dst_offset << ")";
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  return stream->Memcpy(&dst_with_offset, src_with_offset, mem_size_);
}

}  // namespace gpu
}  // namespace xla
