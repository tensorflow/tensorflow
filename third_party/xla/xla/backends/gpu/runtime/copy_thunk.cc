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

#include "absl/container/node_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

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

absl::StatusOr<ThunkProto> CopyThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  CopyThunkProto* copy_thunk_proto = proto.mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source().ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination().ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
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

absl::StatusOr<ThunkProto> DeviceToHostCopyThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  DeviceToHostCopyThunkProto* d2h_copy_thunk_proto =
      proto.mutable_device_to_host_copy_thunk();
  CopyThunkProto* copy_thunk_proto = d2h_copy_thunk_proto->mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source().ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination().ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
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

absl::StatusOr<ThunkProto> HostToDeviceCopyThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  HostToDeviceCopyThunkProto* h2d_copy_thunk_proto =
      proto.mutable_host_to_device_copy_thunk();
  CopyThunkProto* copy_thunk_proto = h2d_copy_thunk_proto->mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source().ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination().ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
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

class DynamicOffsetEvaluator {
 public:
  // Evaluates the clamped array index for the given offset.
  absl::StatusOr<int64_t> EvaluateArrayIndexForOffset(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset) {
    TF_ASSIGN_OR_RETURN(auto call_stack, ComputeCallStack(offset));

    // Walk up the call stack and compute the required parameter's values at
    // each step, using them as the substitutions for the next call. By
    // definition, the first call can only depend on the induction variable.
    TF_ASSIGN_OR_RETURN(auto substitutions,
                        GetInductionVariableSubstitutions(offset));
    HloEvaluator evaluator(/*max_loop_iterations=*/0);
    for (auto it = call_stack.rbegin(), e = call_stack.rend(); it != e; ++it) {
      const HloInstruction* caller = *it;
      VLOG(3) << "Evaluating required operands of caller " << caller->name()
              << ".";
      if (VLOG_IS_ON(4)) {
        VLOG(4) << "Current substitutions:";
        for (auto [instr, value] : substitutions) {
          VLOG(4) << "  " << instr->name() << " -> " << value->ToString();
        }
      }
      absl::flat_hash_map<const HloInstruction*, const LiteralBase*>
          next_substitutions;
      for (auto [parameter, operand] :
           GetRequiredParametersAndOperands(offset, caller)) {
        // Only compute the value if we didn't already need it for a different
        // offset.
        if (!known_values_.contains(operand)) {
          TF_ASSIGN_OR_RETURN(
              known_values_[operand],
              evaluator.Evaluate(operand, {}, true, substitutions));
        }
        next_substitutions[parameter] = &known_values_[operand];
      }

      std::swap(substitutions, next_substitutions);
    }

    // We now have the parameter values for the innermost call, so we can
    // compute the offset.
    TF_ASSIGN_OR_RETURN(
        auto array_index_literal,
        evaluator.Evaluate(offset.offset, {}, true, substitutions));

    std::optional<int64_t> array_index =
        LiteralUtil::LiteralAsScalarInt64(array_index_literal);
    if (!array_index) {
      return absl::InternalError("Failed to evaluate offset");
    }

    int64_t clamped_index =
        std::max<int64_t>(0, std::min(*array_index, offset.dimension_size - 1));
    VLOG(3) << "Computed dynamic array index " << clamped_index << ".";

    return clamped_index;
  }

 private:
  // Computes the call stack between `offset`'s while loop and the derived
  // value. Typically, there will be up to three items in the stack: 1) a
  // fusion, 2) optionally an async-start, 3) optionally a command buffer. The
  // while loop instruction is not included.
  static absl::StatusOr<absl::InlinedVector<HloInstruction*, 4>>
  ComputeCallStack(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset) {
    VLOG(3) << "Computing call stack for " << offset.offset->name() << ".";
    const HloComputation* current_computation = offset.offset->parent();
    const HloComputation* while_body = offset.induction_variable->parent();

    absl::InlinedVector<HloInstruction*, 4> call_stack;
    while (current_computation && current_computation != while_body) {
      VLOG(3) << "Current computation: " << current_computation->name() << ".";
      auto callers = current_computation->caller_instructions();

      // If there isn't a single caller, the thunk was not constructed
      // correctly.
      TF_RET_CHECK(callers.size() == 1);

      call_stack.push_back(callers.front());
      current_computation = callers.front()->parent();
    }

    // If we didn't arrive at the while body, the thunk was not constructed
    // correctly.
    TF_RET_CHECK(current_computation == while_body);
    return call_stack;
  }

  // Returns the pairs of {computation parameter, computation caller operand}
  // that are required in the given computation to compute the given offset.
  static absl::InlinedVector<
      std::pair<const HloInstruction*, const HloInstruction*>, 1>
  GetRequiredParametersAndOperands(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset,
      const HloInstruction* caller) {
    absl::InlinedVector<std::pair<const HloInstruction*, const HloInstruction*>,
                        1>
        result;
    const HloComputation* callee = caller->called_computations().front();
    if (auto maybe_required = offset.required_parameters.find(callee);
        maybe_required != offset.required_parameters.end()) {
      const auto& required_parameters = maybe_required->second;
      for (int i = 0; i < required_parameters.size(); ++i) {
        if (required_parameters[i]) {
          result.push_back(
              {callee->parameter_instruction(i), caller->operand(i)});
        }
      }
    }
    return result;
  }

  absl::StatusOr<absl::flat_hash_map<const HloInstruction*, const LiteralBase*>>
  GetInductionVariableSubstitutions(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset) {
    // Set the value of the induction variable, if it's not known yet.
    if (!known_values_.contains(offset.induction_variable)) {
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
      known_values_[offset.induction_variable] =
          std::move(induction_variable_literal);
    }

    return {{{offset.induction_variable,
              &known_values_.at(offset.induction_variable)}}};
  }

  absl::node_hash_map<const HloInstruction*, Literal> known_values_;
};

absl::StatusOr<int64_t> EvaluateDynamicOffsets(
    absl::Span<const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset>
        offsets) {
  int64_t offset_sum = 0;
  DynamicOffsetEvaluator evaluator;
  for (const auto& offset : offsets) {
    TF_ASSIGN_OR_RETURN(int64_t clamped_index,
                        evaluator.EvaluateArrayIndexForOffset(offset));
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

  TF_ASSIGN_OR_RETURN(int64_t src_offset,
                      EvaluateDynamicOffsets(descriptor_.src_dynamic_offsets));
  src_offset += descriptor_.src_byte_static_offset;

  TF_ASSIGN_OR_RETURN(int64_t dst_offset,
                      EvaluateDynamicOffsets(descriptor_.dst_dynamic_offsets));
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
