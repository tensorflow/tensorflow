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

#include "xla/backends/gpu/runtime/host_execute_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/host_offloading/gpu_host_offloading_allocator.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/host_offloading/host_offloading_allocator.h"
#include "xla/core/host_offloading/host_offloading_buffer.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/core/host_offloading/host_offloading_nanort_executable.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

namespace {

tsl::thread::ThreadPool* GetHostExecuteThreadPool() {
  constexpr int kMaxNumHostExecuteThreads = 32;
  static tsl::thread::ThreadPool* host_offloading_thread_pool =
      new tsl::thread::ThreadPool(
          tsl::Env::Default(), "host-offloading",
          std::min(tsl::port::MaxParallelism(), kMaxNumHostExecuteThreads));
  return host_offloading_thread_pool;
}

bool IsBufferOnDevice(se::Stream* stream, const void* ptr) {
  auto memory_type = stream->parent()->GetPointerMemorySpace(ptr);
  return memory_type.ok() && *memory_type == se::MemoryType::kDevice;
}

class HostExecuteCallFrame {
 public:
  static absl::StatusOr<HostExecuteCallFrame> Create(
      se::Stream* device_to_host_stream, se::Stream* host_to_device_stream,
      const BufferAllocations* buffer_allocations,
      HostOffloadingAllocator& allocator,
      absl::Span<HostExecuteStartThunk::SliceAndShape> args,
      absl::Span<HostExecuteStartThunk::SliceAndShape> result_slices,
      const ProgramShape& program_shape, int64_t gpu_pointer_size);

  absl::Span<const ShapeTree<HostOffloadingBuffer>> parameters() const {
    return parameters_;
  }
  const ShapeTree<HostOffloadingBuffer>& result() const { return result_; }

  absl::Status PublishResult() &&;

 protected:
  HostExecuteCallFrame(
      se::Stream* host_to_device_stream,
      const BufferAllocations* buffer_allocations,
      std::vector<ShapeTree<HostOffloadingBuffer>> parameters,
      ShapeTree<HostOffloadingBuffer> result,
      absl::Span<HostExecuteStartThunk::SliceAndShape> result_slices,
      std::vector<std::unique_ptr<HostOffloadingAllocator::Buffer>> buffers,
      int64_t gpu_pointer_size);

  static absl::Status ValidateArgsAndResults(
      absl::Span<const HostExecuteStartThunk::SliceAndShape> args,
      absl::Span<const HostExecuteStartThunk::SliceAndShape> result_slices,
      const ProgramShape& program_shape);

 private:
  se::Stream* host_to_device_stream_;
  const BufferAllocations* buffer_allocations_;
  std::vector<ShapeTree<HostOffloadingBuffer>> parameters_;
  ShapeTree<HostOffloadingBuffer> result_;

  // This is where results will be published to.
  absl::Span<HostExecuteStartThunk::SliceAndShape> result_slices_;

  // Allocatations used for parameters and results.
  std::vector<std::unique_ptr<HostOffloadingAllocator::Buffer>>
      allocated_buffers_;

  int64_t gpu_pointer_size_;
};

absl::Status HostExecuteCallFrame::ValidateArgsAndResults(
    absl::Span<const HostExecuteStartThunk::SliceAndShape> args,
    absl::Span<const HostExecuteStartThunk::SliceAndShape> result_slices,
    const ProgramShape& program_shape) {
  if (args.size() != program_shape.parameters_size()) {
    return InvalidArgument("Number of arguments does not match program shape.");
  }

  for (int i = 0; i < args.size(); ++i) {
    if (args[i].shape != program_shape.parameters(i)) {
      return InvalidArgument(
          "Argument shape %s does not match program shape %s.",
          args[i].shape.ToString(/*print_layout=*/true),
          program_shape.parameters(i).ToString(/*print_layout=*/true));
    }
  }

  auto program_result_shape = program_shape.result();

  // Iterate through result slices and compare their shapes with the program
  // result shape. If a result slice has a tuple shape, we iterate through the
  // tuple shapes and compare them with the program result shape.
  size_t result_slice_index = 0;
  size_t slice_tuple_index = 0;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      program_result_shape,
      [&result_slices, &slice_tuple_index, &result_slice_index](
          const Shape& expected_shape,
          const ShapeIndex& index) -> absl::Status {
        auto compare_shapes_with_status =
            [](const Shape& shape1, const Shape& shape2) -> absl::Status {
          if (!ShapeUtil::Equal(shape1, shape2)) {
            return InvalidArgument(
                "Result slice shape %s does not match expected program result "
                "shape %s.",
                shape1.ToString(/*print_layout=*/true),
                shape2.ToString(/*print_layout=*/true));
          }
          return absl::OkStatus();
        };

        const auto& [slice, slice_shape] = result_slices[result_slice_index];
        if (slice_shape.IsTuple()) {
          TF_RETURN_IF_ERROR(compare_shapes_with_status(
              slice_shape.tuple_shapes(slice_tuple_index++), expected_shape));
          if (slice_tuple_index == slice_shape.tuple_shapes().size()) {
            slice_tuple_index = 0;
          }
        } else {
          TF_RETURN_IF_ERROR(
              compare_shapes_with_status(slice_shape, expected_shape));
        }

        if (slice_tuple_index == 0) {
          result_slice_index++;
        }

        return absl::OkStatus();
      }));

  return absl::OkStatus();
}

absl::StatusOr<HostExecuteCallFrame> HostExecuteCallFrame::Create(
    se::Stream* device_to_host_stream, se::Stream* host_to_device_stream,
    const BufferAllocations* buffer_allocations,
    HostOffloadingAllocator& allocator,
    absl::Span<HostExecuteStartThunk::SliceAndShape> args,
    absl::Span<HostExecuteStartThunk::SliceAndShape> result_slices,
    const ProgramShape& program_shape, int64_t gpu_pointer_size) {
  TF_RETURN_IF_ERROR(
      ValidateArgsAndResults(args, result_slices, program_shape));

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters;
  std::vector<std::unique_ptr<HostOffloadingAllocator::Buffer>> buffers;
  parameters.reserve(args.size());
  buffers.reserve(args.size() + result_slices.size());

  for (const auto& [slice, shape] : args) {
    auto buffer_allocation = buffer_allocations->GetDeviceAddress(slice);
    if (IsBufferOnDevice(device_to_host_stream, buffer_allocation.opaque())) {
      // Copy device memory to host memory.
      TF_ASSIGN_OR_RETURN(
          buffers.emplace_back(),
          allocator.AllocateTransferBuffer(ShapeUtil::ByteSizeOf(shape)));

      parameters.push_back(ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(buffers.back()->untyped_data(),
                                      buffers.back()->size_bytes())));

      TF_RETURN_IF_ERROR(device_to_host_stream->Memcpy(
          buffers.back()->untyped_data(), buffer_allocation,
          buffers.back()->size_bytes()));
    } else {
      // We don't allocate as buffer is already in host memory.
      parameters.push_back(ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(buffer_allocation.opaque(),
                                      buffer_allocation.size())));
    }
  }

  ShapeTree<HostOffloadingBuffer> result(program_shape.result());

  size_t result_slice_index = 0;
  // We want to consume the entire slice before moving onto the next one.
  size_t current_slice_offset = 0;

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      result.shape(),
      [&result, &result_slice_index, &current_slice_offset, &result_slices,
       &buffer_allocations, &host_to_device_stream, &buffers, &allocator,
       &gpu_pointer_size](const Shape& shape,
                          const ShapeIndex& index) -> absl::Status {
        const auto& [slice, _] = result_slices[result_slice_index];
        auto buffer_allocation = buffer_allocations->GetDeviceAddress(slice);

        void* buffer_ptr = static_cast<uint8_t*>(buffer_allocation.opaque()) +
                           current_slice_offset;
        const int64_t buffer_size = ShapeUtil::ByteSizeOf(shape);
        current_slice_offset +=
            std::max(buffer_size, std::min(slice.size(), gpu_pointer_size));

        // If we've consumed the entire slice, move onto the next entry in the
        // result.
        if (current_slice_offset == slice.size()) {
          current_slice_offset = 0;
          result_slice_index++;
        } else if (current_slice_offset > slice.size()) {
          return InvalidArgument(
              "Current slice offset %d went over the slice size %d.",
              current_slice_offset, slice.size());
        }

        if (IsBufferOnDevice(host_to_device_stream,
                             buffer_allocation.opaque())) {
          TF_ASSIGN_OR_RETURN(
              buffers.emplace_back(),
              allocator.AllocateTransferBuffer(ShapeUtil::ByteSizeOf(shape)));
          *result.mutable_element(index) = HostOffloadingBuffer(
              buffers.back()->untyped_data(), buffers.back()->size_bytes());
        } else {
          // We don't allocate as buffer is already in host memory.
          *result.mutable_element(index) =
              HostOffloadingBuffer(buffer_ptr, buffer_size);
        }

        return absl::OkStatus();
      }));

  return HostExecuteCallFrame(host_to_device_stream, buffer_allocations,
                              std::move(parameters), std::move(result),
                              std::move(result_slices), std::move(buffers),
                              gpu_pointer_size);
}

HostExecuteCallFrame::HostExecuteCallFrame(
    se::Stream* host_to_device_stream,
    const BufferAllocations* buffer_allocations,
    std::vector<ShapeTree<HostOffloadingBuffer>> parameters,
    ShapeTree<HostOffloadingBuffer> result,
    absl::Span<HostExecuteStartThunk::SliceAndShape> result_slices,
    std::vector<std::unique_ptr<HostOffloadingAllocator::Buffer>> buffers,
    int64_t gpu_pointer_size)
    : host_to_device_stream_(host_to_device_stream),
      buffer_allocations_(buffer_allocations),
      parameters_(std::move(parameters)),
      result_(std::move(result)),
      result_slices_(result_slices),
      allocated_buffers_(std::move(buffers)),
      gpu_pointer_size_(gpu_pointer_size) {}

absl::Status HostExecuteCallFrame::PublishResult() && {
  size_t result_slice_index = 0;
  size_t current_slice_offset = 0;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      result_.shape(),
      [this, &result_slice_index, &current_slice_offset](
          const Shape& shape, const ShapeIndex& index) -> absl::Status {
        const auto& [slice, _] = result_slices_[result_slice_index];
        auto buffer_allocation = buffer_allocations_->GetDeviceAddress(slice);

        void* buffer_ptr = static_cast<uint8_t*>(buffer_allocation.opaque()) +
                           current_slice_offset;
        const int64_t buffer_size = ShapeUtil::ByteSizeOf(shape);
        current_slice_offset +=
            std::max(buffer_size, std::min(slice.size(), gpu_pointer_size_));

        // If we've consumed the entire slice, move onto the next entry in the
        // result.
        if (current_slice_offset == slice.size()) {
          current_slice_offset = 0;
          result_slice_index++;
        } else if (current_slice_offset > slice.size()) {
          return InvalidArgument(
              "Current slice offset %d went over the slice size %d.",
              current_slice_offset, slice.size());
        }

        if (!IsBufferOnDevice(host_to_device_stream_,
                              buffer_allocation.opaque())) {
          // No need to copy result since the result is expected to be in host
          // memory and should match the buffer used for execution.
          CHECK(result_.element(index).opaque_base() == buffer_ptr);
          return absl::OkStatus();
        }

        se::DeviceMemoryBase result_device_memory(buffer_ptr, buffer_size);

        TF_RETURN_IF_ERROR(host_to_device_stream_->Memcpy(
            &result_device_memory, result_.element(index).opaque_base(),
            result_.element(index).size_in_bytes()));

        return absl::OkStatus();
      }));

  // Move the backing buffers (allocated_buffers_) to the callback to ensure
  // that they are only destroyed after the memory copies are done.
  TF_RETURN_IF_ERROR(host_to_device_stream_->DoHostCallbackWithStatus(
      [buffers = std::move(allocated_buffers_)]() {
        return absl::OkStatus();
      }));

  return absl::OkStatus();
}

}  // namespace

// HostExecuteAsyncEvents

absl::StatusOr<HostExecuteAsyncEvents::HostExecuteEvent>
HostExecuteAsyncEvents::CreateEvent(se::StreamExecutor* executor,
                                    RunId run_id) {
  VLOG(6) << "Adding event for executor at address " << executor
          << " and event id " << run_id.ToInt();

  TF_ASSIGN_OR_RETURN(auto host_to_device_stream_event,
                      executor->CreateEvent());

  auto event = tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
      std::move(host_to_device_stream_event));

  auto [it, inserted] =
      events_.emplace(std::make_pair(executor, run_id), event);

  if (!inserted) {
    return FailedPrecondition(
        "Event already exists for executor at address %p and event id %d",
        executor, run_id.ToInt());
  }
  return event;
}

absl::StatusOr<HostExecuteAsyncEvents::HostExecuteEvent>
HostExecuteAsyncEvents::ExtractEvent(se::StreamExecutor* executor,
                                     RunId run_id) {
  VLOG(6) << "Extracting event for executor at address " << executor
          << " and event id " << run_id.ToInt();
  auto it = events_.find(std::make_pair(executor, run_id));
  if (it == events_.end()) {
    return FailedPrecondition(
        "Event does not exist for executor at address %p and event id %d",
        executor, run_id.ToInt());
  }
  auto event = std::move(it->second);
  events_.erase(it);
  return event;
}

// HostExecuteStartThunk

HostExecuteStartThunk::HostExecuteStartThunk(
    Thunk::ThunkInfo thunk_info, const HloModule& hlo_module,
    absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4> args,
    absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4> results,
    int64_t pointer_size)
    : Thunk(Thunk::Kind::kHostExecuteStart, std::move(thunk_info)),
      args_(std::move(args)),
      results_(std::move(results)),
      async_events_(std::make_shared<HostExecuteAsyncEvents>()),
      gpu_pointer_size_(pointer_size) {
  HostOffloadingExecutableProto host_offloading_executable_proto;
  *host_offloading_executable_proto.mutable_hlo_module() = hlo_module.ToProto();
  host_offloading_executable_proto.set_executable_type(
      HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);
  executable_proto_ = std::move(host_offloading_executable_proto);
}

std::string HostExecuteStartThunk::ToString(int indent) const { return ""; }

absl::StatusOr<ThunkProto> HostExecuteStartThunk::ToProto() const {
  return Unimplemented("Not implemented yet.");
}

absl::StatusOr<std::unique_ptr<HostExecuteStartThunk>>
HostExecuteStartThunk::FromProto(
    ThunkInfo thunk_info, const HostExecuteStartThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  return Unimplemented("Not implemented yet.");
}

static HostOffloadingAllocator* GetHostOffloadingAllocator(
    se::StreamExecutor* executor) {
  static HostOffloadingAllocator* allocator =
      CreateGpuHostOffloadingAllocator(executor).release();
  return allocator;
}

absl::Status HostExecuteStartThunk::Initialize(const InitializeParams& params) {
  if (!allocator_) {
    allocator_ = GetHostOffloadingAllocator(params.executor);
  }
  // NOTE(basioli): We load the executable here so that we don't get a deadlock
  // when locking llvm command line options.
  absl::Status initialization_status = absl::OkStatus();
  absl::call_once(executable_init_flag_, [this, &initialization_status]() {
    auto executable_or_status =
        HostOffloadingNanoRtExecutable::LoadFromProto(executable_proto_);
    initialization_status = executable_or_status.status();
    if (initialization_status.ok()) {
      executable_ = std::move(executable_or_status.value());
    }
  });

  return initialization_status;
}

absl::Status HostExecuteStartThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::Stream * compute_stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  // Host execute thunk enqueues compute on the device to host stream to allow
  // for the compute stream to be used for device compute.
  se::Stream* device_to_host_stream = params.device_to_host_stream;

  // Wait on the compute stream to finish before copying data to host since we
  // might need to wait for producers to be finished.
  TF_RETURN_IF_ERROR(device_to_host_stream->WaitFor(compute_stream));

  TF_ASSIGN_OR_RETURN(
      auto execute_event,
      async_events_->CreateEvent(params.host_to_device_stream->parent(),
                                 RunId(params.execution_id)));

  TF_ASSIGN_OR_RETURN(
      auto tmp_call_frame,
      HostExecuteCallFrame::Create(
          params.device_to_host_stream, params.host_to_device_stream,
          params.buffer_allocations, *allocator_, absl::MakeSpan(args_),
          absl::MakeSpan(results_), executable_->program_shape(),
          gpu_pointer_size_));

  // We are making a shared pointer here because `execute` needs to be
  // copyable so that it can be scheduled on the thread pool.
  auto call_frame =
      std::make_shared<HostExecuteCallFrame>(std::move(tmp_call_frame));

  auto execute = [this, call_frame = std::move(call_frame), params,
                  shared_execute_event = std::move(execute_event)]() mutable {
    tsl::profiler::TraceMe trace(
        "HostExecuteStartThunk::ExecuteOnStream::execute (host_callback)");
    HostOffloadingExecutable::ExecuteOptions execute_options{
        // TODO(basioli): add device index and context when/if needed.
        /*device_index =*/0,
        /*launch_id =*/static_cast<int32_t>(params.execution_id),
        /*context =*/nullptr};

    auto execute_event = executable_->Execute(
        call_frame->parameters(), call_frame->result(), execute_options);

    tsl::BlockUntilReady(execute_event);
    if (execute_event.IsError()) {
      shared_execute_event.SetError(execute_event.GetError());
      return;
    }
    auto publish_result_status = std::move(*call_frame).PublishResult();
    if (!publish_result_status.ok()) {
      shared_execute_event.SetError(publish_result_status);
      return;
    }
    auto record_event_status = params.host_to_device_stream->RecordEvent(
        shared_execute_event.get().get());
    if (!record_event_status.ok()) {
      shared_execute_event.SetError(record_event_status);
      return;
    }

    shared_execute_event.SetStateConcrete();
  };

  TF_RETURN_IF_ERROR(device_to_host_stream->DoHostCallbackWithStatus(
      [execute = std::move(execute)] {
        GetHostExecuteThreadPool()->Schedule(std::move(execute));
        return absl::OkStatus();
      }));

  return absl::OkStatus();
}

// HostExecuteDoneThunk

HostExecuteDoneThunk::HostExecuteDoneThunk(
    Thunk::ThunkInfo thunk_info,
    std::shared_ptr<HostExecuteAsyncEvents> async_events)
    : Thunk(Thunk::Kind::kHostExecuteDone, std::move(thunk_info)),
      async_events_(std::move(async_events)) {
  CHECK(async_events_) << "async_events must not be null";
}

std::string HostExecuteDoneThunk::ToString(int indent) const { return ""; }

absl::StatusOr<ThunkProto> HostExecuteDoneThunk::ToProto() const {
  return Unimplemented("Not implemented yet.");
}

absl::StatusOr<std::unique_ptr<HostExecuteDoneThunk>>
HostExecuteDoneThunk::FromProto(
    ThunkInfo thunk_info, const HostExecuteDoneThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  return Unimplemented("Not implemented yet.");
}

absl::Status HostExecuteDoneThunk::Initialize(const InitializeParams& params) {
  return absl::OkStatus();
}

absl::Status HostExecuteDoneThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  TF_ASSIGN_OR_RETURN(auto event, async_events_->ExtractEvent(
                                      params.host_to_device_stream->parent(),
                                      RunId(params.execution_id)));

  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  TF_RETURN_IF_ERROR(stream->WaitFor(event.get().get()));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
