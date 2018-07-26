/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {
namespace {

using tensorflow::tracing::ScopedAnnotation;

}  // namespace

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(
    const string& ptx, const std::vector<uint8>& cubin,
    std::pair<int, int> compute_capability,
    std::unique_ptr<const ThunkSchedule> thunk_schedule,
    std::unique_ptr<const HloModule> hlo_module,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      ptx_(ptx),
      cubin_(cubin),
      compute_capability_(compute_capability),
      thunk_schedule_(std::move(thunk_schedule)),
      assignment_(std::move(assignment)) {}

Status GpuExecutable::ExecuteThunks(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();

  std::pair<int, int> stream_compute_compatibility;
  executor->GetDeviceDescription().cuda_compute_capability(
      &stream_compute_compatibility.first,
      &stream_compute_compatibility.second);
  TF_RET_CHECK(stream_compute_compatibility == compute_capability_)
      << "Compute capability mismatch; expected {" << compute_capability_.first
      << ", " << compute_capability_.second << "}, but was {"
      << stream_compute_compatibility.first << ", "
      << stream_compute_compatibility.second << "}";

  bool do_profile = hlo_execution_profile != nullptr;
  if (do_profile) {
    LOG(WARNING) << "PROFILING: profiling is enabled";
  }

  // Stream 0 indicates `main_stream` and substreams start from stream 1.
  std::vector<StreamPool::Ptr> sub_streams;
  sub_streams.reserve(thunk_schedule_->StreamCount() - 1);
  while (sub_streams.size() + 1 < thunk_schedule_->StreamCount()) {
    sub_streams.emplace_back();
    TF_ASSIGN_OR_RETURN(sub_streams.back(),
                        run_options->BorrowStream(executor->device_ordinal()));
  }

  HloExecutionProfiler profiler(do_profile, hlo_execution_profile, main_stream,
                                sub_streams, hlo_module_->entry_computation());
  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  // This top-level trace serves two purposes:
  //  1) It marks the scope of the whole XLA module.
  //  2) It tells us whether tracing is enabled.  We use this to avoid the
  //     expensive HloInstruction::ToString() calls inside the loop below if
  //     tracing is disabled.
  ScopedAnnotation top_level_annotation(hlo_module_->name(), "XLA GPU module");

  std::map<const Thunk*, std::unique_ptr<se::Event>> thunk_to_finish_event;
  for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
    // Annotate execution of this op if tracing was enabled when we started
    // running this module.  If tracing is enabled *while* we're running the
    // module, we won't get any data, but that's probably an OK trade-off.
    //
    // TODO(jlebar): Should we cache the results of HloInstruction::ToString(),
    // since we expect it to be an expensive call?
    tensorflow::gtl::optional<ScopedAnnotation> op_annotation;
    if (top_level_annotation.IsEnabled()) {
      op_annotation.emplace(
          thunk->hlo_instruction() != nullptr
              ? thunk->hlo_instruction()->ToString(HloPrintOptions::Canonical())
              : "<unknown>",
          "XLA op");
    }

    TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
    int32 stream_no =
        thunk_schedule_->StreamNumberForHlo(*thunk->hlo_instruction());
    se::Stream* stream =
        (stream_no == 0 ? main_stream : sub_streams[stream_no - 1].get());

    for (const Thunk* dependency : thunk_schedule_->DependsOn(thunk)) {
      stream->ThenWaitFor(FindOrDie(thunk_to_finish_event, dependency).get());
    }

    // If this thunk requests it, wait for all currently-executing thunks to
    // finish.  This is useful e.g. if the thunk is about to perform autotuning.
    if (thunk->ShouldHaltAllActivityBeforeRunning(stream)) {
      TF_RETURN_IF_ERROR(main_stream->BlockHostUntilDone());
    }

    VLOG(2) << "Executing the thunk for "
            << thunk->hlo_instruction()->ToString() << " on stream "
            << stream_no;
    TF_RETURN_IF_ERROR(
        thunk->ExecuteOnStream(buffer_allocations, stream, &profiler));
    if (thunk_schedule_->Depended(thunk)) {
      auto finish_event = MakeUnique<se::Event>(main_stream->parent());
      finish_event->Init();
      stream->ThenRecordEvent(finish_event.get());
      thunk_to_finish_event[thunk] = std::move(finish_event);
    }
  }

  main_stream->ThenWaitFor(&sub_streams);
  // Make sure kernels are completed before deallocating temporary buffers.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (block_host_until_done) {
    Status block_status = main_stream->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s",
          main_stream, block_status.error_message().c_str());
    }
  }

  profiler.FinishExecution();
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));

    // If hlo profiling was disabled then the cycle count is left empty.
    if (do_profile) {
      execution_profile_.set_compute_cycle_count(
          hlo_execution_profile->total_cycles_executed(
              *module().entry_computation()));
    }
  }

  return Status::OK();
}

StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();

  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  BufferAllocations::Builder buffer_allocations_builder;
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_entry_computation_parameter()) {
      auto param_no = allocation.parameter_number();
      se::DeviceMemoryBase buffer =
          arguments[param_no]->buffer(allocation.param_shape_index());

      // All top-level buffers and sub-buffers must have an explicit, non-null
      // pointer, except for zero-sized buffers, which may be null.
      if (buffer.is_null() && buffer.size() > 0) {
        return FailedPrecondition(
            "Cannot run XLA computation because pointer to (sub-)buffer at "
            "index %s of parameter %lld was null.  All pointers to "
            "(sub-)buffers must not be null, unless the (sub-)buffer has zero "
            "elements.",
            allocation.param_shape_index().ToString().c_str(), param_no);
      }

      buffer_allocations_builder.RegisterBuffer(i, buffer);
    }
  }
  se::StreamExecutor* executor = run_options->stream()->parent();
  TF_ASSIGN_OR_RETURN(
      auto buffer_allocations,
      buffer_allocations_builder.Build(
          assignment_.get(), executor->device_ordinal(), memory_allocator));

  bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();
  TF_RETURN_IF_ERROR(ExecuteThunks(run_options, *buffer_allocations,
                                   block_host_until_done,
                                   hlo_execution_profile));

  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  auto device_ordinal = executor->device_ordinal();
  ScopedShapedBuffer shaped_buffer(root->shape(), root->shape(),
                                   memory_allocator, device_ordinal);

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer.
  std::set<se::DeviceMemoryBase> buffers_in_result;
  TF_RETURN_IF_ERROR(shaped_buffer.buffers().ForEachMutableElementWithStatus(
      [&buffer_allocations, &buffers_in_result, &shaped_buffer, this](
          const ShapeIndex& index, se::DeviceMemoryBase* device_memory) {
        const auto& sources = this->GetRootPointsToSet().element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.size());
        auto src_hlo = sources[0]->instruction();

        VLOG(4) << "Looking at: " << sources[0];

        // The source instruction should have a non-parameter buffer
        // assigned.
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            this->assignment_->GetUniqueSlice(src_hlo, sources[0]->index()));
        CHECK(!slice.allocation()->is_entry_computation_parameter());

        se::DeviceMemoryBase src_base =
            buffer_allocations->GetDeviceAddress(slice.index());
        CHECK(!src_base.is_null() || src_base.size() == 0);
        *device_memory = src_base;
        buffers_in_result.insert(src_base);
        return Status::OK();
      }));
  TF_RETURN_IF_ERROR(buffer_allocations->TearDown(buffers_in_result));

  return std::move(shaped_buffer);
}

StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  // TODO(b/30671675): Implement asynchronous execution mode.
  return Unimplemented(
      "Asynchronous execution on stream is not yet supported on GPU.");
}

const PointsToSet& GpuExecutable::GetRootPointsToSet() const {
  return assignment_->points_to_analysis().GetPointsToSet(
      module().entry_computation()->root_instruction());
}

}  // namespace gpu
}  // namespace xla
