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
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {
namespace {

// A helper class for profiling HLO in the course of GPU program execution.
// All of the profiling is guarded internally, to avoid the caller needing to
// have lots of conditionals sprinkled around.
class HloExecutionProfiler {
 public:
  // If profiling is enabled, start an execution timer running.
  explicit HloExecutionProfiler(bool do_profile, HloExecutionProfile* profile,
                                se::Stream* stream)
      : do_profile_(do_profile), profile_(profile), stream_(stream) {
    if (do_profile_) {
      clock_rate_ghz_ =
          stream->parent()->GetDeviceDescription().clock_rate_ghz();
      execution_timer_.reset(new se::Timer(stream->parent()));
      per_op_timer_.reset(new se::Timer(stream->parent()));
      stream->InitTimer(execution_timer_.get())
          .ThenStartTimer(execution_timer_.get());
      stream->InitTimer(per_op_timer_.get());
    }
  }

  // If profiling is enabled, sets the total cycle count on the profile from the
  // execution timer.
  ~HloExecutionProfiler() {
    if (do_profile_) {
      stream_->ThenStopTimer(execution_timer_.get());
      stream_->BlockHostUntilDone();
      profile_->set_total_cycles_executed(execution_timer_->Nanoseconds() *
                                          clock_rate_ghz_);
    }
  }

  // If profiling is enabled, starts the per-operation timer.
  void StartOperation() {
    if (do_profile_) {
      stream_->ThenStartTimer(per_op_timer_.get());
    }
  }

  // If profiling is enabled, stops the per-operation timer and records the time
  // that the hlo_instruction took to execute in the profile.
  void FinishOperation(const HloInstruction* hlo_instruction) {
    if (do_profile_) {
      stream_->ThenStopTimer(per_op_timer_.get());
      stream_->BlockHostUntilDone();
      profile_->AddProfileResult(
          hlo_instruction, per_op_timer_->Nanoseconds() * clock_rate_ghz_);
    }
  }

 private:
  const bool do_profile_;
  double clock_rate_ghz_;
  HloExecutionProfile* profile_;
  se::Stream* stream_;
  std::unique_ptr<se::Timer> execution_timer_;
  std::unique_ptr<se::Timer> per_op_timer_;
};

}  // namespace

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(tensorflow::StringPiece ptx,
                             std::unique_ptr<ThunkSchedule> thunk_schedule,
                             std::unique_ptr<HloModule> hlo_module,
                             std::unique_ptr<HloModuleConfig> module_config,
                             std::unique_ptr<BufferAssignment> assignment)
    : Executable(std::move(hlo_module), std::move(module_config)),
      ptx_(ptx),
      thunk_schedule_(std::move(thunk_schedule)),
      assignment_(std::move(assignment)) {}

Status GpuExecutable::ExecuteThunks(
    se::Stream* main_stream, const BufferAllocations& buffer_allocations,
    HloExecutionProfile* hlo_execution_profile) {
  bool do_profile = hlo_execution_profile != nullptr;
  if (do_profile) {
    LOG(WARNING) << "PROFILING: profiling is enabled";
  }
  HloExecutionProfiler profiler(do_profile, hlo_execution_profile, main_stream);

  std::vector<std::unique_ptr<se::Stream>> sub_streams;
  // Stream 0 indicates `main_stream` and substreams start from stream 1.
  for (int32 i = 1; i < thunk_schedule_->StreamCount(); ++i) {
    auto sub_stream = MakeUnique<se::Stream>(main_stream->parent());
    sub_stream->Init();
    sub_streams.emplace_back(std::move(sub_stream));
  }

  std::map<const Thunk*, std::unique_ptr<se::Event>> thunk_to_finish_event;
  for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
    TF_RETURN_IF_ERROR(thunk->Initialize(*this));
    int32 stream_no =
        thunk_schedule_->StreamNumberForHlo(*thunk->hlo_instruction());
    se::Stream* stream =
        (stream_no == 0 ? main_stream : sub_streams[stream_no - 1].get());

    for (const Thunk* dependency : thunk_schedule_->DependsOn(thunk)) {
      stream->ThenWaitFor(FindOrDie(thunk_to_finish_event, dependency).get());
    }

    profiler.StartOperation();
    VLOG(2) << "Executing the thunk for "
            << thunk->hlo_instruction()->ToString();
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(buffer_allocations, stream));
    if (thunk_schedule_->Depended(thunk)) {
      auto finish_event = MakeUnique<se::Event>(main_stream->parent());
      finish_event->Init();
      stream->ThenRecordEvent(finish_event.get());
      thunk_to_finish_event[thunk] = std::move(finish_event);
    }
    profiler.FinishOperation(thunk->hlo_instruction());
  }

  main_stream->ThenWaitFor(&sub_streams);
  // Make sure kernels are completed before deallocating temporary buffers.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (!main_stream->BlockHostUntilDone()) {
    return InternalError("Failed to complete all kernels launched on stream %p",
                         main_stream);
  }

  return Status::OK();
}

StatusOr<se::DeviceMemoryBase> GpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  // This ExecuteOnStream overload should only be called if has_hybrid_result is
  // false.
  TF_RET_CHECK(!module_config().has_hybrid_result());

  BufferAllocations::Builder buffer_allocations_builder;
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_entry_computation_parameter()) {
      buffer_allocations_builder.RegisterBuffer(
          i, arguments[allocation.parameter_number()]);
    }
  }
  se::StreamExecutor* executor = stream->parent();
  TF_ASSIGN_OR_RETURN(
      auto buffer_allocations,
      buffer_allocations_builder.Build(*assignment_, executor->device_ordinal(),
                                       memory_allocator));

  TF_RETURN_IF_ERROR(
      ExecuteThunks(stream, *buffer_allocations, hlo_execution_profile));

  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice output_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  se::DeviceMemoryBase output_buffer_address =
      buffer_allocations->GetDeviceAddress(output_slice.index());

  if (ShapeUtil::IsTuple(root->shape())) {
    std::set<se::DeviceMemoryBase> referred_by_output;
    if (GetRootPointsToSet().IsAmbiguous()) {
      // The points-to set of the root is ambiguous so we need to examine the
      // result data to determine which buffers are contained in the result.
      TF_ASSIGN_OR_RETURN(
          TransferManager * transfer_manager,
          TransferManager::GetForPlatform(executor->platform()));
      TF_ASSIGN_OR_RETURN(referred_by_output,
                          transfer_manager->GatherBufferPointersFromTuple(
                              executor, output_buffer_address, root->shape()));
    } else {
      // The points-to set of the root is unambiguous so it's known statically
      // which buffers are in the result. Gather these buffers using the root's
      // points-to set.
      TF_RETURN_IF_ERROR(GetRootPointsToSet().ForEachElement(
          [&referred_by_output, &buffer_allocations, this](
              const ShapeIndex& /*index*/, bool /*is_leaf*/,
              const std::vector<const LogicalBuffer*>& buffers) {
            // The points to set is unambiguous so the set should be a
            // singleton. That is, we know exactly which instruction produced
            // the array at this element.
            CHECK_EQ(1, buffers.size());
            HloInstruction* hlo = buffers[0]->instruction();
            TF_ASSIGN_OR_RETURN(
                const BufferAllocation::Slice slice,
                this->assignment_->GetUniqueSlice(hlo, buffers[0]->index()));
            CHECK(!slice.allocation()->is_entry_computation_parameter());
            referred_by_output.insert(
                buffer_allocations->GetDeviceAddress(slice.index()));
            return Status::OK();
          }));
    }
    TF_RETURN_IF_ERROR(
        buffer_allocations->TearDown(referred_by_output, *assignment_));
  } else {
    // If the computation result is not a tuple, we can delete all temporary
    // buffers that are not the output.
    TF_RETURN_IF_ERROR(
        buffer_allocations->TearDown({output_buffer_address}, *assignment_));
  }
  return output_buffer_address;
}

StatusOr<std::unique_ptr<ShapedBuffer>> GpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  // This ExecuteOnStream overload should only be called by the LocalService
  // which sets has_hybrid_result to true.
  TF_RET_CHECK(module_config().has_hybrid_result());

  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  BufferAllocations::Builder buffer_allocations_builder;
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_entry_computation_parameter()) {
      auto param_no = allocation.parameter_number();
      if (ShapeUtil::IsTuple(arguments[param_no]->shape())) {
        return Unimplemented("Tuple ShapedBuffer arguments not supported");
      }
      buffer_allocations_builder.RegisterBuffer(
          i, arguments[param_no]->buffer(/*index=*/{}));
    }
  }
  se::StreamExecutor* executor = stream->parent();
  TF_ASSIGN_OR_RETURN(
      auto buffer_allocations,
      buffer_allocations_builder.Build(*assignment_, executor->device_ordinal(),
                                       memory_allocator));

  TF_RETURN_IF_ERROR(
      ExecuteThunks(stream, *buffer_allocations, hlo_execution_profile));

  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  auto device_ordinal = executor->device_ordinal();
  TF_ASSIGN_OR_RETURN(auto shaped_buffer,
                      ShapedBuffer::MakeShapedBuffer(
                          root->shape(), executor->platform(), device_ordinal));

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer.
  std::set<se::DeviceMemoryBase> buffers_in_result;
  TF_RETURN_IF_ERROR(
      shaped_buffer->mutable_shape_index_to_buffer_entry()
          ->ForEachMutableElement(
              [&buffer_allocations, &buffers_in_result, &shaped_buffer, this](
                  const ShapeIndex& index, bool is_leaf, size_t* buffer_entry) {
                if (is_leaf) {
                  const std::vector<const LogicalBuffer*>& sources =
                      this->GetRootPointsToSet().element(index);
                  // The points to set is unambiguous so the set should be a
                  // singleton. That is, we know exactly which instruction
                  // produced the array at this element.
                  CHECK_EQ(1, sources.size());
                  auto src_hlo = sources[0]->instruction();

                  VLOG(4) << "Looking at: " << sources[0];

                  // The source instruction should have a non-parameter buffer
                  // assigned.
                  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                                      this->assignment_->GetUniqueSlice(
                                          src_hlo, sources[0]->index()));
                  CHECK(!slice.allocation()->is_entry_computation_parameter());

                  perftools::gputools::DeviceMemoryBase src_base =
                      buffer_allocations->GetDeviceAddress(slice.index());
                  CHECK(!src_base.is_null() || src_base.size() == 0);
                  shaped_buffer->mutable_buffers()->push_back(src_base);
                  *buffer_entry = shaped_buffer->mutable_buffers()->size() - 1;

                  buffers_in_result.insert(src_base);
                }
                return Status::OK();
              }));
  TF_RETURN_IF_ERROR(
      buffer_allocations->TearDown(buffers_in_result, *assignment_));

  return std::move(shaped_buffer);
}

Status GpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    ShapedBuffer* result_buffer, HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  // This ExecuteOnStream overload should only be called by the LocalService
  // which sets has_hybrid_result to true.
  TF_RET_CHECK(module_config().has_hybrid_result());

  // Every array element in the result of the computation must be unambiguously
  // produced by a single instruction.
  // This ensures that the buffers inside result_buffer can be assigned without
  // conflict to the respective instructions because there is a one-to-one
  // correspondence between hlo instructions and array buffers in the result.
  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented(
        "Points-to set of root instruction is ambiguous or not distinct");
  }

  DCHECK(ShapeUtil::Compatible(result_buffer->shape(), result_shape()));

  BufferAllocations::Builder buffer_allocations_builder;
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_entry_computation_parameter()) {
      auto param_no = allocation.parameter_number();
      if (ShapeUtil::IsTuple(arguments[param_no]->shape())) {
        return Unimplemented("Tuple ShapedBuffer arguments not supported");
      }
      buffer_allocations_builder.RegisterBuffer(
          i, arguments[param_no]->buffer(/*index=*/{}));
    }
  }

  // If two tuple elements point to the same buffer, one of the results in the
  // result buffer is considered the canonical location while the other result
  // points to it (instead of, say, making a copy of the result).
  // buffer_index_to_shape_index maps a buffer index to its canonical location
  // in the result buffer.
  std::unordered_map<BufferAllocation::Index, size_t>
      buffer_index_to_shape_index;

  // Register DeviceMemoryBase values in result_buffer to their corresponding
  // buffer indices. These buffers will not be allocated in the call to
  // BufferAllocationsBuilder::Build.
  std::set<se::DeviceMemoryBase> buffers_in_result;
  TF_RETURN_IF_ERROR(
      result_buffer->mutable_shape_index_to_buffer_entry()
          ->ForEachMutableElement(
              [&buffer_allocations_builder, &buffers_in_result,
               &buffer_index_to_shape_index, result_buffer, this](
                  const ShapeIndex& index, bool is_leaf, size_t* buffer_entry) {
                if (is_leaf) {
                  const std::vector<const LogicalBuffer*>& sources =
                      this->GetRootPointsToSet().element(index);
                  // The points to set is unambiguous so the set should be a
                  // singleton. That is, we know exactly which instruction
                  // produced the array at this element.
                  CHECK_EQ(1, sources.size());
                  auto src_hlo = sources[0]->instruction();

                  VLOG(4) << "Looking at: " << sources[0];

                  // The source instruction should have a non-parameter buffer
                  // assigned.
                  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                                      this->assignment_->GetUniqueSlice(
                                          src_hlo, sources[0]->index()));
                  CHECK(!slice.allocation()->is_entry_computation_parameter());

                  auto insert_result = buffer_index_to_shape_index.emplace(
                      slice.index(), *buffer_entry);
                  if (insert_result.second) {
                    // The points-to set is distinct so this buffer should not
                    // have been assigned in a previous invocation of this
                    // lambda.
                    perftools::gputools::DeviceMemoryBase memory_base =
                        result_buffer->buffer(index);
                    CHECK(!memory_base.is_null());
                    buffer_allocations_builder.RegisterBuffer(slice.index(),
                                                              memory_base);
                    buffers_in_result.insert(memory_base);
                  } else {
                    // Record the fact that this tuple element is identical to
                    // some
                    // prior result.
                    *buffer_entry = insert_result.first->second;
                  }
                }
                return Status::OK();
              }));

  se::StreamExecutor* executor = stream->parent();
  auto device_ordinal = executor->device_ordinal();
  TF_ASSIGN_OR_RETURN(auto buffer_allocations,
                      buffer_allocations_builder.Build(
                          *assignment_, device_ordinal, memory_allocator));

  TF_RETURN_IF_ERROR(
      ExecuteThunks(stream, *buffer_allocations, hlo_execution_profile));

  return buffer_allocations->TearDown(buffers_in_result, *assignment_);
}

StatusOr<se::DeviceMemoryBase> GpuExecutable::ExecuteAsyncOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
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
