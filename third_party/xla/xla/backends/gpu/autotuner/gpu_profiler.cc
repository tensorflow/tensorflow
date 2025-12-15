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

#include "xla/backends/gpu/autotuner/gpu_profiler.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla {

namespace gpu {

namespace {

std::vector<ExecutionInput> CreateExecutionInputsFromBuffers(
    absl::Span<se::DeviceAddressBase const> buffers,
    absl::Span<Shape const> shapes) {
  CHECK_EQ(buffers.size(), shapes.size());
  std::vector<ExecutionInput> inputs;
  for (int i = 0; i < buffers.size(); ++i) {
    inputs.emplace_back(shapes.at(i));
    // Our executable doesn't have input-output aliasing, so we can pass
    // unowned input buffers.
    inputs.back().SetUnownedBuffer(
        /*index=*/{}, MaybeOwningDeviceAddress(/*unowned=*/buffers.at(i)));
  }
  return inputs;
}

int GetScratchBytes(const Executable* executable) {
  int scratch_bytes = 0;
  for (const auto* allocation : executable->GetAllocations()) {
    if (allocation->IsPreallocatedTempBuffer()) {
      for (const auto& [buffer, offset] : allocation->assigned_buffers()) {
        // Scratch space is allocated as the second element in the output tuple
        // of the instruction.
        const auto& shape_index = buffer->positions().front().index;
        bool is_second_element_in_output_tuple =
            !shape_index.empty() && shape_index[0] == 1;
        if (is_second_element_in_output_tuple) {
          scratch_bytes += offset.size;
        }
      }
    }
  }
  return scratch_bytes;
}

}  // namespace

std::unique_ptr<GpuProfiler> GpuProfiler::Create(
    se::StreamExecutor* stream_executor, ProfileOptions options,
    se::DeviceAddressAllocator* external_allocator) {
  std::unique_ptr<se::DeviceAddressAllocator> owned_allocator;
  se::DeviceAddressAllocator* active_allocator = external_allocator;

  if (active_allocator == nullptr) {
    owned_allocator =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            stream_executor);
    active_allocator = owned_allocator.get();
  }

  // TODO(b/442997461): Create a new stream using
  // `stream_executor->CreateStream()` instead of reusing the allocator stream
  // once we can handle cuBLAS using multiple streams.
  auto stream = active_allocator->GetStream(stream_executor->device_ordinal());
  if (!stream.ok()) {
    LOG(ERROR) << "Failed to create stream: " << stream.status();
    return nullptr;
  }
  return absl::WrapUnique(new GpuProfiler(stream_executor, active_allocator,
                                          std::move(owned_allocator),
                                          stream.value(), options));
}

absl::StatusOr<std::unique_ptr<InputBuffers>> GpuProfiler::CreateInputBuffers(
    const Executable* executable) {
  TF_ASSIGN_OR_RETURN(
      RedzoneBuffers buffers,
      RedzoneBuffers::FromProgramShape(
          executable->compute_computation_layout().ComputeProgramShape(),
          RedzoneBuffers::BuffersToCreate::kAllInputs,
          options_.should_init_buffers,
          /*should_check_correctness=*/true, options_.redzone_padding_bytes,
          allocator_, stream_));
  auto gpu_buffers = std::make_unique<GpuInputBuffers>();
  gpu_buffers->redzone_buffers = std::move(buffers);
  return gpu_buffers;
}

absl::StatusOr<ProfileResult> GpuProfiler::Profile(
    Executable* executable, const InputBuffers& buffers) {
  const GpuInputBuffers& gpu_buffers =
      tsl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
  ProfileResult result;
  result.scratch_bytes = GetScratchBytes(executable);
  {
    // Warm up run.
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes());
    TF_RETURN_IF_ERROR(Execute(executable, std::move(execution_inputs),
                               /*profile=*/nullptr)
                           .status());

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
  }

  ExecutionProfile profile;
  profile.set_warmup_run_executed(true);
  std::vector<ExecutionInput> execution_inputs =
      CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                       rz_buffers.input_shapes());

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput execution_output,
      Execute(executable, std::move(execution_inputs), &profile));

  result.duration = absl::Nanoseconds(profile.compute_time_ns());
  ScopedShapedBuffer output_buffers = execution_output.Commit().ConsumeResult();
  if (output_buffers.on_device_shape().IsTuple() &&
      !output_buffers.on_device_shape().tuple_shapes().empty()) {
    result.output_buffer = output_buffers.TakeSubTree({0});
  } else {
    result.output_buffer = std::move(output_buffers);
  }

  return result;
}

absl::StatusOr<ExecutionOutput> GpuProfiler::Execute(
    Executable* executable, std::vector<ExecutionInput> inputs,
    ExecutionProfile* profile) {
  // Require exclusive GPU lock to prevent other runs during autotuning.
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_requires_exclusive_lock_on_gpu();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_->device_ordinal());
  run_options.set_stream(stream_);
  run_options.set_allocator(allocator_);
  run_options.set_gpu_executable_run_options(&gpu_opts);
  run_options.set_execution_profile(profile);
  ServiceExecutableRunOptions service_run_options(run_options);
  return executable->ExecuteAsyncOnStreamWrapper(&service_run_options,
                                                 std::move(inputs));
}

absl::Status GpuProfiler::CheckInputBuffers(InputBuffers& buffers) {
  if (options_.redzone_padding_bytes == 0) {
    return absl::OkStatus();
  }
  const GpuInputBuffers& gpu_buffers =
      tsl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
  TF_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
                      rz_buffers.RedzoneAllocator().CheckRedzones());
  if (rz_check_status.ok()) {
    return absl::OkStatus();
  }
  LOG(ERROR) << "Red zone modified";
  return absl::InternalError(rz_check_status.RedzoneFailureMsg());
}

absl::Status GpuProfiler::CheckOutputBuffer(ScopedShapedBuffer& output,
                                            ScopedShapedBuffer& reference,
                                            float rtol) {
  return ShapeUtil::ForEachLeafShapeWithStatus(
      reference.on_device_shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        BufferComparator comparator(subshape, rtol,
                                    /*verbose=*/false);

        TF_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream_, output.buffer(index),
                                    reference.buffer(index)));
        if (outputs_match) {
          return absl::OkStatus();
        }
        return absl::InternalError(
            "Output buffer does not match reference buffer.");
      });
}

}  // namespace gpu

}  // namespace xla
