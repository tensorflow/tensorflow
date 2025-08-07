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
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
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
    absl::Span<se::DeviceMemoryBase const> buffers,
    absl::Span<Shape const> shapes) {
  CHECK_EQ(buffers.size(), shapes.size());
  std::vector<ExecutionInput> inputs;
  for (int i = 0; i < buffers.size(); ++i) {
    inputs.emplace_back(shapes.at(i));
    // Our executable doesn't have input-output aliasing, so we can pass
    // unowned input buffers.
    inputs.back().SetUnownedBuffer(
        /*index=*/{}, MaybeOwningDeviceMemory(/*unowned=*/buffers.at(i)));
  }
  return inputs;
}

}  // namespace

std::unique_ptr<GpuProfiler> GpuProfiler::Create(
    se::StreamExecutor* stream_executor, ProfileOptions options) {
  auto stream = stream_executor->CreateStream();
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorMemoryAllocator>(
          stream_executor);
  if (!stream.ok()) {
    LOG(ERROR) << "Failed to create stream: " << stream.status();
    return nullptr;
  }
  return absl::WrapUnique(new GpuProfiler(
      stream_executor,
      std::make_unique<stream_executor::StreamExecutorMemoryAllocator>(
          stream_executor),
      std::move(stream.value()), options));
}

absl::StatusOr<std::unique_ptr<InputBuffers>> GpuProfiler::CreateInputBuffers(
    const Executable* executable) {
  if (!executable->has_module()) {
    return absl::InvalidArgumentError(
        "Cannot create input buffers, the executable does not have an "
        "attatched HloModule.");
  }
  TF_ASSIGN_OR_RETURN(
      RedzoneBuffers buffers,
      RedzoneBuffers::FromComputation(
          *executable->module().entry_computation(), allocator_.get(),
          stream_.get(), RedzoneBuffers::BuffersToCreate::kAllInputs,
          options_.should_init_buffers,
          /*should_check_correctness=*/true, options_.redzone_padding_bytes));
  auto gpu_buffers = std::make_unique<GpuInputBuffers>();
  gpu_buffers->redzone_buffers = std::move(buffers);
  return gpu_buffers;
}

absl::StatusOr<ProfileResult> GpuProfiler::Profile(
    Executable* executable, const InputBuffers& buffers) {
  const GpuInputBuffers& gpu_buffers =
      tsl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
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

  if (options_.should_populate_output_buffer) {
    return ProfileResult{absl::Nanoseconds(profile.compute_time_ns()),
                         execution_output.Commit().ConsumeResult()};
  }
  return ProfileResult{absl::Nanoseconds(profile.compute_time_ns())};
}

absl::StatusOr<ExecutionOutput> GpuProfiler::Execute(
    Executable* executable, std::vector<ExecutionInput> inputs,
    ExecutionProfile* profile) {
  // Require exclusive GPU lock to prevent other runs during autotuning.
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_requires_exclusive_lock_on_gpu();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_->device_ordinal());
  run_options.set_stream(stream_.get());
  run_options.set_allocator(allocator_.get());
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
  BufferComparator comparator(output.on_device_shape(), rtol);

  TF_ASSIGN_OR_RETURN(
      bool outputs_match,
      comparator.CompareEqual(stream_.get(), output.root_buffer(),
                              reference.root_buffer()));
  if (outputs_match) {
    return absl::OkStatus();
  }
  return absl::InternalError("Output buffer does not match reference buffer.");
}

}  // namespace gpu

}  // namespace xla
