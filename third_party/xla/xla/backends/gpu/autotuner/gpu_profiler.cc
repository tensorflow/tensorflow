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
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

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

absl::StatusOr<std::vector<ProfileResult>>
GpuProfiler::ProfileWithSharedBuffers(
    std::vector<std::unique_ptr<Executable>> executables) {
  std::vector<ProfileResult> results;
  if (executables.empty()) {
    return results;
  }
  TF_ASSIGN_OR_RETURN(
      RedzoneBuffers buffers,
      RedzoneBuffers::FromComputation(
          *executables[0]->module().entry_computation(), allocator_.get(),
          stream_.get(), RedzoneBuffers::BuffersToCreate::kAllInputsAllOutputs,
          options_.should_init_buffers,
          /*should_check_correctness=*/true, options_.redzone_padding_bytes));
  for (auto& executable : executables) {
    TF_ASSIGN_OR_RETURN(ProfileResult result,
                        ProfileInternal(executable.get(), buffers));
    results.push_back(std::move(result));
  }
  return results;
}

absl::StatusOr<ProfileResult> GpuProfiler::ProfileInternal(
    Executable* executable, RedzoneBuffers& buffers) {
  {
    // Warm up run.
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(buffers.input_buffers(),
                                         buffers.input_shapes());
    TF_RETURN_IF_ERROR(Execute(executable, std::move(execution_inputs),
                               /*profile=*/nullptr)
                           .status());

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
  }

  ExecutionProfile profile;
  profile.set_warmup_run_executed(true);
  std::vector<ExecutionInput> execution_inputs =
      CreateExecutionInputsFromBuffers(buffers.input_buffers(),
                                       buffers.input_shapes());

  TF_RETURN_IF_ERROR(
      Execute(executable, std::move(execution_inputs), &profile).status());

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

}  // namespace gpu

}  // namespace xla
