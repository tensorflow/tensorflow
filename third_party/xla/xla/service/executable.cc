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

#include "xla/service/executable.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace xla {

ExecutionInput::~ExecutionInput() {
  for (auto& index : unowned_indices_) {
    auto buffer = buffers_.mutable_element(index)->Release();
    if (buffer) {
      buffer->Release();
    }
  }
}

absl::Status ExecutionInput::SetDynamicShape(Shape dynamic_shape) {
  const Shape& input_shape = shape();
  if (!ShapeUtil::DynamicShapeIsCompatible(input_shape, dynamic_shape)) {
    return tsl::errors::InvalidArgument(
        "Cannot set dynamic shape: ", input_shape.ToString(), " vs. ",
        dynamic_shape.ToString());
  }
  dynamic_shape_ = std::make_unique<Shape>(std::move(dynamic_shape));
  return absl::OkStatus();
}

void ExecutionInput::SetUnownedBuffer(const ShapeIndex& index,
                                      MaybeOwningDeviceMemory buffer) {
  *buffers_.mutable_element(index) = std::move(buffer);
  unowned_indices_.insert(index);
}

absl::StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  absl::StatusOr<ScopedShapedBuffer> result =
      ExecuteAsyncOnStream(run_options, arguments);
  absl::Status blocking_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(blocking_status);
  return result;
}

static ExecutionInput MakeMaybeOwningDeviceMemoryTree(
    const ShapedBuffer& shaped_buffer) {
  ExecutionInput result(shaped_buffer.on_device_shape());
  shaped_buffer.buffers().ForEachElement(
      [&](const ShapeIndex& index, const se::DeviceMemoryBase& mem) {
        result.SetBuffer(index, MaybeOwningDeviceMemory(mem));
      });
  return result;
}

absl::StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  std::vector<ExecutionInput> args;
  args.reserve(arguments.size());
  for (const ShapedBuffer* arg : arguments) {
    args.emplace_back(MakeMaybeOwningDeviceMemoryTree(*arg));
  }
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStream(run_options, std::move(args)));
  return out.ConsumeResult();
}

absl::StatusOr<ExecutionOutput> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  absl::StatusOr<ExecutionOutput> result =
      ExecuteAsyncOnStream(run_options, std::move(arguments));
  absl::Status blocking_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(blocking_status);
  return result;
}

absl::StatusOr<std::vector<ScopedShapedBuffer>> Executable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments) {
  TF_RET_CHECK(run_options.size() == arguments.size());

  std::vector<ScopedShapedBuffer> return_values;
  return_values.reserve(run_options.size());

  if (run_options.size() == 1) {
    TF_ASSIGN_OR_RETURN(auto rv,
                        ExecuteOnStream(&run_options[0], arguments[0]));
    return_values.push_back(std::move(rv));
    return std::move(return_values);
  }

  for (size_t i = 0; i < run_options.size(); ++i) {
    // We cannot BlockHostUntilDone() on the already-launched executions in case
    // of error, since if the executions communicate, the initially launched
    // executions may never complete if not all executions are running.
    TF_ASSIGN_OR_RETURN(auto rv,
                        ExecuteAsyncOnStream(&run_options[i], arguments[i]));
    return_values.push_back(std::move(rv));
  }
  for (const auto& options : run_options) {
    TF_RET_CHECK(options.stream() != nullptr);
    TF_RETURN_IF_ERROR(options.stream()->BlockHostUntilDone());
  }
  return std::move(return_values);
}

absl::StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  absl::StatusOr<ScopedShapedBuffer> result =
      ExecuteAsyncOnStreamWrapper(run_options, arguments);
  absl::Status block_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(block_status);
  return result;
}

absl::StatusOr<ExecutionOutput> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  absl::StatusOr<ExecutionOutput> result =
      ExecuteAsyncOnStreamWrapper(run_options, std::move(arguments));
  absl::Status block_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(block_status);
  return result;
}

struct ExecuteAsyncOnStreamWrapperState {
  ExecutionProfile* profile;
};

static ExecuteAsyncOnStreamWrapperState ExecuteWrapperBeforeExecution(
    const Executable& executable,
    const ServiceExecutableRunOptions* run_options) {
  ExecuteAsyncOnStreamWrapperState state;
  state.profile = run_options->run_options().execution_profile();

  VLOG(1) << "enqueueing executable on stream...";
  return state;
}

absl::Status ExecuteWrapperAfterExecution(
    Executable* executable, const ExecuteAsyncOnStreamWrapperState& state,
    absl::Status return_status, se::Stream* stream) {
  if (!return_status.ok()) {
    if (state.profile != nullptr) {
      absl::Status status = stream->BlockHostUntilDone();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to BlockHostUntilDone: " << status;
      }
    }
    return return_status;
  }

  if (state.profile != nullptr) {
    // We block instead of using an async callback because reading the timer
    // value may call back into the driver on GPU, which is not allowed.
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    const int64_t executable_size_in_bytes =
        executable->SizeOfGeneratedCodeInBytes();
    // Merge in run-time profile information from execution_profile.

    // TODO(b/28447609): The value in compute_and_transfer_time_ns is actually
    // the compute time without the transfer time, so this way we get the
    // correct compute time. We should instead have the correct value for
    // compute_and_transfer_time and set compute_time to the compute time.
    if (state.profile->compute_time_ns() == 0) {
      state.profile->set_compute_time_ns(
          state.profile->compute_and_transfer_time_ns());
    }

    if (executable_size_in_bytes != 0) {
      state.profile->set_executable_size_in_bytes(executable_size_in_bytes);
    }
  }

  return return_status;
}

absl::StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  auto state = ExecuteWrapperBeforeExecution(*this, run_options);
  absl::StatusOr<ScopedShapedBuffer> return_value =
      ExecuteAsyncOnStream(run_options, arguments);
  TF_RETURN_IF_ERROR(ExecuteWrapperAfterExecution(
      this, state, return_value.status(), run_options->stream()));
  return return_value;
}

absl::StatusOr<ExecutionOutput> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  auto state = ExecuteWrapperBeforeExecution(*this, run_options);
  absl::StatusOr<ExecutionOutput> return_value =
      ExecuteAsyncOnStream(run_options, std::move(arguments));
  TF_RETURN_IF_ERROR(ExecuteWrapperAfterExecution(
      this, state, return_value.status(), run_options->stream()));
  return return_value;
}

int64_t Executable::SizeOfGeneratedCodeInBytes() const { return -1; }

void Executable::MarkToBeReleasedArguments(absl::Span<ExecutionInput> arguments,
                                           ExecutionOutput& result) {
  for (ExecutionInput& argument : arguments) {
    for (auto& index_buffer : *argument.MutableBuffers()) {
      if (std::optional<se::OwningDeviceMemory> maybe_owning_buffer =
              index_buffer.second.Release()) {
        result.AddToBeReleased(std::move(*maybe_owning_buffer));
      }
    }
  }
}

}  // namespace xla
