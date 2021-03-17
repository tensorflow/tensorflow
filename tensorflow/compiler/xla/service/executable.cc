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

#include "tensorflow/compiler/xla/service/executable.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/device_description.h"

namespace xla {

ExecutionInput::~ExecutionInput() {
  for (auto& index : unowned_indices_) {
    auto buffer = buffers_.mutable_element(index)->Release();
    if (buffer) {
      buffer->Release();
    }
  }
}

Status ExecutionInput::SetDynamicShape(Shape dynamic_shape) {
  const Shape& input_shape = shape();
  if (!ShapeUtil::DynamicShapeIsCompatible(input_shape, dynamic_shape)) {
    return tensorflow::errors::InvalidArgument(
        "Cannot set dynamic shape: ", input_shape.DebugString(), " vs. ",
        dynamic_shape.DebugString());
  }
  dynamic_shape_ = absl::make_unique<Shape>(std::move(dynamic_shape));
  return Status::OK();
}

void ExecutionInput::SetUnownedBuffer(const ShapeIndex& index,
                                      MaybeOwningDeviceMemory buffer) {
  *buffers_.mutable_element(index) = std::move(buffer);
  unowned_indices_.insert(index);
}

StatusOr<ShapedBuffer> ExecutionInput::ToShapedBuffer(
    se::DeviceMemoryAllocator* allocator, int device_ordinal) const {
  const Shape& input_shape = shape();
  ShapedBuffer shaped_buffer(input_shape, device_ordinal);
  for (const auto& index_buffer : Buffers()) {
    const tensorflow::se::OwningDeviceMemory* mem =
        index_buffer.second.AsOwningDeviceMemory();
    if (mem != nullptr && (mem->allocator() != allocator ||
                           mem->device_ordinal() != device_ordinal)) {
      return tensorflow::errors::InvalidArgument(
          "Device buffer at index ", index_buffer.first.ToString(),
          " has mismatching allocator/device");
    }
    shaped_buffer.set_buffer(index_buffer.second.AsDeviceMemoryBase(),
                             index_buffer.first);
  }
  return std::move(shaped_buffer);
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  StatusOr<ScopedShapedBuffer> result =
      ExecuteAsyncOnStream(run_options, arguments, hlo_execution_profile);
  Status blocking_status = run_options->stream()->BlockHostUntilDone();
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

StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  std::vector<ExecutionInput> args;
  args.reserve(arguments.size());
  for (const ShapedBuffer* arg : arguments) {
    args.emplace_back(MakeMaybeOwningDeviceMemoryTree(*arg));
  }
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStream(run_options, std::move(args),
                                           hlo_execution_profile));
  return out.ConsumeResult();
}

StatusOr<ExecutionOutput> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  StatusOr<ExecutionOutput> result = ExecuteAsyncOnStream(
      run_options, std::move(arguments), hlo_execution_profile);
  Status blocking_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(blocking_status);
  return result;
}

StatusOr<std::vector<ScopedShapedBuffer>> Executable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments) {
  TF_RET_CHECK(run_options.size() == arguments.size());

  std::vector<ScopedShapedBuffer> return_values;
  return_values.reserve(run_options.size());

  if (run_options.size() == 1) {
    TF_ASSIGN_OR_RETURN(auto rv,
                        ExecuteOnStream(&run_options[0], arguments[0],
                                        /*hlo_execution_profile=*/nullptr));
    return_values.push_back(std::move(rv));
    return std::move(return_values);
  }

  for (size_t i = 0; i < run_options.size(); ++i) {
    // We cannot BlockHostUntilDone() on the already-launched executions in case
    // of error, since if the executions communicate, the initially launched
    // executions may never complete if not all executions are running.
    TF_ASSIGN_OR_RETURN(
        auto rv, ExecuteAsyncOnStream(&run_options[i], arguments[i],
                                      /*hlo_execution_profile=*/nullptr));
    return_values.push_back(std::move(rv));
  }
  for (const auto& options : run_options) {
    TF_RET_CHECK(options.stream() != nullptr);
    TF_RETURN_IF_ERROR(options.stream()->BlockHostUntilDone());
  }
  return std::move(return_values);
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  StatusOr<ScopedShapedBuffer> result =
      ExecuteAsyncOnStreamWrapper(run_options, arguments);
  Status block_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(block_status);
  return result;
}

StatusOr<ExecutionOutput> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  StatusOr<ExecutionOutput> result =
      ExecuteAsyncOnStreamWrapper(run_options, std::move(arguments));
  Status block_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(block_status);
  return result;
}

struct ExecuteAsyncOnStreamWrapperState {
  ExecutionProfile* profile;
  std::shared_ptr<se::Timer> timer;
  std::shared_ptr<HloExecutionProfile> profile_ptr;
};

static ExecuteAsyncOnStreamWrapperState ExecuteWrapperBeforeExecution(
    const Executable& executable,
    const ServiceExecutableRunOptions* run_options) {
  ExecuteAsyncOnStreamWrapperState state;
  se::Stream* stream = run_options->stream();
  state.profile = run_options->run_options().execution_profile();
  if (state.profile != nullptr) {
    state.timer = std::make_shared<se::Timer>(stream->parent());
    stream->InitTimer(state.timer.get()).ThenStartTimer(state.timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  state.profile_ptr =
      executable.module_config().debug_options().xla_hlo_profile() &&
              executable.hlo_profiling_enabled()
          ? std::make_shared<HloExecutionProfile>(
                &executable.hlo_profile_printer_data(),
                &executable.hlo_profile_index_map())
          : nullptr;
  return state;
}

Status ExecuteWrapperAfterExecution(
    Executable* executable, const ExecuteAsyncOnStreamWrapperState& state,
    Status return_status, se::Stream* stream) {
  if (!return_status.ok()) {
    if (state.profile != nullptr) {
      // Ensure the ThenStartTimer call has completed before we destroy timer.
      // We already have a failure status to return, so just log this if it
      // fails.
      Status status = stream->BlockHostUntilDone();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to BlockHostUntilDone: " << status;
      }
    }
    return return_status;
  }

  if (state.profile != nullptr) {
    VLOG(1) << "enqueueing 'stop timer' and profiling callback...";
    stream->ThenStopTimer(state.timer.get());

    // We block instead of using an async callback because reading the timer
    // value may call back into the driver on GPU, which is not allowed.
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    const int64 executable_size_in_bytes =
        executable->SizeOfGeneratedCodeInBytes();
    // Merge in run-time profile information from execution_profile.

    // Overall execution time (in nanoseconds) from the executor timer.
    state.profile->set_compute_and_transfer_time_ns(state.timer->Nanoseconds());

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

  if (executable->module_config().debug_options().xla_hlo_profile() &&
      state.profile_ptr != nullptr) {
    DumpToFileInDir(executable->module(), /*file_prefix=*/"",
                    /*file_suffix=*/"hlo_execution_profile_data",
                    state.profile_ptr->ToProto().SerializeAsString());
  }

  if (state.profile_ptr != nullptr) {
    const se::DeviceDescription* device_description =
        &stream->parent()->GetDeviceDescription();
    std::shared_ptr<HloExecutionProfile> profile = state.profile_ptr;
    stream->ThenDoHostCallback([profile, device_description]() {
      XLA_LOG_LINES(tensorflow::INFO,
                    profile->ToString(device_description->clock_rate_ghz()));
    });
  }

  return return_status;
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  auto state = ExecuteWrapperBeforeExecution(*this, run_options);
  StatusOr<ScopedShapedBuffer> return_value =
      ExecuteAsyncOnStream(run_options, arguments, state.profile_ptr.get());
  TF_RETURN_IF_ERROR(ExecuteWrapperAfterExecution(
      this, state, return_value.status(), run_options->stream()));
  return return_value;
}

StatusOr<ExecutionOutput> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  auto state = ExecuteWrapperBeforeExecution(*this, run_options);
  StatusOr<ExecutionOutput> return_value = ExecuteAsyncOnStream(
      run_options, std::move(arguments), state.profile_ptr.get());
  TF_RETURN_IF_ERROR(ExecuteWrapperAfterExecution(
      this, state, return_value.status(), run_options->stream()));
  return return_value;
}

int64 Executable::SizeOfGeneratedCodeInBytes() const { return -1; }

void Executable::MarkToBeReleasedArguments(absl::Span<ExecutionInput> arguments,
                                           ExecutionOutput& result) {
  for (ExecutionInput& argument : arguments) {
    for (auto& index_buffer : *argument.MutableBuffers()) {
      if (absl::optional<se::OwningDeviceMemory> maybe_owning_buffer =
              index_buffer.second.Release()) {
        result.AddToBeReleased(std::move(*maybe_owning_buffer));
      }
    }
  }
}

}  // namespace xla
