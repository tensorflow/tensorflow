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
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/device_description.h"

namespace xla {

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

StatusOr<ExecutionOutput> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ShapeTree<xla::MaybeOwningDeviceMemory>> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  StatusOr<ExecutionOutput> result = ExecuteAsyncOnStream(
      run_options, std::move(arguments), hlo_execution_profile);
  Status blocking_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(blocking_status);
  return result;
}

StatusOr<ExecutionOutput> Executable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* /*run_options*/,
    std::vector<ShapeTree<xla::MaybeOwningDeviceMemory>> /*arguments*/,
    HloExecutionProfile* /*hlo_execution_profile*/) {
  return Unimplemented(
      "MaybeOwningDeviceMemory version of overload is not implemented ");
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

StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  se::Stream* stream = run_options->stream();
  std::shared_ptr<se::Timer> timer;
  ExecutionProfile* profile = run_options->run_options().execution_profile();
  if (profile != nullptr) {
    timer = std::make_shared<se::Timer>(stream->parent());
    stream->InitTimer(timer.get()).ThenStartTimer(timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  std::shared_ptr<HloExecutionProfile> profile_ptr =
      module_config().debug_options().xla_hlo_profile() &&
              hlo_profiling_enabled()
          ? std::make_shared<HloExecutionProfile>(&hlo_profile_printer_data(),
                                                  &hlo_profile_index_map())
          : nullptr;

  StatusOr<ScopedShapedBuffer> return_value =
      ExecuteAsyncOnStream(run_options, arguments, profile_ptr.get());
  if (!return_value.status().ok()) {
    if (profile != nullptr) {
      // Ensure the ThenStartTimer call has completed before we destroy timer.
      // We already have a failure status to return, so just log this if it
      // fails.
      Status status = stream->BlockHostUntilDone();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to BlockHostUntilDone: " << status;
      }
    }
    return return_value.status();
  }

  if (profile != nullptr) {
    VLOG(1) << "enqueueing 'stop timer' and profiling callback...";
    stream->ThenStopTimer(timer.get());

    // We block instead of using an async callback because reading the timer
    // value may call back into the driver on GPU, which is not allowed.
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    const int64 executable_size_in_bytes = SizeOfGeneratedCodeInBytes();
    // Merge in run-time profile information from execution_profile.

    // Overall execution time (in nanoseconds) from the executor timer.
    profile->set_compute_and_transfer_time_ns(timer->Nanoseconds());

    // TODO(b/28447609): The value in compute_and_transfer_time_ns is actually
    // the compute time without the transfer time, so this way we get the
    // correct compute time. We should instead have the correct value for
    // compute_and_transfer_time and set compute_time to the compute time.
    if (profile->compute_time_ns() == 0) {
      profile->set_compute_time_ns(profile->compute_and_transfer_time_ns());
    }

    if (executable_size_in_bytes != 0) {
      profile->set_executable_size_in_bytes(executable_size_in_bytes);
    }
  }

  const auto& dump_path = module_config().debug_options().xla_dump_to();
  if (module_config().debug_options().xla_hlo_profile() &&
      profile_ptr != nullptr && !dump_path.empty()) {
    const std::string full_path =
        tensorflow::io::JoinPath(dump_path, "hlo_execution_profile_data");
    TF_CHECK_OK(tensorflow::WriteStringToFile(
        tensorflow::Env::Default(), full_path,
        profile_ptr->ToProto().SerializeAsString()))
        << "Error saving HloExecutionProfileData to " << full_path;
  }

  if (profile_ptr != nullptr) {
    const se::DeviceDescription* device_description =
        &stream->parent()->GetDeviceDescription();
    stream->ThenDoHostCallback([profile_ptr, device_description]() {
      XLA_LOG_LINES(tensorflow::INFO,
                    profile_ptr->ToString(*device_description));
    });
  }

  return return_value;
}

int64 Executable::SizeOfGeneratedCodeInBytes() { return -1; }

}  // namespace xla
