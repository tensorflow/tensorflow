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
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

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

StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options, ExecutionProfile* profile,
    absl::Span<const ShapedBuffer* const> arguments) {
  se::Stream* stream = run_options->stream();
  std::unique_ptr<se::Timer> timer;
  if (profile != nullptr) {
    timer.reset(new se::Timer(stream->parent()));
    stream->InitTimer(timer.get()).ThenStartTimer(timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  std::unique_ptr<HloExecutionProfile> profile_ptr =
      module_config().debug_options().xla_hlo_profile() &&
              hlo_profiling_enabled()
          ? absl::make_unique<HloExecutionProfile>(&hlo_profile_printer_data(),
                                                   &hlo_profile_index_map())
          : nullptr;

  StatusOr<ScopedShapedBuffer> return_value =
      ExecuteOnStream(run_options, arguments, profile_ptr.get());
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
    VLOG(1) << "enqueueing 'stop timer' and blocking host until done...";
    stream->ThenStopTimer(timer.get());
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    VLOG(1) << "done with block-host-until-done";

    // Merge in run-time profile information from execution_profile.
    //
    // TODO(b/71713097): This is buggy -- even though the mutex takes care of
    // C++ level races, some other concurrent ExecuteOnStreamWrapper call could
    // have rewritten the execution_profile before we get to it.
    profile->MergeFrom(execution_profile());

    // Overall execution time (in nanoseconds) from the executor timer.
    if (stream->ok()) {
      // Don't read timer->Nanoseconds() if the stream isn't OK -- that's
      // illegal.
      profile->set_compute_and_transfer_time_ns(timer->Nanoseconds());
    }

    // TODO(b/28123297): On GPU we end up including transfer time in
    // the compute time this way. Instead, we should get the correct
    // value by measuring it. Setting the field here at least lets
    // benchmarks provide *some* value for GPU computations.
    //
    // TODO(b/28447609): The value in compute_and_transfer_time_ns is actually
    // the compute time without the transfer time, so this way we get the
    // correct compute time. We should instead have the correct value for
    // compute_and_transfer_time and set compute_time to the compute time.
    if (profile->compute_time_ns() == 0) {
      profile->set_compute_time_ns(profile->compute_and_transfer_time_ns());
    }

    const int64 executable_size_in_bytes = SizeInBytes();
    if (executable_size_in_bytes != 0) {
      profile->set_executable_size_in_bytes(executable_size_in_bytes);
    }
  }

  if (profile_ptr != nullptr) {
    XLA_LOG_LINES(
        tensorflow::INFO,
        profile_ptr->ToString(stream->parent()->GetDeviceDescription()));
  }

  return return_value;
}

int64 Executable::SizeInBytes() { return -1; }

}  // namespace xla
