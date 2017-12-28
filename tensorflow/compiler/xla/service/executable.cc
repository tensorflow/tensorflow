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

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"

using tensorflow::gtl::ArraySlice;

namespace xla {

StatusOr<std::vector<std::unique_ptr<ShapedBuffer>>>
Executable::ExecuteOnStreams(
    ArraySlice<const ServiceExecutableRunOptions> run_options,
    ArraySlice<ArraySlice<const ShapedBuffer*>> arguments) {
  TF_RET_CHECK(run_options.size() == arguments.size());

  std::vector<std::unique_ptr<ShapedBuffer>> return_values(run_options.size());

  if (run_options.size() == 1) {
    TF_ASSIGN_OR_RETURN(return_values[0],
                        ExecuteOnStream(&run_options[0], arguments[0],
                                        /*hlo_execution_profile=*/nullptr));
    return std::move(return_values);
  }

  for (size_t i = 0; i < run_options.size(); ++i) {
    // We cannot BlockHostUntilDone() on the already-launched executions in case
    // of error, since if the executions communicate, the initially launched
    // executions may never complete if not all executions are running.
    TF_ASSIGN_OR_RETURN(return_values[i],
                        ExecuteAsyncOnStream(&run_options[i], arguments[i]));
  }
  for (const auto& options : run_options) {
    TF_RET_CHECK(options.stream() != nullptr);
    TF_RETURN_IF_ERROR(options.stream()->BlockHostUntilDone());
  }
  return std::move(return_values);
}

StatusOr<std::unique_ptr<ShapedBuffer>> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options, ExecutionProfile* profile,
    ArraySlice<const ShapedBuffer*> arguments) {
  perftools::gputools::Stream* stream = run_options->stream();
  std::unique_ptr<perftools::gputools::Timer> timer;
  if (profile != nullptr) {
    timer.reset(new perftools::gputools::Timer(stream->parent()));
    stream->InitTimer(timer.get()).ThenStartTimer(timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  std::unique_ptr<HloExecutionProfile> profile_ptr =
      module_config().debug_options().xla_hlo_profile() &&
              hlo_profiling_enabled()
          ? MakeUnique<HloExecutionProfile>(&hlo_profile_printer(),
                                            &hlo_profile_index_map())
          : nullptr;

  StatusOr<std::unique_ptr<ShapedBuffer>> return_value =
      ExecuteOnStream(run_options, arguments, profile_ptr.get());

  if (profile != nullptr) {
    VLOG(1) << "enqueueing 'stop timer' and blocking host until done...";
    stream->ThenStopTimer(timer.get());
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    VLOG(1) << "done with block-host-until-done";

    // Merge in run-time profile information from execution_profile.
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
  }

  if (profile_ptr != nullptr) {
    XLA_LOG_LINES(
        tensorflow::INFO,
        profile_ptr->ToString(stream->parent()->GetDeviceDescription()));
    hlo_graph_dumper::MaybeDumpHloModule(module(), "Service::Execute",
                                         profile_ptr.get());
  }

  return return_value;
}

Status Executable::DumpSessionModule() {
  TF_RET_CHECK(dumping());
  const string& directory_path =
      module_config().debug_options().xla_dump_executions_to();
  VersionedComputationHandle versioned_handle = entry_computation_handle();
  // This filename does not include the version number because the computation
  // is only ever executed at one version.
  string filename = tensorflow::strings::Printf(
      "computation_%lld__%s__execution_%lld", versioned_handle.handle.handle(),
      session_module_->entry().name().c_str(), ++execution_count_);
  return Executable::DumpToDirectory(directory_path, filename,
                                     *session_module_);
}

/* static */ Status Executable::DumpToDirectory(
    const string& directory_path, string filename,
    const SessionModule& session_module) {
  tensorflow::Env* env = tensorflow::Env::Default();
  if (!env->IsDirectory(directory_path).ok()) {
    // NB! CreateDir does not work reliably with multiple XLA threads -- two
    // threads can race to observe the absence of the dump directory and
    // simultaneously try to create it, causing the "losing" thread to get a
    // "directory already exists" error.
    TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(directory_path));
  }
  filename = SanitizeFileName(std::move(filename));
  string file_path = tensorflow::io::JoinPath(directory_path, filename);
  string result;
  TF_RET_CHECK(
      tensorflow::SerializeToStringDeterministic(session_module, &result));
  return tensorflow::WriteStringToFile(tensorflow::Env::Default(), file_path,
                                       result);
}

}  // namespace xla
