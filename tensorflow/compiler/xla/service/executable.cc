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

#include "tensorflow/compiler/xla/legacy_flags/service_flags.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

StatusOr<std::vector<perftools::gputools::DeviceMemoryBase>>
Executable::ExecuteOnStreams(
    tensorflow::gtl::ArraySlice<const ServiceExecutableRunOptions> run_options,
    tensorflow::gtl::ArraySlice<
        tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>>
        arguments) {
  TF_RET_CHECK(run_options.size() == arguments.size());

  if (run_options.size() == 1) {
    TF_ASSIGN_OR_RETURN(auto result,
                        ExecuteOnStream(&run_options[0], arguments[0],
                                        /*hlo_execution_profile=*/nullptr));
    return std::vector<perftools::gputools::DeviceMemoryBase>({result});
  }

  std::vector<perftools::gputools::DeviceMemoryBase> return_values(
      run_options.size());
  for (tensorflow::gtl::ArraySlice<const ExecutableRunOptions>::size_type i = 0;
       i < run_options.size(); ++i) {
    // We cannot BlockHostUntilDone() on the already-launched executions in case
    // of error, since if the executions communicate, the initially launched
    // executions may never complete if not all executions are running.
    TF_ASSIGN_OR_RETURN(return_values[i],
                        ExecuteAsyncOnStream(&run_options[i], arguments[i]));
  }
  for (const auto& options : run_options) {
    TF_RET_CHECK(options.stream() != nullptr);
    options.stream()->BlockHostUntilDone();
  }
  return return_values;
}

Status Executable::DumpSessionModule() {
  TF_RET_CHECK(dumping());
  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  const string& directory_path = flags->xla_dump_executions_to;
  VersionedComputationHandle versioned_handle = entry_computation_handle();
  // This filename does not include the version number because the computation
  // is only ever executed at one version.
  string filename = tensorflow::strings::Printf(
      "computation_%lld__%s__execution_%lld", versioned_handle.handle.handle(),
      session_module_->entry().name().c_str(), ++execution_count_);
  return Executable::DumpToDirectory(directory_path, filename,
                                     *session_module_);
}

// Removes illegal characters from filenames.
static void SanitizeFilename(string* name) {
  for (char& c : *name) {
    if (c == '/' || c == '\\') {
      c = '_';
    }
  }
}

/* static */ Status Executable::DumpToDirectory(
    const string& directory_path, string filename,
    const SessionModule& session_module) {
  tensorflow::Env* env = tensorflow::Env::Default();
  if (!env->IsDirectory(directory_path).ok()) {
    TF_RETURN_IF_ERROR(env->CreateDir(directory_path));
  }
  SanitizeFilename(&filename);
  string file_path = tensorflow::io::JoinPath(directory_path, filename);
  return tensorflow::WriteBinaryProto(env, file_path, session_module);
}

}  // namespace xla
