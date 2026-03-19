/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"

#include <memory>
#include <string>
#include <utility>

namespace tensorflow {

xla::CompileOptions GetPjRtCompileOptions(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result) {
  xla::CompileOptions pjrt_compile_options;
  pjrt_compile_options.argument_layouts = result.xla_input_shapes;
  pjrt_compile_options.executable_build_options =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);
  if (pjrt_compile_options.executable_build_options.num_replicas() > 1 ||
      pjrt_compile_options.executable_build_options.num_partitions() > 1) {
    // Compile executable for sharded program
    pjrt_compile_options.compile_portable_executable = false;
  } else {
    // Compile portable executable for single device compilation.
    pjrt_compile_options.compile_portable_executable = true;
  }
  return pjrt_compile_options;
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
PjRtDeviceCompilerClient::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result) {
  VLOG(2) << "Compiling to xla::PjRtLoadedExecutable.";

  TF_ASSIGN_OR_RETURN(
      auto executable,
      client_->CompileAndLoad(*result.computation,
                              GetPjRtCompileOptions(options, result)));

  VLOG(2) << "Compiled PJRT executable " << executable->name()
          << " num_replicas " << executable->num_replicas()
          << " num_partitions " << executable->num_partitions();

  return std::move(executable);
}

absl::StatusOr<std::string> PjRtDeviceCompilerClient::SerializeExecutable(
    const xla::PjRtLoadedExecutable& executable) {
  VLOG(1) << "Serializing xla::PjRtLoadedExecutable to string.";
  return executable.SerializeExecutable();
}

absl::StatusOr<std::string> PjRtDeviceCompilerClient::BuildSerializedExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result) {
  VLOG(1) << "PJRT currently doesn't support AOT compilation. Compiling to "
             "xla::PjRtLoadedExecutable and serializing it";
  TF_ASSIGN_OR_RETURN(auto executable, BuildExecutable(options, result));
  return executable->SerializeExecutable();
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
PjRtDeviceCompilerClient::LoadExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    const std::string& serialized_executable) {
  VLOG(1) << "Deserializing from string to xla::PjRtLoadedExecutable.";
  return client_->LoadSerializedExecutable(
      serialized_executable, GetPjRtCompileOptions(options, result),
      xla::LoadOptions());
}

void PjRtDeviceCompilerClient::WaitForProgramsToFinish() {
  // TODO(b/255826209): Modify this if PjRtClient exposes a function to wait for
  // programs to finish.
  LOG(INFO) << "Unimplemented: PJRT uses futures and waiting for programs to "
               "finish isn't necessary.";
}

}  // namespace tensorflow
