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

#include "tensorflow/compiler/jit/xla_device_compiler_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/local_client.h"

namespace tensorflow {
namespace {
std::vector<const xla::Shape*> GetShapePointers(
    absl::Span<const xla::Shape> shapes) {
  std::vector<const xla::Shape*> shape_ptrs;
  shape_ptrs.reserve(shapes.size());
  for (const auto& shape : shapes) {
    shape_ptrs.push_back(&shape);
  }
  return shape_ptrs;
}
}  // namespace

StatusOr<std::unique_ptr<xla::LocalExecutable>>
XlaDeviceCompilerClient::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result) {
  VLOG(2) << "Compiling to xla::LocalExecutable.";

  std::vector<const xla::Shape*> argument_layouts =
      GetShapePointers(result.xla_input_shapes);
  xla::ExecutableBuildOptions build_options = GetExecutableBuildOptions(
      options, result, client_->default_device_ordinal());
  TF_ASSIGN_OR_RETURN(
      auto executables,
      client_->Compile(*result.computation, argument_layouts, build_options));
  TF_RET_CHECK(executables.size() == 1);
  return std::move(executables[0]);
}

StatusOr<std::string> XlaDeviceCompilerClient::SerializeExecutable(
    const xla::LocalExecutable& executable) {
  if (executable.executable() == nullptr) {
    return errors::FailedPrecondition(
        "Executable not found for serialization.");
  }

  VLOG(1)
      << "Exporting xla::LocalExecutable as an xla::AotCompilationResult and "
         "serializing it to string.";
  xla::Compiler* compiler = client_->backend().compiler();
  auto exported = compiler->Export(executable.executable());
  if (exported.ok()) {
    return (*exported)->SerializeAsString();
  }

  return exported.status();
}

StatusOr<std::string> XlaDeviceCompilerClient::BuildSerializedExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result) {
  VLOG(2) << "Compiling to xla::AotCompilationResult and serializing it";

  std::vector<const xla::Shape*> argument_layouts =
      GetShapePointers(result.xla_input_shapes);
  xla::ExecutableBuildOptions build_options = GetExecutableBuildOptions(
      options, result, client_->default_device_ordinal());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::AotCompilationResult>> aot_results,
      client_->CompileAheadOfTime(*result.computation, argument_layouts,
                                  build_options));
  TF_RET_CHECK(aot_results.size() == 1);
  return aot_results[0]->SerializeAsString();
}

StatusOr<std::unique_ptr<xla::LocalExecutable>>
XlaDeviceCompilerClient::LoadExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    const std::string& serialized_executable) {
  VLOG(2) << "Loading xla::LocalExecutable from a serialized "
             "xla::AotCompilationResult";

  xla::ExecutableBuildOptions build_options = GetExecutableBuildOptions(
      options, result, client_->default_device_ordinal());
  return client_->Load(serialized_executable, build_options);
}

void XlaDeviceCompilerClient::WaitForProgramsToFinish() {
  for (auto* executor : client_->backend().stream_executors()) {
    bool ok = executor->SynchronizeAllActivity();
    if (!ok) {
      LOG(ERROR) << "Error synchronizing activity while waiting for all "
                    "programs to complete";
    }
  }
}

}  // namespace tensorflow
