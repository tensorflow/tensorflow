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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILER_CLIENT_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILER_CLIENT_H_

#include <optional>
#include <string>
#include <variant>

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/executable_build_options.h"

namespace tensorflow {

template <typename ExecutableType, typename ClientType>
class DeviceCompilerClient {
 public:
  DeviceCompilerClient() = default;
  virtual ~DeviceCompilerClient() = default;

  // Compiles `result` (HLO) to an `ExecutableType` using `ClientType` and
  // returns it.
  virtual StatusOr<std::unique_ptr<ExecutableType>> BuildExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) = 0;

  // Serializes an available `executable` to string using `ClientType` and
  // returns it.
  virtual absl::StatusOr<std::string> SerializeExecutable(
      const ExecutableType& executable) = 0;

  // Compiles `result` (HLO) to a serializable executable (eg.
  // xla::AotCompilationResult) using `ClientType`, serializes it to string and
  // returns it.
  virtual absl::StatusOr<std::string> BuildSerializedExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) = 0;

  // Loads `serialized_executable` into an `ExecutableType` using `ClientType`.
  virtual StatusOr<std::unique_ptr<ExecutableType>> LoadExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result,
      const std::string& serialized_executable) = 0;

  // Waits for the underlying `ClientType` backend's programs to finish
  // executing before returning.
  virtual void WaitForProgramsToFinish() = 0;

  virtual ClientType* client() const = 0;

 private:
  DeviceCompilerClient(const DeviceCompilerClient&) = delete;
  void operator=(const DeviceCompilerClient&) = delete;
};

// Generates the ExecutableBuildOptions for compilation from HLO to
// executable.
xla::ExecutableBuildOptions GetExecutableBuildOptions(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result, int default_device_ordinal);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_COMPILER_CLIENT_H_
