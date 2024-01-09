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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_COMPILER_CLIENT_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_COMPILER_CLIENT_H_

#include <memory>
#include <optional>
#include <string>

#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "xla/client/local_client.h"

namespace tensorflow {

class XlaDeviceCompilerClient
    : public DeviceCompilerClient<xla::LocalExecutable, xla::LocalClient> {
 public:
  explicit XlaDeviceCompilerClient(xla::LocalClient* client)
      : client_(client) {}

  StatusOr<std::unique_ptr<xla::LocalExecutable>> BuildExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) override;

  // Returns a serialized AOT result obtained by exporting the available
  // `executable` using the XlaCompiler.
  StatusOr<std::string> SerializeExecutable(
      const xla::LocalExecutable& executable) override;

  // Returns a serialized AOT result obtained by compiling `result` into an AOT
  // result.
  StatusOr<std::string> BuildSerializedExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) override;

  // Loads a serialized AOT result (`serialized_executable`) into an
  // xla::LocalExecutable and returns it.
  StatusOr<std::unique_ptr<xla::LocalExecutable>> LoadExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result,
      const std::string& serialized_executable) override;

  void WaitForProgramsToFinish() override;

  xla::LocalClient* client() const override { return client_; }

 private:
  xla::LocalClient* const client_;

  XlaDeviceCompilerClient(const XlaDeviceCompilerClient&) = delete;
  void operator=(const XlaDeviceCompilerClient&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_COMPILER_CLIENT_H_
