/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_COMPILER_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_COMPILER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"

namespace xla {
namespace ifrt {
namespace proxy {

class Compiler final : public llvm::RTTIExtends<Compiler, xla::ifrt::Compiler> {
 public:
  Compiler(xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper);

  absl::StatusOr<std::unique_ptr<xla::ifrt::LoadedExecutable>> Compile(
      std::unique_ptr<xla::ifrt::Program> program,
      std::unique_ptr<xla::ifrt::CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      std::unique_ptr<Program> program, const Topology& topology,
      std::unique_ptr<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<xla::ifrt::LoadedExecutable>>
  DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<xla::ifrt::DeserializeExecutableOptions> options)
      override;

  static char ID;  // NOLINT

 private:
  xla::ifrt::Client* client_;
  std::shared_ptr<RpcHelper> rpc_helper_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_COMPILER_H_
