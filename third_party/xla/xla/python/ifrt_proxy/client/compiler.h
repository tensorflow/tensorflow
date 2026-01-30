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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

class Compiler final : public llvm::RTTIExtends<Compiler, xla::ifrt::Compiler> {
 public:
  using xla::ifrt::Compiler::Compile;

  Compiler(xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper);

  tsl::Future<xla::ifrt::LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<xla::ifrt::Program> program,
      std::unique_ptr<xla::ifrt::CompileOptions> options) override;

  tsl::Future<xla::ifrt::ExecutableRef> Compile(
      std::unique_ptr<Program> program, const Topology& topology,
      std::unique_ptr<CompileOptions> options) override;

  absl::Status IsExecutableVersionCompatible(
      const xla::ifrt::ExecutableVersion& executable_version,
      const xla::ifrt::DeviceListRef& devices) const override {
    return absl::UnimplementedError("Not implemented");
  }

  tsl::Future<xla::ifrt::LoadedExecutableRef> DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<xla::ifrt::DeserializeExecutableOptions> options)
      override;

  static char ID;  // NOLINT

 private:
  absl::StatusOr<xla::ifrt::LoadedExecutableRef> CreateExecutableFromResponse(
      std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
          loaded_host_callbacks,
      xla::ifrt::UserContextRef user_context,
      std::shared_ptr<CompileResponse> response);

  xla::ifrt::Client* client_;
  std::shared_ptr<RpcHelper> rpc_helper_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_COMPILER_H_
