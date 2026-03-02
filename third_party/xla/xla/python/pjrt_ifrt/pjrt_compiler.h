/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {

class PjRtClient;

// Compiler that produces PjRt executables.
//
// TODO(hyeontaek): Move executable loading to `PjRtClient` and remove the
// requirement of `PjRtClient`, which will enable ahead-of-time compilation.
class PjRtCompiler final : public llvm::RTTIExtends<PjRtCompiler, Compiler> {
 public:
  PjRtCompiler(PjRtClient* client, int num_threads);

  // Compiler implementation.

  ~PjRtCompiler() override = default;

  tsl::Future<LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<Program> program,
      std::unique_ptr<CompileOptions> options) override;

  tsl::Future<ExecutableRef> Compile(
      std::unique_ptr<Program> program, const Topology& topology,
      std::unique_ptr<CompileOptions> options) override;

  absl::Status IsExecutableVersionCompatible(
      const xla::ifrt::ExecutableVersion& executable_version,
      const xla::ifrt::DeviceListRef& devices) const override {
    return absl::UnimplementedError("Not implemented");
  }

  tsl::Future<LoadedExecutableRef> DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<DeserializeExecutableOptions> options) override;

  static char ID;  // NOLINT

 private:
  PjRtClient* client_;
  std::optional<tsl::thread::ThreadPool> thread_pool_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
