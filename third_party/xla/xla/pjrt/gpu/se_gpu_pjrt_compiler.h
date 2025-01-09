/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
#define XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/stream_executor/platform.h"

namespace xla {
// Implements the interfaces that are needed for the registered compiler.
class StreamExecutorGpuCompiler : public PjRtCompiler {
 public:
  // Constructs a compiler for the default "gpu" platform.
  explicit StreamExecutorGpuCompiler() = default;

  // Constructs a compiler for the given platform.
  explicit StreamExecutorGpuCompiler(stream_executor::Platform::Id platform_id);

  // Setting CompileOptions.TargetConfig field will trigger deviceless
  // compilation, which will not query the GPU attached to the machine.
  // In this case, the `client` argument could be left as `nullptr`.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

 private:
  std::optional<stream_executor::Platform::Id> requested_platform_id_;
};
}  // namespace xla
#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
