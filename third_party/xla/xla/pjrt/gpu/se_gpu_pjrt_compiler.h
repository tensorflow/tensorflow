/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/status/status.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_target_config.h"

namespace xla {
// Implements the interfaces that are needed for the registered compiler.
class StreamExecutorGpuCompiler : public PjRtCompiler {
 public:
  // If `gpu_target_config` is nullopt, the compiler has to compile with device,
  // i.e. calling of `Compile` should depend on the passed-in client's runtime
  // device information.
  explicit StreamExecutorGpuCompiler(const std::optional<gpu::GpuTargetConfig>
                                         gpu_target_config = std::nullopt)
      : gpu_target_config_(gpu_target_config) {}
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

 private:
  // GpuTargetConfig is used by GPU compiler for ahead-of-time (AOT) compilation
  // without device.
  std::optional<gpu::GpuTargetConfig> gpu_target_config_;
};
}  // namespace xla
#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
