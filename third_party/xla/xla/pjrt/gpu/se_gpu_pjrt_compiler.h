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

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {
// Implements the interfaces that are needed for the registered compiler.
class StreamExecutorGpuCompiler : public PjRtCompiler {
 public:
  // Constructs a compiler for the default "gpu" platform.
  explicit StreamExecutorGpuCompiler(PjRtPlatformId pjrt_platform_id)
      : pjrt_platform_id_(pjrt_platform_id) {}

  // Constructs a compiler for the given platform.
  explicit StreamExecutorGpuCompiler(PjRtPlatformId pjrt_platform_id,
                                     stream_executor::PlatformId platform_id);

  // Constructs a compiler with a given XLA compiler instance.
  StreamExecutorGpuCompiler(PjRtPlatformId pjrt_platform_id,
                            std::unique_ptr<Compiler> compiler);

  // Setting CompileOptions.TargetConfig field will trigger deviceless
  // compilation, which will not query the GPU attached to the machine.
  // In this case, the `client` argument could be left as `nullptr`.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, MaybeOwningMlirModule module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  PjRtPlatformId pjrt_platform_id() const { return pjrt_platform_id_; }

  // Returns the target runtime ABI version that the compiled executables will
  // be compatible with.
  absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
  GetTargetRuntimeAbiVersion() override;

 private:
  using LayoutCanonicalizationCallback =
      std::function<absl::StatusOr<std::pair<std::vector<Shape>, Shape>>(
          const HloModule& module)>;

  // Helper function for Compile above.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client,
      LayoutCanonicalizationCallback layout_callback);

  std::optional<stream_executor::Platform::Id> requested_platform_id_;
  mutable absl::Mutex compiler_mutex_;
  std::unique_ptr<Compiler> compiler_ ABSL_GUARDED_BY(compiler_mutex_);

  // Returns an instance of the compiler for the given platform (or the default
  // GPU platform if none is specified). If one does not exist, creates one. The
  // compiler is cached for subsequent calls.
  absl::StatusOr<Compiler*> GetOrCreateCompiler();

  PjRtPlatformId pjrt_platform_id_;
};
}  // namespace xla
#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
