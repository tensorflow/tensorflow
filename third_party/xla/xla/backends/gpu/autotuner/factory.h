/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_FACTORY_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_FACTORY_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

namespace gpu {

// A factory function for getting the codegen backends for a given platform.
// The backend allowlist is a list of backend names (from BackendName enum)
// that are allowed to be returned. If the list is empty, all backends are
// returned.
struct GetCodegenBackends {
  using Type = std::function<std::vector<std::unique_ptr<CodegenBackend>>(
      stream_executor::StreamExecutor*, const DebugOptions*, Compiler*,
      const Compiler::GpuTargetConfig*, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context,
      absl::Span<const autotuner::Backend> backend_allowlist)>;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_FACTORY_H_
