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

#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

namespace gpu {

struct GetCodegenBackends {
  using Type = std::function<std::vector<std::unique_ptr<CodegenBackend>>(
      stream_executor::StreamExecutor*, const DebugOptions*, Compiler*,
      const Compiler::GpuTargetConfig*, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context)>;
};

struct GetFissionBackends {
  using Type = std::function<std::vector<std::unique_ptr<CodegenBackend>>(
      stream_executor::StreamExecutor*, const DebugOptions*, Compiler*,
      const Compiler::GpuTargetConfig*, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context)>;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_FACTORY_H_
