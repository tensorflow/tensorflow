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

#ifndef TENSORFLOW_COMPILER_XLA_BACKENDS_GPU_AUTOTUNER_ROCM_FACTORY_H_
#define TENSORFLOW_COMPILER_XLA_BACKENDS_GPU_AUTOTUNER_ROCM_FACTORY_H_

#include <memory>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/rocblas.h"
#include "xla/backends/gpu/autotuner/triton.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using ::mlir::MLIRContext;

std::vector<std::unique_ptr<CodegenBackend>> GetCodegenBackendsForROCm(
    stream_executor::StreamExecutor* stream_executor,
    const DebugOptions* debug_options, Compiler* compiler,
    const Compiler::GpuTargetConfig* target_config, MLIRContext* mlir_context) {
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<TritonBackend>(
      debug_options, compiler, target_config, mlir_context));
  backends.push_back(std::make_unique<RocblasBackend>(
      stream_executor, debug_options, compiler, target_config));
  return backends;
}

std::vector<std::unique_ptr<CodegenBackend>> GetFissionBackendsForROCm(
    stream_executor::StreamExecutor* stream_executor,
    const DebugOptions* debug_options, Compiler* compiler,
    const Compiler::GpuTargetConfig* target_config, MLIRContext* mlir_context) {
  return {};
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(GetCodegenBackendsROCmRegistration,
                                           GetCodegenBackends,
                                           se::rocm::kROCmPlatformId,
                                           GetCodegenBackendsForROCm);
STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(GetFissionBackendsROCmRegistration,
                                           GetFissionBackends,
                                           se::rocm::kROCmPlatformId,
                                           GetFissionBackendsForROCm);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_BACKENDS_GPU_AUTOTUNER_ROCM_FACTORY_H_
