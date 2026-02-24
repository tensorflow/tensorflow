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

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/hipblaslt.h"
#include "xla/backends/gpu/autotuner/miopen.h"
#include "xla/backends/gpu/autotuner/rocblas.h"
#include "xla/backends/gpu/autotuner/triton.h"
#include "xla/hlo/analysis/alias_info.h"
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
    const Compiler::GpuTargetConfig* target_config, const AliasInfo* alias_info,
    MLIRContext* mlir_context,
    absl::Span<const autotuner::Backend> backend_allowlist) {
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<TritonBackend>(
      debug_options, compiler, target_config, alias_info, mlir_context));
  backends.push_back(std::make_unique<MIOpenBackend>(
      stream_executor, debug_options, compiler, target_config));
  backends.push_back(std::make_unique<RocblasBackend>(
      stream_executor, debug_options, compiler, target_config));
  backends.push_back(std::make_unique<HipblasLtBackend>(
      stream_executor, debug_options, compiler, target_config));
  if (!backend_allowlist.empty()) {
    backends.erase(
        std::remove_if(backends.begin(), backends.end(),
                       [&](const std::unique_ptr<CodegenBackend>& backend) {
                         return !absl::c_any_of(
                             backend_allowlist,
                             [&](autotuner::Backend backend_id) {
                               return backend->backend() == backend_id;
                             });
                       }),
        backends.end());
  }
  return backends;
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(GetCodegenBackendsROCmRegistration,
                                           GetCodegenBackends,
                                           se::rocm::kROCmPlatformId,
                                           GetCodegenBackendsForROCm);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_BACKENDS_GPU_AUTOTUNER_ROCM_FACTORY_H_
