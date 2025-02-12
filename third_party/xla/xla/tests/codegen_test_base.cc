/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/codegen_test_base.h"

#include <memory>

namespace xla {

absl::StatusOr<std::unique_ptr<Executable>>
CodegenTestBase::CompileToExecutable(std::unique_ptr<HloModule> hlo_module,
                                     bool run_optimization_passes) {
  if (run_optimization_passes) {
    TF_ASSIGN_OR_RETURN(hlo_module, backend().compiler()->RunHloPasses(
                                        std::move(hlo_module),
                                        backend().default_stream_executor(),
                                        /*device_allocator=*/nullptr));
  }
  return backend().compiler()->RunBackend(std::move(hlo_module),
                                          backend().default_stream_executor(),
                                          /*device_allocator=*/nullptr);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CodegenTestBase::CompileToAotCompilationResult(
    std::unique_ptr<HloModule> hlo_module,
    const AotCompilationOptions& options) {
  auto module_group = std::make_unique<HloModuleGroup>(std::move(hlo_module));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> results,
      backend().compiler()->CompileAheadOfTime(std::move(module_group),
                                               options));
  return std::move(results.front());
}

}  // namespace xla
