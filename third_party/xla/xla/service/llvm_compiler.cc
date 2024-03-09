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

#include "xla/service/llvm_compiler.h"

#include "tsl/platform/denormal.h"

#ifdef __FAST_MATH__
#error "Don't build XLA with -ffast-math"
#endif

namespace xla {
absl::StatusOr<std::vector<std::unique_ptr<Executable>>> LLVMCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    const CompileOptions& options) {
  // Tensorflow tries to enable the following behaviors in all its threads:
  //
  //  - Denormals are zero (DAZ): roughly, operations treat denormal floats as
  //    zero.
  //  - Flush denormals to zero (FTZ): roughly, operations produce zero instead
  //    of denormal floats.
  //
  // In theory enabling these shouldn't matter since the compiler should ideally
  // not leak its environment into generated code, but we turn off DAZ and FTZ
  // to get some defense-in-depth.
  tsl::port::ScopedDontFlushDenormal dont_flush_denormals;

  std::vector<std::unique_ptr<Executable>> result;
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  for (size_t i = 0; i < modules.size(); i++) {
    TF_ASSIGN_OR_RETURN(modules[i], RunHloPasses(std::move(modules[i]),
                                                 stream_execs[i][0], options));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        RunBackend(std::move(modules[i]), stream_execs[i][0], options));
    result.push_back(std::move(executable));
  }

  return {std::move(result)};
}
}  // namespace xla
