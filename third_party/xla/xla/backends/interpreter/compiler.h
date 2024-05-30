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

#ifndef XLA_BACKENDS_INTERPRETER_COMPILER_H_
#define XLA_BACKENDS_INTERPRETER_COMPILER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/interpreter/platform_id.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status.h"

namespace xla {
namespace interpreter {

// Despite the inherited "compiler" naming, InterpreterCompiler does not
// perform any lowering as other backends do. It operates at HLO-level for
// and is responsible for generating an InterpreterExecutable.
// Refer to interpreter/README.md for more.
class InterpreterCompiler : public Compiler {
 public:
  InterpreterCompiler() {}
  ~InterpreterCompiler() override {}

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& aot_options) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  se::Platform::Id PlatformId() const override;

 private:
  absl::Status RunHloOptimization(HloModule* hlo_module);

  InterpreterCompiler(const InterpreterCompiler&) = delete;
  InterpreterCompiler& operator=(const InterpreterCompiler&) = delete;
};

}  // namespace interpreter
}  // namespace xla

#endif  // XLA_BACKENDS_INTERPRETER_COMPILER_H_
