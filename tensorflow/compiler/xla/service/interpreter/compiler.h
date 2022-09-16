/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_COMPILER_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/interpreter/platform_id.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/core/lib/core/status.h"

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

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& aot_options) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  se::Platform::Id PlatformId() const override;

 private:
  Status RunHloOptimization(HloModule* hlo_module);

  InterpreterCompiler(const InterpreterCompiler&) = delete;
  InterpreterCompiler& operator=(const InterpreterCompiler&) = delete;
};

}  // namespace interpreter
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_COMPILER_H_
